import math
import numpy as np
import os
import torch
import abc

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from collections import OrderedDict
from utils.torch_jit_utils import *
from tasks.base.vec_task import VecTask
from types import SimpleNamespace
from omegaconf import OmegaConf
from collections import deque
from typing import Deque, Dict, Tuple, List
from scipy.spatial.transform import Rotation as R
import enum

from pathlib import Path
import nvisii as nv
from PIL import Image

# ################### #
# Dimensions of robot #
# ################### #


class FrankaDimensions(enum.Enum):
    """
    Dimensions of the Franka with gripper robot.

    """
    # general state
    # cartesian position + quaternion orientation
    PoseDim = 7
    # linear velocity + angular velcoity
    VelocityDim = 6

    # Width 
    WidthDim = 1

    # position of keypoint
    KeypointDim = 3
    TwoDimensionKeypointDim = 2

    # gripper state: pose
    StateDim = 7
    # force + torque
    WrenchDim = 6
    NumFingers = 2
    # for all joints
    JointPositionDim = 9 # the number of joint
    JointVelocityDim = 9 # the number of joint
    JointTorqueDim = 9 # the number of joint
    # joint w/o gripper
    JointWithoutGripperDim = 7

    # generalized coordinates
    GeneralizedCoordinatesDim = JointPositionDim
    GeneralizedVelocityDim = JointVelocityDim
    # for objects
    ObjectPoseDim = 7
    ObjectVelocityDim = 6


class Object:
    def __init__(self, size: Tuple[float, float, float]):
        """Initialize the cuboidal object.

        Args:
            size: The size of the object along x, y, z in meters. 
        """
        self._size = size

    """
    Properties
    """

    @property
    def size(self) -> Tuple[float, float, float]:
        """
        Returns the dimensions of the cuboid object (x, y, z) in meters.
        """
        return self._size

    """
    Configurations
    """

    @size.setter
    def size(self, size: Tuple[float, float, float]):
        """ Set size of the object.

        Args:
            size: The size of the object along x, y, z in meters. 
        """
        self._size = size


class Domain(VecTask):
    """
    Simple card object on the table
    """
    # constants
    # directory where assets for the simulator are present
    _assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
    
    # dimensions of the system
    _dims = FrankaDimensions
    _state_history_len = 2

    #Define limiatation of each component
    _gripper_limits: dict = {
        "gripper_position": SimpleNamespace(
            low=np.array([-1, -1, 0], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32)
        ),
        "gripper_orientation": SimpleNamespace(
            low=-np.ones(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32)
        ),
        "gripper_velocity": SimpleNamespace(
            low=np.full(_dims.VelocityDim.value, -3, dtype=np.float32),
            high=np.full(_dims.VelocityDim.value, 3, dtype=np.float32)
        )
    }
    _object_limits: dict = {
        "position": SimpleNamespace(
            low=np.array([-1, -1, 0], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32)
        ),
        "orientation": SimpleNamespace(
            low=-np.ones(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            default=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        ),
        "velocity": SimpleNamespace(
            low=np.full(_dims.VelocityDim.value, -2, dtype=np.float32),
            high=np.full(_dims.VelocityDim.value, 2, dtype=np.float32),
            default=np.zeros(_dims.VelocityDim.value, dtype=np.float32)
        ),
        "2Dkeypoint": SimpleNamespace(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([320, 240], dtype=np.float32) #TODO: make this to be changed by the config file
        )
    }
    # gripper links state list([num. of instances, num. of fingers, 13]) where 13: (x, y, z, quat, v, omega)
    # The length of list is the history of the state: 0: t, 1: t-1, 2: t-2, ... step.
    _grippers_frames_state_history: Deque[torch.Tensor] = deque(maxlen=_state_history_len)
    # Object prim state [num. of instances, 13] where 13: (x, y, z, quat, v, omega)
    # The length of list is the history of the state: 0: t, 1: t-1, 2: t-2, ... step.
    _object_state_history: Deque[torch.Tensor] = deque(maxlen=_state_history_len)
   
    def __init__(self, cfg, sim_device, graphics_device_id, headless, use_state=False, gym=None):
        """Initializes the card environment configure the buffers.

        Args:
            cfg: Dictionory containing the configuration.
            sim_device: Torch device to store created buffers at (cpu/gpu).
            graphics_device_id: device to render the objects
            headless: if True, it will open the GUI, otherwise, it will just run the server.
        """
        # load default config
        self.cfg = cfg
        self.asymmetric_obs = use_state
        self._set_table_dimension(self.cfg)
        self._object_dims: Object = self._set_object_dimension(self.cfg["env"]["geometry"]["object"])

        # action_dim = 19(residual eef(6) + grriper position(1) + gains(12))
        if self.cfg["env"]["restrict_gripper"]:
            self.action_dim = 20 if self.cfg["env"]["controller"] == "JP" else 18
        else:
            self.action_dim = 19

        self.keypoints_num = int(self.cfg['env']['keypoint']['num'])

        # observations
        self.obs_spec = {
            # robot joint
            "robot_q": self._dims.GeneralizedCoordinatesDim.value,
            # robot joint velocity
            "robot_u": self._dims.GeneralizedVelocityDim.value,
            # object position represented as 2D kepoints
            "object_q": (self._dims.TwoDimensionKeypointDim.value * self.keypoints_num),
            # object goal position represented as 2D kepoints
            "object_q_des": (self._dims.TwoDimensionKeypointDim.value * self.keypoints_num),
            # hand pose
            "hand_state": self._dims.ObjectPoseDim.value,
            # previous action
            "command": self.action_dim
        }

        # states
        if self.asymmetric_obs:
            self.state_spec = self.obs_spec
            self.obs_spec = {
            # robot joint
            "robot_q": self._dims.GeneralizedCoordinatesDim.value,
            # robot joint velocity
            "robot_u": self._dims.GeneralizedVelocityDim.value,
            # object position represented as 2D kepoints
            "object_q": (self._dims.TwoDimensionKeypointDim.value * self.keypoints_num),
            # object goal position represented as 2D kepoints
            "object_q_des": (self._dims.TwoDimensionKeypointDim.value * self.keypoints_num),
            # # gripper (finger) poses
            # "gripper_state": (self._dims.NumFingers.value * self._dims.StateDim.value),
            # hand pose
            "hand_state": self._dims.ObjectPoseDim.value,
            # previous action
            "command": self.action_dim
            }

        # actions
        self.action_spec = {
            "command": self.action_dim
        }

        # student observations
        self.stud_obs_spec = {
            # end-effector pose
            'end-effector_pose': self._dims.PoseDim.value,
            # width
            'width': self._dims.WidthDim.value,
            # joint position (without gripper)
            'joint_position_wo_gripper': self._dims.JointWithoutGripperDim.value,
            # joint velocity (without gripper)
            'joint_velocity_wo_gripper': self._dims.JointWithoutGripperDim.value,
            # previous action
            "command": self.action_dim
        }
        self.cfg["env"]["numObservations"] = sum(self.obs_spec.values())
        if self.asymmetric_obs:
            self.cfg["env"]["numStates"] = sum(self.state_spec.values())
        self.cfg["env"]["numActions"] = sum(self.action_spec.values())
        self.cfg["env"]["numStudentObservations"] = sum(self.stud_obs_spec.values())
        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.random_external_force = self.cfg['task']['random_external_force']
        self.observation_randomize = self.cfg["task"]["observation_randomize"]
        self.image_randomize = self.cfg["task"]["image_randomize"]
        self.env_randomize = self.cfg["task"]["env_randomize"]
        self.torque_randomize = self.cfg["task"]["torque_randomize"]
        self.camera_randomize = self.cfg["task"]["camera_randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.uniform_random_contact = False

        self.dof_pos_offset = self.cfg["env"]["initial_dof_pos_limit"]
        self.dof_vel_offset = self.cfg["env"]["initial_dof_vel_limit"]

        if self.cfg["env"]["adaptive_dof_pos_limit"]["activate"]:
            self.dof_pos_limit_maximum = self.cfg["env"]["adaptive_dof_pos_limit"]["maximum"]
        
        if self.cfg["env"]["adaptive_dof_vel_limit"]["activate"]:
            self.dof_vel_limit_maximum = self.cfg["env"]["adaptive_dof_vel_limit"]["maximum"]

        # define prims present in the scene
        prim_names = ["robot"]
        prim_names += self._get_table_prim_names()
        prim_names += ["object", "goal_object"]
        if self.cfg['env']['scene_randomization']['background']:
            prim_names += ["floor", "back"]
        # mapping from name to asset instance
        self.asset_handles = dict.fromkeys(prim_names)

        # mapping from name to gym rigid body handles
        # name of finger tips links i.e. end-effector frames
        grippers_frames = ["panda_leftfinger", "panda_rightfinger"]
        self._grippers_handles = OrderedDict.fromkeys(grippers_frames, None)
        # mapping from name to gym dof index
        robot_dof_names = list()
        for i in range(1, 8):
            robot_dof_names += [f'panda_joint{i}']
        robot_dof_names += ['panda_finger_joint1', 'panda_finger_joint2']
        self._robot_dof_indices = OrderedDict.fromkeys(robot_dof_names, None)

        # Inductive reward. This is used for the baseline for claim 1 only.
        self.inductive_reward = self.cfg["env"]["inductive_reward"]
        self.energy_reward = self.cfg["env"]["energy_reward"]
        self.controllers={"osc": self.step_osc, "JP": self.step_jp, "cartesian_impedance": self.step_cartesian_impedance}
        self.controller = self.controllers[self.cfg["env"]["controller"]]
        self.compute_target = self.compute_ee_target if not self.cfg["env"]["controller"] == "JP" else self.compute_joint_target
        # Camera sensor
        if self.cfg["env"]["enableCameraSensors"]:
            self.camera_handles = list()
            self._torch_camera_rgba_images: List[torch.Tensor] = list()
            if self.cfg['env']["camera"]["segmentation"]:
                self._torch_camera_segmentation: List[torch.Tensor] = list()
            # if self.cfg['env']["camera"]["depth"]:
            #     self._torch_camera_depth_images: List[torch.Tensor] = list()
        
        # During initialization its parent create_sim is called
        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, gym=gym)

        # initialize the buffers
        self.__initialize()
        
        if self.viewer != None:
            cam_pos = gymapi.Vec3(0.8, 0.0, 0.8)
            cam_target = gymapi.Vec3(0.5, 0.0, 0.4)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        
        for limit_name in self._object_limits:
            # extract limit simple-namespace
            limit_dict = self._object_limits[limit_name].__dict__
            # iterate over namespace attributes
            for prop, value in limit_dict.items():
                limit_dict[prop] = torch.tensor(value, dtype=torch.float, device=self.device)
        for limit_name in self._gripper_limits:
            # extract limit simple-namespace
            limit_dict = self._gripper_limits[limit_name].__dict__
            # iterate over namespace attributes
            for prop, value in limit_dict.items():
                limit_dict[prop] = torch.tensor(value, dtype=torch.float, device=self.device)

        _camera_position = torch.tensor(
            OmegaConf.to_object(self.cfg["env"]["camera"]["position"]), device=self.device
        ).unsqueeze(-1)
        _camera_angle = float(self.cfg["env"]["camera"]["angle"])
        rotation_matrix = torch.tensor((R.from_rotvec(np.array([0.,1.,0.])*np.radians(-90-_camera_angle))*R.from_rotvec(np.array([0.,0.,1.,])*np.radians(90))).inv().as_matrix(),dtype=torch.float).to(self.device)
        self.translation_from_camera_to_object = torch.zeros((3, 4), device=self.device)
        self.translation_from_camera_to_object[:3, :3] = rotation_matrix
        self.translation_from_camera_to_object[:3, 3] = -rotation_matrix.mm(_camera_position)[:, 0]
        self.camera_matrix = self.compute_camera_intrinsics_matrix(
            int(self.cfg["env"]["camera"]["size"][1]), int(self.cfg["env"]["camera"]["size"][0]), 55.368, self.device
        )
        
        # Observation for the student policy
        if self.cfg["env"]["student_obs"] or self.cfg["env"]["enableCameraSensors"]:
            # self.num_student_obs = 69
            self.student_obs = torch.zeros((self.num_envs, 69), dtype=torch.float, device=self.device) # 69 is the extended obs number
        # set the mdp spaces
        self.__configure_mdp_spaces()

        # Initialize for photorealistic rendering
        if self.cfg["env"]["nvisii"]["photorealistic_rendering"]:
            self.__init_photorealistic_rendering(headless)
        
        # save previous smoothed action for interpolation
        # dim=19 because quaternion of the residual is saved
        self.smoothing_coefficient = self.cfg["env"]["smoothing_coefficient"]
        self.previous_smoothed_action = torch.zeros((self.num_envs, 18), dtype=torch.float, device=self.device)

        # save previous keypoints
        self.prev_keypoints = torch.ones(self.num_envs, 8, 3, dtype=torch.float32, device=self.device)

        # table initial pose
        self.table_pose = torch.zeros((self.num_envs, 13), dtype=torch.float32, device=self.device)
        self.is_hole_wide = False
        if self.cfg['name'] == 'Hole_wide': self.is_hole_wide = True
        

    def use_uniform_random_contact(self):
        self.uniform_random_contact = True

    @abc.abstractmethod
    def _set_object_dimension(self, object_dims) -> Object:
        pass

    @abc.abstractmethod
    def _set_table_dimension(self):
        pass

    @abc.abstractmethod
    def _get_table_prim_names(self) -> List[str]:
        pass

    """
    Protected members - Implementation specifics.
    """

    def create_sim(self):
        """
        Setup environment and the simulation scene.
        """
        # define ground plane for simulation
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = 0.0
        # create ground
        self.gym.add_ground(self.sim, plane_params)
        # define scene assets
        self.__create_scene_assets()
        # create environments
        self.__create_envs()

        env_ptr = self.envs[0]
        
        
        # If randomizing, apply once immediately on startup before the fist sim step
        if self.env_randomize:
            self.original_value_copy(self.randomization_params, env_ptr)
            env_ids = list(range(self.num_envs))
            self.env_randomizer(self.randomization_params, env_ids)

    def pre_physics_step(self, actions):
        """
        Setting of input actions into simulator before performing the physics simulation step.
        """
        if self.cfg['env']['scene_randomization']['light']:
            intensity = torch.rand((4, 3), device=self.rl_device) * 0.01 + 0.2
            ambient = torch.rand((4, 3), device=self.rl_device) * 0.01 + 0.8
            intensity[1, :] = 0.2 + (-0.005 + 0.01 * torch.rand(1, device=self.rl_device))
            ambient[1, :] = 0.9 + (-0.005 + 0.01 * torch.rand(1, device=self.rl_device))
            intensity[2, :] = 0.1 + (-0.005 + 0.01 * torch.rand(1, device=self.rl_device))
            ambient[2, :] = 0.9 + (-0.005 + 0.01 * torch.rand(1, device=self.rl_device))
            intensity[3:] = 0.0
            ambient[3:] = 0.0
            direction = torch.tensor([[1.0,-0.05,1.6],[2.4,2.0,3.0],[0.6,0,0.6]], device=self.rl_device)+(-0.005+0.01*torch.rand((3,3), device=self.rl_device))
            self.gym.set_light_parameters(self.sim, 0, gymapi.Vec3(intensity[0,0],intensity[0,0],intensity[0,0]),\
                    gymapi.Vec3(ambient[0,0],ambient[0,0],ambient[0,0]),gymapi.Vec3(*direction[0]) )
            self.gym.set_light_parameters(self.sim, 1, gymapi.Vec3(intensity[1,0],intensity[1,0],intensity[1,0]),\
                    gymapi.Vec3(ambient[1,0],ambient[1,0],ambient[1,0]),gymapi.Vec3(*direction[1]) )
            self.gym.set_light_parameters(self.sim, 2, gymapi.Vec3(intensity[2,0],intensity[2,0],intensity[2,0]),\
                    gymapi.Vec3(ambient[2,0],ambient[2,0],ambient[2,0]),gymapi.Vec3(*direction[2]) )
            self.gym.set_light_parameters(self.sim, 3, gymapi.Vec3(intensity[3,0],intensity[3,0],intensity[3,0]),\
                    gymapi.Vec3(ambient[3,0],ambient[3,0],ambient[3,0]),gymapi.Vec3(0.,-0.1,0.5) )
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        self.env_ids = env_ids
        self.actions = actions.clone().to(self.device)
        self.actions[env_ids, :] = 0.0
        # if normalized_action is true, then denormalize them.
        if self.cfg['env']["normalize_action"]:
            action_transformed = unscale_transform(
                self.actions,
                lower=self._action_scale.low,
                upper=self._action_scale.high
            )
        else:
            action_transformed = self.actions
        self.action_transformed = action_transformed

        ### compute target ee pose or joint pose
        self.compute_target()
        self.mean_energy[:] = 0
        self.max_torque[:] = 0

    def compute_ee_target(self):
        """compute the endeffector target to run the controller"""
        self.sub_goal_pos = self._rigid_body_state[:, self._hand_handle, :3] + self.action_transformed[:, :3]
        rot_residual = self.axisaToquat(self.action_transformed[:, 3:6])
        rot_residual[self.env_ids, 0:3] = 0.0
        rot_residual[self.env_ids, -1] = 1.0
        self.sub_goal_rot = quat_mul(rot_residual, self._rigid_body_state[:, self._hand_handle, 3:7])

    def compute_joint_target(self):
        """compute the joint target to run the controller"""
        delta_joint_position = self._get_delta_dof_pos(
            delta_pose=self.action_transformed[:, :6], ik_method='dls',
            jacobian=self.j_eef.clone(), device=self.device
        )
        delta_joint_position[self.env_ids, :] = 0
        self.desired_joint_position = self.dof_position[:, :-2] + delta_joint_position
        

    def step_osc(self):
        pos_err = self.sub_goal_pos - self._rigid_body_state[:, self._hand_handle, :3]
        orn_err = orientation_error(self.sub_goal_rot, self._rigid_body_state[:, self._hand_handle, 3:7], 3)
        err = torch.cat([pos_err, orn_err], -1)
        
        hand_vel = self._rigid_body_state[:, self._hand_handle:(self._hand_handle + 1), 7:]

        return control_osc(
            self.num_envs, self.j_eef, hand_vel.squeeze(-2), self.mm, err, 
            self.obs_buf[:, 9:18], self.obs_buf[:, :9], self.action_transformed[:, 6:12], self.action_transformed[:, 12:18], 
            True, False, self.device
        )

    def step_cartesian_impedance(self):
        pos_err = self.sub_goal_pos - self._rigid_body_state[:, self._hand_handle, :3]
        orn_err = orientation_error(self.sub_goal_rot, self._rigid_body_state[:, self._hand_handle, 3:7], 3)
        err = torch.cat([pos_err, orn_err], -1)
        
        hand_vel = self._rigid_body_state[:, self._hand_handle, 7:]

        kp = self.action_transformed[:, 6:12]
        kd = 2 * torch.sqrt(kp) * self.action_transformed[:, 12:18]
        xddt = (kp * err - kd * hand_vel).unsqueeze(-1)
        u = torch.transpose(self.j_eef, 1, 2) @ xddt
        
        return u.squeeze(-1)


    def step_jp(self):
        joint_position_error = self.desired_joint_position - self.dof_position[:, :-2]
        p_gains = self.action_transformed[:, 6:13]
        d_gains = torch.sqrt(p_gains) *self.action_transformed[:, 13:]
        ct = p_gains * joint_position_error - d_gains * self.dof_velocity[:, :-2]
        return ct
    
    def step_controller(self):
        computed_torque = torch.zeros(
            self.num_envs, self._dims.JointTorqueDim.value, dtype=torch.float32, device=self.device
        )
        ct = self.controller()
        computed_torque[:, :7] = ct

        if not self.cfg["env"]["restrict_gripper"]:
            computed_torque[:, 7:] = self.franka_dof_stiffness[7] * (-self.dof_position[:, -2:] + self.action_transformed[:, 6:7])
            computed_torque[:, 7:] -= self.franka_dof_damping[7] * self.dof_velocity[:, -2:]
        
        self.computed_torque = computed_torque

        # apply Domain Randomization before saturating torque
        if self.torque_randomize:
            self.torque_randomizer(self.randomization_params, self.computed_torque, self.cfg["env"]["restrict_gripper"])

        applied_torque = saturate(
            self.computed_torque,
            lower=-self.franka_dof_effort_scales,
            upper=self.franka_dof_effort_scales
        )
        applied_torque[self.env_ids, :] = 0
        self.max_torque = torch.maximum(self.max_torque, torch.norm(applied_torque[:,:7],dim=-1))
        self.applied_torque = applied_torque
        # set computed torques to simulator buffer.
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(applied_torque))
        # apply random external force
        if self.random_external_force:
            object_contact_force = self.net_cf[:, 13]
            contact_force_magnitude = torch.norm(object_contact_force, dim=-1)
            existence_of_contact = (contact_force_magnitude < 0.5)
            rigid_body_count: int = self.gym.get_env_rigid_body_count(self.envs[0])
            force_tensor = (torch.rand(size=(self.num_envs, rigid_body_count, 3), device=self.rl_device) - 0.5) * 0.001
            force_tensor[existence_of_contact, 13, :] = 0
            torque_tensor = (torch.rand(size=(self.num_envs, rigid_body_count, 3), device=self.rl_device) - 0.5) * 0.001
            torque_tensor[existence_of_contact, 13, :] = 0
            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                gymtorch.unwrap_tensor(force_tensor.reshape((-1, 3))), 
                gymtorch.unwrap_tensor(torque_tensor.reshape((-1, 3))), 
                gymapi.ENV_SPACE
            )

    def refresh_buffer(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.mean_energy = torch.add(self.mean_energy, torch.norm(self.dof_velocity[:, :7] * self.applied_torque[:, :7], dim=-1))

    def _get_delta_dof_pos(self, delta_pose, ik_method, jacobian, device):
        """Get delta Franka DOF position from delta pose using specified IK method."""
        # References:
        # 1) https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf
        # 2) https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2018/RD_HS2018script.pdf (p. 47)

        if ik_method == 'pinv':  # Jacobian pseudoinverse
            k_val = 1.0
            jacobian_pinv = torch.linalg.pinv(jacobian)
            delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
            delta_dof_pos = delta_dof_pos.squeeze(-1)

        elif ik_method == 'trans':  # Jacobian transpose
            k_val = 1.0
            jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
            delta_dof_pos = k_val * jacobian_T @ delta_pose.unsqueeze(-1)
            delta_dof_pos = delta_dof_pos.squeeze(-1)

        elif ik_method == 'dls':  # damped least squares (Levenberg-Marquardt)
            lambda_val = 0.1
            jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
            lambda_matrix = (lambda_val ** 2) * torch.eye(n=jacobian.shape[1], device=device)
            delta_dof_pos = jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)
            delta_dof_pos = delta_dof_pos.squeeze(-1)

        elif ik_method == 'svd':  # adaptive SVD
            k_val = 1.0
            U, S, Vh = torch.linalg.svd(jacobian)
            S_inv = 1. / S
            min_singular_value = 1.0e-5
            S_inv = torch.where(S > min_singular_value, S_inv, torch.zeros_like(S_inv))
            jacobian_pinv = torch.transpose(Vh, dim0=1, dim1=2)[:, :, :6] @ torch.diag_embed(S_inv) @ torch.transpose(U, dim0=1, dim1=2)
            delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
            delta_dof_pos = delta_dof_pos.squeeze(-1)

        return delta_dof_pos

    def post_physics_step(self):
        """
        Setting of buffers after performing the physics simulation step.

        @note Also need to update the reset buffer for the instances that have terminated.
              The termination conditions to check are besides the episode timeout.
        """
        # count step for each environment
        self.progress_buf += 1
        self.randomize_buf += 1
        # fill observations buffer
        self.compute_observations()
        # compute rewards
        self.compute_reward(self.actions)
        
        # check termination e.g. box is dropped from table.
        self._check_termination()

        if self.observation_randomize: # TODO: change self.randomize option separately
            self.obs_buf[:, 18:34] = self.keypoints_randomizer(self.randomization_params, self.obs_buf[:, 18:34])
            self.obs_buf[:, :18] = self.observation_randomizer(self.randomization_params, self.obs_buf[:, :18])
            self.obs_buf[:, 50:57] = self.observation_randomizer(self.randomization_params, self.obs_buf[:, 50:57])
            # self.extras["regularization"]=self.regularization
        
    def compute_reward(self, actions):
        self.rew_buf[:] = 0
        self.reset_buf[:] = 0
        if self.asymmetric_obs:
            self.rew_buf[:], self.reset_buf[:], self.regularization[:] = compute_card_reward(
                self.states_buf,
                self.reset_buf,
                self.progress_buf,
                self.max_episode_length,
                (self.cfg["env"]["reward_terms"]["object_dist"]["weight1"], self.cfg["env"]["reward_terms"]["object_dist"]["weight2"]),
                (self.cfg["env"]["reward_terms"]["object_dist"]["epsilon"], self.cfg["env"]["reward_terms"]["object_rot"]["epsilon"]),
                self._object_goal_poses_buf,
                self.max_torque,
                self.mean_energy,
                self._object_state_history[0],
                self._grippers_frames_state_history[0],
                self._object_dims.size,
                self.inductive_reward,
                self.energy_reward
            )
        else:
            self.rew_buf[:], self.reset_buf[:], _ = compute_card_reward(
                self.obs_buf,
                self.reset_buf,
                self.progress_buf,
                self.max_episode_length,
                (self.cfg["env"]["reward_terms"]["object_dist"]["weight1"], self.cfg["env"]["reward_terms"]["object_dist"]["weight2"]),
                (self.cfg["env"]["reward_terms"]["object_dist"]["epsilon"], self.cfg["env"]["reward_terms"]["object_rot"]["epsilon"]),
                self._object_goal_poses_buf,
                self.max_torque,
                self.mean_energy,
                self._object_state_history[0],
                self._grippers_frames_state_history[0],
                self._object_dims.size,
                self.inductive_reward,
                self.energy_reward
            )

    """
    Private functions
    """

    def __initialize(self):
        """Allocate memory to various buffers.
        """
        if self.cfg['env']["student_obs"] or self.cfg['env']["enableCameraSensors"]:
            # store the sampled initial poses for the object
            self._object_initial_poses_buf = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float)
        if self.cfg['env']["enableCameraSensors"]:
            rgb_image_shape = [self.num_envs] + OmegaConf.to_object(self.cfg["env"]["camera"]["size"]) + [3]
            self._camera_image = torch.zeros(rgb_image_shape, dtype=torch.uint8, device=self.rl_device)
            segmentation_shape = [self.num_envs] + OmegaConf.to_object(self.cfg["env"]["camera"]["size"]) + [1]
            self._segmentation = torch.zeros(segmentation_shape, dtype=torch.int32, device=self.rl_device)
        if self.cfg['env']["nvisii"]["photorealistic_rendering"]:
            rgb_image_shape = [1] + OmegaConf.to_object(self.cfg["env"]["camera"]["size"]) + [3]
            self._photorealistic_image = torch.zeros(rgb_image_shape, dtype=torch.uint8, device=self.rl_device)
        # store the sampled goal poses for the object: [num. of instances, 7]
        self._object_goal_poses_buf = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float)
        
        # get gym GPU state tensors
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "robot")
        _mm = self.gym.acquire_mass_matrix_tensor(self.sim, "robot")
        _dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        
        # refresh the buffer
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # self.gym.refresh_force_sensor_tensor(self.sim)
        
        # create wrapper tensors for reference (consider everything as pointer to actual memory)
        # DOF
        self.dof_state: torch.Tensor = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self.dof_position = self.dof_state[..., 0]
        self.dof_velocity = self.dof_state[..., 1]
        # rigid body
        self._rigid_body_state: torch.Tensor = gymtorch.wrap_tensor(_rigid_body_tensor).view(self.num_envs, -1, 13)
        # root actors
        self._actors_root_state: torch.Tensor = gymtorch.wrap_tensor(_actor_root_state_tensor).view(-1, 13)
        self.franka_link_dict: Dict[str, int] = self.gym.get_asset_rigid_body_dict(self.asset_handles["robot"])
        franka_hand_index = self.franka_link_dict["panda_hand"]
        # jacobian
        self.jacobian: torch.Tensor = gymtorch.wrap_tensor(_jacobian)
        self.j_eef = self.jacobian[:, (franka_hand_index - 1), :, :7]
        # mass matirx
        self.mm: torch.Tensor = gymtorch.wrap_tensor(_mm)
        self.mm = self.mm[:, :(franka_hand_index - 1), :(franka_hand_index - 1)]
        # joint torques
        self.dof_torque: torch.Tensor = gymtorch.wrap_tensor(_dof_force_tensor).view(self.num_envs, self._dims.JointTorqueDim.value)
        self.net_cf: torch.Tensor = gymtorch.wrap_tensor(_net_cf).view(self.num_envs, -1, 3)

        if self.cfg["env"]["hand_force_limit"]:
            _force_torque_sensor_data = self.gym.acquire_force_sensor_tensor(self.sim)
            self.force_torque_sensor_data = gymtorch.wrap_tensor(_force_torque_sensor_data)


        # frames history
        gripper_handles_indices = list(self._grippers_handles.values())
        object_indices = self.actor_indices["object"]
        # timestep 0 is current tensor
        curr_history_length = 0
        while curr_history_length < self._state_history_len:
            # add tensors to history list
            self._grippers_frames_state_history.append(self._rigid_body_state[:, gripper_handles_indices, :7])
            self._object_state_history.append(self._actors_root_state[object_indices])
            # update current history length
            curr_history_length += 1
        print(f'grippers frames history shape: {self._grippers_frames_state_history[0].shape}')
        
        self._observations_scale = SimpleNamespace(low=None, high=None)
        if self.asymmetric_obs:
            self._states_scale = SimpleNamespace(low=None, high=None)
            self.regularization=torch.zeros(self.num_envs,dtype=torch.float, device=self.device)
        self._action_scale = SimpleNamespace(low=None, high=None)
        if self.cfg["env"]["student_obs"] or self.cfg["env"]["enableCameraSensors"]:
            self._std_obs_scale = SimpleNamespace(low=None, high=None)

        self._successes = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        if self.cfg["env"]["extract_successes"]:
            self.extract = False
            self._successes_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.max_torque = torch.zeros(self.num_envs,dtype=torch.float, device=self.device)
        self.mean_energy = torch.zeros(self.num_envs,dtype=torch.float, device=self.device)

    def __configure_mdp_spaces(self):
        """
        Configures the observations, action spaces.
        """
        # Action scale for the MDP
        # Note: This is order sensitive.

        # action space is residual pose of eef and gripper position
        self.minimum_residual_scale = self.cfg["env"]["adaptive_residual_scale"]["minimum"]
        initial_residual_scale = self.cfg["env"]["initial_residual_scale"]
        self.position_scale = initial_residual_scale[0]
        self.orientation_scale = initial_residual_scale[1]

        residual_scale_low = [-self.position_scale]*3+[-self.orientation_scale]*3
        residual_scale_high = [self.position_scale]*3+[self.orientation_scale]*3

        if self.cfg["env"]["restrict_gripper"]:
            if self.cfg["env"]["controller"]=="JP":
                self._action_scale.low = to_torch(
                    residual_scale_low+[10.0]*7+[0.0]*7, 
                    device=self.device
                )  # TODO: add explanations to this values
                self._action_scale.high = to_torch(
                    residual_scale_high+[200.0]*4+[100.0]*3+[2.0]*7, 
                    device=self.device
                )
            else:
                self._action_scale.low = to_torch(residual_scale_low+[ 10.0]*6          +[0.0]*6, device=self.device)  # TODO: add explanations to this values
                self._action_scale.high = to_torch(residual_scale_high+[200.0]*3+[300.0]*3+[2.0]*6, device=self.device)
        else:
            # plus gain
            self._action_scale.low = to_torch(residual_scale_low+[0.0]+[10.0]*6+[0.0]*6, device=self.device)
            self._action_scale.high = to_torch(residual_scale_high+[0.04]+[300.0]*6+[2.0]*6, device=self.device)

        # Observations scale for the MDP
        # check if policy outputs normalized action [-1, 1] or not.
        if self.cfg['env']["normalize_action"]:
            obs_action_scale = SimpleNamespace(
                low=torch.full((self.action_dim,), -1, dtype=torch.float, device=self.device),
                high=torch.full((self.action_dim,), 1, dtype=torch.float, device=self.device)
            )
        else:
            obs_action_scale = self._action_scale
        # Note: This is order sensitive.
        if self.cfg["env"]["keypoint"]["activate"]:
            self._observations_scale.low = torch.cat([
                self.franka_dof_lower_limits,
                -self.franka_dof_speed_scales,
                self._object_limits["2Dkeypoint"].low.repeat(8),
                self._object_limits["2Dkeypoint"].low.repeat(8),
                self._gripper_limits["gripper_position"].low,
                self._gripper_limits["gripper_orientation"].low,
                obs_action_scale.low
            ])
            self._observations_scale.high = torch.cat([
                self.franka_dof_upper_limits,
                self.franka_dof_speed_scales,
                self._object_limits["2Dkeypoint"].high.repeat(8),
                self._object_limits["2Dkeypoint"].high.repeat(8),
                self._gripper_limits["gripper_position"].high,
                self._gripper_limits["gripper_orientation"].high,
                obs_action_scale.high
            ])
        else:
            # keypoints are always activated. 
            pass
        if self.cfg["env"]["student_obs"] or self.cfg["env"]["enableCameraSensors"]:
            self._std_obs_scale.low = torch.cat([
                self._gripper_limits["gripper_position"].low,
                self._gripper_limits["gripper_orientation"].low,
                torch.tensor([0.0], dtype=torch.float, device=self.device),
                self.franka_dof_lower_limits[:-2],
                -self.franka_dof_speed_scales[:-2],
                obs_action_scale.low,
                torch.tensor([0.0], dtype=torch.float, device=self.device),
                self._object_limits["position"].low,
                self._object_limits["orientation"].low,
                self._object_limits["position"].low,
                self._object_limits["orientation"].low,
                self._object_limits["position"].low,
                self._object_limits["orientation"].low,
                self._object_limits["position"].low,
                self._object_limits["orientation"].low,
            ])
            self._std_obs_scale.high = torch.cat([
                self._gripper_limits["gripper_position"].high,
                self._gripper_limits["gripper_orientation"].high,
                torch.tensor([0.08], dtype=torch.float, device=self.device),
                self.franka_dof_upper_limits[:-2],
                self.franka_dof_speed_scales[:-2],
                obs_action_scale.high,
                torch.tensor([1.0], dtype=torch.float, device=self.device),
                self._object_limits["position"].high,
                self._object_limits["orientation"].high,
                self._object_limits["position"].high,
                self._object_limits["orientation"].high,
                self._object_limits["position"].high,
                self._object_limits["orientation"].high,
                self._object_limits["position"].high,
                self._object_limits["orientation"].high,
            ])
        if self.asymmetric_obs:
            states_dim = sum(self.state_spec.values())
            self._states_scale.low = self._observations_scale.low  
            self._states_scale.high = self._observations_scale.high 
            self._observations_scale.low = torch.cat([
                self.franka_dof_lower_limits,
                -self.franka_dof_speed_scales,
                self._object_limits["2Dkeypoint"].low.repeat(8),
                self._object_limits["2Dkeypoint"].low.repeat(8),
                self._gripper_limits["gripper_position"].low,
                self._gripper_limits["gripper_orientation"].low,
                obs_action_scale.low
            ])
            self._observations_scale.high = torch.cat([
                self.franka_dof_upper_limits,
                self.franka_dof_speed_scales,
                self._object_limits["2Dkeypoint"].high.repeat(8),
                self._object_limits["2Dkeypoint"].high.repeat(8),
                self._gripper_limits["gripper_position"].high,
                self._gripper_limits["gripper_orientation"].high,
                obs_action_scale.high
            ])
        # State scale for the MDP
        obs_dim = sum(self.obs_spec.values())
        action_dim = sum(self.action_spec.values())
        # check that dimensions match
        # observations
        if self._observations_scale.low.shape[0] != obs_dim or self._observations_scale.high.shape[0] != obs_dim:
            msg = f"Observation scaling dimensions mismatch. " \
                  f"\tLow: {self._observations_scale.low.shape[0]}, " \
                  f"\tHigh: {self._observations_scale.high.shape[0]}, " \
                  f"\tExpected: {obs_dim}."
            raise AssertionError(msg)
        # states
        if self.asymmetric_obs:
            if self._states_scale.low.shape[0] != states_dim or self._states_scale.high.shape[0] != states_dim:
                msg = f"States scaling dimensions mismatch. " \
                    f"\tLow: {self._states_scale.low.shape[0]}, " \
                    f"\tHigh: {self._states_scale.high.shape[0]}, " \
                    f"\tExpected: {states_dim}."
                raise AssertionError(msg)
        # actions
        if self._action_scale.low.shape[0] != action_dim or self._action_scale.high.shape[0] != action_dim:
            msg = f"Actions scaling dimensions mismatch. " \
                  f"\tLow: {self._action_scale.low.shape[0]}, " \
                  f"\tHigh: {self._action_scale.high.shape[0]}, " \
                  f"\tExpected: {action_dim}."
            raise AssertionError(msg)
    
    def update_residual_scale(self, position_ratio, orientation_ratio):

        if self.position_scale <= self.minimum_residual_scale[0]: return False

        self.position_scale = self.position_scale * position_ratio
        self.orientation_scale = self.orientation_scale * orientation_ratio

        if self.position_scale <= self.minimum_residual_scale[0]:
            self.position_scale = self.minimum_residual_scale[0]
            self.orientation_scale = self.minimum_residual_scale[1]

        residual_scale_low = [-self.position_scale]*3+[-self.orientation_scale]*3
        residual_scale_high = [self.position_scale]*3+[self.orientation_scale]*3

        print(f"Residual scale is reduced to {residual_scale_high}")

        if self.cfg["env"]["restrict_gripper"]:
            if self.cfg["env"]["controller"]=="JP":
                self._action_scale.low = to_torch(
                    residual_scale_low+[10.0]*7+[0.0]*7, 
                    device=self.device
                )  # TODO: add explanations to this values
                self._action_scale.high = to_torch(
                    residual_scale_high+[200.0]*4+[100.0]*3+[2.0]*7, 
                    device=self.device
                )
            else:
                self._action_scale.low = to_torch(residual_scale_low+[ 10.0]*6          +[0.0]*6, device=self.device)  # TODO: add explanations to this values
                self._action_scale.high = to_torch(residual_scale_high+[200.0]*3+[300.0]*3+[2.0]*6, device=self.device)
        else:
            # plus gain
            self._action_scale.low = to_torch(residual_scale_low+[0.0]+[10.0]*6+[0.0]*6, device=self.device)
            self._action_scale.high = to_torch(residual_scale_high+[0.04]+[300.0]*6+[2.0]*6, device=self.device)
        
        return True

    @abc.abstractmethod
    def _max_dist_btw_obj_and_goal(self) -> torch.Tensor:
        pass

    def __create_scene_assets(self):
        """ Define Gym assets for table, robot and object.
        """
        # define assets
        self.asset_handles["robot"] = self.__define_robot_asset()
        self._define_table_asset()
        if self.cfg['env']['scene_randomization']['background']:
            self.asset_handles["floor"] = self.__define_floor_asset()
            self.asset_handles["back"] = self.__define_back_asset()
        self.asset_handles["object"] = self._define_object_asset()
        self.asset_handles["goal_object"] = self.__define_goal_object_asset()
        # display the properties (only for debugging)
        # robot
        print("Franka Robot Asset: ")
        print(f'\t Number of bodies: {self.gym.get_asset_rigid_body_count(self.asset_handles["robot"])}')
        print(f'\t Number of shapes: {self.gym.get_asset_rigid_shape_count(self.asset_handles["robot"])}')
        print(f'\t Number of dofs: {self.gym.get_asset_dof_count(self.asset_handles["robot"])}')
        print(f'\t Number of actuated dofs: {self._dims.JointTorqueDim.value}')
        # table
        # print("Card table Asset: ")
        # print(f'\t Number of bodies: {self.gym.get_asset_rigid_body_count(self.asset_handles["table"])}')
        # print(f'\t Number of shapes: {self.gym.get_asset_rigid_shape_count(self.asset_handles["table"])}')

    def __create_envs(self):
        """Create various instances for the environment.
        """
        robot_dof_props = self.gym.get_asset_dof_properties(self.asset_handles["robot"])
        # set dof properites based on the control mode
        self.franka_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400, 400, 800, 800], dtype=torch.float, device=self.device)
        self.franka_dof_damping = to_torch([80, 80, 80, 80, 80, 80, 80, 40, 40], dtype=torch.float, device=self.device)

        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self.franka_dof_speed_scales = []
        self.franka_dof_effort_scales = []


        sysid_friction = [0.00174, 0.01, 7.5e-09, 2.72e-07, 0.39, 0.12, 0.9]
        sysid_damping  = [2.12, 2.3, 1.29, 2.8, 0.194, 0.3, 0.46]
        sysid_armature = [0.192, 0.54, 0.128, 0.172, 5.26e-09, 0.08, 0.06]
            
        for k, dof_index in enumerate(self._robot_dof_indices.values()):
            if self.cfg["env"]["restrict_gripper"] and (k >= 7):
                robot_dof_props['driveMode'][dof_index] = gymapi.DOF_MODE_NONE
                robot_dof_props['friction'][dof_index] = 1000
                robot_dof_props['damping'][dof_index] = 1000
                robot_dof_props['armature'][dof_index] = 10
            else:
                robot_dof_props['driveMode'][dof_index] = gymapi.DOF_MODE_EFFORT
                if k < 7:
                    robot_dof_props['friction'][k] = sysid_friction[k]
                    robot_dof_props['damping'][k] = sysid_damping[k]
                    robot_dof_props['armature'][k] = sysid_armature[k]

            self.franka_dof_lower_limits.append(robot_dof_props['lower'][k])
            self.franka_dof_upper_limits.append(robot_dof_props['upper'][k])
            self.franka_dof_speed_scales.append(robot_dof_props['velocity'][k])
            self.franka_dof_effort_scales.append(robot_dof_props['effort'][k])
        
        if self.cfg["env"]["restrict_gripper"]:
            _effort = 1000
        else:
            _effort = 200
        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self.franka_dof_speed_scales = to_torch(self.franka_dof_speed_scales, device=self.device)
        self.franka_dof_speed_scales[[7, 8]] = 0.1
        robot_dof_props['effort'][7] = _effort
        robot_dof_props['effort'][8] = _effort
        self.franka_dof_effort_scales = to_torch(self.franka_dof_effort_scales, device=self.device)
        self.franka_dof_effort_scales[[7, 8]] = _effort

        # define lower and upper region bound for each environment
        env_lower_bound = gymapi.Vec3(-self.cfg['env']["envSpacing"], -self.cfg['env']["envSpacing"], 0.0)
        env_upper_bound = gymapi.Vec3(self.cfg['env']["envSpacing"], self.cfg['env']["envSpacing"], self.cfg['env']["envSpacing"])
        num_instances_per_row = int(np.sqrt(self.num_envs))

        # mapping from name to gym actor indices
        # note: later the list is converted to torch tensor for ease in interfacing with IsaacGym.
        actor_indices: Dict[str, List[int]] = dict()
        for asset_name in self.asset_handles.keys():
            actor_indices[asset_name] = list()
        
        # count number of shapes and bodies
        max_agg_bodies = 0
        max_agg_shapes = 0
        for asset_handle in self.asset_handles.values():
            max_agg_bodies += self.gym.get_asset_rigid_body_count(asset_handle)
            max_agg_shapes += self.gym.get_asset_rigid_shape_count(asset_handle)

        if self.cfg['env']["enableCameraSensors"]:
            camera_position = torch.tensor(OmegaConf.to_object(self.cfg["env"]["camera"]["position"]))
            camera_angle = float(self.cfg["env"]["camera"]["angle"])
        
        self.envs = []
        # iterate and create environment instances
        for env_index in range(self.num_envs):
            # create environment
            env_ptr = self.gym.create_env(self.sim, env_lower_bound, env_upper_bound, num_instances_per_row)
            self.envs.append(env_ptr)

            # begin aggregration
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add robot to environment
            robot_pose = gymapi.Transform()
            robot_pose.p = gymapi.Vec3(0, 0, 0)
            robot_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            Franka_actor = self.gym.create_actor(env_ptr, self.asset_handles["robot"], robot_pose, "robot", env_index, 0, 3)
            self.gym.set_actor_dof_properties(env_ptr, Franka_actor, robot_dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, Franka_actor)
            Franka_idx = self.gym.get_actor_index(env_ptr, Franka_actor, gymapi.DOMAIN_SIM)
            actor_indices["robot"].append(Franka_idx)

            # add table to environment
            self._create_table(env_ptr, env_index, actor_indices)

            # add object to environment
            self._create_object(env_ptr, env_index, actor_indices)

            if self.cfg['env']['scene_randomization']['background']:
                back_pose = gymapi.Transform()
                back_pose.p = gymapi.Vec3(-0.5, 0, 0.5)
                back_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
                back_handle = self.gym.create_actor(env_ptr, self.asset_handles["back"], back_pose, "back", (env_index + self.num_envs * 2), 0, 4)
                back_idx = self.gym.get_actor_index(env_ptr, back_handle, gymapi.DOMAIN_SIM)
                self.gym.set_rigid_body_color(env_ptr, back_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0, 0, 0))
                actor_indices["back"].append(back_idx)
                
                floor_pose = gymapi.Transform()
                floor_pose.p = gymapi.Vec3(0, 0, 0.001)
                floor_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
                floor_handle = self.gym.create_actor(env_ptr, self.asset_handles["floor"], floor_pose, "floor", (env_index + self.num_envs * 3), 0, 5)
                floor_idx = self.gym.get_actor_index(env_ptr, floor_handle, gymapi.DOMAIN_SIM)
                self.gym.set_rigid_body_color(env_ptr, floor_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0, 0, 0))
                actor_indices["floor"].append(floor_idx)
            
            # add goal object to environment
            if not self.cfg['env']["enableCameraSensors"]:
                goal_handle = self.gym.create_actor(
                    env_ptr, self.asset_handles["goal_object"], gymapi.Transform(), "goal_object", (env_index + self.num_envs), 0, 0
                )
                goal_color = gymapi.Vec3(0.3, 0.3, 0.3)
                self.gym.set_rigid_body_color(env_ptr, goal_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, goal_color)
                goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
                actor_indices["goal_object"].append(goal_object_idx)
            
            if self.cfg['env']["enableCameraSensors"]:
                # add camera sensor to environment
                camera_props = gymapi.CameraProperties()
                camera_props.enable_tensors = True
                camera_props.horizontal_fov = 55.368
                camera_props.height = self.cfg["env"]["camera"]["size"][0]
                camera_props.width = self.cfg["env"]["camera"]["size"][1]
                if self.camera_randomize:
                    camera_props, camera_transform = self.camera_randomizer(camera_props, camera_position, camera_angle)
                else:
                    camera_transform = self._get_camera_transform(camera_position, camera_angle)
                camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)
                self.gym.set_camera_transform(camera_handle, env_ptr, camera_transform)
                self.camera_handles.append(camera_handle)
                camera_rgba_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_COLOR)
                self._torch_camera_rgba_images.append(gymtorch.wrap_tensor(camera_rgba_tensor))
                if self.cfg['env']["camera"]["segmentation"]:
                    camera_segmentation_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_SEGMENTATION)
                    self._torch_camera_segmentation.append(gymtorch.wrap_tensor(camera_segmentation_tensor))
            
            # end aggregation
            self.gym.end_aggregate(env_ptr)
        
        # light source
        intensity = [0.2, 0.2, 0.1, 0.]
        ambient = [0.8, 0.9, .9, .0]
        direction = torch.tensor([[1.0, -0.05, 1.6], [2.4, 2.0, 3.0], [0.6, 0, 0.6]], device=self.rl_device)
        if self.cfg['env']['scene_randomization']['light']:
            intensity[:3] =+ (-0.005 + 0.01 * torch.rand(1, device=self.rl_device))
            ambient[:3] =+ (-0.005 + 0.01 * torch.rand(1, device=self.rl_device))
            direction += (-0.005 + 0.01 * torch.rand((3, 3), device=self.rl_device))
        self.gym.set_light_parameters(self.sim, 0, gymapi.Vec3(intensity[0],intensity[0],intensity[0]),\
                gymapi.Vec3(ambient[0], ambient[0], ambient[0]), gymapi.Vec3(*direction[0]))
        self.gym.set_light_parameters(self.sim, 1, gymapi.Vec3(intensity[1],intensity[1],intensity[1]),\
                gymapi.Vec3(ambient[1], ambient[1], ambient[1]), gymapi.Vec3(*direction[1]))
        self.gym.set_light_parameters(self.sim, 2, gymapi.Vec3(intensity[2],intensity[2],intensity[2]),\
                gymapi.Vec3(ambient[2], ambient[2], ambient[2]), gymapi.Vec3(*direction[2]))
        self.gym.set_light_parameters(self.sim, 3, gymapi.Vec3(intensity[3],intensity[3],intensity[3]),\
                gymapi.Vec3(ambient[3], ambient[3], ambient[3]), gymapi.Vec3(0., -0.1, 0.5))
        # convert gym actor indices from list to tensor
        self.actor_indices: Dict[str, torch.Tensor] = dict()
        for asset_name, indices in actor_indices.items():
            self.actor_indices[asset_name] = torch.tensor(indices, dtype=torch.long, device=self.device)
    
    @abc.abstractmethod
    def _create_table(self, env_ptr, env_index: int, actor_indices: Dict[str, List[int]]):
        pass

    @abc.abstractmethod
    def _create_object(self, env_ptr, env_index: int, actor_indices: Dict[str, List[int]]):
        pass

    def _check_termination(self):
        failed_reset = self._check_failure()

        if self.cfg["env"]["adaptive_dof_pos_limit"]["activate"]:
            dof_pos_limit_reset = self._check_dof_position_limit_reset(self.dof_pos_offset)
            failed_reset = torch.logical_or(failed_reset, dof_pos_limit_reset)

        if self.cfg["env"]["adaptive_dof_vel_limit"]["activate"]:
            dof_vel_limit_reset = self._check_dof_velocity_limit_reset(self.dof_vel_offset)
            failed_reset = torch.logical_or(failed_reset, dof_vel_limit_reset)
        
        if self.cfg["env"]["hand_force_limit"]:
            force_limit_reset = torch.norm(self.force_torque_sensor_data[:,:3], p=2, dim=-1) >40.
            #force_limit_reset = torch.max(torch.abs(self.force_torque_sensor_data[:,:3]), dim=-1) >40.
            failed_reset = torch.logical_or(failed_reset, force_limit_reset)

        goal_reached = self._check_success()
        self.rew_buf[goal_reached] += self.cfg["env"]["reward_terms"]["object_dist"]["weight2"]
        self.reset_buf = torch.logical_or(self.reset_buf, failed_reset)
        if self.cfg["env"]["extract_successes"] and self.extract:
            self._successes_count[goal_reached] += 1
        self.reset_buf = torch.logical_or(self.reset_buf, goal_reached)
        self._successes[goal_reached] = 1
        self.reset_buf = self.reset_buf.float()
    
    def add_dof_position_limit_offset(self, offset):
        if self.dof_pos_offset >= self.dof_pos_limit_maximum: return False
        self.dof_pos_offset += offset
        if self.dof_pos_offset >= self.dof_pos_limit_maximum: 
            self.dof_pos_offset = self.dof_pos_limit_maximum
            print("dof_pos_limit is maximum " +str(self.dof_pos_offset))
        else: 
            print("dof_pos_limit is increased to " +str(self.dof_pos_offset))
        return True

    def add_dof_velocity_limit_offset(self, offset):
        if self.dof_vel_offset >= self.dof_vel_limit_maximum: return False
        self.dof_vel_offset += offset
        if self.dof_vel_offset > self.dof_vel_limit_maximum: 
            self.dof_vel_offset = self.dof_vel_limit_maximum
        else: print("dof_vel_limit is increased to " +str(self.dof_vel_offset))
        return True

    def _check_dof_position_limit_reset(self, offset):
        dof_pos_upper_lower_diff = self.franka_dof_upper_limits[:7] - self.franka_dof_lower_limits[:7]
        dof_pos_curr_lower_diff = self.dof_position[:, :7] - self.franka_dof_lower_limits[:7]
        dof_pos_curr_ratio = dof_pos_curr_lower_diff / dof_pos_upper_lower_diff
        dof_pos_curr_lowest_ratio, _ = torch.min(dof_pos_curr_ratio, dim=-1)
        dof_pos_curr_highest_ratio, _ = torch.max(dof_pos_curr_ratio, dim=-1)
        dof_pos_low_envs = torch.le(dof_pos_curr_lowest_ratio, offset)
        dof_pos_high_envs = torch.gt(dof_pos_curr_highest_ratio, 1.0-offset)
        dof_pos_limit_exceeded_envs = torch.logical_or(dof_pos_low_envs, dof_pos_high_envs)
        return dof_pos_limit_exceeded_envs

    def _check_dof_velocity_limit_reset(self, offset):
        dof_vel_upper_lower_diff = 2.*self.franka_dof_speed_scales[:7]
        dof_vel_curr_lower_diff = self.dof_velocity[:, :7] + self.franka_dof_speed_scales[:7]
        dof_vel_curr_ratio = dof_vel_curr_lower_diff / dof_vel_upper_lower_diff
        dof_vel_curr_lowest_ratio, _ = torch.min(dof_vel_curr_ratio, dim=-1)
        dof_vel_curr_highest_ratio, _ = torch.max(dof_vel_curr_ratio, dim=-1)
        dof_vel_low_envs = torch.le(dof_vel_curr_lowest_ratio, offset)
        dof_vel_high_envs = torch.gt(dof_vel_curr_highest_ratio, 1.0-offset)
        dof_vel_limit_exceeded_envs = torch.logical_or(dof_vel_low_envs, dof_vel_high_envs)
        return dof_vel_limit_exceeded_envs

    @abc.abstractmethod
    def _check_failure(self) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def _check_success(self) -> torch.Tensor:
        pass

    def reset_idx(self, env_ids: torch.Tensor):
        # randomization can happen only at reset time, since it can reset actor positions on GPU

        if self.env_randomize:
            self.env_randomizer(self.randomization_params, env_ids)

        # A) Reset episode stats buffers
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.randomize_buf[env_ids] = 0
        self._successes[env_ids] = 0
        self.previous_smoothed_action[env_ids] = 0

        # B) Various randomizations at the start of the episode:
        # -- Robot base position.
        # -- Stage position.
        # -- Coefficient of restituion and friction for robot, object, stage.
        # -- Mass and size of the object
        # -- Mass of robot links
        
        object_initial_state_config = self.cfg["env"]["reset_distribution"]["object_initial_state"]
        # if joint training
        # we must save which data is used to reset environment
        if object_initial_state_config["type"] == "pre_contact_policy":
            indices = torch.arange(
                self._init_data_use_count, (self._init_data_use_count + env_ids.shape[0]), dtype=torch.long, device=self.device
            )
            if self.uniform_random_contact:
                indices %= self._robot_buffer.shape[0]
            self._init_data_use_count += env_ids.shape[0]
            self._init_data_indices[env_ids] = indices

        # -- Sampling of height of the table
        if not self.is_hole_wide:
            self._sample_table_poses(env_ids)
        # -- Sampling of initial pose of the object
        self._sample_object_poses(env_ids)
        # -- Sampling of goal pose of the object
        self._sample_object_goal_poses(env_ids)
        # -- Robot joint state
        self._sample_robot_state(env_ids)

        # C) Extract franka indices to reset
        robot_indices = self.actor_indices["robot"][env_ids].to(torch.int32)
        object_indices = self.actor_indices["object"][env_ids].to(torch.int32)
        if not self.is_hole_wide:
            table_indices = self.actor_indices["table"][env_ids].to(torch.int32)
        if not self.cfg['env']["enableCameraSensors"]:
            goal_object_indices = self.actor_indices["goal_object"][env_ids].to(torch.int32)
            if self.is_hole_wide:
                all_indices = torch.unique(torch.cat([robot_indices, object_indices, goal_object_indices]))
            else:
                all_indices = torch.unique(torch.cat([robot_indices, object_indices, goal_object_indices, table_indices]))
        else:
            if self.is_hole_wide:
                all_indices = torch.unique(torch.cat([robot_indices, object_indices]))
            else:
                all_indices = torch.unique(torch.cat([robot_indices, object_indices, table_indices]))
        # D) Set values into simulator
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(robot_indices), len(robot_indices)
        )
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._actors_root_state), gymtorch.unwrap_tensor(all_indices), len(all_indices)
        )

        # envenv = self.envs[0]
        # robot_handle = self.gym.get_actor_handle(envenv, self.actor_indices["robot"][0])
        # print(self.gym.get_actor_dof_properties(envenv, robot_handle))
        
    
    """
    Helper functions - define assets
    """

    def __define_robot_asset(self):
        """ Define Gym asset for robot.
        """
        # define Franka asset
        robot_asset_options = gymapi.AssetOptions()
        robot_asset_options.flip_visual_attachments = True
        robot_asset_options.fix_base_link = True
        robot_asset_options.collapse_fixed_joints = False
        robot_asset_options.disable_gravity = True
        robot_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        robot_asset_options.thickness = 0.001

        # load Franka asset
        Franka_asset = self.gym.load_asset(self.sim, self._assets_dir, "urdf/Panda/robots/franka_panda.urdf", robot_asset_options)
        Franka_props = self.gym.get_asset_rigid_shape_properties(Franka_asset)
        self._left_finger_handle = self.gym.find_asset_rigid_body_index(Franka_asset, "panda_leftfinger")
        self._right_finger_handle = self.gym.find_asset_rigid_body_index(Franka_asset, "panda_rightfinger")
        cnt=0
        for i, p in enumerate(Franka_props):
            if i==self._left_finger_handle or i==self._right_finger_handle:
                p.friction = 2.0
            else: p.friction = 0.1
            p.restitution = 0.5
        self.gym.set_asset_rigid_shape_properties(Franka_asset, Franka_props)
        after_update_props = self.gym.get_asset_rigid_shape_properties(Franka_asset)
        
        if self.cfg["env"]["hand_force_limit"]:
            hand_idx = self.gym.find_asset_rigid_body_index(Franka_asset, "panda_hand")
            hand_force_sensor_pose = gymapi.Transform(gymapi.Vec3(0., 0.0, 0.0))
            hand_force_sensor_props = gymapi.ForceSensorProperties()
            hand_force_sensor_props.enable_forward_dynamics_forces = False
            hand_force_sensor_props.enable_constraint_solver_forces = True
            hand_force_sensor_props.use_world_frame = True
            hand_force_sensor_idx = self.gym.create_asset_force_sensor(Franka_asset, hand_idx, hand_force_sensor_pose, hand_force_sensor_props)

        for frame_name in self._grippers_handles.keys():
            self._grippers_handles[frame_name] = self.gym.find_asset_rigid_body_index(Franka_asset, frame_name)
            # check valid handle
            if self._grippers_handles[frame_name] == gymapi.INVALID_HANDLE:
                msg = f"Invalid handle received for frame: `{frame_name}`."
                print(msg)
        self._hand_handle = self.gym.find_asset_rigid_body_index(Franka_asset, "panda_hand")
        self._tool_handle = self.gym.find_asset_rigid_body_index(Franka_asset, "panda_tool")

        for dof_name in self._robot_dof_indices.keys():
            self._robot_dof_indices[dof_name] = self.gym.find_asset_dof_index(Franka_asset, dof_name)
            # check valid handle
            if self._robot_dof_indices[dof_name] == gymapi.INVALID_HANDLE:
                msg = f"Invalid index received for DOF: `{dof_name}`."
                print(msg)
        # return the asset
        return Franka_asset

    @abc.abstractmethod
    def _define_table_asset(self):
        pass        

    def __define_floor_asset(self):
        """ Define Gym asset for a floor.
        """
        # define table asset
        floor_asset_options = gymapi.AssetOptions()
        floor_asset_options.disable_gravity = True
        floor_asset_options.fix_base_link = True
        floor_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        floor_asset_options.thickness = 0.001
        # load table asset
        floor_asset = self.gym.create_box(self.sim, 2, 5, 0.002, floor_asset_options)

        return floor_asset

    def __define_back_asset(self):
        """ Define Gym asset for the background.
        """
        # define table asset
        back_asset_options = gymapi.AssetOptions()
        back_asset_options.disable_gravity = True
        back_asset_options.fix_base_link = True
        back_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        back_asset_options.thickness = 0.001
        # load table asset
        back_asset = self.gym.create_box(self.sim, 0.002, 2.5, 1, back_asset_options)

        return back_asset

    @abc.abstractmethod
    def _define_object_asset(self):
        pass

    def __define_goal_object_asset(self):
        """ Define Gym asset for goal object.
        """
        if self.cfg['env']["enableCameraSensors"]: return None # Goal object should not be visible while training student policies
        # define object asset
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.disable_gravity = True
        object_asset_options.fix_base_link = True
        object_asset_options.thickness = 0.001
        object_asset_options.flip_visual_attachments = True

        size = self._object_dims.size
        goal_object_asset = self.gym.create_box(self.sim, size[0], size[1], size[2], object_asset_options)
        
        return goal_object_asset

    """
    Helper functions - MDP
    """
    def compute_observations(self):
        """
        Fills observation and state buffer with the current state of the system.
        """
        self.mean_energy = 0.1 * self.mean_energy
        # extract frame handles
        gripper_handles_indices = list(self._grippers_handles.values())
        object_indices = self.actor_indices["object"]
        # update state histories
        self._grippers_frames_state_history.appendleft(self._rigid_body_state[:, gripper_handles_indices, :7])
        self._object_state_history.appendleft(self._actors_root_state[object_indices])
        # self._hand_vel_history[:,-1]=self._rigid_body_state[:,self._hand_handle,7:]
        # fill the observations and states buffer
        self.__compute_teacher_observations()
        if self.cfg["env"]["enableCameraSensors"]:
            self.__compute_camera_observations()
        if self.cfg["env"]["nvisii"]["photorealistic_rendering"]:
            self.__compute_photorealistic_rendering()
        
        # normalize observations if flag is enabled
        if self.cfg['env']["normalize_obs"]:
            # for normal obs
            self.obs_buf = scale_transform(
                self.obs_buf,
                lower=self._observations_scale.low,
                upper=self._observations_scale.high
            )
            if self.asymmetric_obs:
                self.states_buf = scale_transform(
                    self.states_buf,
                    lower=self._states_scale.low,
                    upper=self._states_scale.high
                )

    def __compute_teacher_observations(self):
        gripper_handles_indices = list(self._grippers_handles.values())
        if self.asymmetric_obs:

            # generalized coordinates
            start_offset = 0
            end_offset = start_offset + self._dims.GeneralizedCoordinatesDim.value
            self.states_buf[:, start_offset:end_offset] = self.dof_position

            # generalized velocities
            start_offset = end_offset
            end_offset = start_offset + self._dims.GeneralizedVelocityDim.value
            self.states_buf[:, start_offset:end_offset] = self.dof_velocity

            # object pose as keypoint
            start_offset = end_offset
            end_offset = start_offset + self._dims.KeypointDim.value*self.keypoints_num
            current_keypoints = gen_keypoints(pose=self._object_state_history[0][:, 0:7], size=self._object_dims.size)
            self.states_buf[:, (start_offset):(end_offset)] = current_keypoints.view(self.num_envs, 24)[:]
            
            # use previous keypoint to mimic delay

            # self.states_buf[:, (start_offset):(end_offset)] = self.prev_keypoints.view(self.num_envs, 24)[:]
            # self.prev_keypoints = current_keypoints

            # object desired pose as keypoint
            start_offset = end_offset
            end_offset = start_offset + self._dims.KeypointDim.value*self.keypoints_num
            goal_keypoints = gen_keypoints(pose=self._object_goal_poses_buf, size=self._object_dims.size)
            self.states_buf[:, start_offset:end_offset] = goal_keypoints.view(self.num_envs, 24)[:]
            
            # object velcity
            start_offset = end_offset
            end_offset = start_offset + self._dims.ObjectVelocityDim.value
            self.states_buf[:, start_offset:end_offset] = self._object_state_history[0][:, 7:13]

            # finger poses
            num_fingertip_states = self._dims.NumFingers.value * self._dims.StateDim.value
            start_offset = end_offset
            end_offset = start_offset + num_fingertip_states
            self.states_buf[:, start_offset:end_offset] = self._rigid_body_state[:, gripper_handles_indices, :7].reshape(self.num_envs, num_fingertip_states)

            # joint torque
            start_offset = end_offset
            end_offset = start_offset + self._dims.JointTorqueDim.value
            self.states_buf[:, start_offset:end_offset] = self.dof_torque

            # distance between object and goal
            start_offset = end_offset
            end_offset = start_offset + 1
            self.states_buf[:, start_offset:end_offset] = torch.norm(self._object_state_history[0][:,0:3]-self._object_goal_poses_buf[:,0:3],2,-1).unsqueeze(-1)

            # angle differences between object and goal
            start_offset = end_offset
            end_offset = start_offset + 1
            self.states_buf[:, start_offset:end_offset] = quat_diff_rad(self._object_state_history[0][:,3:7], self._object_goal_poses_buf[:,3:7]).unsqueeze(-1)

            # previous action
            start_offset = end_offset
            end_offset = start_offset + self.action_dim 
            self.states_buf[:, start_offset:end_offset] = self.actions

            ########################## observation
            # robot pose
            start_state_offset = 0
            end_state_offset = start_state_offset + self._dims.GeneralizedCoordinatesDim.value
            self.obs_buf[:, start_state_offset:end_state_offset] = self.dof_position

            # robot velocity
            start_state_offset = end_state_offset
            end_state_offset = start_state_offset + self._dims.GeneralizedVelocityDim.value
            self.obs_buf[:, start_state_offset:end_state_offset] = self.dof_velocity

            # 2D projected object keypoints
            start_state_offset = end_state_offset
            end_state_offset = start_state_offset + self._dims.TwoDimensionKeypointDim.value*self.keypoints_num
            self.obs_buf[:, start_state_offset:end_state_offset] = compute_projected_points(self.translation_from_camera_to_object, current_keypoints, self.camera_matrix, self.device).reshape(self.num_envs, 16)[:]

            # 2D projected goal keypoints
            start_state_offset = end_state_offset
            end_state_offset = start_state_offset + self._dims.TwoDimensionKeypointDim.value*self.keypoints_num
            self.obs_buf[:, start_state_offset:end_state_offset] = compute_projected_points(self.translation_from_camera_to_object, goal_keypoints, self.camera_matrix, self.device).reshape(self.num_envs, 16)[:]

            # hand pose
            start_state_offset = end_state_offset
            end_state_offset = start_state_offset + self._dims.ObjectPoseDim.value
            self.obs_buf[:, start_state_offset:end_state_offset] = self._rigid_body_state[:, self._tool_handle, :7]

            # previous action
            start_state_offset = end_state_offset
            end_state_offset = start_state_offset + self.action_dim
            self.obs_buf[:, start_state_offset:end_state_offset] = self.actions

        else:
            # generalized coordinates
            start_offset = 0
            end_offset = start_offset + self._dims.GeneralizedCoordinatesDim.value
            self.obs_buf[:, start_offset:end_offset] = self.dof_position

            # generalized velocities
            start_offset = end_offset
            end_offset = start_offset + self._dims.GeneralizedVelocityDim.value
            self.obs_buf[:, start_offset:end_offset] = self.dof_velocity

            # 2D projected object keypoints
            start_offset = end_offset
            end_offset = start_offset + self._dims.TwoDimensionKeypointDim.value * self.keypoints_num
            current_keypoints = gen_keypoints(pose=self._object_state_history[0][:, 0:7], size=self._object_dims.size)
            self.obs_buf[:, start_offset:end_offset] = compute_projected_points(self.translation_from_camera_to_object, current_keypoints, self.camera_matrix, self.device).reshape(self.num_envs, 16)[:]

            # 2D projected goal keypoints
            start_offset = end_offset
            end_offset = start_offset + self._dims.TwoDimensionKeypointDim.value * self.keypoints_num
            goal_keypoints = gen_keypoints(pose=self._object_goal_poses_buf, size=self._object_dims.size)
            self.obs_buf[:, start_offset:end_offset] = compute_projected_points(self.translation_from_camera_to_object, goal_keypoints, self.camera_matrix, self.device).reshape(self.num_envs, 16)[:]

            # hand pose
            start_offset = end_offset
            end_offset = start_offset + self._dims.ObjectPoseDim.value
            self.obs_buf[:, start_offset:end_offset] = self._rigid_body_state[:, self._tool_handle, :7]

            # previous action
            start_offset = end_offset
            end_offset = start_offset + self.action_dim 
            self.obs_buf[:, start_offset:end_offset] = self.actions

    def __compute_camera_observations(self):
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        for i in range(self.num_envs):
            if self.cfg['env']["camera"]["segmentation"]:
                segmentation = self._torch_camera_segmentation[i].unsqueeze(-1)
                img = self._torch_camera_rgba_images[i][..., :3]
                # background and floor are maksed as 4 and 5 respectively. Mask them out.
                img[segmentation.repeat((1, 1, 3)) > 3] = 0
                self._segmentation[i] = segmentation
                self._camera_image[i, ..., :3] = img
            else:
                self._camera_image[i, ..., :3] = self._torch_camera_rgba_images[i][..., :3]
        self.gym.end_access_image_tensors(self.sim)
        
    def __init_photorealistic_rendering(self, headless):
        self.photo_count=0

        nv.initialize(headless=headless, window_on_top = False)

        camera = nv.entity.create(
        name = "camera",
        camera = nv.camera.create("camera"),
        transform = nv.transform.create("camera")
        )
        nv.set_camera_entity(camera)

        nv.enable_denoiser()
        # nv.disable_dome_light_sampling()
        nv.set_dome_light_intensity(self.cfg["env"]["nvisii"]["ambient_intensity"]) # TODO: should implement this in cfg file. normal: 0.8, bright:0.95
        nv.set_dome_light_color((1,1,1))

        light1 = nv.entity.create(
        name = "light1",
        transform = nv.transform.create("light1"),
        mesh = nv.mesh.create_plane("light1", flip_z = True),
        light = nv.light.create("light1")
        )
        light1.get_transform().set_position((2,0,2))
        light1.get_transform().set_scale((.5,.15,.15))
        light1.get_light().set_temperature(3000)
        light1.get_light().set_exposure(2)
        light1.get_light().set_intensity(0.8)

        # light2 = nv.entity.create(
        # name = "light2",
        # transform = nv.transform.create("light2"),
        # mesh = nv.mesh.create_plane("light2", flip_z = True),
        # light = nv.light.create("light2")
        # )
        # light2.get_transform().set_position((2,2,2))
        # light2.get_transform().set_scale((.5,.15,.15))
        # light2.get_light().set_temperature(4000)
        # light2.get_light().set_exposure(2)
        # light2.get_light().set_intensity(0.8)

        # light3 = nv.entity.create(
        # name = "light3",
        # transform = nv.transform.create("light3"),
        # mesh = nv.mesh.create_plane("light3", flip_z = True),
        # light = nv.light.create("light3")
        # )
        # light3.get_transform().set_position((2,2,2))
        # light3.get_transform().set_scale((.5,.15,.15))
        # light3.get_light().set_temperature(4000)
        # light3.get_light().set_exposure(2)
        # light3.get_light().set_intensity(0.8)

        env_ptr = self.envs[0]
        camera_handle = self.camera_handles[0]

        camera.get_transform().look_at(
            eye = (0.96, 0, 0.86),
            # eye = (1.0, -0.5, 1.0), # overall view
            at = (0.5, 0.0, 0.86-0.46*math.tan(43.0*np.pi/180.0)),
            up = (0,0,1)
        )
        self.photorealistic_height = self.cfg["env"]["camera"]["size"][0]
        self.photorealistic_width = self.cfg["env"]["camera"]["size"][1]

        camera.get_camera().set_fov(43.0*np.pi/180.0, self.photorealistic_width/self.photorealistic_height)

        self.urdfDir = Path(__file__).parents[2].joinpath('assets', 'urdf', 'Panda')
        self.textureDir = Path(__file__).parents[2].joinpath('assets', 'datas', 'textures')

        # Floor
        floor = nv.entity.create(
            name = "floor",
            mesh = nv.mesh.create_plane("mesh_floor"),
            transform = nv.transform.create("transform_floor"),
            material = nv.material.create("material_floor")
        )

        # Lets make our floor act as a mirror
        floor_mat = floor.get_material()
        # mat = nv.material.get("material_floor") # <- this also works

        # Mirrors are smooth and "metallic".
        floor_mat.set_base_color((0.0,0.0,0.0)) 
        floor_mat.set_metallic(0.3) 
        floor_mat.set_roughness(0.7)

        # Make the floor large by scaling it
        floor_trans = floor.get_transform()
        floor_trans.set_scale((5,5,1))
        floor_trans.set_position(nv.vec3(0.0, 0.0, 0.0))

        # Wall
        wall_name = "wall"
        wall_mesh = nv.mesh.create_box(name=wall_name, size=nv.vec3(0.1, 5.0, 5.0))
        wall = nv.entity.create(name=wall_name, mesh=wall_mesh, transform=nv.transform.create(wall_name), material=nv.material.create(wall_name))
        wall_pos = nv.vec3(-0.5, 0.0, 0.0)
        wall_rot = nv.quat(0, 0, 0, 1)
        wall.get_transform().set_position(wall_pos)
        wall.get_transform().set_rotation(wall_rot)

        wall_mat = wall.get_material()
        wall_mat.set_base_color((0.0,0.0,0.0)) 
        wall_mat.set_metallic(0.3) 
        wall_mat.set_roughness(0.7)

        # Make the floor large by scaling it
        wall_trans = wall.get_transform()
        wall_trans.set_scale((1,5,5))
        wall_trans.set_position(nv.vec3(-2.0, 0.0, 0.0))

        # Table
        table_name = "table"
        table_mesh = nv.mesh.create_box(name=table_name, size=nv.vec3(0.2, 0.25, 0.2))
        table_obj = nv.entity.create(name=table_name, mesh=table_mesh, transform=nv.transform.create(table_name), material=nv.material.create(table_name))
        table_pos = nv.vec3(0.5, 0.0, 0.2)
        table_rot = nv.quat(0, 0, 0, 1)
        table_obj.get_transform().set_position(table_pos)
        table_obj.get_transform().set_rotation(table_rot)
        table_mat = table_obj.get_material()
        table_mat.set_base_color(nv.vec3(1,1,1))

        if self.cfg["env"]["nvisii"]["table_texture"] != 'None':
            tex_name = self.cfg["env"]["nvisii"]["table_texture"]
            tex_dir = os.path.join(self.textureDir, tex_name+'.png')
            tex = nv.texture.create_from_file("tex", tex_dir)
            table_tex = nv.texture.create_hsv("table", tex, hue = 0, saturation = .5, value = 1.0, mix = 1.0)
            table_mat.set_roughness_texture(tex)
            table_mat.set_base_color_texture(table_tex)
            
        elif self.cfg["env"]["nvisii"]["table_reflection"]:
            table_mat.set_metallic(0.5)
            table_mat.set_roughness(0.1)
            table_mat.set_specular(0.9)

        # Panda
        self.nv_links = self.update_joint_angle()

        # Object
        self.card_db = nv.import_scene(
            file_path = str(self.urdfDir.joinpath('meshes/collision/coloredcard.obj')),
            position = nv.vec3(0, 0, 0),
            scale = (1.0, 1.0, 1.0),
            rotation = nv.quat(1, 0, 0, 0)
        )

    def __compute_photorealistic_rendering(self):

        # Only rendering 0th environment.
        self.photo_count = self.photo_count+1

        # Panda
        self.nv_links = self.update_joint_angle(nv_objects=self.nv_links)

        # Object
        self.card_db = self.update_object_pose(nv_object=self.card_db)

        # convert linear RGB to sRGB
        linear_RGB_data = np.asarray(nv.render(self.photorealistic_width, self.photorealistic_height, 128)).reshape((self.photorealistic_height, self.photorealistic_width, -1))[:, :, :3]
        linear_RGB_data = np.flip(linear_RGB_data, 0)
        sRGB_data = self.convert_linear_to_RGB(linear_RGB_data)
        formatted = (sRGB_data * 255.0).astype('uint8')

        im = Image.fromarray(formatted)
        # im.show()

        # Change 0th env camera image into photorealistic image
        self._camera_image[0, :, :, :] = torch.from_numpy(formatted[:, :, :])

    def convert_linear_to_RGB(self, linear: np.array):
            return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * (pow(linear, (1.0 / 2.4))) - 0.055)

    def update_joint_angle(self, nv_objects=None):

        if nv_objects is None:
            nv_objects = { }

        overall_link_pose = self._rigid_body_state[0, :, :7].clone().detach().cpu().numpy()
        for link_name, link_idx in self.franka_link_dict.items():

            if link_idx > 10 : continue # ignore tool frame

            link_pose = overall_link_pose[link_idx, :]
            link_pos = nv.vec3(link_pose[0], link_pose[1], link_pose[2])
            link_rot = nv.quat(link_pose[6], link_pose[3], link_pose[4], link_pose[5])

            obj_file_name = ''
            
            if link_idx < 7:
                obj_file_name = 'link'+str(link_idx)+'.fbx'
            elif link_idx == 7:
                obj_file_name = 'link7.fbx'
            elif link_idx == 8:
                obj_file_name = 'hand.fbx'
            else:
                obj_file_name = 'finger.fbx'

            objDir = self.urdfDir.joinpath('meshes/visual/'+obj_file_name)

            object_name = f"{link_name}_{link_idx}"
            if object_name not in nv_objects:
                # Create mesh component if not yet made
                try:
                    nv_objects[object_name] = nv.import_scene(
                        file_path = str(objDir),
                        position = link_pos,
                        rotation = link_rot
                    )
                        
                except Exception as e:
                    print(e)
                
            if object_name not in nv_objects: continue

            # Link transform
            m1 = nv.translate(nv.mat4(1), link_pos)
            m1 = m1 * nv.mat4_cast(link_rot)

            if link_idx == 10:
                # Visual frame transform for right finger
                m1 = m1 * nv.mat4_cast(nv.quat(0, 0, 0, 1))

            # import scene directly with mesh files
            if isinstance(nv_objects[object_name], nv.scene):
                # Set root transform of visual objects collection to above transform
                nv_objects[object_name].transforms[0].set_transform(m1)

        return nv_objects
    
    def update_object_pose(self, nv_object):
        object_indices = self.actor_indices["object"]
        card_pose = self._actors_root_state[object_indices].cpu().numpy()[0]
        card_pos = nv.vec3(card_pose[0], card_pose[1], card_pose[2])
        card_rot = nv.quat(card_pose[6], card_pose[3], card_pose[4], card_pose[5])

        m1 = nv.translate(nv.mat4(1), card_pos)
        m1 = m1 * nv.mat4_cast(card_rot)
        nv_object.transforms[0].set_transform(m1)
        return nv_object

    def _sample_robot_state(self, reset_env_indices: torch.Tensor):
        """Samples the robot DOF state based on the settings.

        Type of robot initial state distribution: ["default", "random"]
             - "default" means that robot is in default configuration.
             - "pre_contact_policy" means that robot pose is made by pi_pre

        Args:
            reset_env_indices: A tensor contraining indices of environments to reset.
            distribution: Name of distribution to sample initial state from: ['default', 'random']
        """
        self.dof_position[reset_env_indices] = self._robot_buffer[self._init_data_indices[reset_env_indices]]
        self.dof_velocity[reset_env_indices] = torch.zeros(
            (reset_env_indices.shape[0], self._dims.GeneralizedVelocityDim.value), device=self.device
        )
        # reset robot grippers state history
        for idx in range(1, self._state_history_len):
            self._grippers_frames_state_history[idx][reset_env_indices] = 0.0
    
    def _sample_table_poses(self, reset_env_indices: torch.Tensor):
        """Sample poses for the table.

        Args:
            reset_env_indices: A tensor contraining indices of environment instances to reset.
        """
        table_indices = self.actor_indices["table"][reset_env_indices]
        self.table_pose[reset_env_indices, :] = 0.
        self.table_pose[reset_env_indices, 0] = 0.5
        self.table_pose[reset_env_indices, 1] = 0.

        # if self.env_randomize:
        #     self.table_pose[reset_env_indices, 2] = torch.tensor(self.table_z_randomizer(self.randomization_params, reset_env_indices),  dtype=torch.float32, device=self.device)
        # else:
        #     self.table_pose[reset_env_indices, 2] = 0.2
        
        self.table_pose[reset_env_indices, 2] = 0.2

        self.table_pose[reset_env_indices, 3:7] = torch.tensor([0., 0., 0., 1.], device=self.device)

        # root actor buffer
        self._actors_root_state[table_indices] = self.table_pose[reset_env_indices, :]

    def _sample_object_poses(self, reset_env_indices: torch.Tensor):
        """Sample poses for the object.

        Args:
            reset_env_indices: A tensor contraining indices of environment instances to reset.
        """
        object_indices = self.actor_indices["object"][reset_env_indices]
        self._object_state_history[0][reset_env_indices, :7] = self._initial_object_pose_buffer[self._init_data_indices[reset_env_indices]]
        self._object_state_history[0][reset_env_indices, 7:13] = 0

        # if self.env_randomize:
        #     self._object_state_history[0][reset_env_indices, 2] = self.table_pose[reset_env_indices, 2]+0.2025

        # reset object state history
        for idx in range(1, self._state_history_len):
            self._object_state_history[idx][reset_env_indices] = 0.0
        # root actor buffer
        self._actors_root_state[object_indices] = self._object_state_history[0][reset_env_indices]
        if self.cfg['env']["student_obs"] or self.cfg['env']["enableCameraSensors"]:
            self._object_initial_poses_buf[reset_env_indices, :] = self._object_state_history[0][reset_env_indices, :7]

    def _sample_object_goal_poses(self, reset_env_indices: torch.Tensor):
        """Sample goal poses for the object and sets them into the desired goal pose buffer.

        Args:
            reset_env_indices: A tensor contraining indices of environments to reset.
        """
        self._object_goal_poses_buf[reset_env_indices] = self._goal_buffer[self._init_data_indices[reset_env_indices]]
        if not self.cfg['env']["enableCameraSensors"]:
            goal_object_indices = self.actor_indices["goal_object"][reset_env_indices]
            self._actors_root_state[goal_object_indices, 0:7] = self._object_goal_poses_buf[reset_env_indices]

            # if self.env_randomize:
            #     self._actors_root_state[goal_object_indices, 2] = self.table_pose[reset_env_indices, 2]+0.2025

    def push_data(self, initial_object_pose: torch.Tensor, goal_object_pose: torch.Tensor, initial_joint_position: torch.Tensor):
        """
            Fill the object pose, goal pose, and robot configuration made by pi_pre and sampler
        """
        if initial_object_pose.dim() == 3:
            initial_object_pose = initial_object_pose.squeeze(1)
            goal_object_pose = goal_object_pose.squeeze(1)
            initial_joint_position = initial_joint_position.squeeze(1)
        self._robot_buffer = initial_joint_position
        self._initial_object_pose_buffer = initial_object_pose
        self._goal_buffer = goal_object_pose
        self._init_data_use_count = 0
        self._init_data_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
    def axisaToquat(self, axisA: torch.Tensor):

        num_rotations = axisA.shape[0]
        angle = torch.norm(axisA, dim=-1)
        small_angle = (angle <= 1e-3)
        large_angle = ~small_angle

        scale = torch.empty((num_rotations,), device=self.device, dtype=torch.float)
        scale[small_angle] = (0.5 - angle[small_angle] ** 2 / 48 +
                            angle[small_angle] ** 4 / 3840)
        scale[large_angle] = (torch.sin(angle[large_angle] / 2) /
                            angle[large_angle])
        quat = torch.empty((num_rotations, 4), device=self.device, dtype=torch.float)
        quat[:,:3] = scale[:, None] * axisA
        quat[:,-1] = torch.cos(angle/2)
        return quat
    
    def compute_camera_intrinsics_matrix(self, image_width, image_heigth, horizontal_fov, device):
        vertical_fov = (image_heigth / image_width * horizontal_fov) * np.pi / 180
        horizontal_fov *= np.pi / 180

        f_x = (image_width / 2.0) / np.tan(horizontal_fov / 2.0)
        f_y = (image_heigth / 2.0) / np.tan(vertical_fov / 2.0)

        K = torch.tensor([[f_x, 0.0, image_width / 2.0], [0.0, f_y, image_heigth / 2.0], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)

        return K

    @property
    def camera_image(self) -> torch.Tensor:
        image = self._camera_image.detach().clone()
        if self.image_randomize:
            randomized_image: torch.Tensor = self.image_randomizer(self.randomization_params, image.permute((0, 3, 1, 2)), self.segmentation)
            return randomized_image.permute((0, 2, 3, 1))
        else:
            return image

    @property
    def segmentation(self) -> torch.Tensor:
        return self._segmentation.detach().clone()

    @property
    def env_steps_count(self) -> int:
        """Returns the total number of environment steps aggregated across parallel environments."""
        return self.gym.get_frame_count(self.sim) * self.num_envs
    
    @property
    def env_succeed(self) -> torch.Tensor:
        """Returns the succeded infromation of each environment."""
        return self._successes.detach()
    
    @property
    def env_pointing_indices(self) -> torch.Tensor:
        """Returns the indices of data used to reset environments."""
        return self._init_data_indices.detach()
    
    @property
    def env_succeed_count(self) -> torch.Tensor:
        """Returns the total number of environment succeded aggregated across parallel environments."""
        return self._successes_count.detach()

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def axis_angle_from_quat(quat: torch.Tensor, eps: float = 1.0e-6):
    """Convert tensor of quaternions to tensor of axis-angles."""

    mag = torch.linalg.norm(quat[:, 0:3], dim=1)
    half_angle = torch.atan2(mag, quat[:, 3])
    angle = 2.0 * half_angle
    sin_half_angle_over_angle = torch.where(torch.abs(angle) > eps,
                                            torch.sin(half_angle) / angle,
                                            1 / 2 - angle ** 2.0 / 48)
    axis_angle = quat[:, 0:3] / sin_half_angle_over_angle.unsqueeze(-1)

    return axis_angle

@torch.jit.script
def orientation_error(desired: torch.Tensor, current: torch.Tensor, version: int = 2):
    if version==1:
        cc = quat_conjugate(current)
        q_r = quat_mul(desired, cc)
        return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)
    elif version==2:
        quat_norm = quat_mul(current, quat_conjugate(current))[:, 3]  # scalar component
        quat_inv = quat_conjugate(current) / quat_norm.unsqueeze(-1)
        quat_error = quat_mul(desired, quat_inv)

        # Convert to axis-angle error
        axis_angle_error = axis_angle_from_quat(quat_error)
        return axis_angle_error
    else:
        return -(current[:,:3]*desired[:,-1:]-desired[:,:3]*current[:,-1:]+\
            torch.cross(desired[:,:3], current[:,:3],-1))

    
@torch.jit.script
def gen_keypoints(pose: torch.Tensor, num_keypoints: int = 8, size: Tuple[float, float, float] = (0.065, 0.065, 0.065)):
    num_envs = pose.shape[0]
    keypoints_buf = torch.ones(num_envs, num_keypoints, 3, dtype=torch.float32, device=pose.device)
    for i in range(num_keypoints):
        # which dimensions to negate
        n = [((i >> k) & 1) == 0 for k in range(3)]
        corner_loc = [(1 if n[k] else -1) * s / 2 for k, s in enumerate(size)],
        corner = torch.tensor(corner_loc, dtype=torch.float32, device=pose.device) * keypoints_buf[:, i, :]
        keypoints_buf[:, i, :] = local_to_world_space(corner, pose)
    return keypoints_buf

@torch.jit.script
def compute_card_reward(
    obs_buf: torch.Tensor,
    reset_buf: torch.Tensor,
    progress_buf: torch.Tensor,
    episode_length: int,
    object_dist_weight: Tuple[float, float],
    epsilon: Tuple[float, float],
    object_goal_poses_buf: torch.Tensor,
    max_torque: torch.Tensor,
    mean_energy: torch.Tensor,
    object_state: torch.Tensor,
    gripper_state: torch.Tensor,
    size: Tuple[float, float, float],
    use_inductive_reward: bool,
    use_energy_reward: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # distance from each finger to the centroid of the object, shape (N, 3).
    gripper_state = torch.mean(gripper_state, 1)
    curr_norms = torch.norm(gripper_state[:, 0:3] - object_state[:, 0:3], p=2, dim=-1)
    
    last_action = torch.norm(obs_buf[:, -14:-7], 2, -1)
    residual = torch.norm(obs_buf[:, :7], 2, -1)

    # Reward for object distance
    object_keypoints = gen_keypoints(pose=object_state[:, 0:7], size=size)
    goal_keypoints = gen_keypoints(pose=object_goal_poses_buf[:, 0:7], size=size)
    delta = object_keypoints - goal_keypoints
    dist = torch.norm(delta, p=2, dim=-1)
    object_dist_reward = torch.sum((object_dist_weight[0] / (dist + epsilon[0])), -1)
    obj_reward = object_dist_reward

    if use_energy_reward:
        total_reward = obj_reward - 0.01 * mean_energy
    else:
        total_reward = obj_reward - 1. * last_action #- 0.5 * residual

    if use_inductive_reward:
        total_reward += 0.03 / (curr_norms + epsilon[0])

    # reset agents
    reset = torch.zeros_like(reset_buf)
    reset = torch.where((progress_buf >= episode_length - 1), torch.ones_like(reset_buf), reset)

    return total_reward, reset, last_action

@torch.jit.script
def control_osc(num_envs: int, j_eef, hand_vel, mm, dpose, dof_vel, dof_pos, kp, damping_ratio, variable: bool, decouple: bool, device: str):
    null=False
    kd = 2.0*torch.sqrt(kp)*damping_ratio if variable else 2.0 *math.sqrt(kp)
    kp_null = 10.
    kd_null = 2.0 *math.sqrt(kp_null)
    mm_inv = torch.inverse(mm)
    error = (kp * dpose - kd * hand_vel).unsqueeze(-1)

    if decouple:
        m_eef_pos_inv = j_eef[:, :3, :] @ mm_inv @ torch.transpose(j_eef[:, :3, :], 1, 2)
        m_eef_ori_inv = j_eef[:, 3:, :] @ mm_inv @ torch.transpose(j_eef[:, 3:, :], 1, 2)
        m_eef_pos = torch.inverse(m_eef_pos_inv)
        m_eef_ori = torch.inverse(m_eef_ori_inv)
        wrench_pos = m_eef_pos @ error[:, :3, :]
        wrench_ori = m_eef_ori @ error[:, 3:, :]
        wrench = torch.cat([wrench_pos, wrench_ori], dim=1)
    else:
        m_eef_inv = j_eef@ mm_inv @ torch.transpose(j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)
        wrench = m_eef @ error

    u = torch.transpose(j_eef, 1, 2) @ wrench
    
    return u.squeeze(-1)

@torch.jit.script
def compute_projected_points(T_matrix: torch.Tensor, keypoints: torch.Tensor, camera_matrix: torch.Tensor, device: str, num_points: int=8):
    num_envs=keypoints.shape[0]
    p_CO=torch.matmul(T_matrix, torch.cat([keypoints,torch.ones((num_envs, num_points,1),device=device)],-1).transpose(1,2))
    image_coordinates=torch.matmul(camera_matrix, p_CO).transpose(1,2)
    mapped_coordinates=image_coordinates[:,:,:2]/(image_coordinates[:,:,2].unsqueeze(-1))
    return mapped_coordinates

@torch.jit.script
def compute_friction(num_envs: int, joint_vel:torch.Tensor, rho1: torch.Tensor, rho2: torch.Tensor, rho3: torch.Tensor, device:str="cuda:0") ->torch.Tensor:
    friction_torque=rho1*(torch.sigmoid(rho2*(joint_vel+rho3))-torch.sigmoid(rho2*rho3))
    return friction_torque
