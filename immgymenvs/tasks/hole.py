import pickle
from shutil import move
import math
import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from collections import OrderedDict

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from utils.torch_jit_utils import *
from tasks.base.vec_task import VecTask
from types import SimpleNamespace
from collections import deque
from typing import Deque, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt

# python
import enum
import glob
import random
import time
# ################### #
# Dimensions of robot #
# ################### #


class FrankaDimensions(enum.Enum):
    """
    Dimensions of the Franka with gripper robot.

    """
    # general state
    # cartesian position + quaternion orientation
    PoseDim = 7,
    # linear velocity + angular velcoity
    VelocityDim = 6

    #position of keypoint
    KeypointDim=3

    # state: pose + velocity
    StateDim = 13
    # force + torque
    WrenchDim = 6
    NumFingers = 2
    # for all joints
    #TODO put exact size for UR5 with gripper
    JointPositionDim = 9 # the number of joint
    JointVelocityDim = 9 # the number of joint
    JointTorqueDim = 9 # the number of joint
    # generalized coordinates
    GeneralizedCoordinatesDim = JointPositionDim
    GeneralizedVelocityDim = JointVelocityDim
    # for objects
    ObjectPoseDim = 7
    ObjectVelocityDim = 6

# ################# #
# Different objects #
# ################# #

goal_pos=[     0.4723,     -0.0006,      0.7245]
goal_rot=[-0.6928, -0.2801,  0.2632,  0.6101]
# initial=[    -0.0050,     -0.2170,      0.0065,     -2.1196,      0.0004,
#               2.0273,      0.7912,      0.0000,      0.0000]
initial=[    -0.0702,
             -0.3993,
              0.0599,
             -2.2159,
              0.0026,
              2.2675,
              0.7870,
              0.0000,
              0.0000]
#Table size


class Object:
    """
    Fields for a cuboidal shape object.
    """
    # 3D radius of the cuboid
    radius_3d: float
    # distance from wall to the center
    max_com_distance_to_center: float
    # height for spawning the object
    max_height: float
    min_height: float
    max_width: float
    max_length: float

    def __init__(self, size: Tuple[float, float, float]):
        """Initialize the cuboidal object.

        Args:
            size: The size of the object along x, y, z in meters. 
        """
        self._size = size       
        # compute remaning attributes

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
       
        # compute attributes





class Hole(VecTask):
    """
    The current environment supports the following objects:
        - cuboid: object used in the phase 3 of the competition
    """
    # constants
    # directory where assets for the simulator are present
    _Franka_assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
    # robot urdf (path relative to `_Franka_assets_dir`)
    _robot_urdf_file = "urdf/Panda/robots/franka_panda.urdf"
    # Define size of the obejct
    # dimensions of the system
    _dims = FrankaDimensions
    _state_history_len = 2
    _gripper_limits: dict={
            "gripper_position": SimpleNamespace(
            low=-np.array([-1, -1,0], dtype=np.float32),
            high=np.array([1, 1,1], dtype=np.float32),
            ),
            "gripper_orientation": SimpleNamespace(
                low=-np.ones(4, dtype=np.float32),
                high=np.ones(4, dtype=np.float32),
            ),
            "gripper_velocity": SimpleNamespace(
                low=np.full(_dims.VelocityDim.value, -3, dtype=np.float32),
                high=np.full(_dims.VelocityDim.value, 3, dtype=np.float32),
            ),
        }
    _object_limits: dict = {
            # "position": SimpleNamespace(
            #     low=np.array([table_pose.p.x-0.5*table_dims.x, -0.5*table_dims.y, 0.5], dtype=np.float32),
            #     high=np.array([table_pose.p.x+0.5*table_dims.x, 0.5*table_dims.y, 1], dtype=np.float32),
            #     default=np.array([table_pose.p.x, 0, table_dims.z+_object_dims.size[2]*0.5], dtype=np.float32)
            # ),
            "position": SimpleNamespace(
                low=np.array([-1,-1,0], dtype=np.float32),
                high=np.array([1,1,1], dtype=np.float32),
            ),
            "orientation": SimpleNamespace(
                low=-np.ones(4, dtype=np.float32),
                high=np.ones(4, dtype=np.float32),
                default=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            ),
            "velocity": SimpleNamespace(
                low=np.full(_dims.VelocityDim.value, -1, dtype=np.float32),
                high=np.full(_dims.VelocityDim.value, 1, dtype=np.float32),
                default=np.zeros(_dims.VelocityDim.value, dtype=np.float32)
            ),
        }
    # buffers to store the simulation data
    # goal poses for the object [num. of instances, 7] where 7: (x, y, z, quat)
    _object_goal_poses_buf: torch.Tensor
    # buffer to store the per-timestep rotation of the goal position, if this is enabled

    # DOF state of the system [num. of instances, num. of dof, 2] where last index: pos, vel
    _dof_state: torch.Tensor
    # Rigid body state of the system [num. of instances, num. of bodies, 13] where 13: (x, y, z, quat, v, omega)
    _rigid_body_state: torch.Tensor
    # Root prim states [num. of actors, 13] where 13: (x, y, z, quat, v, omega)
    _actors_root_state: torch.Tensor
    # Force-torque sensor array [num. of instances, num. of bodies * wrench]
    #_ft_sensors_values: torch.Tensor
    # DOF position of the system [num. of instances, num. of dof]
    _dof_position: torch.Tensor
    # DOF velocity of the system [num. of instances, num. of dof]
    _dof_velocity: torch.Tensor
    # DOF torque of the system [num. of instances, num. of dof]
    _dof_torque: torch.Tensor
    # gripper links state list([num. of instances, num. of fingers, 13]) where 13: (x, y, z, quat, v, omega)
    # The length of list is the history of the state: 0: t, 1: t-1, 2: t-2, ... step.
    _grippers_frames_state_history: Deque[torch.Tensor] = deque(maxlen=_state_history_len)
    _hand_frames_state_history: Deque[torch.Tensor] = deque(maxlen=_state_history_len)
    # Object prim state [num. of instances, 13] where 13: (x, y, z, quat, v, omega)
    # The length of list is the history of the state: 0: t, 1: t-1, 2: t-2, ... step.
    _object_state_history: Deque[torch.Tensor] = deque(maxlen=_state_history_len)
    # keeps track of the number of goal resets
    _successes: torch.Tensor

    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        """Initializes the card environment configure the buffers.

        Args:
            cfg: Dictionory containing the configuration (default: /home/user/rl/IsaacGymEnvs/isaacgymenvs/cfg/task/Card.yaml).
            sim_device: Torch device to store created buffers at (cpu/gpu).
            graphics_device_id: device to render the objects
            headless: if True, it will open the GUI, otherwise, it will just run the server.
        """
        # load default config
        self.cfg=cfg
        # define spaces for the environment
        object_dims=self.cfg["env"]["geometry"]["object"]

        self. _object_dims = Object((object_dims["width"], object_dims["length"], object_dims["height"]))
        self.action_dim = 19 if self.cfg["env"]["command_mode"] == 'variable' else 7

        # observations
        if self.cfg['env']['keypoint']['activate']:
            self.keypoints_num=int(self.cfg['env']['keypoint']['num'])
            self.obs_spec = {
            # robot joint
            "robot_q": self._dims.GeneralizedCoordinatesDim.value,
            # robot joint velocity
            "robot_u": self._dims.GeneralizedVelocityDim.value,
            # object position by kepoints
            "object_q": self._dims.KeypointDim.value*self.keypoints_num,
            # object goal position by kepoints
            "object_q_des": self._dims.KeypointDim.value*self.keypoints_num,
            # object velocity(+ angular velocity)
            "object_u": self._dims.ObjectVelocityDim.value,
            # gripper pose + velocity 13 dimes
            "gripper_state": self._dims.NumFingers.value * self._dims.StateDim.value,
            # robot joint acceleration
            "robot_a": self._dims.GeneralizedVelocityDim.value,
            #"gripper_wrench": self._dims.NumFingers.value * self._dims.WrenchDim.value,
            #distance from object to goal
            "dist_diff": 1,
            #radian differnces between object and goal
            "radian_diff": 1,
            # previous policy command
            "command": self.action_dim
        }
        else:
            self.obs_spec = {
                # robot joint
                "robot_q": self._dims.GeneralizedCoordinatesDim.value,
                # robot joint velocity
                "robot_u": self._dims.GeneralizedVelocityDim.value,
                # object position(+ rotation)
                "object_q": self._dims.ObjectPoseDim.value,
                # object goal position(+ rotation)
                "object_q_des": self._dims.ObjectPoseDim.value,
                # object velocity(+ angular velocity)
                "object_u": self._dims.ObjectVelocityDim.value,
                # gripper pose + velocity 13 dimes
                "gripper_state": self._dims.NumFingers.value * self._dims.StateDim.value,
                # robot joint acceleration
                "robot_a": self._dims.GeneralizedVelocityDim.value,
                #"gripper_wrench": self._dims.NumFingers.value * self._dims.WrenchDim.value,
                #distance from object to goal
                "dist_diff": 1,
                #radian differnces between object and goal
                "radian_diff": 1,
                # previous policy command
                "command": self.action_dim
            }
        # state is same
        #self.state_spec = self.obs_spec
        # actions
        self.action_spec = {
            "command":  self.action_dim
        }
        self.cfg["env"]["numObservations"] = sum(self.obs_spec.values())
        #self.cfg["env"]["numStates"] = sum(self.state_spec.values())
        self.cfg["env"]["numActions"] = sum(self.action_spec.values())
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.randomize = False
        # define prims present in the scene
        self.boxes=self.cfg["env"]["geometry"]["boxes"]
        prim_names = ["robot", "object", "goal_object"]+list(self.boxes.keys())
        # mapping from name to asset instance
        self._gym_assets = dict.fromkeys(prim_names)
        # mapping from name to gym indices
        self.gym_indices = dict.fromkeys(prim_names)
        # mapping from name to gym rigid body handles
        # name of finger tips links i.e. end-effector frames
        grippers_frames = ["panda_leftfinger", "panda_rightfinger"]
        self._grippers_handles = OrderedDict.fromkeys(grippers_frames, None)
        # mapping from name to gym dof index
        robot_dof_names = list()
        for i in range(1,8):
            robot_dof_names+=[f'panda_joint{i}']
        robot_dof_names+=['panda_finger_joint1', 'panda_finger_joint2']
        self._robot_dof_indices = OrderedDict.fromkeys(robot_dof_names, None)
        
            
        #During initialization its parent create_sim is called
        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

        # initialize the buffers
        self.__initialize()
        
        if self.viewer != None:
            cam_pos = gymapi.Vec3(0.7, 0.0, 0.7)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
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
        
        # tensor to store whether polciy succeded task; it's not useful now
        self._successes = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # set the mdp spaces
        self.__configure_mdp_spaces()

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

    def pre_physics_step(self, actions):
        """
        Setting of input actions into simulator before performing the physics simulation step.

        """
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()

        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        self.actions = actions.clone().to(self.device)
        self.actions[env_ids,:]=0.0
        # if normalized_action is true, then denormalize them.

        if self.cfg['env']["normalize_action"]:
            action_transformed = unscale_transform(
                self.actions,
                lower=self._action_scale.low,
                upper=self._action_scale.high
            )
        else:
            action_transformed = self.actions

        # compute command on the basis of mode selected
        if self.cfg["env"]["command_mode"] == 'ik':
            # command is the change in hand calculate 
            computed_torque=torch.zeros(self.num_envs, self._dims.JointTorqueDim.value,
                dtype=torch.float32, device=self.device)
            ik=control_ik(self.num_envs, self.j_eef, action_transformed[:, :6], self.device)
            computed_torque[:,:7] = self.franka_dof_stiffness[:7]*ik
            computed_torque[:,7:] = self.franka_dof_stiffness[7]*action_transformed[:,6:7]
            computed_torque-=self.franka_dof_damping* self._dof_velocity

        elif self.cfg["env"]["command_mode"] == 'osc':
            # command is the desired joint positions
            computed_torque=torch.zeros(self.num_envs, self._dims.JointTorqueDim.value,
                dtype=torch.float32, device=self.device)
            hand_vel=self._rigid_body_state[:,self._hand_handel:self._hand_handel+1,7:]
            ct = control_osc(self.num_envs, self.j_eef, hand_vel.squeeze(-2), self.mm, action_transformed[:, :6],
                self.obs_buf[:, 9:18], self.obs_buf[:, :9], 150, 1, self.cfg["env"]["command_mode"] == 'variable', self.device)
            computed_torque[:,:7]=ct
            # compute torque to apply
            computed_torque[:,7:] = self.franka_dof_stiffness[7]*action_transformed[:,6:7]
            computed_torque[:,7:]-=self.franka_dof_damping[7]* self._dof_velocity[:,6:7]
        elif self.cfg["env"]["command_mode"] == 'variable':
            # command is the desired joint positions
            computed_torque=torch.zeros(self.num_envs, self._dims.JointTorqueDim.value,
                dtype=torch.float32, device=self.device)
            hand_vel=self._rigid_body_state[:,self._hand_handel:self._hand_handel+1,7:]
            ct = control_osc(self.num_envs, self.j_eef, hand_vel.squeeze(-2), self.mm, action_transformed[:, :6],
                self.obs_buf[:, 9:18], self.obs_buf[:, :9], action_transformed[:, 6:12], action_transformed[:, 12:18],
                True, self.device)
            computed_torque[:,:7]=ct
            # compute torque to apply
            computed_torque[:,7:] = self.franka_dof_stiffness[7]*action_transformed[:,6:7]
            computed_torque[:,7:]-=self.franka_dof_damping[7]* self._dof_velocity[:,-2:]
        # apply clamping of computed torque to actuator limits
        self.computed_torque=computed_torque
        applied_torque = saturate(
            computed_torque,
            lower=-self.franka_dof_effort_scales,
            upper=self.franka_dof_effort_scales
        )
        #in oreder to bind both fingers together    
        # set computed torques to simulator buffer.
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(applied_torque))

    def post_physics_step(self):
        """
        Setting of buffers after performing the physics simulation step.

        @note Also need to update the reset buffer for the instances that have terminated.
              The termination conditions to check are besides the episode timeout.

        """
        # count step for each environment
        self.progress_buf += 1
        # fill observations buffer
        self.compute_observations()

        # compute rewards
        self.compute_reward(self.actions)
        
        # check termination e.g. box is dropped from table.
        self._check_termination()

    def compute_reward(self, actions):
        self.rew_buf[:] = 0.
        self.reset_buf[:] = 0.
        self.rew_buf[:], self.reset_buf[:], _ = compute_card_reward(
            self.obs_buf,
            self.reset_buf,
            self.progress_buf,
            self.max_episode_length,
            self.cfg["sim"]["dt"],
            self.cfg['env']["reward_terms"]['gripper_reach_object_rate']['weight'],
            (self.cfg["env"]["reward_terms"]["object_dist"]["weight1"], self.cfg["env"]["reward_terms"]["object_dist"]["weight2"]),
            (self.cfg["env"]["reward_terms"]["object_rot"]["weight1"], self.cfg["env"]["reward_terms"]["object_rot"]["weight2"]),
            (self.cfg["env"]["reward_terms"]["obejct_move"]["weight1"], self.cfg["env"]["reward_terms"]["obejct_move"]["weight2"]),
            (self.cfg["env"]["reward_terms"]["object_dist"]["th"], self.cfg["env"]["reward_terms"]["object_rot"]["th"]),
            (self.cfg["env"]["reward_terms"]["object_dist"]["epsilon"], self.cfg["env"]["reward_terms"]["object_rot"]["epsilon"]),
            self.env_steps_count,
            self._object_goal_poses_buf,
            self._object_state_history[0],
            self._object_state_history[1],
            self._grippers_frames_state_history[0],
            self._grippers_frames_state_history[1],
            self.cfg["env"]["keypoint"]["activate"],
            self._object_dims.size,
            self.computed_torque,
        )      

    """
    Private functions
    """

    def __initialize(self):
        """Allocate memory to various buffers.
        """
        # store the sampled goal poses for the object: [num. of instances, 7]
        self._object_goal_poses_buf = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float)
        
        # get force torque sensor if enabled
        if self.cfg['env']["enable_ft_sensors"]:
            # joint torques
            dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            self._dof_torque = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs,
                                                                           self._dims.JointTorqueDim.value)
            # force-torque sensor
            # num_ft_dims = self._dims.NumFingers.value * self._dims.WrenchDim.value
            # sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            # self._ft_sensors_values = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, num_ft_dims)
        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        jacobian = self.gym.acquire_jacobian_tensor(self.sim, "robot")
        mm=self.gym.acquire_mass_matrix_tensor(self.sim, "robot")
        
        # refresh the buffer 
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        
        # create wrapper tensors for reference (consider everything as pointer to actual memory)
        # DOF
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(self.num_envs, -1, 2)
        self._dof_position = self._dof_state[..., 0]
        self._dof_velocity = self._dof_state[..., 1]
        # rigid body
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        # root actors
        self._actors_root_state = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        # jacobian
        self._jacobian = gymtorch.wrap_tensor(jacobian)
        self.j_eef = self._jacobian[:,self.franka_hand_index-1,:,:7]
        # mass matirx
        self.mm = gymtorch.wrap_tensor(mm)
        self.mm = self.mm[:, :self.franka_hand_index-1, :self.franka_hand_index-1]

        # frames history
        gripper_handles_indices = list(self._grippers_handles.values())
        object_indices = self.gym_indices["object"]
        # timestep 0 is current tensor
        curr_history_length = 0

        while curr_history_length < self._state_history_len:
            # add tensors to history list
            self._grippers_frames_state_history.append(self._rigid_body_state[:, gripper_handles_indices])
            self._object_state_history.append(self._actors_root_state[object_indices])
            # update current history length
            curr_history_length += 1
        self._observations_scale = SimpleNamespace(low=None, high=None)
        self._action_scale = SimpleNamespace(low=None, high=None)

        self._successes = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
    
    
    def __configure_mdp_spaces(self):
        """
        Configures the observations, action spaces.
        """
        # Action scale for the MDP
        # Note: This is order sensitive.
       
        # action space is residual pose of eef and gripper position
        if self.cfg["env"]["command_mode"] == 'variable':
            #plus gain
            self._action_scale.low = to_torch([-0.1]*3+[-0.5]*3+[0.0]+[10.0]*6+[0.0]*6,device=self.device)
            self._action_scale.high = to_torch([0.1]*3+[0.5]*3+[0.04]+[300.0]*6+[2.0]*6,device=self.device)
        else:
            self._action_scale.low = to_torch([-0.1]*3+[-0.5]*3+[0.0],device=self.device)
            self._action_scale.high = to_torch([0.1]*3+[0.5]*3+[0.04],device=self.device)

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
            self._object_limits["position"].low.repeat(8),
            self._object_limits["position"].low.repeat(8),
            self._object_limits["velocity"].low,
            self._gripper_limits["gripper_position"].low,
            self._gripper_limits["gripper_orientation"].low,
            self._gripper_limits["gripper_velocity"].low,
            self._gripper_limits["gripper_position"].low,
            self._gripper_limits["gripper_orientation"].low,
            self._gripper_limits["gripper_velocity"].low,
            torch.zeros(1,dtype=torch.float32, device=self.device),
            torch.zeros(1,dtype=torch.float32, device=self.device),
            -self.franka_dof_effort_scales,
            #self._robot_limits["gripper_wrench"].low.repeat(self._dims.NumFingers.value),
            obs_action_scale.low
            ])
            self._observations_scale.high = torch.cat([
                self.franka_dof_upper_limits,
                self.franka_dof_speed_scales,
                self._object_limits["position"].high.repeat(8),
                self._object_limits["position"].high.repeat(8),
                self._object_limits["velocity"].high,
                self._gripper_limits["gripper_position"].high,
                self._gripper_limits["gripper_orientation"].high,
                self._gripper_limits["gripper_velocity"].high,
                self._gripper_limits["gripper_position"].high,
                self._gripper_limits["gripper_orientation"].high,
                self._gripper_limits["gripper_velocity"].high,
                self.franka_dof_effort_scales,
                #self._robot_limits["gripper_wrench"].high.repeat(self._dims.NumFingers.value),
                np.sqrt(0.4**2+0.5**2+0.1**2)*torch.ones(1,dtype=torch.float32, device=self.device),
                2*np.pi*torch.ones(1,dtype=torch.float32, device=self.device),
                obs_action_scale.high
            ])
        else:
            self._observations_scale.low = torch.cat([
            self.franka_dof_lower_limits,
            -self.franka_dof_speed_scales,
            self._object_limits["position"].low,
            self._object_limits["orientation"].low,
            self._object_limits["position"].low,
            self._object_limits["orientation"].low,
            self._object_limits["velocity"].low,
            self._gripper_limits["gripper_position"].low,
            self._gripper_limits["gripper_orientation"].low,
            self._gripper_limits["gripper_velocity"].low,
            self._gripper_limits["gripper_position"].low,
            self._gripper_limits["gripper_orientation"].low,
            self._gripper_limits["gripper_velocity"].low,
            -self.franka_dof_effort_scales,
            #self._robot_limits["gripper_wrench"].low.repeat(self._dims.NumFingers.value),
            torch.zeros(1,dtype=torch.float32, device=self.device),
            torch.zeros(1,dtype=torch.float32, device=self.device),
            obs_action_scale.low
            ])
            self._observations_scale.high = torch.cat([
                self.franka_dof_upper_limits,
                self.franka_dof_speed_scales,
                self._object_limits["position"].high,
                self._object_limits["orientation"].high,
                self._object_limits["position"].high,
                self._object_limits["orientation"].high,
                self._object_limits["velocity"].high,
                self._gripper_limits["gripper_position"].high,
                self._gripper_limits["gripper_orientation"].high,
                self._gripper_limits["gripper_velocity"].high,
                self._gripper_limits["gripper_position"].high,
                self._gripper_limits["gripper_orientation"].high,
                self._gripper_limits["gripper_velocity"].high,
                self.franka_dof_effort_scales,
                #self._robot_limits["gripper_wrench"].high.repeat(self._dims.NumFingers.value),
                np.sqrt(0.4**2+0.5**2+0.1**2)*torch.ones(1,dtype=torch.float32, device=self.device),
                2*np.pi*torch.ones(1,dtype=torch.float32, device=self.device),
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
        # state
        # actions
        if self._action_scale.low.shape[0] != action_dim or self._action_scale.high.shape[0] != action_dim:
            msg = f"Actions scaling dimensions mismatch. " \
                  f"\tLow: {self._action_scale.low.shape[0]}, " \
                  f"\tHigh: {self._action_scale.high.shape[0]}, " \
                  f"\tExpected: {action_dim}."
            raise AssertionError(msg)
        # print the scaling
        print(f'MDP Raw observation bounds\n'
                    f'\tLow: {self._observations_scale.low}\n'
                    f'\tHigh: {self._observations_scale.high}')
        print(f'MDP Raw action bounds\n'
                    f'\tLow: {self._action_scale.low}\n'
                    f'\tHigh: {self._action_scale.high}')

    def __create_scene_assets(self):
        """ Define Gym assets for table, robot and object.
        """
        # define assets
        self._gym_assets["robot"] = self.__define_robot_asset()
        self.__define_table_asset()
        self._gym_assets["object"] = self.__define_object_asset()
        self._gym_assets["goal_object"] = self.__define_goal_object_asset()
        # display the properties (only for debugging)
        # robot
        print("Card Robot Asset: ")
        print(f'\t Number of bodies: {self.gym.get_asset_rigid_body_count(self._gym_assets["robot"])}')
        print(f'\t Number of shapes: {self.gym.get_asset_rigid_shape_count(self._gym_assets["robot"])}')
        print(f'\t Number of dofs: {self.gym.get_asset_dof_count(self._gym_assets["robot"])}')
        print(f'\t Number of actuated dofs: {self._dims.JointTorqueDim.value}')
        # table
        #print("Card table Asset: ")
        # print(f'\t Number of bodies: {self.gym.get_asset_rigid_body_count(self._gym_assets["table"])}')
        # print(f'\t Number of shapes: {self.gym.get_asset_rigid_shape_count(self._gym_assets["table"])}')

    def __create_envs(self):
        """Create various instances for the environment.
        """
        robot_dof_props = self.gym.get_asset_dof_properties(self._gym_assets["robot"])
        # set dof properites based on the control mode
        self.franka_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400, 400, 1.0e6, 1.0e6], dtype=torch.float, device=self.device)
        self.franka_dof_damping = to_torch([80, 80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self.franka_dof_speed_scales=[]
        self.franka_dof_effort_scales=[]
        for k, dof_index in enumerate(self._robot_dof_indices.values()):
            robot_dof_props['driveMode'][dof_index] = gymapi.DOF_MODE_EFFORT
            robot_dof_props['stiffness'][dof_index] = 0.0
            robot_dof_props['damping'][dof_index] = 0.0
            self.franka_dof_lower_limits.append(robot_dof_props['lower'][k])
            self.franka_dof_upper_limits.append(robot_dof_props['upper'][k])
            self.franka_dof_speed_scales.append(robot_dof_props['velocity'][k])
            self.franka_dof_effort_scales.append(robot_dof_props['effort'][k])
        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self.franka_dof_speed_scales = to_torch(self.franka_dof_speed_scales, device=self.device)
        self.franka_dof_speed_scales[[7, 8]] = 0.1
        robot_dof_props['effort'][7] = 200
        robot_dof_props['effort'][8] = 200
        self.franka_dof_effort_scales=to_torch(self.franka_dof_effort_scales, device=self.device)
        self.franka_dof_effort_scales[[7,8]]=200
        franka_mids = 0.3 * (self.franka_dof_lower_limits + self.franka_dof_upper_limits)
        self.franka_default_dof_pos=franka_mids
        self.franka_default_dof_pos[7:]=self.franka_dof_lower_limits[7:]
        self.franka_default_dof_pos = to_torch(self.franka_default_dof_pos, device=self.device)
        self.envs = []
        # define lower and upper region bound for each environment
        env_lower_bound = gymapi.Vec3(-self.cfg['env']["envSpacing"], -self.cfg['env']["envSpacing"], 0.0)
        env_upper_bound = gymapi.Vec3(self.cfg['env']["envSpacing"], self.cfg['env']["envSpacing"], self.cfg['env']["envSpacing"])
        num_instances_per_row = int(np.sqrt(self.num_envs))
        # initialize gym indices buffer as a list
        # note: later the list is converted to torch tensor for ease in interfacing with IsaacGym.
        for asset_name in self.gym_indices.keys():
            self.gym_indices[asset_name] = list()
        # count number of shapes and bodies
        max_agg_bodies = 0
        max_agg_shapes = 0
        for asset in self._gym_assets.values():
            max_agg_bodies += self.gym.get_asset_rigid_body_count(asset)
            max_agg_shapes += self.gym.get_asset_rigid_shape_count(asset)
        # iterate and create environment instances
        
        for env_index in range(self.num_envs):
            # create environment
            env_ptr = self.gym.create_env(self.sim, env_lower_bound, env_upper_bound, num_instances_per_row)
            # begin aggregration mode if enabled
            if self.cfg['env']["aggregateMode"]:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            # add trifinger robot to environment
            robot_pose=gymapi.Transform()
            robot_pose.p=gymapi.Vec3(0,0,0)
            robot_pose.r=gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            Franka_actor = self.gym.create_actor(env_ptr, self._gym_assets["robot"], robot_pose,
                                                     "robot", env_index, 0, 0)
            Franka_idx = self.gym.get_actor_index(env_ptr, Franka_actor, gymapi.DOMAIN_SIM)
            # add table to environment
            table_color = gymapi.Vec3(0.73, 0.68, 0.72)
            for key, item in self.boxes.items():
                if key=="box1":
                    x=0.7-(0.2-self.holelength/2)/2
                    y=item["y"]
                    z=item["z"]
                elif key=="box2":
                    x=0.3+(0.2-self.holelength/2)/2
                    y=item["y"]
                    z=item["z"]
                elif key=="box3":
                    x=item["x"]
                    y=0.25-(0.25-self.holelength/2)/2
                    z=item["z"]
                elif key=="box4":
                    x=item["x"]
                    y=-0.25+(0.25-self.holelength/2)/2
                    z=item["z"]
                else:
                    x=item["x"]
                    y=item["y"]
                    z=item["z"]
                box_pose=gymapi.Transform()
                box_pose.p=gymapi.Vec3(x,y,z)
                box_pose.r=gymapi.Quat(0.0, 0.0, 0.0, 1.0)
                box_handle = self.gym.create_actor(env_ptr, self._gym_assets[key], box_pose,
                                                   key, env_index, 1, 1)
                box_idx = self.gym.get_actor_index(env_ptr, box_handle, gymapi.DOMAIN_SIM)
                self.gym.set_rigid_body_color(env_ptr, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, table_color)
                self.gym_indices[key].append(box_idx)
            # add object to environment
            object_handle = self.gym.create_actor(env_ptr, self._gym_assets["object"], gymapi.Transform(),
                                                   "object", env_index, 0, 2)
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            # add goal object to environment
            goal_handle = self.gym.create_actor(env_ptr, self._gym_assets["goal_object"], gymapi.Transform(),
                                                 "goal_object", env_index + self.num_envs, 0, 2)
            goal_color=gymapi.Vec3(0.3, 0.3, 0.3)
            self.gym.set_rigid_body_color(env_ptr, goal_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, goal_color)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            if self.cfg['env']["enable_ft_sensors"]:
                self.gym.enable_actor_dof_force_sensors(env_ptr, Franka_actor)
                # add force-torque sensor to finger tips
                # for gripper_handle in self._grippers_handles.values():
                #     self.gym.create_force_sensor(env_ptr, gripper_handle, gymapi.Transform())
            # change settings of DOF
            self.gym.set_actor_dof_properties(env_ptr, Franka_actor, robot_dof_props)
            # end aggregation mode if enabled
            if self.cfg['env']["aggregateMode"]:
                self.gym.end_aggregate(env_ptr)
            # add instances to list
            self.envs.append(env_ptr)
            self.gym_indices["robot"].append(Franka_idx)
            self.gym_indices["object"].append(object_idx)
            self.gym_indices["goal_object"].append(goal_object_idx)
        # convert gym indices from list to tensor
        for asset_name, asset_indices in self.gym_indices.items():
            self.gym_indices[asset_name] = torch.tensor(asset_indices, dtype=torch.long, device=self.device)
        franka_link_dict = self.gym.get_asset_rigid_body_dict(self._gym_assets["robot"])
        self.franka_hand_index = franka_link_dict["panda_hand"]


    def _check_termination(self):
        
        # log theoretical number of r eseats
        failed_resety = torch.logical_or(torch.lt(self._object_state_history[0][:, 1], -self.holelength/2),torch.gt(self._object_state_history[0][:, 1], self.holelength/2))
        failed_resetx = torch.logical_or(torch.lt(self._object_state_history[0][:, 0], 0.5-self.holelength/2),torch.gt(self._object_state_history[0][:, 0], 0.5+self.holelength/2))
        failed_reset=torch.logical_and(failed_resetx, failed_resety)
        #self.rew_buf[failed_reset]-=10
        delta=self._object_state_history[0][:,0:3]-self._object_goal_poses_buf[:,0:3]
        dist= torch.norm(delta, dim=-1)
        goal_position_reached = torch.le(dist, self.cfg["env"]["reward_terms"]["object_dist"]["th"])

        goal_reached=goal_position_reached
        #goal_reached=goal_position_reached
        self.rew_buf[goal_reached]+=self.cfg["env"]["reward_terms"]["object_dist"]["weight2"]
        self.reset_buf=torch.logical_or(self.reset_buf, failed_reset)
        self.reset_buf=torch.logical_or(self.reset_buf, goal_reached)
        self.reset_buf=self.reset_buf.float()
        

    def reset_idx(self, env_ids):
        # A) Reset episode stats buffers
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self._successes[env_ids] = 0
        # B) Various randomizations at the start of the episode:
        # -- Robot base position.
        # -- Stage position.
        # -- Coefficient of restituion and friction for robot, object, stage.
        # -- Mass and size of the object
        # -- Mass of robot links
        # -- Robot joint state
        robot_initial_state_config = self.cfg["env"]["reset_distribution"]["robot_initial_state"]
        object_initial_state_config = self.cfg["env"]["reset_distribution"]["object_initial_state"]
        if object_initial_state_config["type"]=="pre_contact_policy":
            indices=torch.arange(self._counting, self._counting+env_ids.shape[0], dtype=torch.long, device=self.device)%self._robot_buffer.shape[0]
            self._counting+=env_ids.shape[0]
            self._reset_indices[env_ids]=indices

        self._sample_robot_state(
            env_ids,
            distribution=robot_initial_state_config["type"],
            dof_pos_stddev=robot_initial_state_config["dof_pos_stddev"],
            dof_vel_stddev=robot_initial_state_config["dof_vel_stddev"]
        )
        # -- Sampling of initial pose of the object
        self._sample_object_poses(
            env_ids,
            distribution=object_initial_state_config["type"],
            difficulty=self.cfg["env"]["difficulty"]
        )
        # -- Sampling of goal pose of the object
        self._sample_object_goal_poses(
            env_ids,
            difficulty=self.cfg["env"]["difficulty"]
        )
        # C) Extract franka indices to reset
        robot_indices = self.gym_indices["robot"][env_ids].to(torch.int32)
        object_indices = self.gym_indices["object"][env_ids].to(torch.int32)
        goal_object_indices = self.gym_indices["goal_object"][env_ids].to(torch.int32)
        all_indices = torch.unique(torch.cat([robot_indices, object_indices, goal_object_indices]))
        # D) Set values into simulator
        # -- DOF
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state),
                                               gymtorch.unwrap_tensor(robot_indices), len(robot_indices))
        # -- actor root states
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._actors_root_state),
                                                      gymtorch.unwrap_tensor(all_indices), len(all_indices))
   
   
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
        Franka_asset = self.gym.load_asset(self.sim, self._Franka_assets_dir,
                                               self._robot_urdf_file, robot_asset_options)
       
        Franka_props = self.gym.get_asset_rigid_shape_properties(Franka_asset)
        for p in Franka_props:
            p.friction = 1.0
            p.torsion_friction = 0.8
            p.restitution = 0.8
        self.gym.set_asset_rigid_shape_properties(Franka_asset, Franka_props)
        for frame_name in self._grippers_handles.keys():
            self._grippers_handles[frame_name] = self.gym.find_asset_rigid_body_index(Franka_asset,
                                                                                         frame_name)
            # check valid handle
            if self._grippers_handles[frame_name] == gymapi.INVALID_HANDLE:
                msg = f"Invalid handle received for frame: `{frame_name}`."
                print(msg)
        self._hand_handel=self.gym.find_asset_rigid_body_index(Franka_asset, "panda_hand")
        # sensor_pose = gymapi.Transform()
        # for gripper_handle in self._gripper_handles.values():
        #     self.gym.create_asset_force_sensor(Franka_asset, gripper_handle, sensor_pose)
        # extract the dof indices
        # Note: need to write actuated dofs manually since the system contains fixed joints as well which show up.
        for dof_name in self._robot_dof_indices.keys():
            self._robot_dof_indices[dof_name] = self.gym.find_asset_dof_index(Franka_asset, dof_name)
            # check valid handle
            if self._robot_dof_indices[dof_name] == gymapi.INVALID_HANDLE:
                msg = f"Invalid index received for DOF: `{dof_name}`."
                print(msg)
        # return the asset
        return Franka_asset

    def __define_table_asset(self):
        """ Define Gym asset for table.
        """
        self.holelength=self.cfg["env"]["geometry"]["holelength"]

        # define table asset
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = True
        table_asset_options.fix_base_link = True
        table_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        table_asset_options.thickness = 0.001
        # table_asset_options.vhacd_enabled=True
        # table_asset_options.vhacd_params = gymapi.VhacdParams()
        # table_asset_options.vhacd_params.resolution = 64000000
        # table_asset_options.vhacd_params.alpha = 0.04
        # table_asset_options.vhacd_params.beta = 1.0
        # table_asset_options.vhacd_params.max_convex_hulls = 128
        # table_asset_options.vhacd_params.max_num_vertices_per_ch = 1024
        # load table asset
        for key, item in self.boxes.items():
            if key=="box1" or key=="box2":
                table_asset=self.gym.create_box(self.sim,
                    0.2-self.holelength/2, item["length"], item["height"], table_asset_options)
            elif key=="box3" or key=="box4":
                table_asset=self.gym.create_box(self.sim,
                    self.holelength, (0.25-self.holelength/2) ,item["height"], table_asset_options)
            else:
                table_asset=self.gym.create_box(self.sim,
                    item["width"], item["length"], item["height"], table_asset_options)
            table_props = self.gym.get_asset_rigid_shape_properties(table_asset)
            for p in table_props:
                p.friction = 0.5
                p.torsion_friction = 0.3
            self.gym.set_asset_rigid_shape_properties(table_asset, table_props)
            self._gym_assets[key]=table_asset

    def __define_object_asset(self):
        """ Define Gym asset for object.
        """
        # define object asset
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.disable_gravity = False
        object_asset_options.thickness = 0.001
        object_asset_options.density=10
        object_asset_options.flip_visual_attachments = True

        # load object asset
        # object_asset = self.gym.load_asset(self.sim, self._Franka_assets_dir,
        #                                     self._object_urdf_file, object_asset_options)
        # set object properties
        # Ref: https://github.com/rr-learning/rrc_simulation/blob/master/python/rrc_simulation/collision_objects.py#L96
        size=self._object_dims.size
        object_asset=self.gym.create_box(self.sim, size[0], size[1],size[2], object_asset_options)
        object_props = self.gym.get_asset_rigid_shape_properties(object_asset)
        # for p in object_props:
        #     p.friction = 1.0
        #     p.torsion_friction = 0.001
        #     p.restitution = 0.0
        if self.cfg["env"]["extract_contact"]:
            sensor_options = gymapi.ForceSensorProperties()
            sensor_options.enable_forward_dynamics_forces = True
            sensor_options.enable_constraint_solver_forces = True
            sensor_options.use_world_frame = True
            sensor_pose=gymapi.Transform()
            sensor_pose.p=gymapi.Vec3(0,0,size[2]/2)
            self.gym.create_asset_force_sensor(object_asset, 0, sensor_pose, sensor_options)
        # return the asset
        return object_asset

    def __define_goal_object_asset(self):
        """ Define Gym asset for goal object.
        """
        # define object asset
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.disable_gravity = True
        object_asset_options.fix_base_link = True
        object_asset_options.thickness = 0.001
        object_asset_options.flip_visual_attachments = True

        # load object asset
        # goal_object_asset = self.gym.load_asset(self._sim, self._Franka_assets_dir,
        #                                          self._object_urdf_file, object_asset_options)
        size=self._object_dims.size
        goal_object_asset=self.gym.create_box(self.sim, size[0], size[1],size[2], object_asset_options)
        # return the asset
        return goal_object_asset

    """
    Helper functions - MDP
    """

    def compute_observations(self):
        """
        Fills observation and state buffer with the current state of the system.
        """
        # refresh memory buffers
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        if self.cfg['env']["enable_ft_sensors"] :
            self.gym.refresh_dof_force_tensor(self.sim)
            joint_torques = self._dof_torque
            #self.gym.refresh_force_sensor_tensor(self.sim)
        # if self.cfg["env"]["extract_contact"]:
        #     self.gym.refresh_net_contact_force_tensor(self.sim)
        #     self.gym.refresh_force_sensor_tensor(self.sim)
        # extract frame handles
        gripper_handles_indices = list(self._grippers_handles.values())
        object_indices = self.gym_indices["object"]
        # update state histories
        self._grippers_frames_state_history.appendleft(self._rigid_body_state[:, gripper_handles_indices])
        self._object_state_history.appendleft(self._actors_root_state[object_indices])
        # fill the observations and states buffer
        self.__compute_franka_observations()
        
        # normalize observations if flag is enabled
        if self.cfg['env']["normalize_obs"]:
            # for normal obs
            self.obs_buf = scale_transform(
                self.obs_buf,
                lower=self._observations_scale.low,
                upper=self._observations_scale.high
            )

    def __compute_franka_observations(self):
        """
        Fills observation buffer with the current state of the system.
        """
        # generalized coordinates
        start_offset = 0
        end_offset = start_offset + self._dims.GeneralizedCoordinatesDim.value
        self.obs_buf[:, start_offset:end_offset] = self._dof_position
        # generalized velocities
        start_offset = end_offset
        end_offset = start_offset + self._dims.GeneralizedVelocityDim.value
        self.obs_buf[:, start_offset:end_offset] = self._dof_velocity
        if self.cfg['env']['keypoint']['activate']:
            # object pose as keypoint
            start_offset = end_offset
            end_offset = start_offset + self._dims.KeypointDim.value*self.keypoints_num
            curent_keypoints=gen_keypoints(pose=self._object_state_history[0][:, 0:7], size=self._object_dims.size)
            self.obs_buf[:, (start_offset):(end_offset)] = curent_keypoints.view(self.num_envs,24)[:]
            # object desired pose as keypoint
            start_offset = end_offset
            end_offset = start_offset + self._dims.KeypointDim.value*self.keypoints_num
            goal_keypoints=gen_keypoints(pose=self._object_goal_poses_buf, size=self._object_dims.size)
            self.obs_buf[:, start_offset:end_offset] = goal_keypoints.view(self.num_envs,24)[:]
        else:
            # object pose
            start_offset = end_offset
            end_offset = start_offset + self._dims.ObjectPoseDim.value
            self.obs_buf[:, start_offset:end_offset] = self._object_state_history[0][:, 0:7]
            # object desired pose
            start_offset = end_offset
            end_offset = start_offset + self._dims.ObjectPoseDim.value
            self.obs_buf[:, start_offset:end_offset] = self._object_goal_poses_buf
        # object velcity
        start_offset = end_offset
        end_offset = start_offset + self._dims.ObjectVelocityDim.value
        self.obs_buf[:, start_offset:end_offset] = self._object_state_history[0][:, 7:13]
        # gripper state
        num_fingerip_states = self._dims.NumFingers.value * self._dims.StateDim.value
        start_offset = end_offset
        end_offset = start_offset + num_fingerip_states
        self.obs_buf[:, start_offset:end_offset] = \
            self._grippers_frames_state_history[0].reshape(self.num_envs, num_fingerip_states)
        # joint torque
        start_offset = end_offset
        end_offset = start_offset + self._dims.JointTorqueDim.value
        self.obs_buf[:, start_offset:end_offset] = self._dof_torque
        # force-torque sensors
        # start_offset = end_offset
        # end_offset = start_offset + self._dims.NumFingers.value * self._dims.WrenchDim.value
        # self.obs_buf[:, start_offset:end_offset] = self._ft_sensors_values
        # distance between object and goal
        start_offset = end_offset
        end_offset = start_offset + 1
        self.obs_buf[:, start_offset:end_offset] = torch.norm(self._object_state_history[0][:,0:3]-self._object_goal_poses_buf[:,0:3],2,-1).unsqueeze(-1)
        # angle differences between object and goal
        start_offset = end_offset
        end_offset = start_offset + 1
        self.obs_buf[:, start_offset:end_offset] = quat_diff_rad(self._object_state_history[0][:,3:7], self._object_goal_poses_buf[:,3:7]).unsqueeze(-1)
        # previous action from policy
        start_offset = end_offset
        end_offset = start_offset + self.action_dim 
        self.obs_buf[:, start_offset:end_offset] = self.actions
        # observation for pi_pre object pose(7)/goal pose(7) 
        # self.pre_obs[:, :7]=self._object_state_history[0][:, 0:7]
        # self.pre_obs[:, 7:14]=self._object_goal_poses_buf

    def _sample_robot_state(self, instances: torch.Tensor, distribution: str = 'default',
                             dof_pos_stddev: float = 0.0, dof_vel_stddev: float = 0.0):
        """Samples the robot DOF state based on the settings.

        Type of robot initial state distribution: ["default", "random"]
             - "default" means that robot is in default configuration.
             - "random" means that noise is added to default configuration
             - "pre_contact_policy" means that robot pose is made by pi_pre
             - "none" means that robot is configuration is not reset between episodes.

        Args:
            instances: A tensor contraining indices of environment instances to reset.
            distribution: Name of distribution to sample initial state from: ['default', 'random']
            dof_pos_stddev: Noise scale to DOF position (used if 'type' is 'random')
            dof_vel_stddev: Noise scale to DOF velocity (used if 'type' is 'random')
        """
        # number of samples to generate
        num_samples = instances.size()[0]
        # sample dof state based on distribution type
        if distribution == "none":
            return
        elif distribution == "default":
            # set to default configuration
            self._dof_position[instances] = self.franka_default_dof_pos
            self._dof_velocity[instances] = torch.zeros_like(self.franka_default_dof_pos,device=self.device)
        elif distribution == "random":
            # sample uniform random from (-1, 1)
            dof_state_dim = self._dims.JointPositionDim.value 
            dof_state_noise = 2 * torch.rand((num_samples, dof_state_dim,), dtype=torch.float,
                                             device=self.device) - 1
            # set to default configuration
            self._dof_position[instances] = self.franka_default_dof_pos
            self._dof_velocity[instances] = torch.zeros_like(self.franka_default_dof_pos,device=self.device)
            # DOF position
            self._dof_position[instances] += dof_pos_stddev * dof_state_noise
            # DOF velocity
            start_offset = end_offset
            end_offset += self._dims.JointVelocityDim.value
            self._dof_velocity[instances] += dof_vel_stddev * dof_state_noise[:, start_offset:end_offset]
        elif distribution == "pre_contact_policy": # Use trained pre-contact policy for initialization
            self._dof_position[instances]=self._robot_buffer[self._reset_indices[instances]]
        else:
            msg = f"Invalid robot initial state distribution. Input: {distribution} not in set."
            raise ValueError(msg)
        # reset robot grippers state history
        for idx in range(1, self._state_history_len):
            self._grippers_frames_state_history[idx][instances] = 0.0

    def _sample_object_poses(self, instances: torch.Tensor, distribution: str, difficulty: int):
        """Sample poses for the cube.

        Type of distribution: ["default", "random", "none"]
             - "default" means that pose is default configuration.
             - "random" means that pose is randomly sampled on the table.
             - "pre_contact_policy" means that pose is randomly sampled by pi_pre.
             - "none" means no resetting of object pose between episodes.

        Args:
            instances: A tensor contraining indices of environment instances to reset.
            distribution: Name of distribution to sample initial state from: ['default', 'random']
        """
        # number of samples to generate
        num_samples = instances.size()[0]
        # sample poses based on distribution type
        if distribution == "none":
            return
        elif distribution == "offline":
            pos_x = self.initial_poses[instances,9]
            pos_y = self.initial_poses[instances,10]
            pos_z = self._object_dims.size[2] / 2+0.3
            orientation=self.initial_poses[instances,12:-1]
        elif distribution == "pre_contact_policy": # Use fully-trained pre-contact policy for initialization
            pos_x = self._object_buffer[self._reset_indices[instances],0]
            pos_y = self._object_buffer[self._reset_indices[instances],1]
            pos_z = self._object_buffer[self._reset_indices[instances],2]
            orientation=self._object_buffer[self._reset_indices[instances],3:]
       
        else:
            msg = f"Invalid object initial state distribution. Input: {distribution} " \
                  "not in [`default`, `random`, `none`]."
            raise ValueError(msg)
        # set buffers into simulator
        # extract indices for goal object
        object_indices = self.gym_indices["object"][instances]
        # set values into buffer
        # object buffer
        self._object_state_history[0][instances, 0] = pos_x
        self._object_state_history[0][instances, 1] = pos_y
        self._object_state_history[0][instances, 2] = pos_z
        self._object_state_history[0][instances, 3:7] = orientation
        self._object_state_history[0][instances, 7:13] = 0
        # reset object state history
        for idx in range(1, self._state_history_len):
            self._object_state_history[idx][instances] = 0.0
        # root actor buffer
        self._actors_root_state[object_indices] = self._object_state_history[0][instances]

    def _sample_object_goal_poses(self, instances: torch.Tensor, difficulty: int):
        """Sample goal poses for the cube and sets them into the desired goal pose buffer.

        Args:
            instances: A tensor contraining indices of environment instances to reset.
            difficulty: Difficulty level. The higher, the more difficult is the goal.

        Possible levels are:
            - -1: Fixed goal position (gol_pose, gol_rot) at the top.
            - 1: Teaching policy how to touch object in random 2D position and only yaw rotation on the table. 
            - 2: Move obeject to random to the goal with random 2D postion and only yaw rotation on the table.
            - 3: Random goal position in the air, only yaw rotation.
            - 4: Random goal pose in the air, including orientation.
        """
        # number of samples to generate
        num_samples = instances.size()[0]
        # sample poses based on task difficulty
        if difficulty == 1:
            # no need to change goal postion.
            pos_x, pos_y, pos_z, orientation=self._fixed_goal
        elif difficulty == 2:
            # Move obeject to random to the goal with random 2D postion and only yaw rotation on the table.
            pos_x, pos_y = random_xy(num_samples, self._object_dims.max_width,self._object_dims.max_length, self.device)
            pos_z = 0.3+self._object_dims.size[2]*0.5
            #orientation = random_yaw_orientation(num_samples, self.device)
            orientation=default_orientation(num_samples, self.device)
        elif difficulty == 3:
            # Random goal position in the air, no orientation.
            pos_x, pos_y = random_xy(num_samples, self._object_dims.max_width,self._object_dims.max_length, self.device)
            pos_z = 0.3+self._object_dims.size[2]*0.5
            orientation = random_yaw_orientation(num_samples, self.device)
        elif difficulty == 4 :
            # Random goal pose in the air, including orientation.
            # Note: Set minimum height such that the cube does not intersect with the
            #       ground in any orientation
            pos_x, pos_y = random_xy(num_samples, 1,1, self.device)
            pos_z = 0.3+self._object_dims.size[2]*0.5
        elif difficulty == "pre_contact_policy": # Use fully-trained pre-contact policy for initialization
            pos_x = self._goal_buffer[self._reset_indices[instances],0]
            pos_y = self._goal_buffer[self._reset_indices[instances],1]
            pos_z = self._goal_buffer[self._reset_indices[instances],2]
            orientation=self._goal_buffer[self._reset_indices[instances],3:]
        elif difficulty == -1:
            #Fixed goal position (gol_pose, gol_rot) at the top.
            pos_x, pos_y, pos_z, orientation=self._fixed_goal

        else:
            msg = f"Invalid difficulty index for task: {difficulty}."
            raise ValueError(msg)

        
        # extract indices for goal object
        goal_object_indices = self.gym_indices["goal_object"][instances]
        # set values into buffer
        # object goal buffer
        self._object_goal_poses_buf[instances, 0] = pos_x
        self._object_goal_poses_buf[instances, 1] = pos_y
        self._object_goal_poses_buf[instances, 2] = pos_z
        self._object_goal_poses_buf[instances, 3:7] = orientation
        # root actor buffer
        self._actors_root_state[goal_object_indices, 0:7] = self._object_goal_poses_buf[instances]

    def push_data(self, T_O, T_G, q):
        """
            Fill the object pose, goal pose, and robot configuration made by pi_pre and sampler
            
            args:
                T_O: object pose
                T_G: goal pose,
                q: robot configuration
        """
        #print(T_O.shape)
        if T_O.dim()==3:
            T_O=T_O.squeeze(1)
            T_G=T_G.squeeze(1)
            q=q.squeeze(1)
        self._robot_buffer=q
        self._object_buffer=T_O
        self._goal_buffer=T_G
        self._counting=0
        self._reset_indices=torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    
    def interupt(self, idx, q, quasi_static=False):
        """
            interupt execution of pi_post and update contacts to new contacts
            
            args:
                idx: indices to change contact
                q: robot configuration with new contact
                quasi_static: assume the object as fixed during contact change (bool)
        """
        self.interuption[idx]=1
        self._dof_position[idx]=q
        self._dof_velocity[idx] = torch.zeros_like(self.franka_default_dof_pos,dtype=torch.float, device=self.device)
        robot_indices = self.gym_indices["robot"][idx].to(torch.int32)
        object_indices = self.gym_indices["object"][idx].to(torch.int32)
        goal_object_indices = self.gym_indices["goal_object"][idx].to(torch.int32)
        all_indices = torch.unique(torch.cat([robot_indices, object_indices, goal_object_indices]))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state),
                                               gymtorch.unwrap_tensor(robot_indices), len(robot_indices))
        if quasi_static:
            self._actors_root_state[object_indices.to(torch.long),7:13]=0
        

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
        return self._reset_indices.detach()
    
    @property
    def env_succeed_count(self) -> torch.Tensor:
        """Returns the total number of environment succeded aggregated across parallel environments."""
        return self._successes_count.detach()

#####################################################################
###=========================jit functions=========================###
#####################################################################
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
        dt: float,
        finger_reach_object_weight: float,
        object_dist_weight: Tuple[float, float],
        object_rot_weight: Tuple[float, float],
        object_move_weight: Tuple[float, float],
        th: Tuple[float, float],
        epsilon: Tuple[float, float],
        env_steps_count: int,
        object_goal_poses_buf: torch.Tensor,
        object_state: torch.Tensor,
        last_object_state: torch.Tensor,
        gripper_state: torch.Tensor,
        last_gripper_state: torch.Tensor,
        keypoint: bool,
        size: Tuple[float, float, float],
        last_torque: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:

    ft_sched_start = 0
    vel_sched_start=0
    ft_sched_end = 5e7
    vel_sched_end=1e8
    # # Reward penalising finger movement

    # gripper_vel = (gripper_state[:, :, 0:3] - last_gripper_state[:, :, 0:3]) / dt

    # finger_movement_penalty = finger_move_penalty_weight * gripper_vel.pow(2).view(-1, 9).sum(dim=-1)

    # Reward for finger reaching the object

    # distance from each finger to the centroid of the object, shape (N, 3).
    gripper_state=torch.mean(gripper_state,1)
    last_gripper_state=torch.mean(last_gripper_state,1)
    curr_norms = torch.norm(gripper_state[:, 0:3] - object_state[:, 0:3], p=2, dim=-1)

    # end_effector postion and rotation changes.
    last_action=torch.norm(obs_buf[:,-7:-1],2,-1)    
    # distance from each finger to the centroid of the object in the last timestep, shape (N, 3).
    prev_norms = torch.norm(last_gripper_state[:, 0:3] - last_object_state[:, 0:3], p=2, dim=-1)
        

    ft_sched_val = 1.0 if ft_sched_start <= env_steps_count <= ft_sched_end else 0.0
    #ft_sched_val=1
    #finger_reach_object_reward = finger_reach_object_weight * ft_sched_val * (curr_norms - prev_norms)
    finger_reach_object_reward=torch.where((curr_norms - prev_norms)>=0.01,finger_reach_object_weight * ft_sched_val, 0.0) 
    #finger_reach_object_reward=ft_sched_val*finger_reach_object_weight/(torch.sum(curr_norms,-1)+epsilon[0])
    # Reward for object distance
    if keypoint:
        object_keypoints = gen_keypoints(pose=object_state[:, 0:7], size=size)
        goal_keypoints = gen_keypoints(pose=object_goal_poses_buf[:, 0:7], size=size)
        delta=object_keypoints- goal_keypoints
        dist = torch.norm(delta, p=2, dim=-1)
        delta_max, _=torch.max(torch.abs(delta),-1)
        goal_position_reached = torch.le(delta_max, th[0])
        object_dist_reward=torch.sum(object_dist_weight[0]/(dist+epsilon[0]),-1)
        #+object_dist_weight[1]*goal_position_reached
        obj_reward = object_dist_reward
    else:
        delta=object_state[:, 0:3] - object_goal_poses_buf[:, 0:3]
        dist = torch.norm(delta, p=2, dim=-1)
        delta_max, _=torch.max(torch.abs(delta),-1)
        goal_position_reached = torch.le(delta_max, th[0])
        #object_dist_reward = object_dist_weight * dt * lgsk_kernel(object_dist, scale=50., eps=2.)
        object_dist_reward=object_dist_weight[0]/(dist+epsilon[0])
        # Reward for object rotation

        # extract quaternion orientation
        # quat_a = object_state[:, 3:7]
        # quat_b = object_goal_poses_buf[:, 3:7]

        # angles = quat_diff_rad(quat_a, quat_b)
        # rot_rew = (torch.abs(angles) + epsilon[1])
        # goal_rotation_reached = torch.le(torch.abs(angles), th[1])
        # object_rot_reward=object_rot_weight[0]/rot_rew
        obj_reward = object_dist_reward 
    
    total_reward= obj_reward-0.5*last_action
    #total_reward= obj_reward-0.5*last_action+0.03*ft_sched_val/(curr_norms+epsilon[0])
   
    reset = torch.zeros_like(reset_buf)
    reset = torch.where(progress_buf >= episode_length - 1, torch.ones_like(reset_buf), reset)

    info: Dict[str, torch.Tensor] = {
        'obj_goal_reward': obj_reward,
        'reward': total_reward,
    }

    return total_reward, reset, info

@torch.jit.script
def default_position(num: int, device: str) -> torch.Tensor:
    """Returns identity rotation transform."""
    xyz = torch.zeros((num, 4,), dtype=torch.float, device=device)
    return xyz

@torch.jit.script
def random_xy(num: int, max_width: float, max_length: float, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns sampled uniform positions in circle (https://stackoverflow.com/a/50746409)"""
    # sample radius of circle
    # radius = torch.sqrt(torch.rand(num, dtype=torch.float, device=device))
    # radius *= max_com_distance_to_center
    # # sample theta of point
    # theta = np.pi * torch.rand(num, dtype=torch.float, device=device)
    # # x,y-position of the cube
    x = torch.rand(num, dtype=torch.float, device=device)*(max_width)-max_width/2+0.5
    y = torch.rand(num, dtype=torch.float, device=device)*(max_length)-max_length/2

    return x, y


@torch.jit.script
def random_z(num: int, min_height: float, max_height: float, device: str) -> torch.Tensor:
    """Returns sampled height of the goal object."""
    z = torch.rand(num, dtype=torch.float, device=device)
    z = (max_height - min_height) * z + min_height

    return z


@torch.jit.script
def default_orientation(num: int, device: str) -> torch.Tensor:
    """Returns identity rotation transform."""
    quat = torch.zeros((num, 4,), dtype=torch.float, device=device)
    quat[..., -1] = 1.0

    return quat


@torch.jit.script
def random_orientation(num: int, device: str) -> torch.Tensor:
    """Returns sampled rotation in 3D as quaternion.
    Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.random.html
    """
    # sample random orientation from normal distribution
    quat = torch.randn((num, 4,), dtype=torch.float, device=device)
    # normalize the quaternion
    quat = torch.nn.functional.normalize(quat, p=2., dim=-1, eps=1e-12)

    return quat

@torch.jit.script
def random_yaw_orientation(num: int, device: str) -> torch.Tensor:
    """Returns sampled rotation around z-axis."""
    roll = torch.zeros(num, dtype=torch.float, device=device)
    pitch = torch.zeros(num, dtype=torch.float, device=device)
    yaw = 2 * np.pi * torch.rand(num, dtype=torch.float, device=device)

    return quat_from_euler_xyz(roll, pitch, yaw)

@torch.jit.script
def control_ik(num_envs: int, j_eef, dpose, device: str):
    # solve damped least squares
    damping = 0.05
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = torch.matmul(torch.matmul(j_eef_T,torch.inverse(torch.matmul(j_eef, j_eef_T) + lmbda)), dpose.unsqueeze(-1)).view(num_envs, 7)
    return u

@torch.jit.script
def control_osc(num_envs: int, j_eef, hand_vel, mm, dpose, dof_vel, dof_pos, kp, damping_ratio, variable: bool, device: str):
    null=True
    kd = 2.0*torch.sqrt(kp)*damping_ratio if variable else 2.0 *math.sqrt(kp)
    kp_null = 10.
    kd_null = 2.0 *math.sqrt(kp_null)
    mm_inv = torch.inverse(mm)
    m_eef_inv = torch.matmul(torch.matmul(j_eef, mm_inv), torch.transpose(j_eef, 1, 2))
    m_eef = torch.inverse(m_eef_inv)
    u = torch.matmul(torch.matmul(torch.transpose(j_eef, 1, 2), m_eef), (
        kp * dpose - kd * hand_vel).unsqueeze(-1))
    default_dof_pos_tensor=torch.tensor([    -0.0050,     -0.2170,      0.0065,     -2.1196,      0.0004,
              2.0273,      0.7912,      0.0000,      0.0000], dtype=torch.float, device=device)
    default_dof_pos_tensor=torch.stack([default_dof_pos_tensor]*num_envs, 0)
    # Nullspace control torques `u_null` prevents large changes in joint configuration
    # They are added into the nullspace of OSC so that the end effector orientation remains constant
    # roboticsproceedings.org/rss07/p31.pdf
    if null:
        j_eef_inv = torch.matmul(torch.matmul(m_eef,j_eef), mm_inv)
        u_null = kd_null * -dof_vel + kp_null * (
            (default_dof_pos_tensor - dof_pos + np.pi) % (2 * np.pi) - np.pi)
        u_null = u_null[:, :7]
        u_null = torch.matmul(mm, u_null.unsqueeze(-1))
        u += torch.matmul((torch.eye(7, device=device).unsqueeze(0) - torch.matmul(torch.transpose(j_eef, 1, 2), j_eef_inv)), u_null)
    return u.squeeze(-1)