from tasks.domain import Domain, Object
from typing import List, Dict
from isaacgym import gymapi
import numpy as np
import torch
from utils.torch_jit_utils import quat_diff_rad


class HiddenCard(Domain):
    def __init__(self, cfg, sim_device, graphics_device_id, headless, use_state=False):
        super().__init__(cfg, sim_device, graphics_device_id, headless, use_state)

    def _set_table_dimension(self):
        table_dims = gymapi.Vec3(0.4, 0.5, 0.4)
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.5, 0.0, (0.5 * table_dims.z))
        self.table_dims = table_dims
        self.table_pose = table_pose

        self.table2_dims = gymapi.Vec3(0.1, 0.25, 0.1)
        self.table2_pose = gymapi.Transform()
        self.table2_pose.p = gymapi.Vec3(0.65, 0, (table_dims.z + self.table2_dims.z / 2.0))
        
        self.table3_dims = gymapi.Vec3(0.2, 0.25, 0.04)
        self.table3_pose = gymapi.Transform()
        self.table3_pose.p = gymapi.Vec3(0.6, 0, (table_dims.z + self.table2_dims.z + self.table3_dims.z / 2.0))

    def _get_table_prim_names(self) -> List[str]:
        prim_names = ["table", "table1", "table2"]
        return prim_names

    def _set_object_dimension(self, object_dims) -> Object:
        return Object((object_dims["width"], object_dims["length"], object_dims["height"]))

    def _create_table(self, env_ptr, env_index: int, actor_indices: Dict[str, List[int]]):
        table_handle = self.gym.create_actor(env_ptr, self.asset_handles["table"], self.table_pose, "table", env_index, 1, 1)
        table_color = gymapi.Vec3(0.54, 0.57, 0.59)
        self.gym.set_rigid_body_color(env_ptr, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, table_color)
        table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
        actor_indices["table"].append(table_idx)

        table2_handle = self.gym.create_actor(env_ptr, self.asset_handles["table1"], self.table2_pose, "table2", env_index, 1, 1)
        self.gym.set_rigid_body_color(env_ptr, table2_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, table_color)
        table2_idx = self.gym.get_actor_index(env_ptr, table2_handle, gymapi.DOMAIN_SIM)
        actor_indices["table1"].append(table2_idx)

        table3_handle = self.gym.create_actor(env_ptr, self.asset_handles["table2"], self.table3_pose, "table3", env_index, 1, 1)
        self.gym.set_rigid_body_color(env_ptr, table3_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, table_color)
        table3_idx = self.gym.get_actor_index(env_ptr, table3_handle, gymapi.DOMAIN_SIM)
        actor_indices["table2"].append(table3_idx)

    def _create_object(self, env_ptr, env_index: int, actor_indices: Dict[str, List[int]]):
        object_handle = self.gym.create_actor(env_ptr, self.asset_handles["object"], gymapi.Transform(), "object", env_index, 0, 2)
        object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
        actor_indices["object"].append(object_idx)

    def _max_dist_btw_obj_and_goal(self) -> torch.Tensor:
        return np.sqrt(self.table_dims.x ** 2 + self.table_dims.y ** 2) * torch.ones(1, dtype=torch.float32, device=self.device)

    def _define_table_asset(self):
        """ Define Gym asset for table. This function returns nothing.
        """
        # define table asset
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = True
        table_asset_options.fix_base_link = True
        table_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        table_asset_options.thickness = 0.001
        # load table asset
        table_asset = self.gym.create_box(self.sim, self.table_dims.x, self.table_dims.y, self.table_dims.z, table_asset_options)
        # set table properties
        table_props = self.gym.get_asset_rigid_shape_properties(table_asset)
        # iterate over each mesh
        for p in table_props:
            p.friction = 0.5
            # p.torsion_friction = 0.3
        self.gym.set_asset_rigid_shape_properties(table_asset, table_props)
        self.asset_handles["table"] = table_asset

        table2_asset = self.gym.create_box(self.sim, self.table2_dims.x, self.table2_dims.y, self.table2_dims.z, table_asset_options)
        table2_props = self.gym.get_asset_rigid_shape_properties(table2_asset)
        for p in table2_props:
            p.friction = 0.3
        self.gym.set_asset_rigid_shape_properties(table2_asset, table2_props)
        self.asset_handles["table1"] = table2_asset

        table3_asset = self.gym.create_box(self.sim, self.table3_dims.x, self.table3_dims.y, self.table3_dims.z, table_asset_options)
        table3_props = self.gym.get_asset_rigid_shape_properties(table3_asset)
        for p in table3_props:
            p.friction = 0.3
        self.gym.set_asset_rigid_shape_properties(table3_asset, table3_props)
        self.asset_handles["table2"] = table3_asset

    def _define_object_asset(self):
        """ Define Gym asset for object.
        """
        # define object asset
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.disable_gravity = False
        object_asset_options.thickness = 0.001
        object_asset_options.density = self.cfg["env"]["geometry"]["object"]["density"]
        object_asset = self.gym.load_asset(self.sim, self._assets_dir, 'urdf/Panda/Coloredcard.urdf', object_asset_options)
        object_props = self.gym.get_asset_rigid_shape_properties(object_asset)
        for p in object_props:
            p.friction = 0.5
        #     p.torsion_friction = 0.001
        #     p.restitution = 0.0

        return object_asset

    def _check_failure(self) -> torch.Tensor:
        failed_envs = torch.le(self._object_state_history[0][:, 2], (0.8 * self.table_dims.z))
        return failed_envs

    def _check_success(self) -> torch.Tensor:
        delta = self._object_state_history[0][:, 0:3] - self._object_goal_poses_buf[:, 0:3]
        dist = torch.norm(delta, p=2, dim=-1)
        goal_position_reached = torch.le(dist, self.cfg["env"]["reward_terms"]["object_dist"]["th"])
        quat_a = self._object_state_history[0][:, 3:7]
        quat_b = self._object_goal_poses_buf[:, 3:7]
        angles = quat_diff_rad(quat_a, quat_b)
        goal_rotation_reached = torch.le(torch.abs(angles), self.cfg["env"]["reward_terms"]["object_rot"]["th"])
        goal_reached = torch.logical_and(goal_rotation_reached, goal_position_reached)
        return goal_reached
