from tasks.domain import Domain, Object
from typing import List, Dict
from isaacgym import gymapi
import numpy as np
import torch
from utils.torch_jit_utils import quat_diff_rad


class Bookshelf(Domain):
    def __init__(self, cfg, sim_device, graphics_device_id, headless, use_state=False):
        super().__init__(cfg, sim_device, graphics_device_id, headless, use_state)

    def _set_table_dimension(self):
        self.bookshelf_dims = self.cfg["env"]["geometry"]["bookshelf"] # x, y, z, width, height

    def _set_object_dimension(self, object_dims) -> Object:
        x = self.bookshelf_dims["book_size_x"]
        y = self.bookshelf_dims["book_size_y"]
        z = self.bookshelf_dims["book_size_z"]
        return Object((x, y, z)) # TODO: Check whether this value is right

    def _get_table_prim_names(self) -> List[str]:
        prim_names = [
            "bookshelf_1", "bookshelf_2", "bookshelf_3", "bookshelf_4", "bookshelf_5", "bookshelf_6", "bookshelf_7", "bookshelf_8", "bookshelf_9"
        ]
        return prim_names

    def _create_table(self, env_ptr, env_index: int, actor_indices: Dict[str, List[int]]):
        x = self.bookshelf_dims["book_size_x"]
        y = self.bookshelf_dims["book_size_y"]
        z = self.bookshelf_dims["book_size_z"]
        width = self.bookshelf_dims["width"]
        height = self.bookshelf_dims["height"]

        bookshelf_1_pose = gymapi.Transform()
        bookshelf_1_pose.p = gymapi.Vec3(0.5, 0.0, 0.2)
        bookshelf_1_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        bookshelf_1_handle = self.gym.create_actor(env_ptr, self.asset_handles["bookshelf_1"], bookshelf_1_pose, "bookshelf_1", env_index, 1, 1)
        bookshelf_1_idx = self.gym.get_actor_index(env_ptr, bookshelf_1_handle, gymapi.DOMAIN_SIM)
        actor_indices["bookshelf_1"] = bookshelf_1_idx
        self.gym.set_rigid_body_color(env_ptr, bookshelf_1_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.1, 0.1, 0.8))

        bookshelf_2_pose = gymapi.Transform()
        bookshelf_2_pose.p = gymapi.Vec3(0.5, (0.2 - 0.01), (0.4 + height / 2))
        bookshelf_2_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        bookshelf_2_handle = self.gym.create_actor(env_ptr, self.asset_handles["bookshelf_2"], bookshelf_2_pose, "bookshelf_2", env_index, 1, 1)
        bookshelf_2_idx = self.gym.get_actor_index(env_ptr, bookshelf_2_handle, gymapi.DOMAIN_SIM)
        actor_indices["bookshelf_2"] = bookshelf_2_idx
        self.gym.set_rigid_body_color(env_ptr, bookshelf_2_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.1, 0.1, 0.8))

        bookshelf_3_pose = gymapi.Transform()
        bookshelf_3_pose.p = gymapi.Vec3(0.5, (-0.2 + 0.01), (0.4 + height / 2))
        bookshelf_3_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        bookshelf_3_handle = self.gym.create_actor(env_ptr, self.asset_handles["bookshelf_3"], bookshelf_3_pose, "bookshelf_3", env_index, 1, 1)
        bookshelf_3_idx = self.gym.get_actor_index(env_ptr, bookshelf_3_handle, gymapi.DOMAIN_SIM)
        actor_indices["bookshelf_3"] = bookshelf_3_idx
        self.gym.set_rigid_body_color(env_ptr, bookshelf_3_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.1, 0.1, 0.8))

        bookshelf_4_pose = gymapi.Transform()
        bookshelf_4_pose.p = gymapi.Vec3((0.6 - 0.01), 0, (0.4 + height / 2))
        bookshelf_4_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        bookshelf_4_handle = self.gym.create_actor(env_ptr, self.asset_handles["bookshelf_4"], bookshelf_4_pose, "bookshelf_4", env_index, 1, 1)
        bookshelf_4_idx = self.gym.get_actor_index(env_ptr, bookshelf_4_handle, gymapi.DOMAIN_SIM)
        actor_indices["bookshelf_4"] = bookshelf_4_idx
        self.gym.set_rigid_body_color(env_ptr, bookshelf_4_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.1, 0.1, 0.8))

        bookshelf_5_pose = gymapi.Transform()
        bookshelf_5_pose.p = gymapi.Vec3(0.5, 0.0, (0.4 + height + 0.01))
        bookshelf_5_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        bookshelf_5_handle = self.gym.create_actor(env_ptr, self.asset_handles["bookshelf_5"], bookshelf_5_pose, "bookshelf_5", env_index, 1, 1)
        bookshelf_5_idx = self.gym.get_actor_index(env_ptr, bookshelf_5_handle, gymapi.DOMAIN_SIM)
        actor_indices["bookshelf_5"] = bookshelf_5_idx
        self.gym.set_rigid_body_color(env_ptr, bookshelf_5_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.1, 0.1, 0.8))

        bookshelf_6_pose = gymapi.Transform()
        bookshelf_6_pose.p = gymapi.Vec3((0.4 + x / 2), (y + width), (0.4 + z / 2))
        bookshelf_6_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        bookshelf_6_handle = self.gym.create_actor(env_ptr, self.asset_handles["bookshelf_6"], bookshelf_6_pose, "bookshelf_6", env_index, 1, 1)
        bookshelf_6_idx = self.gym.get_actor_index(env_ptr, bookshelf_6_handle, gymapi.DOMAIN_SIM)
        actor_indices["bookshelf_6"] = bookshelf_6_idx
        self.gym.set_rigid_body_color(env_ptr, bookshelf_6_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.1, 0.1, 0.8))

        bookshelf_7_pose = gymapi.Transform()
        bookshelf_7_pose.p = gymapi.Vec3((0.4 + x / 2), (- y - width), (0.4 + z / 2))
        bookshelf_7_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        bookshelf_7_handle = self.gym.create_actor(env_ptr, self.asset_handles["bookshelf_7"], bookshelf_7_pose, "bookshelf_7", env_index, 1, 1)
        bookshelf_7_idx = self.gym.get_actor_index(env_ptr, bookshelf_7_handle, gymapi.DOMAIN_SIM)
        actor_indices["bookshelf_7"] = bookshelf_7_idx
        self.gym.set_rigid_body_color(env_ptr, bookshelf_7_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.1, 0.1, 0.8))

        bookshelf_8_pose = gymapi.Transform()
        bookshelf_8_pose.p = gymapi.Vec3((0.4 + x / 2), (2 * y + 2 * width), (0.4 + z / 2))
        bookshelf_8_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        bookshelf_8_handle = self.gym.create_actor(env_ptr, self.asset_handles["bookshelf_8"], bookshelf_8_pose, "bookshelf_8", env_index, 1, 1)
        bookshelf_8_idx = self.gym.get_actor_index(env_ptr, bookshelf_8_handle, gymapi.DOMAIN_SIM)
        actor_indices["bookshelf_8"] = bookshelf_8_idx
        self.gym.set_rigid_body_color(env_ptr, bookshelf_8_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.1, 0.1, 0.8))

        bookshelf_9_pose = gymapi.Transform()
        bookshelf_9_pose.p = gymapi.Vec3((0.4 + x / 2), (- 2 * y - 2 * width), (0.4 + z / 2))
        bookshelf_9_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        bookshelf_9_handle = self.gym.create_actor(env_ptr, self.asset_handles["bookshelf_9"], bookshelf_9_pose, "bookshelf_9", env_index, 1, 1)
        bookshelf_9_idx = self.gym.get_actor_index(env_ptr, bookshelf_9_handle, gymapi.DOMAIN_SIM)
        actor_indices["bookshelf_9"] = bookshelf_9_idx
        self.gym.set_rigid_body_color(env_ptr, bookshelf_9_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.1, 0.1, 0.8))

    def _create_object(self, env_ptr, env_index: int, actor_indices: Dict[str, List[int]]):
        x = self.bookshelf_dims["book_size_x"]
        z = self.bookshelf_dims["book_size_z"]
        book_pose = gymapi.Transform()
        book_pose.p = gymapi.Vec3((0.4 + x / 2), 0, (0.4 + z / 2))
        book_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        object_handle = self.gym.create_actor(env_ptr, self.asset_handles["object"], book_pose, "object", env_index, 0, 2)
        object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
        actor_indices["object"].append(object_idx)

    def _max_dist_btw_obj_and_goal(self) -> torch.Tensor:
        return np.sqrt(0.25 ** 2 + 0.2 ** 2) * torch.ones(1, dtype=torch.float32, device=self.device)

    def _define_table_asset(self):
        """ Define Gym asset for table.
        """
        # define table asset
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = True
        table_asset_options.fix_base_link = True
        table_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        table_asset_options.thickness = 0.001

        # load table asset
        x = self.bookshelf_dims["book_size_x"]
        y = self.bookshelf_dims["book_size_y"]
        z = self.bookshelf_dims["book_size_z"]
        width = self.bookshelf_dims["width"]
        height = self.bookshelf_dims["height"]

        bookshelf_1_asset = self.gym.create_box(self.sim, 0.2, 0.4, 0.4, table_asset_options)
        bookshelf_1_props = self.gym.get_asset_rigid_shape_properties(bookshelf_1_asset)
        for p in bookshelf_1_props:
            p.friction = 0.5
            p.torsion_friction = 0.3
        self.gym.set_asset_rigid_shape_properties(bookshelf_1_asset, bookshelf_1_props)
        self.asset_handles["bookshelf_1"] = bookshelf_1_asset

        bookshelf_2_asset = self.gym.create_box(self.sim, 0.2, 0.02, height, table_asset_options)
        bookshelf_2_props = self.gym.get_asset_rigid_shape_properties(bookshelf_2_asset)
        for p in bookshelf_2_props:
            p.friction = 0.5
            p.torsion_friction = 0.3
        self.gym.set_asset_rigid_shape_properties(bookshelf_2_asset, bookshelf_2_props)
        self.asset_handles["bookshelf_2"] = bookshelf_2_asset

        bookshelf_3_asset = self.gym.create_box(self.sim, 0.2, 0.02, height , table_asset_options)
        bookshelf_3_props = self.gym.get_asset_rigid_shape_properties(bookshelf_3_asset)
        for p in bookshelf_3_props:
            p.friction = 0.5
            p.torsion_friction = 0.3
        self.gym.set_asset_rigid_shape_properties(bookshelf_3_asset, bookshelf_3_props)
        self.asset_handles["bookshelf_3"] = bookshelf_3_asset

        bookshelf_4_asset = self.gym.create_box(self.sim, 0.02, 0.4, height, table_asset_options)
        bookshelf_4_props = self.gym.get_asset_rigid_shape_properties(bookshelf_4_asset)
        for p in bookshelf_4_props:
            p.friction = 0.5
            p.torsion_friction = 0.3
        self.gym.set_asset_rigid_shape_properties(bookshelf_4_asset, bookshelf_4_props)
        self.asset_handles["bookshelf_4"] = bookshelf_4_asset

        # shelf
        bookshelf_5_asset = self.gym.create_box(self.sim, 0.2, 0.4, 0.02, table_asset_options)
        bookshelf_5_props = self.gym.get_asset_rigid_shape_properties(bookshelf_5_asset)
        for p in bookshelf_5_props:
            p.friction = 0.5
            p.torsion_friction = 0.3
        self.gym.set_asset_rigid_shape_properties(bookshelf_5_asset, bookshelf_5_props)
        self.asset_handles["bookshelf_5"] = bookshelf_5_asset

        bookshelf_6_asset = self.gym.create_box(self.sim, x, y, z, table_asset_options)
        bookshelf_6_props = self.gym.get_asset_rigid_shape_properties(bookshelf_6_asset)
        for p in bookshelf_6_props:
            p.friction = 0.5
            p.torsion_friction = 0.3
        self.gym.set_asset_rigid_shape_properties(bookshelf_6_asset, bookshelf_6_props)
        self.asset_handles["bookshelf_6"] = bookshelf_6_asset

        bookshelf_7_asset = self.gym.create_box(self.sim, x, y, z, table_asset_options)
        bookshelf_7_props = self.gym.get_asset_rigid_shape_properties(bookshelf_7_asset)
        for p in bookshelf_7_props:
            p.friction = 0.5
            p.torsion_friction = 0.3
        self.gym.set_asset_rigid_shape_properties(bookshelf_7_asset, bookshelf_7_props)
        self.asset_handles["bookshelf_7"] = bookshelf_7_asset

        bookshelf_8_asset = self.gym.create_box(self.sim, x, y, z, table_asset_options)
        bookshelf_8_props = self.gym.get_asset_rigid_shape_properties(bookshelf_8_asset)
        for p in bookshelf_8_props:
            p.friction = 0.5
            p.torsion_friction = 0.3
        self.gym.set_asset_rigid_shape_properties(bookshelf_8_asset, bookshelf_8_props)
        self.asset_handles["bookshelf_8"] = bookshelf_8_asset

        bookshelf_9_asset = self.gym.create_box(self.sim, x, y, z, table_asset_options)
        bookshelf_9_props = self.gym.get_asset_rigid_shape_properties(bookshelf_9_asset)
        for p in bookshelf_9_props:
            p.friction = 0.5
            p.torsion_friction = 0.3
        self.gym.set_asset_rigid_shape_properties(bookshelf_9_asset, bookshelf_9_props)
        self.asset_handles["bookshelf_9"] = bookshelf_9_asset

    def _define_object_asset(self):
        """ Define Gym asset for object.
        """
        # define object asset
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.disable_gravity = False
        object_asset_options.thickness = 0.001
        object_asset_options.flip_visual_attachments = True
        obj_density = self.cfg["env"]["geometry"]["object"]["density"]
        object_asset_options.density = obj_density
        x = self.bookshelf_dims["book_size_x"]
        y = self.bookshelf_dims["book_size_y"]
        z = self.bookshelf_dims["book_size_z"]
        object_asset = self.gym.create_box(self.sim, x, y, z, object_asset_options)
        
        return object_asset

    def _check_failure(self) -> torch.Tensor:
        failed_resetz = torch.le(self._object_state_history[0][:, 2], (0.8 * 0.4))
        # failed_resety = torch.logical_or(torch.lt(self._object_state_history[0][:, 1], -0.23), torch.gt(self._object_state_history[0][:, 1], 0.27))
        # failed_resetx = torch.logical_or(torch.lt(self._object_state_history[0][:, 0], 0.28), torch.gt(self._object_state_history[0][:, 0], 0.72))
        # failed_reset = torch.logical_or(failed_resetx, failed_resety)
        # failed_reset = torch.logical_or(failed_reset, failed_resetz)
        return failed_resetz

    def _check_success(self) -> torch.Tensor:
        quat_a = self._object_state_history[0][:, 3:7]
        quat_b = self._object_goal_poses_buf[:, 3:7]
        angles = quat_diff_rad(quat_a, quat_b)
        goal_rotation_reached = torch.le(torch.abs(angles), self.cfg["env"]["reward_terms"]["object_rot"]["th"])
        goal_reached = goal_rotation_reached
        return goal_reached
