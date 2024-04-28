from tasks.domain import Domain, Object
from typing import List, Dict
from isaacgym import gymapi
import numpy as np
import torch
from utils.torch_jit_utils import quat_diff_rad


class Hole(Domain):
    def __init__(self, cfg, sim_device, graphics_device_id, headless, use_state=False, gym=None):
        super().__init__(cfg, sim_device, graphics_device_id, headless, use_state, gym=gym)

    def _set_table_dimension(self, cfg):
        pass

    def _get_table_prim_names(self) -> List[str]:
        self.boxes: Dict[str, Dict[str, float]] = self.cfg["env"]["geometry"]["boxes"]
        return list(self.boxes.keys())

    def _set_object_dimension(self, object_dims) -> Object:
        return Object((object_dims["width"], object_dims["length"], object_dims["height"]))

    def _create_table(self, env_ptr, env_index: int, actor_indices: Dict[str, List[int]]):
        for key, item in self.boxes.items():
            box_color = gymapi.Vec3(0.85, 0.85, 0.85)
            if key == "box1":
                x = item["x"]
                y = item["y"]
                z = item["z"]
            elif key == "box2":
                x = item["x"]
                y = item["y"]
                z = item["z"]
            # elif key == "box3":
            #     x = item["x"]
            #     y = item["y"]
            #     z = item["z"]
            # elif key == "box4":
            #     x = item["x"]
            #     y = item["y"]
            #     z = item["z"]
            # else:
            #     x = item["x"]
            #     y = item["y"]
            #     z = item["z"]
            box_pose = gymapi.Transform()
            box_pose.p = gymapi.Vec3(x, y, z)
            box_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            box_handle = self.gym.create_actor(env_ptr, self.asset_handles[key], box_pose, key, env_index, 2, 1)
            box_idx = self.gym.get_actor_index(env_ptr, box_handle, gymapi.DOMAIN_SIM)
            self.gym.set_rigid_body_color(env_ptr, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, box_color)
            actor_indices[key].append(box_idx)

    def _create_object(self, env_ptr, env_index: int, actor_indices: Dict[str, List[int]]):
        object_handle = self.gym.create_actor(env_ptr, self.asset_handles["object"], gymapi.Transform(), "object", env_index, 0, 2)
        object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
        actor_indices["object"].append(object_idx)

    def _max_dist_btw_obj_and_goal(self) -> torch.Tensor:
        return np.sqrt(0.4 ** 2 + 0.5 ** 2 + 0.1 ** 2) * torch.ones(1, dtype=torch.float32, device=self.device)

    def _define_table_asset(self):
        """ Define Gym asset for table. This function returns nothing.
        """
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = True
        table_asset_options.fix_base_link = True
        table_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        table_asset_options.thickness = 0.001
        for key, item in self.boxes.items():
            if key == "box1" or key == "box2":
                table_asset = self.gym.create_box(self.sim, item["width"], item["length"], item["height"], table_asset_options)
            # elif key == "box3" or key == "box4":
            #     table_asset = self.gym.create_box(self.sim, item["width"], item["length"], item["height"], table_asset_options)
            # else:
            #     table_asset = self.gym.create_box(self.sim, item["width"], item["length"], item["height"], table_asset_options)
            table_props = self.gym.get_asset_rigid_shape_properties(table_asset)
            for p in table_props:
                p.friction = 0.5
                p.restitution = 0.5
            self.gym.set_asset_rigid_shape_properties(table_asset, table_props)
            self.asset_handles[key] = table_asset

    def _define_object_asset(self):
        """ Define Gym asset for object.
        """
        # define object asset
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.disable_gravity = False
        object_asset_options.thickness = 0.001
        obj_density = self.cfg["env"]["geometry"]["object"]["density"]
        object_asset_options.density = obj_density
        object_asset = self.gym.load_asset(self.sim, self._assets_dir, 'urdf/Panda/Coloredbox.urdf', object_asset_options)
        object_props = self.gym.get_asset_rigid_shape_properties(object_asset)
        for p in object_props:
            p.friction = 0.5
            p.restitution = 0.5
        self.gym.set_asset_rigid_shape_properties(object_asset, object_props)

        return object_asset

    def _check_failure(self) -> torch.Tensor: # TODO: set failure criterion
        failed_envs = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.rl_device)
        return failed_envs

    def _check_success(self) -> torch.Tensor:
        delta = self._object_state_history[0][:, :2] - self._object_goal_poses_buf[:, :2]
        dist = torch.norm(delta, p=2, dim=-1)
        goal_position_reached = torch.le(dist, self.cfg["env"]["reward_terms"]["object_dist"]["th"])
        goal_reached = goal_position_reached
        return goal_reached
