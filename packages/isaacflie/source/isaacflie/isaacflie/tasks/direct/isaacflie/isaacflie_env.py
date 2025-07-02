from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import matrix_from_quat
from isaaclab.markers import CUBOID_MARKER_CFG

from .isaacflie_env_cfg import IsaacflieEnvCfg


class IsaacflieEnv(DirectRLEnv):
    cfg: IsaacflieEnvCfg

    def __init__(self, cfg: IsaacflieEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._rotor_positions = torch.tensor(
            self.cfg.rotor_positions, device=self.device
        )
        self._rotor_thrust_directions = torch.tensor(
            self.cfg.rotor_thrust_directions, device=self.device
        )
        self._rotor_torque_directions = torch.tensor(
            self.cfg.rotor_torque_directions, device=self.device
        )

        self._actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            device=self.device,
        )
        self._action_history = torch.zeros(
            (self.num_envs, 32, self._actions.shape[1]),
            device=self.device,
        )
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._torque = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._prev_lin_vel = torch.zeros_like(self._robot.data.root_lin_vel_b)
        self._prev_ang_vel = torch.zeros_like(self._robot.data.root_ang_vel_b)
        self._rpm = torch.zeros_like(self._actions)

        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

        with self._window.ui_window_elements["main_vstack"]:
            with self._window.ui_window_elements["debug_frame"]:
                with self._window.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._window._create_debug_vis_ui_element("targets", self)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # setup ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _update_rpms(self, actions: torch.Tensor):
        # Assuming current RPMs are self._rpm, and `action` represents the desired RPM changes
        k1 = self._rpm.clone()
        # Compute the change in RPMs based on the action (adjustment), using time constant
        k1 = (actions - self._rpm) / self.cfg.rpm_time_constant

        # Calculate the second step (k2) with a similar method, using k1 as the intermediate adjustment
        k2 = self._rpm + (self.step_dt / 2) * k1
        k2 = (actions - k2) / self.cfg.rpm_time_constant

        # Third step (k3), use k2 for adjustments
        k3 = self._rpm + (self.step_dt / 2) * k2
        k3 = (actions - k3) / self.cfg.rpm_time_constant

        # Fourth step (k4), full step using k3
        k4 = self._rpm + self.step_dt * k3
        k4 = (actions - k4) / self.cfg.rpm_time_constant

        # Final update of RPMs, weighted average of k1, k2, k3, k4
        self._rpm += (self.step_dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _update_thrust_and_torque(self, rpms: torch.Tensor):
        a, b, c = self.cfg.thrust_constants
        thrust_mag = a + b * rpms + c * rpms**2  # (B, 4)

        rotor_thrusts = (
            thrust_mag.unsqueeze(-1) * self._rotor_thrust_directions
        )  # (B, 4, 3)

        self._thrust[:, 0, :] = rotor_thrusts.sum(dim=1)

        drag_torques = (
            thrust_mag.unsqueeze(-1)
            * self.cfg.torque_constant
            * self._rotor_torque_directions
        )

        moment_arms = torch.cross(
            self._rotor_positions.unsqueeze(0), rotor_thrusts, dim=-1
        )  # (B, 4, 3)

        self._torque[:, 0, :] = drag_torques.sum(dim=1) + moment_arms.sum(dim=1)

    def _pre_physics_step(self, actions: torch.Tensor):
        # save current velocities for reward computation later
        self._prev_lin_vel = self._robot.data.root_lin_vel_b.clone()
        self._prev_ang_vel = self._robot.data.root_ang_vel_b.clone()

        self._actions = actions.clone()
        self._action_history = torch.roll(self._action_history, shifts=-1, dims=1)
        self._action_history[:, -1, :] = self._actions

        half_range = 0.5 * (
            self.cfg.action_limits["max"] - self.cfg.action_limits["min"]
        )
        mean_rpm = half_range + self.cfg.action_limits["min"]

        if self.cfg.action_noise > 0.0:
            noise = torch.randn_like(self._actions) * self.cfg.action_noise
            self._actions += noise

        self._actions = self._actions.clamp(-1.0, 1.0)
        scaled_actions = self._actions * half_range + mean_rpm  # shape: (B, 4)

        self._update_thrust_and_torque(scaled_actions)
        self._update_rpms(scaled_actions)

    def _apply_action(self):
        self._robot.set_external_force_and_torque(
            self._thrust, self._torque, body_ids=self._body_id
        )

    def _get_observations(self) -> dict:
        position = self._robot.data.root_pos_w - self._terrain.env_origins
        rot_mat = matrix_from_quat(self._robot.data.root_quat_w)
        rot_mat_flat = rot_mat.view(self.num_envs, 9)

        common_obs = torch.cat(
            [
                position,
                rot_mat_flat,
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
            ],
            dim=-1,
        )

        actor_obs = torch.cat(
            [
                common_obs,
                self._action_history.view(self.num_envs, -1),
            ],
            dim=-1,
        )

        rpms = (self._rpm - self.cfg.action_limits["min"]) / (
            self.cfg.action_limits["max"] - self.cfg.action_limits["min"]
        ) * 2.0 - 1.0

        critic_obs = torch.cat(
            [
                common_obs,
                rpms,
            ],
            dim=-1,
        )

        observations = {"policy": actor_obs, "critic": critic_obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        pos = self._robot.data.root_pos_w - self._terrain.env_origins
        quat = self._robot.data.root_quat_w
        lin_vel = self._robot.data.root_lin_vel_b
        ang_vel = self._robot.data.root_ang_vel_b

        lin_acc = (lin_vel - self._prev_lin_vel) / self.step_dt
        ang_acc = (ang_vel - self._prev_ang_vel) / self.step_dt

        action_diff = self._actions - self.cfg.reward_params["action_baseline"]

        rp = self.cfg.reward_params

        pos_sq = torch.square(pos).sum(dim=-1)
        orient_cost = 1.0 - torch.square(quat[..., 0])
        lin_vel_sq = torch.square(lin_vel).sum(dim=-1)
        ang_vel_sq = torch.square(ang_vel).sum(dim=-1)
        lin_acc_sq = torch.square(lin_acc).sum(dim=-1)
        ang_acc_sq = torch.square(ang_acc).sum(dim=-1)
        action_diff_sq = torch.square(action_diff).sum(dim=-1)

        weighted_cost = (
            rp["position"] * pos_sq
            + rp["orientation"] * orient_cost
            + rp["linear_velocity"] * lin_vel_sq
            + rp["angular_velocity"] * ang_vel_sq
            + rp["linear_acceleration"] * lin_acc_sq
            + rp["angular_acceleration"] * ang_acc_sq
            + rp["action"] * action_diff_sq
        )

        scaled_cost = rp["scale"] * weighted_cost
        reward = -scaled_cost + rp["constant"]
        if rp["non_negative"]:
            reward = torch.clamp_min(reward, 0.0)

        reward = torch.where(self.reset_terminated, rp["termination_penalty"], reward)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        pos_thresh = self.cfg.termination_params["position_threshold"]
        lin_vel_thresh = self.cfg.termination_params["linear_velocity_threshold"]
        ang_vel_thresh = self.cfg.termination_params["angular_velocity_threshold"]

        position = self._robot.data.root_pos_w - self._terrain.env_origins

        too_low = torch.abs(position[:, 2]) < 0.1  # (num_envs,)
        too_far = torch.any(torch.abs(position) > pos_thresh, dim=-1)  # (num_envs,)
        pos_exceeded = torch.logical_or(too_low, too_far)  # (num_envs,)

        lin_vel_exceeded = (
            torch.abs(self._robot.data.root_lin_vel_b) > lin_vel_thresh
        ).any(dim=-1)
        ang_vel_exceeded = (
            torch.abs(self._robot.data.root_ang_vel_b) > ang_vel_thresh
        ).any(dim=-1)

        died = pos_exceeded | lin_vel_exceeded | ang_vel_exceeded

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        self._actions[env_ids] = 0.0
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(
            torch.zeros(self.num_envs, 3, device=self.device)
        )
