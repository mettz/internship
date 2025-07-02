from isaaclab_assets.robots import CRAZYFLIE_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg


@configclass
class IsaacflieEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 2
    action_space = 4
    observation_space = 18 + 4 * 32
    state_space = 18 + 4
    debug_vis = True

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True
    )

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # robot parameters
    arm_length = 0.028
    rotor_positions = [
        [arm_length, -arm_length, 0.0],
        [-arm_length, -arm_length, 0.0],
        [-arm_length, arm_length, 0.0],
        [arm_length, arm_length, 0.0],
    ]
    rotor_thrust_directions = [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ]
    rotor_torque_directions = [
        [0.0, 0.0, -1.0],
        [0.0, 0.0, +1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, +1.0],
    ]
    thrust_constants = [0.0, 0.0, 3.16e-10]
    torque_constant = 0.005964552
    rpm_time_constant = 0.15
    action_limits = {
        "min": 0.0,
        "max": 21702.0,
    }

    # mdp parameters
    action_noise = 0.0
    observation_noise = {
        "position": 0.001,
        "velocity": 0.001,
        "linear_velocity": 0.002,
        "angular_velocity": 0.002,
    }

    reward_params = {
        "non_negative": False,
        "action_baseline": 0.334,
        "scale": 0.5,
        "constant": 2,
        "termination_penalty": 0,
        "position": 5,
        "orientation": 5,
        "linear_velocity": 0.01,
        "angular_velocity": 0,
        "linear_acceleration": 0,
        "angular_acceleration": 0,
        "action": 0.01,
    }

    termination_params = {
        "position_threshold": 0.6,
        "linear_velocity_threshold": 1000,
        "angular_velocity_threshold": 1000,
    }
