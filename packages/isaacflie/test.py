import torch
from types import SimpleNamespace

class DummyMultirotor:
    def __init__(self, batch_size=1):
        self.batch_size = batch_size

        # Output buffers
        self._thrust = torch.zeros((batch_size, 1, 3))
        self._torque = torch.zeros((batch_size, 1, 3))

        # Configuration values
        self.cfg = SimpleNamespace()

        # Normalized action limits
        self.cfg.action_limits = {
            "min": 0.0,        # RPM min
            "max": 1500.0      # RPM max
        }

        self.cfg.action_noise = 0.0  # Set to >0.0 to test noise

        # Quadratic thrust model: thrust = a + b*rpm + c*rpm^2
        self.cfg.thrust_constants = [0.0, 0.0, 1e-6]  # Simple squared model

        self.cfg.torque_constant = 1e-7

        # Rotor layout (quad-X)
        L = 0.2  # Distance from center
        self.cfg.rotor_positions = torch.tensor([
            [ L,  L, 0.0],
            [-L,  L, 0.0],
            [-L, -L, 0.0],
            [ L, -L, 0.0],
        ])  # (4, 3)

        self.cfg.rotor_thrust_directions = torch.tensor([
            [0.0, 0.0, 1.0],  # All thrust up
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ])  # (4, 3)

        self.cfg.rotor_torque_directions = torch.tensor([
            [0.0, 0.0, -1.0],  # CW
            [0.0, 0.0,  1.0],  # CCW
            [0.0, 0.0, -1.0],
            [0.0, 0.0,  1.0],
        ])  # (4, 3)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()

        half_range = 0.5 * (self.cfg.action_limits["max"] - self.cfg.action_limits["min"])
        mean_rpm = half_range + self.cfg.action_limits["min"]

        if self.cfg.action_noise > 0.0:
            noise = torch.randn_like(self._actions) * self.cfg.action_noise
            self._actions += noise

        self._actions = self._actions.clamp(-1.0, 1.0)
        rpms = self._actions * half_range + mean_rpm  # shape: (B, 4)

        a, b, c = self.cfg.thrust_constants
        thrust_mag = a + b * rpms + c * rpms**2  # (B, 4)

        rotor_thrust_dirs = self.cfg.rotor_thrust_directions.to(rpms.device)  # (4, 3)
        rotor_thrusts = thrust_mag.unsqueeze(-1) * rotor_thrust_dirs  # (B, 4, 3)

        self._thrust[:, 0, :] = rotor_thrusts.sum(dim=1)

        rotor_torque_dirs = self.cfg.rotor_torque_directions.to(rpms.device)  # (4, 3)
        drag_torques = thrust_mag.unsqueeze(-1) * self.cfg.torque_constant * rotor_torque_dirs

        rotor_positions = self.cfg.rotor_positions.to(rpms.device)  # (4, 3)
        moment_arms = torch.cross(rotor_positions.unsqueeze(0), rotor_thrusts, dim=-1)  # (B, 4, 3)

        self._torque[:, 0, :] = drag_torques.sum(dim=1) + moment_arms.sum(dim=1)


if __name__ == "__main__":
    # Create dummy instance
    robot = DummyMultirotor(batch_size=1)

    # Define normalized action: all rotors at 50% power (0.0 normalized)
    actions = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # (B, 4)

    # Run rotor logic
    robot._pre_physics_step(actions)

    # Print results
    print("Thrust:\n", robot._thrust)
    print("Torque:\n", robot._torque)
