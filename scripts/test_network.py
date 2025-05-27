import logging

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger

# URI to the Crazyflie to connect to
uri = 'radio://0/88/2M/E7E7E7E7EF'

# Only output errors from the logging framework
logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    # Initialize the low-level drivers
    cflib.crtp.init_drivers()

    lg_stab = LogConfig(name='px4rl', period_in_ms=10)
    lg_stab.add_variable('px4rl_p.target_pos_x', 'float')
    lg_stab.add_variable('px4rl_p.target_pos_y', 'float')
    lg_stab.add_variable('px4rl_p.target_pos_z', 'float')
    lg_stab.add_variable('px4rl_v.vel_x', 'float')
    lg_stab.add_variable('px4rl_v.vel_y', 'float')
    lg_stab.add_variable('px4rl_v.vel_z', 'float')
    lg_stab.add_variable('px4rl_av.ang_vel_x', 'float')
    lg_stab.add_variable('px4rl_av.ang_vel_y', 'float')
    lg_stab.add_variable('px4rl_av.ang_vel_z', 'float')
    lg_stab.add_variable('px4rl_g.gravity_b_x', 'float')
    lg_stab.add_variable('px4rl_g.gravity_b_y', 'float')
    lg_stab.add_variable('px4rl_g.gravity_b_z', 'float')
    lg_stab.add_variable('px4rl_des.desired_x', 'float')
    lg_stab.add_variable('px4rl_des.desired_y', 'float')
    lg_stab.add_variable('px4rl_des.desired_z', 'float')
    lg_stab.add_variable('px4rl_out.thrust', 'float')
    lg_stab.add_variable('px4rl_out.torque_x', 'float')
    lg_stab.add_variable('px4rl_out.torque_y', 'float')
    lg_stab.add_variable('px4rl_out.torque_z', 'float')

    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        print('Connected to %s' % uri)
        with SyncLogger(scf, lg_stab) as logger:
            for log_entry in logger:

                timestamp = log_entry[0]
                data = log_entry[1]

                print('[%d]: %s' % (timestamp, data))