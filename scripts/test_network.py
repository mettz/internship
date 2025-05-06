import logging

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger

# URI to the Crazyflie to connect to
uri = 'radio://0/88/2M/E7E7E7E7EF'

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)

if __name__ == '__main__':
    # Initialize the low-level drivers
    cflib.crtp.init_drivers()

    lg_stab = LogConfig(name='Actions', period_in_ms=10)
    lg_stab.add_variable('actions.action_0', 'float')
    lg_stab.add_variable('actions.action_1', 'float')
    lg_stab.add_variable('actions.action_2', 'float')
    lg_stab.add_variable('actions.action_3', 'float')

    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        with SyncLogger(scf, lg_stab) as logger:
            for log_entry in logger:

                timestamp = log_entry[0]
                data = log_entry[1]

                print('[%d]: %s' % (timestamp, data))