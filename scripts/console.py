import time

import cflib.crtp  # noqa
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper

# Reads the CFLIB_URI environment variable for URI or uses default
uri = uri_helper.uri_from_env(default='radio://0/88/2M/E7E7E7E7EF')


def console_callback(text: str):
    '''A callback to run when we get console text from Crazyflie'''
    # We do not add newlines to the text received, we get them from the
    # Crazyflie at appropriate places.
    print(text, end='')


if __name__ == '__main__':
    # Initialize the low-level drivers
    cflib.crtp.init_drivers()

    # Create Crazyflie object, with cache to avoid re-reading param and log TOC
    cf = Crazyflie(rw_cache='./cache')

    # Add a callback to whenever we receive 'console' text from the Crazyflie
    # This is the output from DEBUG_PRINT calls in the firmware.
    #
    # This could also be a Python lambda, something like:
    #   cf.console.receivedChar.add_callback(lambda text: logger.info(text))
    cf.console.receivedChar.add_callback(console_callback)

    # This will connect the Crazyflie with the URI specified above.
    # You might have to restart your Crazyflie in order to get output
    # from console, since not much is written during regular uptime.
    with SyncCrazyflie(uri, cf=cf) as scf:
        print('[host] Connected, use ctrl-c to quit.')

        while True:
            time.sleep(1)