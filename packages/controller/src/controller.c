#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "controller.h"
#include "controller_pid.h"

#define DEBUG_MODULE "PX4RL"
#include "debug.h"

void controllerOutOfTreeInit() {
  DEBUG_PRINT("Initializing controller...\n");

  controllerPidInit();
}

bool controllerOutOfTreeTest() {
  // Always return true
  return true;
}

void controllerOutOfTree(control_t *control, const setpoint_t *setpoint,
                         const sensorData_t *sensors, const state_t *state,
                         const uint32_t tick) {
  controllerPid(control, setpoint, sensors, state, tick);
}
