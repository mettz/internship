#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "controller.h"
#include "controller_pid.h"
#include "log.h"
#include "math3d.h"
#include "platform_defaults.h"
#include "usec_time.h"

#include "network/network.h"
#include "network/network_data.h"

#define DEBUG_MODULE "PX4RL"
#include "debug.h"

#define POS_DISTANCE_LIMIT 0.5f // Limit for position distance in meters

/* Global handle to reference the instantiated C-model */
STAI_NETWORK_CONTEXT_DECLARE(m_network, STAI_NETWORK_CONTEXT_SIZE)

/* Array to store the data of the activation buffers */
STAI_ALIGNED(STAI_NETWORK_ACTIVATION_1_ALIGNMENT)
static uint8_t activations_1[STAI_NETWORK_ACTIVATION_1_SIZE_BYTES];

/* Array to store the data of the input tensors */
/* -> data_in_1 is allocated in activations buffer */

/* Array to store the data of the output tensors */
/* -> data_out_1 is allocated in activations buffer */

static stai_ptr m_inputs[STAI_NETWORK_IN_NUM];
static stai_ptr m_outputs[STAI_NETWORK_OUT_NUM];
static stai_ptr m_acts[STAI_NETWORK_ACTIVATIONS_NUM];

static float observations[STAI_NETWORK_IN_1_SIZE];
static float actions[STAI_NETWORK_OUT_1_SIZE];

static float target_pos[3] = {0.0f, 0.0f, 0.0f};
static struct vec gravity_w;
static const float thrust_to_weight = 1.9f; // 50% thrust to weight ratio
static const float robot_weight =
    CF_MASS * 9.81f; // kg, adjust as per your robot's weight
static const float moment_scale = 0.01f; // Nm, scale for the moment actions

static float thrust = 0.0f;
static float torques[3] = {0.0f, 0.0f, 0.0f};

static inline float clip(float v, float low, float high) {
  if (v < low) {
    return low;
  } else {
    if (v > high) {
      return high;
    } else {
      return v;
    }
  }
}

void controllerOutOfTreeInit() {
  stai_size _dummy;

  /* -- Create and initialize the c-model */

  /* Initialize the instance */
  stai_network_init(m_network);

  /* -- Set the @ of the activation buffers */

  /* Activation buffers are allocated in the user/app space */
  m_acts[0] = (stai_ptr)activations_1;
  stai_network_set_activations(m_network, m_acts, STAI_NETWORK_ACTIVATIONS_NUM);
  stai_network_get_activations(m_network, m_acts, &_dummy);

  /* -- Set the @ of the input/output buffers */

  /* Input buffers are allocated in the activations buffer */
  stai_network_get_inputs(m_network, m_inputs, &_dummy);
  memset(actions, 0.0f, sizeof(actions));

  /* Output buffers are allocated in the activations buffer */
  stai_network_get_outputs(m_network, m_outputs, &_dummy);
  memset(observations, 0.0f, sizeof(observations));

  gravity_w = mkvec(0.0f, 0.0f, -9.81f); // world gravity

  controllerPidInit();
}

bool controllerOutOfTreeTest() { return controllerPidTest(); }

void controllerOutOfTree(control_t *control, const setpoint_t *setpoint,
                         const sensorData_t *sensors, const state_t *state,
                         const stabilizerStep_t stabilizerStep) {

  if (!RATE_DO_EXECUTE(ATTITUDE_RATE, stabilizerStep)) {
    // If not executing the attitude rate, just return
    return;
  }

  if (stabilizerStep % 1000 == 0) {
    DEBUG_PRINT("SETPOINT:\n");
    DEBUG_PRINT("  timestamp: %lu\n", setpoint->timestamp);
    DEBUG_PRINT("  attitude: { roll: %.2f, pitch: %.2f, yaw: %.2f }\n",
                setpoint->attitude.roll, setpoint->attitude.pitch,
                setpoint->attitude.yaw);
    DEBUG_PRINT("  attitudeRate: { roll: %.2f, pitch: %.2f, yaw: %.2f }\n",
                setpoint->attitudeRate.roll, setpoint->attitudeRate.pitch,
                setpoint->attitudeRate.yaw);
    DEBUG_PRINT(
        "  attitudeQuaternion: { q0: %.2f, q1: %.2f, q2: %.2f, q3: %.2f }\n",
        setpoint->attitudeQuaternion.q0, setpoint->attitudeQuaternion.q1,
        setpoint->attitudeQuaternion.q2, setpoint->attitudeQuaternion.q3);
    DEBUG_PRINT("  thrust: %.2f\n", setpoint->thrust);
    DEBUG_PRINT("  position: { x: %.2f, y: %.2f, z: %.2f }\n",
                setpoint->position.x, setpoint->position.y,
                setpoint->position.z);
    DEBUG_PRINT("  velocity: { x: %.2f, y: %.2f, z: %.2f }\n",
                setpoint->velocity.x, setpoint->velocity.y,
                setpoint->velocity.z);
    DEBUG_PRINT("  acceleration: { x: %.2f, y: %.2f, z: %.2f }\n",
                setpoint->acceleration.x, setpoint->acceleration.y,
                setpoint->acceleration.z);
    DEBUG_PRINT("  jerk: { x: %.2f, y: %.2f, z: %.2f }\n", setpoint->jerk.x,
                setpoint->jerk.y, setpoint->jerk.z);
    DEBUG_PRINT("  velocity_body: %s\n",
                setpoint->velocity_body ? "true" : "false");
    DEBUG_PRINT("  mode: { x: %d, y: %d, z: %d, roll: %d, pitch: %d, yaw: %d, "
                "quat: %d }\n",
                setpoint->mode.x, setpoint->mode.y, setpoint->mode.z,
                setpoint->mode.roll, setpoint->mode.pitch, setpoint->mode.yaw,
                setpoint->mode.quat);
  }

  if (setpoint->mode.x == modeAbs) {
    target_pos[0] = setpoint->position.x;
  } else {
    target_pos[0] = 0;
  }

  if (setpoint->mode.y == modeAbs) {
    target_pos[1] = setpoint->position.y;
  } else {
    target_pos[1] = 0;
  }

  if (setpoint->mode.z == modeAbs) {
    target_pos[2] = setpoint->position.z;
  } else {
    target_pos[2] = 0;
  }

  struct vec p_robot =
      mkvec(state->position.x, state->position.y, state->position.z);
  struct vec p_desired = mkvec(target_pos[0], target_pos[1], target_pos[2]);

  struct vec delta_w = vsub(p_robot, p_desired);

  struct quat q =
      mkquat(setpoint->attitudeQuaternion.q0, setpoint->attitudeQuaternion.q1,
             setpoint->attitudeQuaternion.q2, setpoint->attitudeQuaternion.q3);
  struct quat q_inv = qinv(q);

  struct vec delta_b = qvrot(q_inv, delta_w);
  struct vec projected_gravity_b = qvrot(q_inv, gravity_w);

  observations[0] = state->velocity.x;
  observations[1] = state->velocity.y;
  observations[2] = state->velocity.z;
  observations[3] = state->attitude.roll;
  observations[4] = state->attitude.pitch;
  observations[5] = state->attitude.yaw;
  observations[6] = projected_gravity_b.x;
  observations[7] = projected_gravity_b.y;
  observations[8] = projected_gravity_b.z;
  observations[9] = clip(delta_b.x, -POS_DISTANCE_LIMIT, POS_DISTANCE_LIMIT);
  observations[10] = clip(delta_b.y, -POS_DISTANCE_LIMIT, POS_DISTANCE_LIMIT);
  observations[11] = clip(delta_b.z, -POS_DISTANCE_LIMIT, POS_DISTANCE_LIMIT);

  memcpy(m_inputs[0], observations, sizeof(observations));

  uint64_t start = usecTimestamp();
  stai_return_code ret = stai_network_run(m_network, STAI_MODE_SYNC);
  uint64_t end = usecTimestamp();

  if (ret != STAI_SUCCESS) {
    DEBUG_PRINT("STAI network run failed with error code: %d\n", ret);
  }

  memcpy(actions, m_outputs[0], sizeof(actions));

  if (stabilizerStep % 1000 == 0) {
    DEBUG_PRINT("STAI network run took %llu us\n", end - start);
  }

  for (int i = 0; i < STAI_NETWORK_OUT_1_SIZE; i++) {
    actions[i] = clip(actions[i], -1.0f, 1.0f);
  }

  thrust = thrust_to_weight * robot_weight * (actions[0] + 1.0f) / 2.0f;
  torques[0] = moment_scale * actions[1]; // Torque X
  torques[1] = moment_scale * actions[2]; // Torque Y
  torques[2] = moment_scale * actions[3]; // Torque Z

  control->controlMode = controlModeForceTorque;
  control->thrustSi = thrust;
  control->torqueX = torques[0];
  control->torqueY = torques[1];
  control->torqueZ = torques[2];

  // controllerPid(control, setpoint, sensors, state, stabilizerStep);
}

LOG_GROUP_START(px4rl_p)
LOG_ADD(LOG_FLOAT, target_pos_x, &target_pos[0])
LOG_ADD(LOG_FLOAT, target_pos_y, &target_pos[1])
LOG_ADD(LOG_FLOAT, target_pos_z, &target_pos[2])
LOG_GROUP_STOP(px4rl_p)

LOG_GROUP_START(px4rl_v)
LOG_ADD(LOG_FLOAT, vel_x, &observations[0])
LOG_ADD(LOG_FLOAT, vel_y, &observations[1])
LOG_ADD(LOG_FLOAT, vel_z, &observations[2])
LOG_GROUP_STOP(px4rl_v)

LOG_GROUP_START(px4rl_av)
LOG_ADD(LOG_FLOAT, ang_vel_x, &observations[3])
LOG_ADD(LOG_FLOAT, ang_vel_y, &observations[4])
LOG_ADD(LOG_FLOAT, ang_vel_z, &observations[5])
LOG_GROUP_STOP(px4rl_av)

LOG_GROUP_START(px4rl_g)
LOG_ADD(LOG_FLOAT, gravity_b_x, &observations[6])
LOG_ADD(LOG_FLOAT, gravity_b_y, &observations[7])
LOG_ADD(LOG_FLOAT, gravity_b_z, &observations[8])
LOG_GROUP_STOP(px4rl_g)

LOG_GROUP_START(px4rl_des)
LOG_ADD(LOG_FLOAT, desired_x, &observations[9])
LOG_ADD(LOG_FLOAT, desired_y, &observations[10])
LOG_ADD(LOG_FLOAT, desired_z, &observations[11])
LOG_GROUP_STOP(px4rl_des)

LOG_GROUP_START(px4rl_out)
LOG_ADD(LOG_FLOAT, thrust, &thrust)
LOG_ADD(LOG_FLOAT, torque_x, &torques[0])
LOG_ADD(LOG_FLOAT, torque_y, &torques[1])
LOG_ADD(LOG_FLOAT, torque_z, &torques[2])
LOG_GROUP_STOP(px4rl_out)