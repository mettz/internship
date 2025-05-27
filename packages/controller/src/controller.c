#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "controller.h"
#include "log.h"
#include "usec_time.h"

#include "network/network.h"
#include "network/network_data.h"

#define DEBUG_MODULE "PX4RL"
#include "debug.h"

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

void controllerOutOfTreeInit() {
  DEBUG_PRINT("Initializing controller...\n");

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
}

bool controllerOutOfTreeTest() {
  // Always return true
  return true;
}

void controllerOutOfTree(control_t *control, const setpoint_t *setpoint,
                         const sensorData_t *sensors, const state_t *state,
                         const stabilizerStep_t stabilizerStep) {

  if (!RATE_DO_EXECUTE(ATTITUDE_RATE, stabilizerStep)) {
    return;
  }

  stai_return_code res = STAI_SUCCESS;

  memcpy(m_inputs[0], observations, sizeof(observations));

  uint64_t start = usecTimestamp();
  res = stai_network_run(m_network, STAI_MODE_SYNC);
  uint64_t end = usecTimestamp();

  if (RATE_DO_EXECUTE(RATE_SUPERVISOR, stabilizerStep)) {
    DEBUG_PRINT("Network run took %llu us\n", end - start);
  }

  if (res != STAI_SUCCESS) {
    DEBUG_PRINT("Error running network: %d\n", res);
  } else {
    memcpy(actions, m_outputs[0], sizeof(actions));
    // for (int i = 0; i < STAI_NETWORK_OUT_1_SIZE; i++) {
    //   DEBUG_PRINT("Action %d: %f\n", i, (double)actions[i]);
    // }
  }
}