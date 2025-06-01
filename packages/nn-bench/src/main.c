#include <stdint.h>
#include <string.h>

#include "stm32f4xx.h"

#include "FreeRTOS.h"
#include "event_groups.h"
#include "task.h"

#include "static_mem.h"
#include "system.h"

#include "log.h"
#include "param.h"
#include "usec_time.h"

#define DEBUG_MODULE "APPMAIN"
#include "debug.h"

#include "network/network.h"
#include "network/network_data.h"

#define AI_EVENT_START_BIT (1 << 0)
#define AI_EVENT_DONE_BIT (1 << 1)
#define AI_EVENT_FAIL_BIT (1 << 2)

STAI_NETWORK_CONTEXT_DECLARE(m_network, STAI_NETWORK_CONTEXT_SIZE)

STAI_ALIGNED(STAI_NETWORK_ACTIVATION_1_ALIGNMENT)
static uint8_t activations_1[STAI_NETWORK_ACTIVATION_1_SIZE_BYTES];

static stai_ptr m_inputs[STAI_NETWORK_IN_NUM];
static stai_ptr m_outputs[STAI_NETWORK_OUT_NUM];
static stai_ptr m_acts[STAI_NETWORK_ACTIVATIONS_NUM];

static float observations[STAI_NETWORK_IN_1_SIZE];
static float actions[STAI_NETWORK_OUT_1_SIZE];

static uint32_t ai_cycles = 0;
static uint32_t ai_start = 0;
static TaskHandle_t ai_task_handle = NULL; // To be initialized later
static EventGroupHandle_t ai_event_group = NULL;

void dwt_init(void) {
  // Initialize the DWT (Data Watchpoint and Trace) unit for cycle counting
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk; // Enable trace
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;            // Enable cycle counter
  DWT->CYCCNT = 0;                                // Reset cycle counter
}

uint32_t dwt_get_cycles(void) {
  return DWT->CYCCNT; // Return the current cycle count
}

void on_task_switched_in(void) {
  if (xTaskGetCurrentTaskHandle() == ai_task_handle) {
    ai_start = dwt_get_cycles();
  }
}

void on_task_switched_out(void) {
  if (xTaskGetCurrentTaskHandle() == ai_task_handle) {
    ai_cycles += dwt_get_cycles() - ai_start;
  }
}

static void aiTask(void *pvParameters) {
  (void)pvParameters; // Unused parameter

  EventBits_t bits;

  while (1) {
    bits = xEventGroupWaitBits(ai_event_group, AI_EVENT_START_BIT, pdTRUE,
                               pdFALSE, portMAX_DELAY);

    if (bits & AI_EVENT_START_BIT) {
      stai_return_code res = stai_network_run(m_network, STAI_MODE_SYNC);
      if (res == STAI_SUCCESS) {
        xEventGroupSetBits(ai_event_group, AI_EVENT_DONE_BIT);
      } else {
        xEventGroupSetBits(ai_event_group, AI_EVENT_FAIL_BIT);
      }
    }
  }
}

void appMain() {
  dwt_init();

  stai_size _dummy;

  stai_network_init(m_network);

  m_acts[0] = (stai_ptr)activations_1;
  stai_network_set_activations(m_network, m_acts, STAI_NETWORK_ACTIVATIONS_NUM);
  stai_network_get_activations(m_network, m_acts, &_dummy);

  stai_network_get_inputs(m_network, m_inputs, &_dummy);
  memset(observations, 0.0f, sizeof(observations));

  stai_network_get_outputs(m_network, m_outputs, &_dummy);
  memset(actions, 0.0f, sizeof(actions));

  ai_event_group = xEventGroupCreate();
  if (ai_event_group == NULL) {
    DEBUG_PRINT("Failed to create AI event group\n");
    goto err_event_group_creation;
  }

  if (xTaskCreate(aiTask, "AI", 1000, NULL, 0, &ai_task_handle) != pdPASS) {
    DEBUG_PRINT("Failed to create AI task\n");
    goto err_task_creation;
  }

  while (1) {
    vTaskDelay(M2T(2000));

    ai_cycles = 0;
    memcpy(m_inputs[0], observations, sizeof(observations));
    xEventGroupSetBits(ai_event_group, AI_EVENT_START_BIT);

    EventBits_t bits = xEventGroupWaitBits(
        ai_event_group, AI_EVENT_DONE_BIT | AI_EVENT_FAIL_BIT, pdTRUE, pdFALSE,
        portMAX_DELAY);

    if (bits & AI_EVENT_DONE_BIT) {
      memcpy(actions, m_outputs[0], sizeof(actions));
      float duration_ms = ai_cycles * 1000.0f / configCPU_CLOCK_HZ;
      DEBUG_PRINT("STAI network run took %.2f ms\n", (double)duration_ms);
      for (int i = 0; i < STAI_NETWORK_OUT_1_SIZE; i++) {
        DEBUG_PRINT("Action %d: %.2f\n", i, (double)actions[i]);
      }
    } else {
      DEBUG_PRINT("stai_network_run failed.\n");
    }
  }

err_task_creation:
  vEventGroupDelete(ai_event_group);
err_event_group_creation:
  DEBUG_PRINT("Exiting appMain due to errors\n");
  while (1) {
    vTaskDelay(portMAX_DELAY);
  }
}