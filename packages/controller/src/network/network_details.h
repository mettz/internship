/**
  ******************************************************************************
  * @file    network.h
  * @date    2025-05-13T11:15:47+0200
  * @brief   ST.AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */
#ifndef STAI_NETWORK_DETAILS_H
#define STAI_NETWORK_DETAILS_H

#include "stai.h"
#include "layers.h"

const stai_network_details g_network_details = {
  .tensors = (const stai_tensor[8]) {
   { .size_bytes = 68, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 17}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "obs_output" },
   { .size_bytes = 1024, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 256}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_actor_net_net_0_Gemm_output_0_output" },
   { .size_bytes = 1024, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 256}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_actor_net_net_1_Elu_output_0_output" },
   { .size_bytes = 1024, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 256}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_actor_net_net_2_Gemm_output_0_output" },
   { .size_bytes = 1024, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 256}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_actor_net_net_3_Elu_output_0_output" },
   { .size_bytes = 512, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 128}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_actor_net_net_4_Gemm_output_0_output" },
   { .size_bytes = 512, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 128}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_actor_net_net_5_Elu_output_0_output" },
   { .size_bytes = 16, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 4}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "actions_output" }
  },
  .nodes = (const stai_node_details[7]){
    {.id = 1, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {1, (const int32_t[1]){1}} }, /* _actor_net_net_0_Gemm_output_0 */
    {.id = 2, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){2}} }, /* _actor_net_net_1_Elu_output_0 */
    {.id = 3, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){2}}, .output_tensors = {1, (const int32_t[1]){3}} }, /* _actor_net_net_2_Gemm_output_0 */
    {.id = 4, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){3}}, .output_tensors = {1, (const int32_t[1]){4}} }, /* _actor_net_net_3_Elu_output_0 */
    {.id = 5, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){4}}, .output_tensors = {1, (const int32_t[1]){5}} }, /* _actor_net_net_4_Gemm_output_0 */
    {.id = 6, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){5}}, .output_tensors = {1, (const int32_t[1]){6}} }, /* _actor_net_net_5_Elu_output_0 */
    {.id = 7, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){6}}, .output_tensors = {1, (const int32_t[1]){7}} } /* actions */
  },
  .n_nodes = 7
};
#endif