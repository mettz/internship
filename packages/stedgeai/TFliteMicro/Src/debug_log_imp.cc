/**
 ******************************************************************************
 * @file    debug_log_imp.cc
 * @author  MCD/AIS Team
 * @brief   Debug log implementation for TFL for MicroControllers runtime
 ******************************************************************************
 * @attention
 *
 * <h2><center>&copy; Copyright (c) 2019,2021 STMicroelectronics.
 * All rights reserved.</center></h2>
 *
 * This software is licensed under terms that can be found in the LICENSE file in
 * the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */

#include "tensorflow/lite/micro/debug_log.h"

#include <string.h>
#include <stdint.h>
#include <stdio.h>

extern "C" int tflm_io_write(const void *buff, uint16_t count);

extern "C" void DebugLog(const char* format, va_list args)
{
#ifndef TF_LITE_STRIP_ERROR_STRINGS
  constexpr int kMaxLogLen = 256;
  char log_buffer[kMaxLogLen];

  if (!format)
    return;

  vsnprintf(log_buffer, kMaxLogLen, format, args);

#if defined(USE_PRINTF)
#include <stdio.h>
	printf("%s", log_buffer);
#else 
  size_t sl = strlen(log_buffer);
  if (sl)
	  tflm_io_write(log_buffer, (uint16_t)sl);
#endif
#endif
}

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#ifndef TF_LITE_STRIP_ERROR_STRINGS
#include <stdio.h>
#endif

#ifndef TF_LITE_STRIP_ERROR_STRINGS
// Only called from MicroVsnprintf (micro_log.h)
int DebugVsnprintf(char* buffer, size_t buf_size, const char* format,
                   va_list vlist) {
  return vsnprintf(buffer, buf_size, format, vlist);
}
#endif

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus



