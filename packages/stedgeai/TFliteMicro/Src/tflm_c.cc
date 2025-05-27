/**
 ******************************************************************************
 * @file    tflm_c.cc
 * @author  MCD/AIS Team
 * @brief   Light C-wrapper for C++ TFL for MicroControllers runtime interface
 ******************************************************************************
 * @attention
 *
 * <h2><center>&copy; Copyright (c) 2019,2021,2024 STMicroelectronics.
 * All rights reserved.</center></h2>
 *
 * This software is licensed under terms that can be found in the LICENSE file in
 * the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */

/*
 * see header file
 *
 * Notes: (implementation consideration)
 * - when an instance of the interpreter is created, a context object (~300 Bytes)
 *   is created, dynamically allocated (see tflm_c_create() fct).
 *   This allocation is done, through the new C++ operator. To be able to
 *   monitor the usage of the heap, new operator is expected to be
 *   based on the C-malloc/free functions.
 */

// if (=0), resolver is created with all built-in operators
#if !defined(TFLM_RUNTIME_USE_ALL_OPERATORS)
#define TFLM_RUNTIME_USE_ALL_OPERATORS 1
#endif

#if defined(TFLM_RUNTIME_USE_ALL_OPERATORS) && TFLM_RUNTIME_USE_ALL_OPERATORS == 1
#include "all_ops_resolver.h"
#endif

#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro//memory_helpers.h"
#include "tensorflow/lite/micro/memory_planner/greedy_memory_planner.h"


#define TF_MAJOR_VERSION      0
#define TF_MINOR_VERSION      0
#define TF_PATCH_VERSION      0

#define TF_BUILD_VERSION      (0x722913ea)
#define TF_DESC_VERSION       "tflite-micro github"


#undef _IS_AC5_COMPILER
#undef _IS_AC6_COMPILER
#undef _IS_GHS_COMPILER
#undef _IS_HTC_COMPILER
#undef _IS_GCC_COMPILER
#undef _IS_IAR_COMPILER

/* ARM Compiler 5 tool-chain */
#if defined ( __CC_ARM )
// #if ((__ARMCC_VERSION >= 5000000) && (__ARMCC_VERSION < 6000000))
#define _IS_AC5_COMPILER    1

/* ARM Compiler 6 tool-chain */
#elif defined (__ARMCC_VERSION) && (__ARMCC_VERSION >= 6010050)
#define _IS_AC6_COMPILER    1

/* GHS tool-chain */
#elif defined(__ghs__)
#define _IS_GHS_COMPILER    1

/* HIGHTEC tool-chain */
#elif defined(__clang__)
#define _IS_HTC_COMPILER    1

/* GCC tool-chain */
#elif defined ( __GNUC__ )
#define _IS_GCC_COMPILER    1

/* IAR tool-chain */
#elif defined ( __ICCARM__ )
#define _IS_IAR_COMPILER    1

#else
#define _IS_UNDEFINED_COMPILER    1
#endif

#undef _IS_ACx_COMPILER
#if defined(_IS_AC5_COMPILER) && _IS_AC5_COMPILER   \
    || defined(_IS_AC6_COMPILER) && _IS_AC6_COMPILER
#define _IS_ACx_COMPILER    1
#endif

#if defined(_IS_AC6_COMPILER) && _IS_AC6_COMPILER
#define _STAI_COMPILER_ID   4
#elif defined(_IS_GHS_COMPILER) && _IS_GHS_COMPILER
#define _STAI_COMPILER_ID   6
#elif defined(_IS_HTC_COMPILER) && _IS_HTC_COMPILER
#define _STAI_COMPILER_ID   5
#elif defined(_IS_GCC_COMPILER) && _IS_GCC_COMPILER
#define _STAI_COMPILER_ID   1
#elif defined(_IS_IAR_COMPILER) && _IS_IAR_COMPILER
#define _STAI_COMPILER_ID   2
#elif defined(_IS_AC5_COMPILER) && _IS_AC5_COMPILER
#define _STAI_COMPILER_ID   3
#else
#define _STAI_COMPILER_ID   0
#endif

#include "tflm_c.h"


// forward declaration
class CTfLiteInterpreterContext;

class CTfLiteProfiler : public tflite::MicroProfilerInterface {
public:
  CTfLiteProfiler(CTfLiteInterpreterContext* interp) : ctx_(interp), options_(nullptr),
    event_starts_(0), event_ends_(0) {}
  ~CTfLiteProfiler() override = default;

  uint32_t BeginEvent(const char* tag) override {
    if (options_)
      return begin_event(tag);
    else
      return 0;
  }

  void EndEvent(uint32_t event_handle) override {
    if (options_)
      end_event(event_handle);
  }

  TfLiteStatus register_cb(struct tflm_c_observer_options* options) {
    /* a get_time cb is mandatory to use the profiler */
    if (options_ || options == nullptr || options->get_time == nullptr)
      return kTfLiteError;
    options_ = options;
    start();
    return kTfLiteOk;
  }

  TfLiteStatus unregister_cb(struct tflm_c_observer_options* options) {
    if (options_ == options) {
      options_ = nullptr;
      return kTfLiteOk;
    }
    return kTfLiteError;
  }

  TfLiteStatus start(void) {
    cb_dur_ = 0;
    node_dur_ = 0;
    node_idx_ = -1;
    node_ts_begin_ = 0;
    event_starts_ = 0;
    event_ends_ = 0;
    return kTfLiteOk;
  }

  TfLiteStatus reset(void) {
    node_idx_ = -1;
    return kTfLiteOk;
  }

  TfLiteStatus info(struct tflm_c_profile_info* p_info) {
    if (!p_info) {
      return kTfLiteError;
    }
    p_info->cb_dur = cb_dur_;
    p_info->node_dur = node_dur_;
    p_info->n_events = event_starts_;
    return kTfLiteOk;
  }

protected:
  uint32_t begin_event(const char* tag);

  void end_event(uint32_t event_handle);

  int event_starts() { return event_starts_; }
  int event_ends() { return event_ends_; }

private:
  CTfLiteInterpreterContext* ctx_;
  struct tflm_c_observer_options* options_;
  int event_starts_;
  int event_ends_;
  uint64_t node_ts_begin_;
  const char* node_tag_;
  int32_t node_idx_;
  uint64_t cb_dur_;
  uint64_t node_dur_;

  TF_LITE_REMOVE_VIRTUAL_DELETE
};

class CTfLiteInterpreterContext {
public:
  CTfLiteInterpreterContext(const tflite::Model* model,
      const tflite::MicroOpResolver& op_resolver,
      uint8_t* tensor_arena, size_t tensor_arena_size): model_(model), profiler(this),
          interpreter(model, op_resolver, tensor_arena, tensor_arena_size,
              nullptr, (tflite::MicroProfilerInterface *)&profiler),
          n_invoks(0) {}

public:
  const tflite::Model *model_;
  CTfLiteProfiler profiler;
  tflite::MicroInterpreter interpreter;
  int n_invoks;

public:
  static TfLiteStatus input(const uint32_t hdl, int32_t index, struct tflm_c_tensor_info* t_info) {
    CTfLiteInterpreterContext *ctx = reinterpret_cast<CTfLiteInterpreterContext *>(hdl);
    const TfLiteTensor* tens = ctx->interpreter.input(index);
    return ctx->tflitetensor_to(tens, t_info, -1);
  }

  static TfLiteStatus output(const uint32_t hdl, int32_t index, struct tflm_c_tensor_info* t_info) {
    CTfLiteInterpreterContext *ctx = reinterpret_cast<CTfLiteInterpreterContext *>(hdl);
    const TfLiteTensor* tens = ctx->interpreter.output(index);
    return ctx->tflitetensor_to(tens, t_info, -1);
  }

  static TfLiteStatus invoke(const uint32_t hdl) {
    CTfLiteInterpreterContext *ctx = reinterpret_cast<CTfLiteInterpreterContext *>(hdl);
    ctx->profiler.reset();
    ctx->n_invoks++;
    return ctx->interpreter.Invoke();
  }

  static TfLiteStatus reset_all_variables(const uint32_t hdl) {
    CTfLiteInterpreterContext *ctx = reinterpret_cast<CTfLiteInterpreterContext *>(hdl);
    ctx->profiler.reset();
    return ctx->interpreter.Reset();
  }

  static TfLiteStatus observer_register(const uint32_t hdl, struct tflm_c_observer_options* options)
  {
    CTfLiteInterpreterContext *ctx = reinterpret_cast<CTfLiteInterpreterContext *>(hdl);
    return ctx->profiler.register_cb(options);
  }

  static TfLiteStatus observer_unregister(const uint32_t hdl, struct tflm_c_observer_options* options)
  {
    CTfLiteInterpreterContext *ctx = reinterpret_cast<CTfLiteInterpreterContext *>(hdl);
    return ctx->profiler.unregister_cb(options);
  }

  static TfLiteStatus observer_start(const uint32_t hdl)
  {
    CTfLiteInterpreterContext *ctx = reinterpret_cast<CTfLiteInterpreterContext *>(hdl);
    ctx->n_invoks = 0;
    return ctx->profiler.start();
 }

  static TfLiteStatus observer_info(const uint32_t hdl, struct tflm_c_profile_info* p_info)
  {
    CTfLiteInterpreterContext *ctx = reinterpret_cast<CTfLiteInterpreterContext *>(hdl);
    TfLiteStatus res = ctx->profiler.info(p_info);
    if (res == kTfLiteOk)
      p_info->n_invoks = ctx->n_invoks;
    return res;
  }

  uint32_t get_handle() {
    return (uint32_t)this;
  }

private:
  TfLiteStatus tflitetensor_to(const TfLiteTensor* tfls, struct tflm_c_tensor_info* t_info, int32_t idx=-1);

protected:
 friend class CTfLiteProfiler;
};


TfLiteStatus CTfLiteInterpreterContext::tflitetensor_to(const TfLiteTensor* tft,
    struct tflm_c_tensor_info* ti, int32_t idx)
{
  if (!tft || !ti)
    return kTfLiteError;

  ti->type = tft->type;
  ti->idx = idx;
  ti->bytes = tft->bytes;

  if (tft->dims->size > TFLM_C_MAX_DIM)
    return kTfLiteError;

  memset(&ti->shape.data, 0, sizeof(uint32_t) * TFLM_C_MAX_DIM);
  ti->shape.size = tft->dims->size;
  for (size_t i=0; i<ti->shape.size; i++) {
    ti->shape.data[i] = tft->dims->data[i];
  }

  ti->data = (void *)tft->data.uint8;

  if (tft->quantization.type == kTfLiteAffineQuantization) {
    TfLiteAffineQuantization *quant = (TfLiteAffineQuantization *)tft->quantization.params;
    ti->scale = *(quant->scale->data);
    ti->zero_point = *(quant->zero_point->data);
  } else {
    ti->scale = 0.0f; // nullptr;
    ti->zero_point = 0; // nullptr;
  }

  return kTfLiteOk;
}

// static tflite::MicroErrorReporter micro_error_reporter;


#ifdef __cplusplus
extern "C" {
#endif

TfLiteStatus tflm_c_create(const uint8_t *model_data,
    uint8_t *tensor_arena,
    const uint32_t tensor_arena_size,
    uint32_t *hdl)
{
  TfLiteStatus status;

  if (!hdl || !tensor_arena_size || !tensor_arena || !model_data)
    return kTfLiteError;

  *hdl = 0;

  const tflite::Model* model = ::tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    printf("Invalid expected TFLite model version %d instead %d\r\n",
        (int)model->version(), (int)TFLITE_SCHEMA_VERSION);
    return kTfLiteError;
  }


#if defined(TFLM_RUNTIME_USE_ALL_OPERATORS) && TFLM_RUNTIME_USE_ALL_OPERATORS == 1
  static tflite::AllOpsResolver _resolver;
#else
  static tflite::MicroMutableOpResolver<23> _resolver;
  _resolver.AddAveragePool2D(tflite::Register_AVERAGE_POOL_2D_INT8());
  _resolver.AddConv2D(tflite::Register_CONV_2D_INT8());
  _resolver.AddDepthwiseConv2D(tflite::Register_DEPTHWISE_CONV_2D_INT8());
  _resolver.AddFullyConnected(tflite::Register_FULLY_CONNECTED_INT8());
  _resolver.AddReshape();
  _resolver.AddMean();
  _resolver.AddPad();
  _resolver.AddLogistic();
  _resolver.AddMaxPool2D(tflite::Register_MAX_POOL_2D_INT8()); // Register_MAX_POOL_2D_INT8
  _resolver.AddResizeNearestNeighbor();
  _resolver.AddResizeBilinear();
  _resolver.AddTranspose();
  _resolver.AddMul();
  _resolver.AddSub();
  _resolver.AddShape();
  _resolver.AddTransposeConv();
  _resolver.AddPack();
  _resolver.AddAdd();
  _resolver.AddQuantize();
  _resolver.AddDequantize();
  _resolver.AddStridedSlice();
  _resolver.AddConcatenation();
  _resolver.AddSoftmax(tflite::Register_SOFTMAX_INT8());
 #endif

  CTfLiteInterpreterContext *ctx = new CTfLiteInterpreterContext(
      model,
      _resolver,
      tensor_arena,
      tensor_arena_size
  );

  // Allocate the resources
  status = ctx->interpreter.AllocateTensors();
  if (status != kTfLiteOk) {
    printf("AllocateTensors() fails\r\n");
    delete ctx;
    return kTfLiteError;
  }

  // tflite::ErrorReporter* error_reporter = &micro_error_reporter;
  // error_reporter->Report("hello %d\n\t", sizeof(CTfLiteInterpreterContext));
  // error_reporter->Report("hello %d\n\t", sizeof(tflite::MicroProfiler));
  // error_reporter->Report("hello %d\n\t", sizeof(tflite::MicroInterpreter));

  *hdl = ctx->get_handle();

  return kTfLiteOk;
}

TfLiteStatus tflm_c_destroy(uint32_t hdl)
{
  CTfLiteInterpreterContext *ctx = reinterpret_cast<CTfLiteInterpreterContext *>(hdl);
  delete ctx;
  return kTfLiteOk;
}

int32_t tflm_c_inputs_size(const uint32_t hdl)
{
  CTfLiteInterpreterContext *ctx = reinterpret_cast<CTfLiteInterpreterContext *>(hdl);
  return ctx->interpreter.inputs_size();
}

int32_t tflm_c_outputs_size(const uint32_t hdl)
{
  CTfLiteInterpreterContext *ctx = reinterpret_cast<CTfLiteInterpreterContext *>(hdl);
  return ctx->interpreter.outputs_size();
}

TfLiteStatus tflm_c_input(const uint32_t hdl, int32_t index, struct tflm_c_tensor_info *t_info)
{
  return CTfLiteInterpreterContext::input(hdl, index, t_info);
}

TfLiteStatus tflm_c_output(const uint32_t hdl, int32_t index, struct tflm_c_tensor_info *t_info)
{
  return CTfLiteInterpreterContext::output(hdl, index, t_info);
}

TfLiteStatus tflm_c_invoke(const uint32_t hdl)
{
  return CTfLiteInterpreterContext::invoke(hdl);
}

TfLiteStatus tflm_c_reset_all_variables(const uint32_t hdl)
{
  return CTfLiteInterpreterContext::reset_all_variables(hdl);
}

int32_t tflm_c_operators_size(const uint32_t hdl)
{
  CTfLiteInterpreterContext *ctx = reinterpret_cast<CTfLiteInterpreterContext *>(hdl);
  return ctx->model_->subgraphs()->Get(0)->operators()->size();
}

int32_t tflm_c_tensors_size(const uint32_t hdl)
{
  CTfLiteInterpreterContext *ctx = reinterpret_cast<CTfLiteInterpreterContext *>(hdl);
  return ctx->model_->subgraphs()->Get(0)->tensors()->size();
}

int32_t tflm_c_operator_codes_size(const uint32_t hdl)
{
  CTfLiteInterpreterContext *ctx = reinterpret_cast<CTfLiteInterpreterContext *>(hdl);
  return ctx->model_->operator_codes()->size();
}

int32_t tflm_c_arena_used_bytes(const uint32_t hdl)
{
  CTfLiteInterpreterContext *ctx = reinterpret_cast<CTfLiteInterpreterContext *>(hdl);
  return ctx->interpreter.arena_used_bytes();
}

const char* tflm_c_TfLiteTypeGetName(TfLiteType type)
{
  return TfLiteTypeGetName(type);
}

void tflm_c_rt_version(struct tflm_c_version *version)
{
  if (version) {
    version->major = TF_MAJOR_VERSION;
    version->minor = TF_MINOR_VERSION;
    version->patch = TF_PATCH_VERSION;
    version->schema = TFLITE_SCHEMA_VERSION;
    version->build = TF_BUILD_VERSION;
    version->desc = TF_DESC_VERSION;
    version->tools_id = _STAI_COMPILER_ID;
  }
}

TfLiteStatus tflm_c_observer_register(const uint32_t hdl, struct tflm_c_observer_options* options)
{
  return CTfLiteInterpreterContext::observer_register(hdl, options);
}

TfLiteStatus tflm_c_observer_unregister(const uint32_t hdl, struct tflm_c_observer_options* options)
{
  return CTfLiteInterpreterContext::observer_unregister(hdl, options);
}

TfLiteStatus tflm_c_observer_start(const uint32_t hdl)
{
  return CTfLiteInterpreterContext::observer_start(hdl);
}

TfLiteStatus tflm_c_observer_info(const uint32_t hdl, struct tflm_c_profile_info* p_info)
{
  return CTfLiteInterpreterContext::observer_info(hdl, p_info);
}

uint32_t CTfLiteProfiler::begin_event(const char* tag)
{
  volatile uint64_t ts = options_->get_time(0);
  event_starts_++;
  node_tag_ = tag;
  node_idx_++;
  node_ts_begin_ = options_->get_time(0);  // ts before node execution
  cb_dur_ += (node_ts_begin_ - ts);
  return 0;
}

void CTfLiteProfiler::end_event(uint32_t event_handle)
{
  volatile uint64_t ts = options_->get_time(0);
  event_ends_++;
  if (options_->notify) {
    struct tflm_c_node node;
    node.node_info.name = node_tag_;
    node.node_info.idx = node_idx_;
    node.node_info.dur = ts - node_ts_begin_;
    node.node_info.n_outputs = 0;
    node.output = nullptr;
    node_dur_ += node.node_info.dur;
    options_->notify(options_->cookie, options_->flags, &node);
  }
  cb_dur_ += (options_->get_time(1) - ts);
}

#ifdef __cplusplus
}
#endif
