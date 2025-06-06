/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "all_ops_resolver.h"

#include "tensorflow/lite/micro/kernels/micro_ops.h"

namespace tflite {

AllOpsResolver::AllOpsResolver() {
  AddAbs();
  AddAdd();
  AddAddN();
  AddArgMax();
  AddArgMin();
  AddAssignVariable();
  AddAveragePool2D();
  AddBatchMatMul();
  AddBatchToSpaceNd();
  AddBroadcastArgs();
  AddBroadcastTo();
  AddCallOnce();
  AddCast();
  AddCeil();
  AddCircularBuffer();
  AddConcatenation();
  AddConv2D();
  AddCos();
  AddCumSum();
  AddDelay();
  AddDepthToSpace();
  AddDepthwiseConv2D();
  AddDequantize();
  AddDetectionPostprocess();
  AddDiv();
  AddEmbeddingLookup();
  AddEnergy();
  AddElu();
  AddEqual();
  AddEthosU();
  AddExp();
  AddExpandDims();
  AddFftAutoScale();
  AddFill();
  AddFilterBank();
  AddFilterBankLog();
  AddFilterBankSquareRoot();
  AddFilterBankSpectralSubtraction();
  AddFloor();
  AddFloorDiv();
  AddFloorMod();
  AddFramer();
  AddFullyConnected();
  AddGather();
  AddGatherNd();
  AddGreater();
  AddGreaterEqual();
  AddHardSwish();
  AddIf();
  AddIrfft();
  AddL2Normalization();
  AddL2Pool2D();
  AddLeakyRelu();
  AddLess();
  AddLessEqual();
  AddLog();
  AddLogicalAnd();
  AddLogicalNot();
  AddLogicalOr();
  AddLogistic();
  AddLogSoftmax();
  AddMaximum();
  AddMaxPool2D();
  AddMirrorPad();
  AddMean();
  AddMinimum();
  AddMul();
  AddNeg();
  AddNotEqual();
  AddOverlapAdd();
  AddPack();
  AddPad();
  AddPadV2();
  AddPCAN();
  AddPrelu();
  AddQuantize();
  AddReadVariable();
  AddReduceMax();
  AddRelu();
  AddRelu6();
  AddReshape();
  AddResizeBilinear();
  AddResizeNearestNeighbor();
  AddRfft();
  AddRound();
  AddRsqrt();
  AddSelectV2();
  AddShape();
  AddSin();
  AddSlice();
  AddSoftmax();
  AddSpaceToBatchNd();
  AddSpaceToDepth();
  AddSplit();
  AddSplitV();
  AddSqueeze();
  AddSqrt();
  AddSquare();
  AddSquaredDifference();
  AddStridedSlice();
  AddStacker();
  AddSub();
  AddSum();
  AddSvdf();
  AddTanh();
  AddTransposeConv();
  AddTranspose();
  AddUnpack();
  AddUnidirectionalSequenceLSTM();
  AddVarHandle();
  AddWhile();
  AddWindow();
  AddZerosLike();
}

}  // namespace tflite
