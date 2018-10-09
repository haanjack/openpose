#ifndef OPENPOSE_NET_NET_TENSORRT_BASE_HPP
#define OPENPOSE_NET_NET_TENSORRT_BASE_HPP

#include <openpose/core/common.hpp>
#include <openpose/pose/enumClasses.hpp>

#include <cuda_runtime_api.h>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include <openpose/utilities/tensorRT.hpp>

namespace op 
{
    OP_API ICudaEngine *createEngine();
}

#endif // OPENPOSE_NET_NET_TENSORRT_BASE_HPP