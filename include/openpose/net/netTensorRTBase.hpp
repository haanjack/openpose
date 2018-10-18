#ifndef OPENPOSE_NET_NET_TENSORRT_BASE_HPP
#define OPENPOSE_NET_NET_TENSORRT_BASE_HPP

#include <openpose/core/common.hpp>
#include <openpose/pose/enumClasses.hpp>

#include <cuda_runtime_api.h>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include <openpose/utilities/tensorRT.hpp>

#include "boost/shared_ptr.hpp"

namespace op 
{
    OP_API ICudaEngine *createEngine(const std::string &caffeProto, const std::string &caffeTrainedModel,
                          std::vector<std::string> &inputs, std::vector<std::string> &outputs,
                          const int batchSize, const int workspaceSize, bool fp16,
                          const Logger logger);
    
    OP_API float* createTrtMemory(const ICudaEngine* engine, const int batchSize, const std::string& name);
    OP_API boost::shared_ptr<float> createOutputBlob(const ICudaEngine* engine, const int batchSize, const std::string& name);
}

#endif // OPENPOSE_NET_NET_TENSORRT_BASE_HPP