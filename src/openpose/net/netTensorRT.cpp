#include <numeric> // std::accumulate
#ifdef USE_CAFFE
    #include <atomic>
    #include <mutex>
    #include <caffe/net.hpp>
    #include <glog/logging.h> // google::InitGoogleLogging
#endif
#ifdef USE_CUDA
    #include <openpose/gpu/cuda.hpp>
#endif
#ifdef USE_OPENCL
    #include <openpose/gpu/opencl.hcl>
    #include <openpose/gpu/cl2.hpp>
#endif
#include <openpose/utilities/fileSystem.hpp>
#include <openpose/utilities/standard.hpp>
#include <openpose/utilities/tensorRT.hpp>
#include <openpose/net/netTensorRT.hpp>

#include <cuda_runtime_api.h>

#include "NvInfer.h"
#include "NvCaffeParser.h"


using namespace nvinfer1;
using namespace nvcaffeparser1;

//*****


namespace op
{
    std::mutex sMutexNetTensorRT;
    //std::atomic<bool> sGoogleLoggingInitialized{false};
    #ifdef USE_OPENCL
        std::atomic<bool> sOpenCLInitialized{false};
    #endif

    struct NetTensorRT::ImplNetTensorRT
    {
        #ifdef USE_TENSORRT
        // Init with constructor
        const int mGpuId;
        const std::string mCaffeproto;
        const std::string mCaffetraindModel;
        const std::string mLastBlobName;
        std::vector<int> mNetInputSize4D;
        // Init with thread
        //std::unique_ptr<caffe::Net<float>> upCaffeNet;
        //boost::shared_ptr<caffe::Blob<float>> spOutputBlob;
        boost::shared_ptr<float> spOutputBlob;

        ImplNetTensorRT(const std::string &caffeProto,
                        const std::string &caffeTrainedModel, const int gpuId,
                        const bool enableGoogleLogging, const std::string &lastBlobName) : 
                        mGpuId{gpuId},
                        mCaffeProto{caffeProto},
                        mCaffeTrainedModel{caffeTrainedModel},
                        mLastBlobName { lastBlobName }
        {
            const std::string message{".\nPossible causes:\n\t1. Not downloading the OpenPose trained models."
                                      "\n\t2. Not running OpenPose from the same directory where the `model`"
                                      " folder is located.\n\t3. Using paths with spaces."};
            if (!existFile(mCaffeProto))
                error("Prototxt file not found: " + mCaffeProto + message, __LINE__, __FUNCTION__, __FILE__);
            if (!existFile(mCaffeTrainedModel))
                error("Caffe trained model file not found: " + mCaffeTrainedModel + message,
                      __LINE__, __FUNCTION__, __FILE__);
            // Double if condition in order to speed up the program if it is called several times
            if (enableGoogleLogging && !sGoogleLoggingInitialized)
            {
                std::lock_guard<std::mutex> lock{sMutexNetTensorRT};
                if (enableGoogleLogging && !sGoogleLoggingInitialized)
                {
                    google::InitGoogleLogging("OpenPose");
                    sGoogleLoggingInitialized = true;
                }
            }
            createEngine();
        }
        #endif
    };

    #ifdef USE_CAFFE
        inline void reshapeNetTensorRT(caffe::Net<float>* caffeNet, const std::vector<int>& dimensions)
        {
            try
            {
                caffeNet->blobs()[0]->Reshape(dimensions);
                caffeNet->Reshape();
                #ifdef USE_CUDA
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                #endif
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }
    #endif

    NetTensorRT::NetTensorRT(const std::string& caffeProto, const std::string& caffeTrainedModel, const int gpuId,
                       const bool enableGoogleLogging, const std::string& lastBlobName)
        #ifdef USE_TENSORRT
            : upImpl{new ImplNetTensorRT{caffeProto, caffeTrainedModel, gpuId, enableGoogleLogging,
                                      lastBlobName}}
        #endif
    {
        try
        {
            #ifndef USE_TENSORRT
                //UNUSED(netInputSize4D);
                UNUSED(caffeProto);
                UNUSED(caffeTrainedModel);
                UNUSED(gpuId);
                UNUSED(lastBlobName);
                error("OpenPose must be compiled with the `USE_CAFFE` macro definition in order to use this"
                      " functionality.", __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    NetTensorRT::~NetTensorRT()
    {
    }

    void NetTensorRT::initializationOnThread()
    {
        try
        {
            #ifdef USE_TENSORRT
                // Initialize net
                #ifdef USE_CUDA
                    //caffe::Caffe::set_mode(caffe::Caffe::GPU);
                    //caffe::Caffe::SetDevice(upImpl->mGpuId);
                    cudaSetDevice(upImpl->mGpuId);
                    //upImpl->upCaffeNet.reset(new caffe::Net<float>{upImpl->mCaffeProto, caffe::TEST});
                #else
                    caffe::Caffe::set_mode(caffe::Caffe::CPU);
                    #ifdef _WIN32
                        upImpl->upCaffeNet.reset(new caffe::Net<float>{upImpl->mCaffeProto, caffe::TEST,
                                                                        caffe::Caffe::GetCPUDevice()});
                    #else
                        upImpl->upCaffeNet.reset(new caffe::Net<float>{upImpl->mCaffeProto, caffe::TEST});
                    #endif
                #endif

                //upImpl->upCaffeNet->CopyTrainedLayersFrom(upImpl->mCaffeTrainedModel);
                #ifdef USE_CUDA
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                #endif
                // Set spOutputBlob
                //upImpl->spOutputBlob = upImpl->upCaffeNet->blob_by_name(upImpl->mLastBlobName);
                if (upImpl->spOutputBlob == nullptr)
                    error("The output blob is a nullptr. Did you use the same name than the prototxt? (Used: "
                          + upImpl->mLastBlobName + ").", __LINE__, __FUNCTION__, __FILE__);
                #ifdef USE_CUDA
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                #endif
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void NetTensorRT::forwardPass(const Array<float>& inputData) const
    {
        try
        {
            #ifdef USE_TENSORRT
                // Sanity checks
                if (inputData.empty())
                    error("The Array inputData cannot be empty.", __LINE__, __FUNCTION__, __FILE__);
                if (inputData.getNumberDimensions() != 4 || inputData.getSize(1) != 3)
                    error("The Array inputData must have 4 dimensions: [batch size, 3 (RGB), height, width].",
                          __LINE__, __FUNCTION__, __FILE__);
                std::cout << upImpl->mCaffeproto << std::endl;
                // Reshape Caffe net if required
                if (!vectorsAreEqual(upImpl->mNetInputSize4D, inputData.getSize()))
                {
                    upImpl->mNetInputSize4D = inputData.getSize();
                    reshapeNetTensorRT(upImpl->upCaffeNet.get(), inputData.getSize());
                }
                // Copy frame data to GPU memory
                #ifdef USE_CUDA
                    auto* gpuImagePtr = upImpl->upCaffeNet->blobs().at(0)->mutable_gpu_data();
                    cudaMemcpy(gpuImagePtr, inputData.getConstPtr(), inputData.getVolume() * sizeof(float),
                               cudaMemcpyHostToDevice);
                #elif defined USE_OPENCL
                    auto* gpuImagePtr = upImpl->upCaffeNet->blobs().at(0)->mutable_gpu_data();
                    cl::Buffer imageBuffer = cl::Buffer((cl_mem)gpuImagePtr, true);
                    OpenCL::getInstance(upImpl->mGpuId)->getQueue().enqueueWriteBuffer(imageBuffer, true, 0,
                                                                                       inputData.getVolume() * sizeof(float),
                                                                                       inputData.getConstPtr());
                #else
                    auto* cpuImagePtr = upImpl->upCaffeNet->blobs().at(0)->mutable_cpu_data();
                    std::copy(inputData.getConstPtr(), inputData.getConstPtr() + inputData.getVolume(), cpuImagePtr);
                #endif
                // Perform deep network forward pass
                upImpl->upCaffeNet->ForwardFrom(0);
                // Cuda checks
                #ifdef USE_CUDA
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                #endif
            #else
                UNUSED(inputData);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    boost::shared_ptr<float> NetTensorRT::getOutputBlob() const
    {
        try
        {
            #ifdef USE_TENSORRT
                return upImpl->spOutputBlob;
            #else
                return nullptr;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }
}
