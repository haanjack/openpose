#include <openpose/net/netTensorRTBase.hpp>
#include <openpose/net/netTensorRTPlugin.hpp>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvCaffeParser.h"

#include "boost/shared_ptr.hpp"

using namespace nvinfer1;
using namespace nvcaffeparser1;

namespace op 
{
ICudaEngine *caffeToTRTModel(const std::string &caffeProto, const std::string &caffeTrainedModel,
                             std::vector<std::string> &inputs,
                             std::vector<std::string> &outputs,
                             const int batchSize, const int workspaceSize, bool fp16,
                             Logger logger)
{
    // create API root class - must span the lifetime of the engine usage
    IBuilder *builder = createInferBuilder(logger);

    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition *network = builder->createNetwork();
    ICaffeParser *parser = createCaffeParser();

    // Parse Plugin Layers
    PluginFactory parserPluginFactory;
    parser->setPluginFactoryExt(&parserPluginFactory);

    const IBlobNameToTensor *blobNameToTensor =
        parser->parse(caffeProto.c_str(), // caffe deploy file
                      caffeTrainedModel.c_str(),  // caffe model file
                      *network,                   // network definition that the parser will populate
                      fp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT );
    if (!blobNameToTensor)
        return nullptr;

    // TODO: Input??
    for (int i = 0, n = network->getNbInputs(); i < n; i++)
    {
        Dims3 dims = static_cast<Dims3 &&>(network->getInput(i)->getDimensions());
        inputs.push_back(network->getInput(i)->getName());
        std::cout << "Input \"" << network->getInput(i)->getName() << "\": " << dims.d[0] << "x" << dims.d[1] << "x" << dims.d[2] << std::endl;
    }

    // specify which tensors are outputs
    // the caffe file has no notion of outputs,
    // so we need to manually say which tensors the engine should generate
    for (auto &s : outputs)
    {
        if (blobNameToTensor->find(s.c_str()) == nullptr)
        {
            std::cout << "could not find output blob " << s << std::endl;
            return nullptr;
        }
        network->markOutput(*blobNameToTensor->find(s.c_str()));
    }

    for (int i = 0, n = network->getNbOutputs(); i < n; i++)
    {
        Dims3 dims = static_cast<Dims3 &&>(network->getOutput(i)->getDimensions());
        std::cout << "Output \"" << network->getOutput(i)->getName() << "\": " << dims.d[0] << "x" << dims.d[1] << "x"
                  << dims.d[2] << std::endl;
    }

    // Build the engine
    builder->setMaxBatchSize(batchSize);
    builder->setMaxWorkspaceSize(size_t(workspaceSize) << 20);
    builder->setFp16Mode(fp16);

    // RndInt8Calibrator calibrator(1, gParams.calibrationCache);
    // if (gParams.int8)
    // {
    //     builder->setInt8Mode(true);
    //     builder->setInt8Calibrator(&calibrator);
    // }

    // if (gParams.useDLA > 0)
    // {
    //     builder->setDefaultDeviceType(static_cast<DeviceType>(gParams.useDLA));
    //     if (gParams.allowGPUFallback)
    //         builder->allowGPUFallback(gParams.allowGPUFallback);
    // }

    ICudaEngine *engine = builder->buildCudaEngine(*network);
    if (engine == nullptr)
        std::cout << "could not build engine" << std::endl;

    // we don't need the network any more, and we can destroy the parser
    parserPluginFactory.destroyPlugin();
    parser->destroy();
    network->destroy();
    builder->destroy();

    return engine;
}

ICudaEngine *createEngine(const std::string &caffeProto, const std::string &caffeTrainedModel,
                          std::vector<std::string> &inputs, std::vector<std::string> &outputs,
                          const int batchSize, const int workspaceSize, bool fp16,
                          const Logger logger)
{
    ICudaEngine *engine;
    if ((!caffeProto.empty()))
    {
        // Create engine (caffe)
        engine = caffeToTRTModel(caffeProto, caffeTrainedModel,
                                 inputs,
                                 outputs,
                                 batchSize,
                                 workspaceSize,
                                 fp16,
                                 logger); // load prototxt & caffemodel files
        if (!engine)
        {
            std::cerr << "Engine could not be created" << std::endl;
            return nullptr;
        }

        // write plan file if it is specified
        // if (!gParams.engine.empty())
        // {
        //     std::ofstream p(gParams.engine);
        //     if (!p)
        //     {
        //         std::cerr << "could not open plan output file" << std::endl;
        //         return nullptr;
        //     }
        //     IHostMemory *ptr = engine->serialize();
        //     assert(ptr);
        //     p.write(reinterpret_cast<const char *>(ptr->data()), ptr->size());
        //     ptr->destroy();
        // }
        return engine;
    }

    // load directlry from serialized engine file if deploy not specified
    // if (!gParams.engine.empty())
    // {
    //     char *trtModelStream{nullptr};
    //     size_t size{0};
    //     std::ifstream file(gParams.engine, std::ios::binary);
    //     if (file.good())
    //     {
    //         file.seekg(0, file.end);
    //         size = file.tellg();
    //         file.seekg(0, file.beg);
    //         trtModelStream = new char[size];
    //         assert(trtModelStream);
    //         file.read(trtModelStream, size);
    //         file.close();
    //     }

    //     IRuntime *infer = createInferRuntime(gLogger);
    //     PluginFactory pluginFactory;
    //     engine = infer->deserializeCudaEngine(trtModelStream, size, &pluginFactory);
    //     //pluginFactory.destroyPlugin();
    //     if (trtModelStream)
    //         delete[] trtModelStream;

    //     gParams.inputs.empty() ? gInputs.push_back("image") : gInputs.push_back(gParams.inputs.c_str());
    //     return engine;
    // }

    // complain about empty deploy file
    std::cerr << "Deploy file not specified" << std::endl;
    return nullptr;
    }

    float* createTrtMemory(const ICudaEngine* engine, const int batchSize, const std::string& name)
    {
        float *blobPtr;

        Dims3 dim = static_cast<Dims3&&>(engine->getBindingDimensions((int)engine->getBindingIndex(name.c_str())));
        size_t elemCount = dim.d[0] * dim.d[1] * dim.d[2] * batchSize;
        cudaMalloc((void**)&blobPtr, elemCount * sizeof(float));
        
        return blobPtr;
    }

    boost::shared_ptr<float> createOutputBlob(const ICudaEngine* engine, const int batchSize, const std::string& name)
    {
        auto _cudaMalloc = [](size_t size) { void *ptr; cudaMalloc((void**)&ptr, size); return ptr; };
        auto deleter = [](void *ptr) { cudaFree(ptr); };

        Dims3 dim = static_cast<Dims3&&>(engine->getBindingDimensions((int)engine->getBindingIndex(name.c_str())));
        size_t elemCount = dim.d[0] * dim.d[1] * dim.d[2] * batchSize;
        boost::shared_ptr<float> blobPtr((float*)_cudaMalloc(elemCount * sizeof(float)), deleter);
        
        return blobPtr;
    }
}