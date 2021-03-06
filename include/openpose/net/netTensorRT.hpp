#ifndef OPENPOSE_NET_NET_TENSORRT_HPP
#define OPENPOSE_NET_NET_TENSORRT_HPP

#include <openpose/core/common.hpp>
#include <openpose/net/net.hpp>
#include <openpose/net/netTensorRTBase.hpp>

namespace op
{
    class OP_API NetTensorRT : public Net
    {
    public:
        NetTensorRT(const std::string& caffeProto, 
                const std::string& caffeTrainedModel, 
                const int gpuId = 0,
                const bool enableGoogleLogging = true, 
                const std::string& inputBlobName = "image",
                const std::string& lastBlobName = "net_output");

        virtual ~NetTensorRT();

        void initializationOnThread();

        void forwardPass(const Array<float>& inputNetData) const;

        boost::shared_ptr<float> getOutputBlob() const;

    private:
        // PIMPL idiom
        // http://www.cppsamples.com/common-tasks/pimpl.html
        struct ImplNetTensorRT;
        std::unique_ptr<ImplNetTensorRT> upImpl;

        // PIMP requires DELETE_COPY & destructor, or extra code
        // http://oliora.github.io/2015/12/29/pimpl-and-rule-of-zero.html
        DELETE_COPY(NetTensorRT);
    };
}

#endif // OPENPOSE_NET_NET_TENSORRT_HPP
