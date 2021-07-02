/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TRT_NORMALIZE_PLUGIN_H
#define TRT_NORMALIZE_PLUGIN_H
#include "cudnn.h"
#include "kernel.h"
#include "plugin.h"
#include <cublas_v2.h>
#include <string>
#include <vector>

//接口封装，使其符合TensorRT自定义层的使用规范
namespace nvinfer1
{
namespace plugin
{

class Normalize : public IPluginV2Ext //插件类，具体实现
{
public:
    Normalize(const Weights* weights, int nbWeights, bool acrossSpatial, bool channelShared, float eps);//prase阶段调用构造函数

    Normalize(
        const Weights* weights, int nbWeights, bool acrossSpatial, bool channelShared, float eps, int C, int H, int W);//clone用构造函数

    Normalize(const void* buffer, size_t length);//deserialize时调用

    ~Normalize() override = default;

    int getNbOutputs() const override;//返回输出Tensor个数

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;//返回输出Tensor维度，存在多个output Tensor时怎么处理？

    int initialize() override;//初始化函数

    void terminate() override;//终止

    size_t getWorkspaceSize(int maxBatchSize) const override;//获取算子所需中间显存大小

    int enqueue(
        int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;//运算操作

    size_t getSerializationSize() const override;//获取序列化所需空间大小

    void serialize(void* buffer) const override;//序列化操作

    bool supportsFormat(DataType type, PluginFormat format) const override;//支持的数据格式

    const char* getPluginType() const override;//自定义层名

    const char* getPluginVersion() const override;//自定义层版本

    void destroy() override;//销毁

    IPluginV2Ext* clone() const override;

    void setPluginNamespace(const char* pluginNamespace) override;

    const char* getPluginNamespace() const override;

    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;//获取输出类型

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const override;

    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override; //使用tensorrt提供的其他Context

    void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
        const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) override;//判断输入和输出类型数量是否正确

    void detachFromContext() override; //tensorrt Context 解绑

private:
    Weights copyToDevice(const void* hostData, size_t count);//主机数据拷贝到显存
    void serializeFromDevice(char*& hostBuffer, Weights deviceWeights) const;//从显存中获取序列化数据
    Weights deserializeToDevice(const char*& hostBuffer, size_t count);//将序列化数据写入显存

    cublasHandle_t mCublas;//Cublas handle

    Weights mWeights{};  //私有成员变量，保存算子所需权重、形状信息
    int mNbWeights{};
    bool acrossSpatial{};
    bool channelShared{};
    float eps{};
    int C{};
    int H{};
    int W{};
    std::string mPluginNamespace;
};

class NormalizePluginCreator : public BaseCreator  //插件工厂类，根据需求实例化插件(类似JAVA中的组件工厂？)
{
public:
    NormalizePluginCreator();//创建一个空的mPluginAttributes初始化mFC,功能不太理解

    ~NormalizePluginCreator() override = default;

    const char* getPluginName() const override;//获取自定义组件名

    const char* getPluginVersion() const override;//获取自定义组件版本号

    const PluginFieldCollection* getFieldNames() override;//获取mFC

    IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) override;//创建自定义层，解析name获取自定义构造函数参数，创建插件

    IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;//

private:
    static PluginFieldCollection mFC;
    bool mAcrossSpatial{};
    bool mChannelShared{};
    float mEps{};
    int mNbWeights{};
    static std::vector<PluginField> mPluginAttributes;
};
} // namespace plugin
} // namespace nvinfer1

#endif // TRT_NORMALIZE_PLUGIN_H
