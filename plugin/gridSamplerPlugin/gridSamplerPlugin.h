/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef TRT_GRID_SAMPLER_PLUGIN_H
#define TRT_GRID_SAMPLER_PLUGIN_H

#include <cuda_runtime.h>

#include <string>
#include <vector>

#include <NvInfer.h>

namespace nvinfer1::plugin
{

// The same enumeration as the pytorch functions, the arguments for pytorch are converted
// to enumeration upon export, therefore we must also use this when parsing the parameters,
// rather than string arguments.
namespace GridSampler
{
enum class Interpolation
{
    Bilinear,
    Nearest
};
enum class Padding
{
    Zeros,
    Border,
    Reflection
};
} // namespace GridSampler

using GridSampler::Interpolation;
using GridSampler::Padding;

class GridSamplerPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    GridSamplerPlugin(const nvinfer1::PluginFieldCollection& fc);

    GridSamplerPlugin(const void* data, size_t length);

    ~GridSamplerPlugin() override = default;

    int getNbOutputs() const override
    {
        return 1;
    }

    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const override;

    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) override;

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) override;

    size_t getSerializationSize() const override;

    void serialize(void* buffer) const override;

    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override;

    const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    void destroy() override;

    IPluginV2DynamicExt* clone() const override;

    void setPluginNamespace(const char* pluginNamespace) override;

    const char* getPluginNamespace() const override;

    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator) override;

    void detachFromContext() override;

private:
    bool mAlignCorners;
    GridSampler::Interpolation mInterpolationMode;
    GridSampler::Padding mPaddingMode;
    const char* mPluginNamespace;
};

class GridSamplerPluginCreator : public nvinfer1::IPluginCreator
{
public:
    GridSamplerPluginCreator();

    ~GridSamplerPluginCreator() override = default;

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const nvinfer1::PluginFieldCollection* getFieldNames() override;

    nvinfer1::IPluginV2DynamicExt* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) override;

    nvinfer1::IPluginV2DynamicExt* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) override;

    void setPluginNamespace(const char* libNamespace) override
    {
        mNamespace = libNamespace;
    }

    const char* getPluginNamespace() const override
    {
        return mNamespace.c_str();
    }

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace nvinfer1::plugin

#endif // TRT_GRID_SAMPLER_PLUGIN_H
