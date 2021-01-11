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

#include "gridSamplerPlugin.h"
#include "serialize.hpp"
#include <cassert>

using nvinfer1::plugin::GridSamplerPlugin;
using nvinfer1::plugin::GridSamplerPluginCreator;

namespace
{
const char* GRID_SAMPLER_PLUGIN_VERSION{"1"};
const char* GRID_SAMPLER_PLUGIN_NAME{"grid_sampler"};
} // namespace

nvinfer1::PluginFieldCollection GridSamplerPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> GridSamplerPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(GridSamplerPluginCreator);

GridSamplerPlugin::GridSamplerPlugin(const PluginFieldCollection& fc)
    : mAlignCorners{false}
    , mInterpolationMode{Interpolation::Bilinear}
    , mPaddingMode{Padding::Border}
{
    for (int i = 0; i < fc.nbFields; ++i)
    {
        std::string attrName(fc.fields[i].name);
        if (attrName.compare("align_corners") == 0)
        {
            assert(fc.fields[i].type == nvinfer1::PluginFieldType::kINT32);
            mAlignCorners = *(static_cast<const bool*>(fc.fields[i].data));
        }
        else if (attrName.compare("interpolation_mode") == 0)
        {
            assert(fc.fields[i].type == nvinfer1::PluginFieldType::kINT32);
            mInterpolationMode = *(static_cast<const GridSampler::Interpolation*>(fc.fields[i].data));
        }
        else if (attrName.compare("padding_mode") == 0)
        {
            assert(fc.fields[i].type == nvinfer1::PluginFieldType::kINT32);
            mPaddingMode = *(static_cast<const GridSampler::Padding*>(fc.fields[i].data));
        }
    }
}

GridSamplerPlugin::GridSamplerPlugin(const void* data, size_t length)
{
    deserialize_value(&data, &length, &mAlignCorners);
    deserialize_value(&data, &length, &mInterpolationMode);
    deserialize_value(&data, &length, &mPaddingMode);
}

void GridSamplerPlugin::serialize(void* buffer) const
{
    serialize_value(&buffer, mAlignCorners);
    serialize_value(&buffer, static_cast<int>(mInterpolationMode));
    serialize_value(&buffer, static_cast<int>(mPaddingMode));
}

size_t GridSamplerPlugin::getSerializationSize() const
{
    size_t serializationSize = 0;

    serializationSize += sizeof(mAlignCorners);
    serializationSize += sizeof(static_cast<int>(mInterpolationMode));
    serializationSize += sizeof(static_cast<int>(mPaddingMode));

    return serializationSize;
}

int GridSamplerPlugin::initialize()
{
    return 0;
}

void GridSamplerPlugin::terminate() {}

size_t GridSamplerPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const
{
    // No additional workspace required, ops done inplace.
    return 0;
}

nvinfer1::DimsExprs GridSamplerPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder)
{
    // Only one input and one output which should match dimensions
    assert(outputIndex == 0 && nbInputs == 2);
    return inputs[0];
}

void GridSamplerPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* GridSamplerPlugin::getPluginNamespace() const
{
    return mPluginNamespace;
}

// Return the DataType of the plugin output at the requested index
nvinfer1::DataType GridSamplerPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    // Two inputs and one output, only kFLOAT and kHALF Supported
    assert(index == 0 && nbInputs == 2);
    for (int i = 0; i < nbInputs; ++i)
    {
        assert(inputTypes[i] == nvinfer1::DataType::kFLOAT || inputTypes[i] == nvinfer1::DataType::kHALF);
    }
    return inputTypes[index];
}

bool GridSamplerPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    // Two inputs and one output, only kFLOAT and kHALF Supported
    assert(nbOutputs == 1 && nbInputs == 2);
    bool condition = 1;
    // Should be a bog standard tensor however format doesn't really matter I guess?
    condition &= inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
    // Only kFLOAT and kHALF supported
    condition &= (inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF);
    // Input and output has same type except if the end is dynamic
    condition &= (inOut[pos].type == inOut[nbInputs].type || (int32_t) inOut[nbInputs].type == -1);
    condition &= (inOut[0].dims.d[2] == inOut[1].dims.d[1]);
    condition &= (inOut[0].dims.d[3] == inOut[1].dims.d[2]);
    condition &= (inOut[1].dims.d[3] == 2);
    if (pos == 2)
    {
        condition &= inOut[0].type == inOut[2].type && inOut[0].type == inOut[1].type;
    }
    return condition;
}

void GridSamplerPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs)
{
    assert(in && nbInputs == 2);
    assert(out && nbOutputs == 1);
    assert(in[0].desc.type == in[1].desc.type && in[0].desc.type == out[0].desc.type);

    assert(in[0].desc.dims.nbDims == in[1].desc.dims.nbDims);

    // Input1: NCHW, Input2: NHW2
    assert(in[0].desc.dims.d[0] == in[1].desc.dims.d[0]);
    assert(in[0].desc.dims.d[2] == in[1].desc.dims.d[1]);
    assert(in[0].desc.dims.d[3] == in[1].desc.dims.d[2]);
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void GridSamplerPlugin::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void GridSamplerPlugin::detachFromContext() {}

const char* GridSamplerPlugin::getPluginType() const
{
    return GRID_SAMPLER_PLUGIN_NAME;
}

const char* GridSamplerPlugin::getPluginVersion() const
{
    return GRID_SAMPLER_PLUGIN_VERSION;
}

void GridSamplerPlugin::destroy()
{
    delete this;
}

// Clone the plugin
nvinfer1::IPluginV2DynamicExt* GridSamplerPlugin::clone() const
{
    GridSamplerPlugin* p = new GridSamplerPlugin(*this);
    p->setPluginNamespace(mPluginNamespace);
    return p;
}

GridSamplerPluginCreator::GridSamplerPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(
        nvinfer1::PluginField("align_corners", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(
        nvinfer1::PluginField("interpolation_mode", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(
        nvinfer1::PluginField("padding_mode", nullptr, nvinfer1::PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* GridSamplerPluginCreator::getPluginName() const
{
    return GRID_SAMPLER_PLUGIN_NAME;
}

const char* GridSamplerPluginCreator::getPluginVersion() const
{
    return GRID_SAMPLER_PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* GridSamplerPluginCreator::getFieldNames()
{
    return &mFC;
}

nvinfer1::IPluginV2DynamicExt* GridSamplerPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc)
{
    GridSamplerPlugin* obj = new GridSamplerPlugin(*fc);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

nvinfer1::IPluginV2DynamicExt* GridSamplerPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength)
{
    GridSamplerPlugin* obj = new GridSamplerPlugin(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}
