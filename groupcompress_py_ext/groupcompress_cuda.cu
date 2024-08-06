#include "groupcompress_cpu.hpp"
#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT_CUDA(x) \
    CHECK_CUDA(x);          \
    CHECK_CONTIGUOUS(x);

template <size_t numBits, size_t numCombos>
__global__ void count_bit_patterns_cuda_kernel(
    const uint8_t *inputs,
    const uint64_t *indices,
    uint64_t *outputs,
    size_t inputSize,
    size_t numInputs,
    size_t numIndices)
{
    uint64_t permutation[numBits];
    uint64_t output[numCombos];

    const size_t permIndex = threadIdx.x + blockIdx.x * blockDim.x;
    const bool mask = permIndex < numIndices;
    for (size_t i = 0; i < numBits; i++) {
        permutation[i] = mask ? indices[i + permIndex * numBits] : 0;
    }

    for (size_t i = 0; i < numCombos; i++) {
        output[i] = 0;
    }

    for (size_t i = 0; i < numInputs; i++) {
        const uint8_t *input = &inputs[i * inputSize];
        uint8_t value = 0;
        for (size_t j = 0; j < numBits; j++) {
            value |= input[permutation[j]] << j;
        }
        output[value]++;
    }

    uint64_t *subOutput = &outputs[numCombos * permIndex];
    for (size_t i = 0; i < numCombos; i++) {
        if (mask) {
            subOutput[i] = output[i];
        }
    }
}

void count_bit_patterns_cuda(
    const torch::Tensor inputs,
    const torch::Tensor indices,
    torch::Tensor output)
{
    CHECK_INPUT_CUDA(inputs);
    CHECK_INPUT_CUDA(indices);
    CHECK_INPUT_CUDA(output);
    CHECK_BYTE(inputs);
    CHECK_LONG(indices);
    CHECK_LONG(output);

    size_t numBits = indices.size(1);
    assert(output.size(0) == indices.size(0));
    assert(output.size(1) == (1 << numBits));
    assert(numBits != 0);
    AT_ASSERTM(numBits > 0 && numBits <= 8, "kernel does not support this number of bits");

    int blockSize = min(1024, 16384 >> numBits);
    int numBlocks = indices.size(0) / blockSize;
    if (indices.size(0) % blockSize) {
        numBlocks++;
    }

    const dim3 threads(blockSize, 1, 1);
    const dim3 blocks(numBlocks, 1, 1);

#define DISPATCH(x, y)                                         \
    count_bit_patterns_cuda_kernel<x, y><<<blocks, threads>>>( \
        (const uint8_t *)inputs.data_ptr(),                    \
        (const uint64_t *)indices.data_ptr(),                  \
        (uint64_t *)output.data_ptr(),                         \
        inputs.size(1),                                        \
        inputs.size(0),                                        \
        indices.size(0));

    switch (numBits) {
    case 1:
        DISPATCH(1, 2);
        break;
    case 2:
        DISPATCH(2, 4);
        break;
    case 3:
        DISPATCH(3, 8);
        break;
    case 4:
        DISPATCH(4, 16);
        break;
    case 5:
        DISPATCH(5, 32);
        break;
    case 6:
        DISPATCH(6, 64);
        break;
    case 7:
        DISPATCH(7, 128);
        break;
    case 8:
        DISPATCH(8, 256);
        break;
    default:
        break;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("count_bit_patterns_cpu", &count_bit_patterns_cpu, "Count bit patterns extracted from inputs");
    m.def("count_bit_patterns_cuda", &count_bit_patterns_cuda, "Count bit patterns extracted from inputs");
}
