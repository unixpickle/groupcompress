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

template <size_t numBits, size_t numCombos>
__global__ void count_bit_patterns_cuda_kernel_prefetch(
    const uint8_t *inputs,
    const uint64_t *indices,
    uint64_t *outputs,
    size_t inputSize,
    size_t numInputs,
    size_t numIndices)
{
    __shared__ uint32_t prefetched[1024];
    uint64_t permutation[numBits];
    uint64_t output[numCombos];

    uint8_t *prefetchedBytes = (uint8_t *)prefetched;

    const size_t permIndex = threadIdx.x + blockIdx.x * blockDim.x;
    const bool mask = permIndex < numIndices;
    for (size_t i = 0; i < numBits; i++) {
        permutation[i] = mask ? indices[i + permIndex * numBits] : 0;
    }

    for (size_t i = 0; i < numCombos; i++) {
        output[i] = 0;
    }

    for (size_t i = 0; i < numInputs; i++) {
        const uint32_t *input = (const uint32_t *)&inputs[i * inputSize];
        for (size_t j = 0; j * 4 < inputSize; j += blockDim.x) {
            const size_t inputOffset = j + threadIdx.x;
            if (inputOffset * 4 < inputSize) {
                prefetched[inputOffset] = input[inputOffset];
            }
        }
        __syncthreads();

        uint8_t value = 0;
        for (size_t j = 0; j < numBits; j++) {
            value |= prefetchedBytes[permutation[j]] << j;
        }
        output[value]++;
        __syncthreads();
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

#define DISPATCH(x, y)                                                                             \
    if (inputs.size(1) <= 4096 && inputs.size(1) % 4 == 0 && ((long)inputs.data_ptr()) % 4 == 0) { \
        count_bit_patterns_cuda_kernel_prefetch<x, y><<<blocks, threads>>>(                        \
            (const uint8_t *)inputs.data_ptr(),                                                    \
            (const uint64_t *)indices.data_ptr(),                                                  \
            (uint64_t *)output.data_ptr(),                                                         \
            inputs.size(1),                                                                        \
            inputs.size(0),                                                                        \
            indices.size(0));                                                                      \
    } else {                                                                                       \
        count_bit_patterns_cuda_kernel<x, y><<<blocks, threads>>>(                                 \
            (const uint8_t *)inputs.data_ptr(),                                                    \
            (const uint64_t *)indices.data_ptr(),                                                  \
            (uint64_t *)output.data_ptr(),                                                         \
            inputs.size(1),                                                                        \
            inputs.size(0),                                                                        \
            indices.size(0));                                                                      \
    }

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

__device__ float permutation_entropy_cuda(
    const uint64_t *counts,
    const uint64_t *perm,
    size_t numBits)
{
    uint64_t bitCount[8];
    for (size_t i = 0; i < numBits; i++) {
        bitCount[i] = 0;
    }
    uint64_t totalCount = 0;
    for (size_t i = 0; i < (1 << numBits); i++) {
        uint64_t value = perm[i];
        uint64_t count = counts[i];
        for (size_t j = 0; j < numBits; j++) {
            if (value & (1 << j)) {
                bitCount[j] += count;
            }
        }
        totalCount += count;
    }
    float result = 0.0;
    for (size_t i = 0; i < numBits; i++) {
        uint64_t count = bitCount[i];
        if (count == 0 || count == totalCount) {
            continue;
        }
        float prob = ((float)count) / ((float)totalCount);
        if (prob > 0) {
            result -= prob * log(prob);
        }
        if (1 - prob > 0) {
            result -= (1 - prob) * log(1 - prob);
        }
    }
    return result;
}

__global__ void greedy_permutation_search_cuda_kernel(
    const uint64_t *counts,
    uint64_t *permOut,
    uint64_t *inverseOut,
    size_t batchSize,
    size_t numBits)
{
    const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= batchSize) {
        return;
    }

    const size_t offset = idx << numBits;
    const uint64_t *subCounts = &counts[offset];
    uint64_t *inverse = &inverseOut[offset];
    uint64_t *perm = &permOut[offset];

    for (size_t j = 0; j < (1 << numBits); j++) {
        perm[j] = j;
        inverse[j] = j;
    }

    float entropy = permutation_entropy_cuda(subCounts, perm, numBits);

    // Greedily swap each example for each bit.
    for (size_t bit = 0; bit < numBits; bit++) {
        for (uint64_t i = 0; i < (1 << numBits); i++) {
            uint64_t pattern = perm[i];
            uint64_t other = pattern ^ (1 << bit);
            uint64_t otherIdx = inverse[other];

            perm[i] = other;
            perm[otherIdx] = pattern;
            float newEntropy = permutation_entropy_cuda(subCounts, perm, numBits);

            if (newEntropy > entropy) {
                perm[i] = pattern;
                perm[otherIdx] = other;
            } else {
                inverse[pattern] = otherIdx;
                inverse[other] = i;
                entropy = newEntropy;
            }
        }
    }
}

void greedy_permutation_search_cuda(
    const torch::Tensor counts,
    torch::Tensor output,
    torch::Tensor invOutput)
{
    CHECK_INPUT_CUDA(counts);
    CHECK_INPUT_CUDA(output);
    CHECK_INPUT_CUDA(invOutput);
    CHECK_LONG(counts);
    CHECK_LONG(output);
    CHECK_LONG(invOutput);

    AT_ASSERTM(counts.size(0) == output.size(0), "mismatching output shapes");
    AT_ASSERTM(counts.size(1) == output.size(1), "mismatching output shapes");
    AT_ASSERTM(counts.size(1) == invOutput.size(1), "mismatching output shapes");
    AT_ASSERTM(counts.size(1) <= 256, "unsupported number of bits");

    size_t numCombos = output.size(1);
    size_t numBits = 0;
    for (size_t i = 1; i < 9; i++) {
        if (1 << i == numCombos) {
            numBits = i;
        }
    }
    assert(numBits != 0);

    int blockSize = 1024;
    int numBlocks = counts.size(0) / blockSize;
    if (counts.size(0) % blockSize) {
        numBlocks++;
    }

    const dim3 threads(blockSize, 1, 1);
    const dim3 blocks(numBlocks, 1, 1);

    greedy_permutation_search_cuda_kernel<<<blocks, threads>>>(
        (const uint64_t *)counts.data_ptr(),
        (uint64_t *)output.data_ptr(),
        (uint64_t *)invOutput.data_ptr(),
        counts.size(0),
        numBits);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("count_bit_patterns_cpu", &count_bit_patterns_cpu, "Count bit patterns extracted from inputs");
    m.def("greedy_permutation_search_cpu", &greedy_permutation_search_cpu, "Search for permutations to reduce bitwise entropy");
    m.def("count_bit_patterns_cuda", &count_bit_patterns_cuda, "Count bit patterns extracted from inputs");
    m.def("greedy_permutation_search_cuda", &greedy_permutation_search_cuda, "Search for permutations to reduce bitwise entropy");
}
