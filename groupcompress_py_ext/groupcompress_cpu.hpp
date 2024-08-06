#include <torch/extension.h>

#define CHECK_CPU(x) AT_ASSERTM(x.device().is_cpu(), #x " must be a CPU tensor")
#define CHECK_LONG(x) AT_ASSERTM(x.scalar_type() == torch::ScalarType::Long, #x " must be a Long tensor")
#define CHECK_BYTE(x) AT_ASSERTM(x.scalar_type() == torch::ScalarType::Byte, #x " must be a Byte tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CPU(x);      \
    CHECK_CONTIGUOUS(x);

void count_bit_patterns_cpu_kernel(
    const uint8_t *inputs,
    const uint64_t *indices,
    uint64_t *outputs,
    size_t inputSize,
    size_t numInputs,
    size_t numBits,
    size_t numIndices)
{
    size_t inputsPerBlock = 16384 / inputSize;
    if (!inputsPerBlock) {
        inputsPerBlock = 1;
    }
    size_t indexPerBlock = 16384 / numBits;
    size_t numValues = 1 << numBits;

    for (size_t i = 0; i < numInputs; i += inputsPerBlock) {
        for (size_t k = 0; k < numIndices; k += indexPerBlock) {
            for (size_t j = i; j < i + inputsPerBlock && j < numInputs; j++) {
                const uint8_t *input = &inputs[j * inputSize];
                for (size_t l = k; l < k + indexPerBlock && l < numIndices; l++) {
                    const uint64_t *curIndex = &indices[l * numBits];
                    uint8_t result = 0;
                    for (size_t m = 0; m < numBits; m++) {
                        if (input[curIndex[m]]) {
                            result |= 1 << m;
                        }
                    }
                    outputs[l * numValues + (size_t)result] += 1;
                }
            }
        }
    }
}

void count_bit_patterns_cpu(
    const torch::Tensor inputs,
    const torch::Tensor indices,
    torch::Tensor output)
{
    CHECK_INPUT(inputs);
    CHECK_INPUT(indices);
    CHECK_INPUT(output);
    CHECK_BYTE(inputs);
    CHECK_LONG(indices);
    CHECK_LONG(output);

    size_t numBits = indices.size(1);
    assert(output.size(0) == indices.size(0));
    assert(output.size(1) == (1 << numBits));
    assert(numBits != 0);

    count_bit_patterns_cpu_kernel(
        (const uint8_t *)inputs.data_ptr(),
        (const uint64_t *)indices.data_ptr(),
        (uint64_t *)output.data_ptr(),
        inputs.size(1),
        inputs.size(0),
        numBits,
        indices.size(0));
}