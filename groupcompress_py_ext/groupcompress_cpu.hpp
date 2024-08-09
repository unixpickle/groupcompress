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

float permutation_entropy(
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

void greedy_permutation_search_cpu_kernel(
    const uint64_t *counts,
    uint64_t *outputs,
    size_t batchSize,
    size_t numBits)
{
    for (size_t i = 0; i < batchSize; i++) {
        const uint64_t *subCounts = &counts[i * (1 << numBits)];
        uint64_t *subOutputs = &outputs[i * (1 << numBits)];

        // These sizes are upper-bounded to avoid extra templates.
        uint64_t perm[256];
        uint64_t inverse[256];
        for (size_t j = 0; j < (1 << numBits); j++) {
            perm[j] = j;
            inverse[j] = j;
        }

        float entropy = permutation_entropy(subCounts, perm, numBits);

        // Greedily swap each example for each bit.
        for (size_t bit = 0; bit < numBits; bit++) {
            for (uint64_t i = 0; i < (1 << numBits); i++) {
                uint64_t pattern = perm[i];
                uint64_t other = pattern ^ (1 << bit);
                uint64_t otherIdx = inverse[other];

                perm[i] = other;
                perm[otherIdx] = pattern;
                float newEntropy = permutation_entropy(subCounts, perm, numBits);

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

        for (size_t i = 0; i < (1 << numBits); i++) {
            subOutputs[i] = perm[i];
        }
    }
}

void greedy_permutation_search_cpu(
    const torch::Tensor counts,
    torch::Tensor output)
{
    CHECK_INPUT(counts);
    CHECK_INPUT(output);
    CHECK_LONG(counts);
    CHECK_LONG(output);

    AT_ASSERTM(counts.size(0) == output.size(0), "mismatching output shapes");
    AT_ASSERTM(counts.size(1) == output.size(1), "mismatching output shapes");
    AT_ASSERTM(counts.size(1) <= 256, "unsupported number of bits");

    size_t numCombos = output.size(1);
    size_t numBits = 0;
    for (size_t i = 1; i < 9; i++) {
        if (1 << i == numCombos) {
            numBits = i;
        }
    }
    assert(numBits != 0);

    greedy_permutation_search_cpu_kernel(
        (const uint64_t *)counts.data_ptr(),
        (uint64_t *)output.data_ptr(),
        counts.size(0),
        numBits);
}