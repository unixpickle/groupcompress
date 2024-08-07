#include "groupcompress_cpu.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("count_bit_patterns_cpu", &count_bit_patterns_cpu, "Count bit patterns extracted from inputs");
    m.def("greedy_permutation_search_cpu", &greedy_permutation_search_cpu, "Search for permutations to reduce bitwise entropy");
}
