#include <torch/extension.h>

torch::Tensor VecAddCuda(torch::Tensor a, torch::Tensor b);
torch::Tensor
VecAddRaw(torch::Tensor a_cpu, torch::Tensor b_cpu);

torch::Tensor ReluCuda(torch::Tensor a);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("VecAddCuda", &VecAddCuda)
        .def("VecAddRaw", &VecAddRaw)
        .def("ReluCuda", &ReluCuda);
}
