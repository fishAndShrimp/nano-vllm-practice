#include <torch/extension.h>

torch::Tensor VecAddCuda(torch::Tensor a, torch::Tensor b);
torch::Tensor
VecAddRaw(torch::Tensor a_cpu, torch::Tensor b_cpu);

torch::Tensor ReluCuda(torch::Tensor a);

torch::Tensor TransposeCuda(torch::Tensor a);

torch::Tensor ReduceCuda(torch::Tensor a);

torch::Tensor PrefixSumCuda(torch::Tensor a);

torch::Tensor SafeSoftmaxCuda(torch::Tensor a);
torch::Tensor OnlineSoftmaxCuda(torch::Tensor a);
torch::Tensor BatchedOnlineSoftmaxCuda(torch::Tensor a);

torch::Tensor GemmCuda(torch::Tensor a, torch::Tensor b);
torch::Tensor
GemmRowWiseCuda(torch::Tensor a, torch::Tensor b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("VecAddCuda", &VecAddCuda)
        .def("ReluCuda", &ReluCuda)
        .def("TransposeCuda", &TransposeCuda)
        .def("ReduceCuda", &ReduceCuda)
        .def("PrefixSumCuda", &PrefixSumCuda)
        .def("SafeSoftmaxCuda", &SafeSoftmaxCuda)
        .def("OnlineSoftmaxCuda", &OnlineSoftmaxCuda)
        .def(
            "BatchedOnlineSoftmaxCuda",
            &BatchedOnlineSoftmaxCuda
        )
        .def("GemmCuda", &GemmCuda)
        .def("GemmRowWiseCuda", &GemmRowWiseCuda)
        .def("VecAddRaw", &VecAddRaw);
}
