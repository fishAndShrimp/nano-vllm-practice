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

torch::Tensor FlashAttentionCuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v
);
torch::Tensor FlashAttentionCoalescedCuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v
);

torch::Tensor PagedAttentionGemmCuda(
    torch::Tensor q,
    torch::Tensor k_pool,
    torch::Tensor v_pool,
    torch::Tensor cu_seqlens,
    int q_len_max,
    torch::Tensor kv_page_tables,
    torch::Tensor kv_lens
);

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
        .def("FlashAttentionCuda", &FlashAttentionCuda)
        .def(
            "FlashAttentionCoalescedCuda",
            &FlashAttentionCoalescedCuda
        )
        .def(
            "PagedAttentionGemmCuda",
            &PagedAttentionGemmCuda
        )
        .def("VecAddRaw", &VecAddRaw);
}
