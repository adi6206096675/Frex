#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "frex_core.hpp"

namespace py = pybind11;

// Hardware Detection Function
bool is_cuda_available() {
#ifdef FREX_USE_CUDA
    return true; // Compiled with GPU support
#else
    return false; // Compiled as CPU-only
#endif
}

PYBIND11_MODULE(_C, m) {
    m.doc() = "Frex Engine C++ Virtual Memory Core Engine Interface";

    // Expose hardware detection to Python
    m.def("is_cuda_available", &is_cuda_available, "Check if Frex is running on an NVIDIA GPU");

    py::class_<FrexEngine>(m, "FrexEngine")
        // Updated constructor to support UVM Tiered Memory Architecture (Fast + Slow)
        .def(py::init<size_t, size_t>(), py::arg("fast_mb"), py::arg("slow_mb"))
        
        // Tensor Lifecycle Operations
        .def("allocate_tensor", &FrexEngine::allocate_tensor, py::arg("name"), py::arg("shape"), py::arg("element_size"), py::arg("requires_grad") = false)
        .def("free_tensor", &FrexEngine::free_tensor, py::arg("name"))
        .def("fill_tensor", &FrexEngine::fill_tensor, py::arg("name"), py::arg("value"), py::arg("fill_grad") = false)
        .def("read_value", &FrexEngine::read_value, py::arg("name"), py::arg("index"), py::arg("read_grad") = false)

        // Add these right under fused_linear_relu
        .def("extract_grad", &FrexEngine::extract_grad, py::arg("source_name"), py::arg("target_name"))
        .def("accumulate_grad", &FrexEngine::accumulate_grad, py::arg("source_name"), py::arg("target_name"))
        
        // Core Math Ops
        .def("matmul", &FrexEngine::matmul, py::arg("A_name"), py::arg("B_name"), py::arg("Out_name"))
        .def("transpose", &FrexEngine::transpose, py::arg("input_name"), py::arg("output_name"))
        // Add this under your matmul/transpose bindings
        .def("scale", &FrexEngine::scale, py::arg("input_name"), py::arg("scalar"), py::arg("output_name"))

        // Add this line under your other math bindings
        .def("softmax", &FrexEngine::softmax, py::arg("input_name"), py::arg("output_name"))
        // Add these right under .def("softmax", ...)
        .def("cross_entropy", &FrexEngine::cross_entropy, py::arg("probs_name"), py::arg("targets_name"), py::arg("out_loss_name"))
        .def("cross_entropy_backward", &FrexEngine::cross_entropy_backward, py::arg("probs_name"), py::arg("targets_name"), py::arg("grad_name"))

        // Add these at the bottom of your module definitions
        .def("sgd_step", &FrexEngine::sgd_step, py::arg("name"), py::arg("learning_rate"))
        .def("zero_grad", &FrexEngine::zero_grad, py::arg("name"))

        .def("add", &FrexEngine::add, py::arg("A_name"), py::arg("B_name"), py::arg("Out_name"))
        .def("gelu", &FrexEngine::gelu, py::arg("input_name"), py::arg("output_name"))
        .def("layer_norm", &FrexEngine::layer_norm, py::arg("input_name"), py::arg("weight_name"), py::arg("bias_name"), py::arg("output_name"))

        // Fused Layer to bypass the PyTorch Memory Wall
        .def("fused_linear_relu", &FrexEngine::fused_linear_relu, py::arg("X_name"), py::arg("W_name"), py::arg("b_name"), py::arg("Out_name"))
        
        // Memory Telemetry & Monitoring
        .def("get_free_pages", &FrexEngine::get_free_pages)
        .def("get_total_pages", &FrexEngine::get_total_pages)
        .def("print_memory_map", &FrexEngine::print_memory_map);
}