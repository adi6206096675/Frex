#include "frex_core.hpp"
#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>

// --- UVM INITIALIZATION ---
FrexEngine::FrexEngine(size_t fast_mb, size_t slow_mb) {
    fast_pool_pages = fast_mb;
    slow_pool_pages = slow_mb;
    total_pages = fast_mb + slow_mb; 
    
    size_t pool_size_bytes = total_pages * PAGE_SIZE;
    physical_memory_pool = new uint8_t[pool_size_bytes];
    std::fill(physical_memory_pool, physical_memory_pool + pool_size_bytes, 0);
    
    page_bitmap.assign(total_pages, false);
    std::cout << "[Frex UVM] Unified Engine Booting...\n" 
              << "  -> Fast VRAM Tier: " << fast_pool_pages << " MB\n"
              << "  -> Slow RAM Tier:  " << slow_pool_pages << " MB\n"
              << "  -> Total Capacity: " << total_pages << " MB" << std::endl;
}

FrexEngine::~FrexEngine() {
    delete[] physical_memory_pool;
}

float* FrexEngine::get_physical_address(const VirtualTensor& tensor, size_t linear_index, bool is_grad) {
    size_t byte_offset = linear_index * sizeof(float);
    size_t logical_page = byte_offset / PAGE_SIZE;
    size_t page_offset = byte_offset % PAGE_SIZE;
    
    size_t physical_page = is_grad ? tensor.grad_page_table[logical_page] : tensor.page_table[logical_page];
    return reinterpret_cast<float*>(physical_memory_pool + (physical_page * PAGE_SIZE) + page_offset);
}

// --- UVM ALLOCATION PIPELINE ---
bool FrexEngine::allocate_tensor(const std::string& name, const std::vector<int64_t>& shape, size_t element_size, bool requires_grad) {
    if (tensor_registry.find(name) != tensor_registry.end()) return false;

    size_t total_elements = 1;
    for (int64_t dim : shape) total_elements *= dim;
    size_t total_bytes = total_elements * element_size;
    
    size_t pages_needed = (total_bytes + PAGE_SIZE - 1) / PAGE_SIZE;
    size_t total_pages_requested = requires_grad ? pages_needed * 2 : pages_needed;

    size_t free_pages_count = 0;
    for (bool bit : page_bitmap) if (!bit) free_pages_count++;

    if (total_pages_requested > free_pages_count) {
        std::cout << "[Frex UVM Error] Absolute OOM. Cannot even fit in slow RAM." << std::endl;
        return false; 
    }

    VirtualTensor tensor;
    tensor.name = name;
    tensor.total_bytes = total_bytes;
    tensor.num_pages = pages_needed;
    tensor.shape = shape;
    tensor.requires_grad = requires_grad;

    auto allocate_pages = [&](std::vector<size_t>& table) {
        size_t allocated = 0;
        size_t fast_used = 0, slow_used = 0;
        
        for (size_t i = 0; i < fast_pool_pages && allocated < pages_needed; ++i) {
            if (!page_bitmap[i]) {
                page_bitmap[i] = true;
                table.push_back(i);
                allocated++;
                fast_used++;
            }
        }
        
        for (size_t i = fast_pool_pages; i < total_pages && allocated < pages_needed; ++i) {
            if (!page_bitmap[i]) {
                page_bitmap[i] = true;
                table.push_back(i);
                allocated++;
                slow_used++;
            }
        }
        return std::make_pair(fast_used, slow_used);
    };

    auto data_usage = allocate_pages(tensor.page_table);
    std::cout << "[Frex UVM] Mapped '" << name << "'. Fast VRAM: " << data_usage.first << "MB | Spilled to System RAM: " << data_usage.second << "MB" << std::endl;

    if (requires_grad) {
        auto grad_usage = allocate_pages(tensor.grad_page_table);
        std::cout << "[Frex UVM] Mapped '" << name << "_GRAD'. Fast VRAM: " << grad_usage.first << "MB | Spilled to System RAM: " << grad_usage.second << "MB" << std::endl;
    }

    tensor_registry[name] = tensor;
    return true;
}

bool FrexEngine::free_tensor(const std::string& name) {
    if (tensor_registry.find(name) == tensor_registry.end()) return false;
    
    const auto& tensor = tensor_registry[name];
    for (size_t page : tensor.page_table) {
        page_bitmap[page] = false;
    }
    if (tensor.requires_grad) {
        for (size_t page : tensor.grad_page_table) {
            page_bitmap[page] = false;
        }
    }
    
    std::cout << "[Frex Dealloc] Released tensor '" << name << "'." << std::endl;
    tensor_registry.erase(name);
    return true;
}

void FrexEngine::fill_tensor(const std::string& name, float value, bool fill_grad) {
    auto& tensor = tensor_registry[name];
    size_t total_elements = tensor.total_bytes / sizeof(float);
    
    #pragma omp parallel for
    for (size_t i = 0; i < total_elements; ++i) {
        *get_physical_address(tensor, i, fill_grad) = value;
    }
}

float FrexEngine::read_value(const std::string& name, size_t index, bool read_grad) {
    auto& tensor = tensor_registry[name];
    return *get_physical_address(tensor, index, read_grad);
}

// --- MATH OPS: MATMUL ---
bool FrexEngine::matmul(const std::string& A_name, const std::string& B_name, const std::string& Out_name) {
    auto& A = tensor_registry[A_name];
    auto& B = tensor_registry[B_name];
    auto& Out = tensor_registry[Out_name];

    int64_t M = A.shape[0];
    int64_t K = A.shape[1];
    int64_t N = B.shape[1];

    #pragma omp parallel for collapse(2)
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += (*get_physical_address(A, i * K + k)) * (*get_physical_address(B, k * N + j));
            }
            *get_physical_address(Out, i * N + j) = sum;
        }
    }
    return true;
}

// --- MATH OPS: TRANSPOSE ---
bool FrexEngine::transpose(const std::string& input_name, const std::string& output_name) {
    auto& in_tensor = tensor_registry[input_name];
    auto& out_tensor = tensor_registry[output_name];

    int64_t rows = in_tensor.shape[0];
    int64_t cols = in_tensor.shape[1];

    if (out_tensor.shape[0] != cols || out_tensor.shape[1] != rows) {
        std::cerr << "[Frex Math Error] Transpose shape mismatch." << std::endl;
        return false;
    }

    std::cout << "[Frex Math] Executing Parallel Transpose (" << rows << "x" << cols << ") -> scattered pages..." << std::endl;

    #pragma omp parallel for collapse(2)
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            float val = *get_physical_address(in_tensor, i * cols + j, false);
            *get_physical_address(out_tensor, j * rows + i, false) = val;
        }
    }
    return true;
}

// --- FUSED OPERATOR: Linear + Bias + ReLU ---
bool FrexEngine::fused_linear_relu(const std::string& X_name, const std::string& W_name, const std::string& b_name, const std::string& Out_name) {
    auto& X = tensor_registry[X_name];
    auto& W = tensor_registry[W_name];
    auto& b = tensor_registry[b_name];
    auto& Out = tensor_registry[Out_name];

    int64_t M = X.shape[0]; 
    int64_t K = X.shape[1]; 
    int64_t N = W.shape[1]; 

    std::cout << "[Frex Fused Math] Executing Fused Linear+ReLU (" << M << "x" << N << ") avoiding memory walls..." << std::endl;

    #pragma omp parallel for collapse(2)
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            
            for (int64_t k = 0; k < K; ++k) {
                float x_val = *get_physical_address(X, i * K + k);
                float w_val = *get_physical_address(W, k * N + j);
                sum += x_val * w_val;
            }
            
            float bias_val = *get_physical_address(b, j);
            sum += bias_val;
            
            if (sum < 0.0f) {
                sum = 0.0f;
            }
            
            *get_physical_address(Out, i * N + j) = sum;
        }
    }
    return true;
}

size_t FrexEngine::get_free_pages() const {
    size_t count = 0;
    for (bool bit : page_bitmap) if (!bit) count++;
    return count;
}

size_t FrexEngine::get_total_pages() const {
    return total_pages;
}

// --- UVM VISUALIZER ---
void FrexEngine::print_memory_map() const {
    std::cout << "\n[Frex Hardware State]\n";
    std::cout << "--- TIER 1: FAST VRAM ---\n[";
    for (size_t i = 0; i < fast_pool_pages; ++i) {
        std::cout << (page_bitmap[i] ? "█" : "□");
        if ((i + 1) % 50 == 0 && i < fast_pool_pages - 1) std::cout << "]\n[";
    }
    std::cout << "]\n";
    
    std::cout << "--- TIER 2: SLOW SYSTEM RAM ---\n[";
    for (size_t i = fast_pool_pages; i < total_pages; ++i) {
        std::cout << (page_bitmap[i] ? "▒" : "□"); 
        if ((i + 1 - fast_pool_pages) % 50 == 0 && i < total_pages - 1) std::cout << "]\n[";
    }
    std::cout << "]\n\n";
}

// --- GRADIENT ROUTING BRIDGES ---

bool FrexEngine::extract_grad(const std::string& source_name, const std::string& target_name) {
    auto& source = tensor_registry[source_name];
    auto& target = tensor_registry[target_name];
    size_t elements = source.total_bytes / sizeof(float);
    
    // Copy Gradient Pages (Source) -> Data Pages (Target)
    #pragma omp parallel for
    for (size_t i = 0; i < elements; ++i) {
        *get_physical_address(target, i, false) = *get_physical_address(source, i, true);
    }
    return true;
}

bool FrexEngine::accumulate_grad(const std::string& source_name, const std::string& target_name) {
    auto& source = tensor_registry[source_name];
    auto& target = tensor_registry[target_name];
    size_t elements = source.total_bytes / sizeof(float);
    
    // Add Data Pages (Source) -> Gradient Pages (Target)
    #pragma omp parallel for
    for (size_t i = 0; i < elements; ++i) {
        float computed_grad = *get_physical_address(source, i, false);
        float current_grad = *get_physical_address(target, i, true);
        *get_physical_address(target, i, true) = current_grad + computed_grad;
    }
    return true;
}

// --- NEURAL PRIMITIVES ---

bool FrexEngine::softmax(const std::string& input_name, const std::string& output_name) {
    auto& in_tensor = tensor_registry[input_name];
    auto& out_tensor = tensor_registry[output_name];
    
    int64_t rows = in_tensor.shape[0]; // Batch size (e.g., number of words)
    int64_t cols = in_tensor.shape[1]; // Vocab size (e.g., 50,000 possible words)

    std::cout << "[Frex Math] Executing Parallel Softmax (" << rows << "x" << cols << ")..." << std::endl;

    // Process each row (batch) independently
    #pragma omp parallel for
    for (int64_t i = 0; i < rows; ++i) {
        // 1. Find Max Value (For Numerical Stability)
        float max_val = -INFINITY;
        for (int64_t j = 0; j < cols; ++j) {
            float val = *get_physical_address(in_tensor, i * cols + j, false);
            if (val > max_val) max_val = val;
        }
        
        // 2. Compute Exponentials and Sum
        float sum_exp = 0.0f;
        for (int64_t j = 0; j < cols; ++j) {
            float val = *get_physical_address(in_tensor, i * cols + j, false);
            float e_val = std::exp(val - max_val);
            *get_physical_address(out_tensor, i * cols + j, false) = e_val; // Temporarily store e^x
            sum_exp += e_val;
        }
        
        // 3. Normalize to Probabilities (Divide by Sum)
        for (int64_t j = 0; j < cols; ++j) {
            *get_physical_address(out_tensor, i * cols + j, false) /= sum_exp;
        }
    }
    return true;
}

// --- OPTIMIZERS ---

bool FrexEngine::sgd_step(const std::string& name, float learning_rate) {
    auto& tensor = tensor_registry[name];
    if (!tensor.requires_grad) return false;
    
    size_t elements = tensor.total_bytes / sizeof(float);
    
    // Weight = Weight - (Learning_Rate * Gradient)
    #pragma omp parallel for
    for (size_t i = 0; i < elements; ++i) {
        float weight = *get_physical_address(tensor, i, false);
        float grad = *get_physical_address(tensor, i, true);
        
        *get_physical_address(tensor, i, false) = weight - (learning_rate * grad);
    }
    return true;
}

bool FrexEngine::zero_grad(const std::string& name) {
    auto& tensor = tensor_registry[name];
    if (!tensor.requires_grad) return false;
    
    size_t elements = tensor.total_bytes / sizeof(float);
    
    // Wipe gradients clean for the next training loop
    #pragma omp parallel for
    for (size_t i = 0; i < elements; ++i) {
        *get_physical_address(tensor, i, true) = 0.0f;
    }
    return true;
}
// --- LOSS FUNCTIONS ---

bool FrexEngine::cross_entropy(const std::string& probs_name, const std::string& targets_name, const std::string& out_loss_name) {
    auto& probs = tensor_registry[probs_name];
    auto& targets = tensor_registry[targets_name];
    auto& loss = tensor_registry[out_loss_name];

    size_t elements = probs.total_bytes / sizeof(float);
    float total_loss = 0.0f;

    for (size_t i = 0; i < elements; ++i) {
        float p = *get_physical_address(probs, i, false);
        float t = *get_physical_address(targets, i, false);
        if (t > 0.0f) {
            // Calculate: -Target * log(Probability)
            // We add 1e-7f to prevent std::log(0) from crashing the C++ engine
            total_loss -= t * std::log(p + 1e-7f);
        }
    }
    
    // Write the scalar loss to the output tensor
    *get_physical_address(loss, 0, false) = total_loss;
    return true;
}

bool FrexEngine::cross_entropy_backward(const std::string& probs_name, const std::string& targets_name, const std::string& grad_name) {
    auto& probs = tensor_registry[probs_name];
    auto& targets = tensor_registry[targets_name];
    auto& grad = tensor_registry[grad_name];

    size_t elements = probs.total_bytes / sizeof(float);

    // The Beautiful Derivative: Gradient = Probabilities - Targets
    #pragma omp parallel for
    for (size_t i = 0; i < elements; ++i) {
        float p = *get_physical_address(probs, i, false);
        float t = *get_physical_address(targets, i, false);
        *get_physical_address(grad, i, false) = p - t;
    }
    return true;
}

bool FrexEngine::scale(const std::string& input_name, float scalar, const std::string& output_name) {
    auto& in_tensor = tensor_registry[input_name];
    auto& out_tensor = tensor_registry[output_name];
    size_t elements = in_tensor.total_bytes / sizeof(float);
    
    #pragma omp parallel for
    for (size_t i = 0; i < elements; ++i) {
        float val = *get_physical_address(in_tensor, i, false);
        *get_physical_address(out_tensor, i, false) = val * scalar;
    }
    return true;
}

// --- LLM PRIMITIVES (Add, GeLU, LayerNorm) ---

bool FrexEngine::add(const std::string& A_name, const std::string& B_name, const std::string& Out_name) {
    auto& A = tensor_registry[A_name];
    auto& B = tensor_registry[B_name];
    auto& Out = tensor_registry[Out_name];
    size_t elements = A.total_bytes / sizeof(float);
    
    #pragma omp parallel for
    for (size_t i = 0; i < elements; ++i) {
        *get_physical_address(Out, i, false) = *get_physical_address(A, i, false) + *get_physical_address(B, i, false);
    }
    return true;
}

bool FrexEngine::gelu(const std::string& input_name, const std::string& output_name) {
    auto& in_tensor = tensor_registry[input_name];
    auto& out_tensor = tensor_registry[output_name];
    size_t elements = in_tensor.total_bytes / sizeof(float);
    
    // GeLU Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    #pragma omp parallel for
    for (size_t i = 0; i < elements; ++i) {
        float x = *get_physical_address(in_tensor, i, false);
        float cube = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * cube);
        *get_physical_address(out_tensor, i, false) = 0.5f * x * (1.0f + std::tanh(inner));
    }
    return true;
}

bool FrexEngine::layer_norm(const std::string& input_name, const std::string& weight_name, const std::string& bias_name, const std::string& output_name) {
    auto& in = tensor_registry[input_name];
    auto& w = tensor_registry[weight_name];
    auto& b = tensor_registry[bias_name];
    auto& out = tensor_registry[output_name];
    
    int64_t rows = in.shape[0];
    int64_t cols = in.shape[1];
    float eps = 1e-5f;

    #pragma omp parallel for
    for (int64_t i = 0; i < rows; ++i) {
        float mean = 0.0f;
        for (int64_t j = 0; j < cols; ++j) mean += *get_physical_address(in, i * cols + j, false);
        mean /= cols;
        
        float var = 0.0f;
        for (int64_t j = 0; j < cols; ++j) {
            float diff = *get_physical_address(in, i * cols + j, false) - mean;
            var += diff * diff;
        }
        var /= cols;
        float inv_std = 1.0f / std::sqrt(var + eps);
        
        for (int64_t j = 0; j < cols; ++j) {
            float norm_val = (*get_physical_address(in, i * cols + j, false) - mean) * inv_std;
            float weight_val = *get_physical_address(w, j, false);
            float bias_val = *get_physical_address(b, j, false);
            *get_physical_address(out, i * cols + j, false) = (norm_val * weight_val) + bias_val;
        }
    }
    return true;
}