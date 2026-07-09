#ifndef FREX_CORE_HPP
#define FREX_CORE_HPP

#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cstddef>
#include <string>

const size_t PAGE_SIZE = 1024 * 1024; // 1 MB pages

struct VirtualTensor {
    std::string name;
    size_t total_bytes;
    size_t num_pages;
    std::vector<int64_t> shape;
    std::vector<size_t> page_table; 
    
    bool requires_grad;
    std::vector<size_t> grad_page_table; 
};

class FrexEngine {
private:
    size_t fast_pool_pages; // The "GPU VRAM"
    size_t slow_pool_pages; // The "CPU RAM"
    size_t total_pages;
    
    uint8_t* physical_memory_pool;
    std::vector<bool> page_bitmap;
    std::unordered_map<std::string, VirtualTensor> tensor_registry;

    float* get_physical_address(const VirtualTensor& tensor, size_t linear_index, bool is_grad = false);

public:
    // Constructor now takes both Fast and Slow limits
    FrexEngine(size_t fast_mb, size_t slow_mb);
    ~FrexEngine();

    bool allocate_tensor(const std::string& name, const std::vector<int64_t>& shape, size_t element_size, bool requires_grad);
    bool free_tensor(const std::string& name);
    
    void fill_tensor(const std::string& name, float value, bool fill_grad = false);
    float read_value(const std::string& name, size_t index, bool read_grad = false);

    bool scale(const std::string& input_name, float scalar, const std::string& output_name);
    
    bool matmul(const std::string& A_name, const std::string& B_name, const std::string& Out_name);
    bool transpose(const std::string& input_name, const std::string& output_name);
    bool fused_linear_relu(const std::string& X_name, const std::string& W_name, const std::string& b_name, const std::string& Out_name);

    // Gradient Routing Bridges
    bool extract_grad(const std::string& source_name, const std::string& target_name);
    bool accumulate_grad(const std::string& source_name, const std::string& target_name);

    // Optimizer Step
    bool sgd_step(const std::string& name, float learning_rate);
    bool zero_grad(const std::string& name);

    bool add(const std::string& A_name, const std::string& B_name, const std::string& Out_name);
    bool gelu(const std::string& input_name, const std::string& output_name);
    bool layer_norm(const std::string& input_name, const std::string& weight_name, const std::string& bias_name, const std::string& output_name);

    // Neural Primitives
    bool softmax(const std::string& input_name, const std::string& output_name);
    // Loss Functions
    bool cross_entropy(const std::string& probs_name, const std::string& targets_name, const std::string& out_loss_name);
    bool cross_entropy_backward(const std::string& probs_name, const std::string& targets_name, const std::string& grad_name);

    size_t get_free_pages() const;
    size_t get_total_pages() const;
    void print_memory_map() const;
};

#endif // FREX_CORE_HPP