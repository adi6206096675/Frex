import frex

print("--- Booting Frex Autograd Preparation ---")
ctx = frex.FrexContext(memory_pool_mb=64)

# Create a tensor that tracks gradients
print("\n--- Creating Weight Matrix ---")
W = ctx.create_tensor("Weights", [2000, 1000], init_val=5.0, requires_grad=True)

print(f"\nOriginal Matrix Shape: {W.shape}")
print(f"Top-Left Value: {W.item(0,0)}")

# Transpose it across the scattered memory pages
print("\n--- Transposing Matrix for Backward Pass ---")
W_T = W.T

print(f"\nTransposed Matrix Shape: {W_T.shape}")
print(f"Top-Left Value in Transposed Matrix: {W_T.item(0,0)}")

ctx.display_hardware_state()