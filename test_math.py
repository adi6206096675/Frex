import frex
import time

# 1. Initialize Frex with 1024 MB (1 GB) of RAM
print("--- Booting Frex Math Engine (1GB Mode) ---")
ctx = frex.FrexContext(memory_pool_mb=1024)

# 2. Let's create some MASSIVE tensors
# A 4000x4000 float32 matrix requires exactly 64 Megabytes (64 Pages)
print("\n--- Allocating Massive Tensors ---")
A = ctx.create_tensor("Matrix_A", [4000, 4000], 2.0)
B = ctx.create_tensor("Matrix_B", [4000, 4000], 3.0)

ctx.display_hardware_state()

# 3. Do the Math!
print("\n--- Executing Massive C = A @ B ---")
start_time = time.time()

# Output C will also be 64 MB
C = A @ B

end_time = time.time()

print(f"\nResulting Tensor: {C}")
print(f"Top-Left Value in Tensor C: {C.item(0,0)} (Expected: 24000.0)")
print(f"Time taken for MatMul: {end_time - start_time:.2f} seconds")

print("\n--- Final Memory State ---")
ctx.display_hardware_state()