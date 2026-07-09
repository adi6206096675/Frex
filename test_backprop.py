import frex

# Boot the engine with managed storage tiers
ctx = frex.FrexContext(fast_vram_mb=32, slow_ram_mb=128)

print("\n--- Initializing Network Nodes ---")
# Inputs (e.g., Batch of 2 tokens, 3 hidden features)
X = ctx.create_tensor("Input_X", [2, 3], init_val=1.0, requires_grad=False)

# Weights of Layer 1
W1 = ctx.create_tensor("Weights_W1", [3, 2], init_val=2.0, requires_grad=True)

# Forward Pass: Layer 1
# H = X @ W1
print("\n--- Forward Pass: Hidden Layer ---")
H = X @ W1
print(f"Hidden Layer Output Value (0,0): {H.item(0,0)}") # Should be 1*2 + 1*2 + 1*2 = 6.0

# Backward Pass execution
print("\n--- Executing Backpropagation Graph ---")
H.backward()

print("\n--- Verification Matrix ---")
# Verification math: dH/dW1 = X^T
# Since H grad seed is 1.0 everywhere, dW1 (0,0) must equal the sum of inputs affecting it.
print(f"Computed Gradient for W1 (0,0): {W1.grad(0,0)}")

ctx.display_hardware_state()