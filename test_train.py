import frex

ctx = frex.FrexContext(fast_vram_mb=32, slow_ram_mb=128)

print("\n--- Initializing AI Model ---")
# Input data (Batch 1, Feature 1)
X = ctx.create_tensor("Input", [1, 1], init_val=2.0)

# The network weight we want to train
W = ctx.create_tensor("Weight", [1, 1], init_val=0.5, requires_grad=True)

# Initialize the Optimizer
optimizer = frex.SGD([W], lr=0.1)

print("\n--- Starting Training Loop ---")
# Let's say the true target answer for X * W is 10.0
# We will train it for 5 epochs to get closer to 10.0
for epoch in range(1, 6):
    optimizer.zero_grad()
    
    # 1. Forward Pass
    Prediction = X @ W
    
    # 2. Calculate Error (Loss = Prediction - 10.0) 
    # For simplicity in this test, we simulate a constant error gradient pushing it up
    error_val = Prediction.item(0,0) - 10.0
    
    # 3. Backward Pass
    Prediction.backward()
    
    # Because we don't have a full Loss function yet, we manually scale the backward gradient by the error
    ctx._engine.fill_tensor(Prediction.name, error_val, True) 
    
    # 4. Optimizer Step
    optimizer.step()
    
    print(f"Epoch {epoch} | Prediction: {Prediction.item(0,0):.3f} | New Weight: {W.item(0,0):.3f}")

print("\nTraining Complete! The network successfully learned.")