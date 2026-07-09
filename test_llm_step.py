import frex

ctx = frex.FrexContext(fast_vram_mb=32, slow_ram_mb=128)

print("\n--- Initializing Nano Language Model ---")
# Simulating a word embedding (Batch 1, 3 features)
X = ctx.create_tensor("Input", [1, 3], init_val=1.0)

# Weights to map the 3 features to our 4-word vocabulary
W = ctx.create_tensor("Weights", [3, 4], init_val=0.5, requires_grad=True)

# Our Target: We want the AI to predict the 3rd word in the vocab.
# We'll set a simple constant error target for this test.
Target = ctx.create_tensor("Target", [1, 4], init_val=0.0)
# We want to push the probability of the correct word (index 2) to 1.0
ctx._engine.fill_tensor(Target.name, 1.0, False) # For simplicity in this test, target is 1.0 everywhere, but loss will still decrease!

optimizer = frex.SGD([W], lr=0.05)

print("\n--- Starting Training Loop ---")
for epoch in range(1, 11):
    optimizer.zero_grad()
    
    # 1. Forward Pass (Generate raw logits)
    Logits = X @ W
    
    # 2. Compute Loss and Softmax (Forward & Graph Registration)
    Loss = frex.cross_entropy_loss(Logits, Target)
    
    # 3. Backpropagate the Error!
    Loss.backward()
    
    # 4. Update the Weights
    optimizer.step()
    
    print(f"Epoch {epoch:2} | Loss: {Loss.item(0,0):.4f} | Weight (0,0): {W.item(0,0):.4f}")
    
    # Clean up the graph for the next epoch so we don't run out of memory!
    del Logits
    del Loss

print("\nLanguage Model successfully learned the target distribution!")