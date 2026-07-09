import frex

ctx = frex.FrexContext(fast_vram_mb=32, slow_ram_mb=128)

print("\n--- Testing Softmax Primitive ---")
# Simulate a network outputting raw logits for a vocabulary of 4 words.
# We will initialize them all to 2.0 to see if it correctly distributes the probability.
logits = ctx.create_tensor("Raw_Logits", [1, 4], init_val=2.0)

print(f"Raw Logits: [{logits.item(0,0)}, {logits.item(0,1)}, {logits.item(0,2)}, {logits.item(0,3)}]")

# Convert raw numbers into a probability distribution
probs = logits.softmax()

print(f"Probabilities: [{probs.item(0,0):.4f}, {probs.item(0,1):.4f}, {probs.item(0,2):.4f}, {probs.item(0,3):.4f}]")

# Math Check
total_prob = probs.item(0,0) + probs.item(0,1) + probs.item(0,2) + probs.item(0,3)
print(f"Total Probability Sum (Should be 1.0): {total_prob:.4f}")