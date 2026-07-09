import frex
import frex.nn as nn

ctx = frex.FrexContext(fast_vram_mb=64, slow_ram_mb=256)

print("\n--- Booting Frex Self-Attention Block ---")
# Let's say our embedding dimension is 8 (a tiny LLM!)
attention_layer = nn.SelfAttention(ctx, embed_dim=8)

# Simulating a sentence with 4 words (Sequence Length 4, Embed Dim 8)
# e.g., ["The", "cat", "sat", "down"]
Sequence = ctx.create_tensor("Input_Sequence", [4, 8], init_val=1.5)

print("\n--- Forward Pass: Computing Attention ---")
# The words will now mathematically communicate with each other!
ContextOutput = attention_layer(Sequence)

print(f"\nFinal Context Shape: {ContextOutput.shape}")
print(f"Top-Left Context Value: {ContextOutput.item(0,0):.4f}")

ctx.display_hardware_state()