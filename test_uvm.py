import frex

print("\n--- Booting Frex UVM ---")
# Simulating a tiny 16MB Fast GPU and 64MB of System RAM
ctx = frex.FrexContext(fast_vram_mb=16, slow_ram_mb=64)

# A 2000x2500 matrix requires exactly 20 MB of space.
# It is physically impossible to fit this in the 16MB Fast VRAM.
print("\n--- Allocating Massive Tensor ---")
BigMatrix = ctx.create_tensor("Massive_Matrix", [2000, 2500], 1.5)

ctx.display_hardware_state()

# The math engine doesn't even know it's split across hardware boundaries!
print(f"Matrix Value Test: {BigMatrix.item(0,0)}")