import frex

# Initialize Frex engine managing a tight 64MB memory limits boundary environment
print("Initializing Frex runtime engine...")
ctx = frex.FrexContext(memory_pool_mb=64)

# Allocate some varying tensor blocks to fragments the memory structure 
print("\nAllocating fragmented tensor structures...")
ctx.tensor("weight_matrix_A", [2000, 2000]) # Uses ~16MB (16 Pages)
ctx.tensor("weight_matrix_B", [1000, 1000]) # Uses ~4MB (4 Pages)
ctx.tensor("weight_matrix_C", [3000, 3000]) # Uses ~36MB (36 Pages)

ctx.display_hardware_state()

# Delete middle chunk to create an intentional memory hole
print("\nReleasing middle tensor node to simulate data lifecycle memory hole fragmentation...")
ctx.delete("weight_matrix_B")

ctx.display_hardware_state()

# Standard framework would throw an OOM if trying to fit a chunk across scattered nodes
print("\nAttempting memory execution edge request fitting into structural page spaces...")
ctx.tensor("weight_matrix_D", [2000, 1000]) # Needs ~8MB (8 Pages). Left spaces: 4MB (old hole) + 8MB (tail) = 12MB total.

ctx.display_hardware_state()