import frex
import frex.nn as nn

ctx = frex.FrexContext(fast_vram_mb=32, slow_ram_mb=128)

# 1. Define a Deep Neural Network using our new abstraction!
class DeepNetwork(nn.Module):
    def __init__(self, ctx):
        super().__init__()
        # 3 input features -> 16 hidden -> 4 output classes
        self.layer1 = nn.Linear(ctx, in_features=3, out_features=16, name="L1")
        self.layer2 = nn.Linear(ctx, in_features=16, out_features=4, name="L2")
        
        self.register_module(self.layer1)
        self.register_module(self.layer2)

    def forward(self, x):
        # Pass through Layer 1, then Layer 2
        hidden = self.layer1(x)
        out = self.layer2(hidden)
        return out

print("\n--- Booting Frex Deep Neural Network ---")
model = DeepNetwork(ctx)

# Pass ALL the automatically discovered parameters to the Optimizer
optimizer = frex.SGD(model.parameters(), lr=0.01)

# Create some dummy input data
X = ctx.create_tensor("Input_Data", [1, 3], init_val=1.0)
Target = ctx.create_tensor("Target_Data", [1, 4], init_val=0.0)
# Make it a true one-hot target (we cheat with a fill command)
ctx._engine.fill_tensor(Target.name, 1.0, False) # Just for the test

print("\n--- Training the Deep Network ---")
for epoch in range(1, 4):
    optimizer.zero_grad()
    
    # Notice how beautiful and clean this is now!
    Logits = model(X) 
    
    Loss = frex.cross_entropy_loss(Logits, Target)
    Loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch} | Loss: {Loss.item(0,0):.4f}")
    
    # Clean memory graph
    del Logits
    del Loss

print("\nNetwork Abstraction Successful!")