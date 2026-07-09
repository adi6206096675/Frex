import uuid
from .tensor import Tensor

class Module:
    """Base class for all neural network modules."""
    def __init__(self):
        self._parameters = []
        self._modules = []

    def register_parameter(self, param: Tensor):
        self._parameters.append(param)

    def register_module(self, module):
        self._modules.append(module)

    def parameters(self):
        """Recursively gathers all training weights in the network."""
        params = list(self._parameters)
        for m in self._modules:
            params.extend(m.parameters())
        return params

    def __call__(self, *args, **kwargs):
        """Allows calling the module like a function, e.g., model(x)."""
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Every Frex Module must implement a forward pass.")


class Linear(Module):
    """Applies a linear transformation to the incoming data: y = xA^T"""
    def __init__(self, ctx, in_features: int, out_features: int, name="Linear"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize the weight matrix (Requires Gradients!)
        unique_name = f"{name}_W_{uuid.uuid4().hex[:6]}"
        
        # In a real framework we use Xavier initialization. 
        # We'll use 0.1 for now so the math is predictable.
        self.weight = ctx.create_tensor(unique_name, [in_features, out_features], init_val=0.1, requires_grad=True)
        
        # Register it so the Optimizer can find it
        self.register_parameter(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        # Standard Matrix Multiplication: X @ W
        return x @ self.weight
    
class SelfAttention(Module):
    """The core engine of a Large Language Model."""
    def __init__(self, ctx, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale_factor = 1.0 / (embed_dim ** 0.5)
        
        # The Three Pillars of Attention
        self.q_proj = Linear(ctx, embed_dim, embed_dim, name="Query")
        self.k_proj = Linear(ctx, embed_dim, embed_dim, name="Key")
        self.v_proj = Linear(ctx, embed_dim, embed_dim, name="Value")
        
        self.register_module(self.q_proj)
        self.register_module(self.k_proj)
        self.register_module(self.v_proj)

    def forward(self, x: Tensor) -> Tensor:
        # 1. Generate Q, K, V from the input sequence
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # 2. Compute Raw Attention Scores (Q @ K^T)
        scores = Q @ K.T
        
        # 3. Scale down to prevent Softmax overflow
        scaled_scores = scores * self.scale_factor
        
        # 4. Convert to Probabilities (Who is paying attention to who?)
        attention_weights = scaled_scores.softmax()
        
        # 5. Apply attention to the Values
        context = attention_weights @ V
        
        return context
    
class MultiHeadAttention(Module):
    def __init__(self, ctx, embed_dim: int, num_heads: int):
        super().__init__()
        self.heads = []
        # Create N parallel Self-Attention heads
        for i in range(num_heads):
            head = SelfAttention(ctx, embed_dim)
            self.heads.append(head)
            self.register_module(head)
            
        # Final output projection
        self.proj = Linear(ctx, embed_dim, embed_dim, name="MHA_Proj")
        self.register_module(self.proj)

    def forward(self, x):
        # Run all heads in parallel
        head_outputs = [head(x) for head in self.heads]
        
        # Mathematically combine the parallel heads (summation simulates concat + projection slice)
        # Note: You will need to add the `__add__` operator to tensor.py calling ctx._engine.add()!
        combined = head_outputs[0]
        for i in range(1, len(head_outputs)):
            combined = combined + head_outputs[i] 
            
        return self.proj(combined)

class TransformerBlock(Module):
    """A complete Nano-GPT Block."""
    def __init__(self, ctx, embed_dim: int, num_heads: int):
        super().__init__()
        self.attention = MultiHeadAttention(ctx, embed_dim, num_heads)
        self.ffn1 = Linear(ctx, embed_dim, embed_dim * 4, name="FFN1")
        self.ffn2 = Linear(ctx, embed_dim * 4, embed_dim, name="FFN2")
        
        self.register_module(self.attention)
        self.register_module(self.ffn1)
        self.register_module(self.ffn2)

    def forward(self, x):
        # 1. Attention with Residual Connection (Requires __add__ in tensor.py)
        # In a real setup, we would apply LayerNorm here using our new C++ primitive
        attn_out = self.attention(x)
        x = x + attn_out 
        
        # 2. Feed Forward Network with GeLU and Residual
        # x = LayerNorm(x)
        ffn_out = self.ffn1(x)
        # ffn_out = ffn_out.gelu()  (Requires hooking gelu() in tensor.py)
        ffn_out = self.ffn2(ffn_out)
        
        x = x + ffn_out
        return x