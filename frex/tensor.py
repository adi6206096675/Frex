import uuid

class Tensor:
    def __init__(self, context, name: str, shape: list, init_val: float = 0.0, requires_grad: bool = False, creator=None):
        self.context = context
        self.name = name
        self.shape = shape
        self.requires_grad = requires_grad
        self.creator = creator 
        
        self.context._engine.allocate_tensor(self.name, self.shape, 4, self.requires_grad)
        if init_val != 0.0:
            self.context._engine.fill_tensor(self.name, init_val, False)
        
        if self.requires_grad:
            self.context._engine.fill_tensor(self.name, 0.0, True)

    @property
    def T(self):
        out_shape = [self.shape[1], self.shape[0]]
        out_name = f"transpose_{self.name}_{uuid.uuid4().hex[:8]}"
        out_requires_grad = self.requires_grad
        
        out_tensor = Tensor(self.context, out_name, out_shape, 0.0, out_requires_grad, creator=TransposeBackward(self))
        self.context._engine.transpose(self.name, out_name)
        return out_tensor

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("Frex can only multiply with another Frex Tensor.")
        
        out_shape = [self.shape[0], other.shape[1]]
        out_name = f"matmul_{self.name}_x_{other.name}_{uuid.uuid4().hex[:8]}"
        
        out_requires_grad = self.requires_grad or other.requires_grad
        creator = MatMulBackward(self, other) if out_requires_grad else None
        
        out_tensor = Tensor(self.context, out_name, out_shape, 0.0, out_requires_grad, creator=creator)
        self.context._engine.matmul(self.name, other.name, out_name)
        return out_tensor

    def __mul__(self, scalar: float):
        if not isinstance(scalar, (int, float)):
            raise TypeError("Frex only supports scaling by a standard number right now.")
            
        out_name = f"scale_{self.name}_{uuid.uuid4().hex[:8]}"
        out_requires_grad = self.requires_grad
        creator = ScaleBackward(self, scalar) if out_requires_grad else None
        
        out_tensor = Tensor(self.context, out_name, self.shape, 0.0, out_requires_grad, creator=creator)
        self.context._engine.scale(self.name, float(scalar), out_name)
        return out_tensor

    def softmax(self):
        out_name = f"softmax_{self.name}_{uuid.uuid4().hex[:8]}"
        out_requires_grad = self.requires_grad
        
        out_tensor = Tensor(self.context, out_name, self.shape, 0.0, out_requires_grad)
        self.context._engine.softmax(self.name, out_name)
        return out_tensor

    def linear_relu(self, weight, bias):
        out_shape = [self.shape[0], weight.shape[1]]
        out_name = f"fused_linear_{uuid.uuid4().hex[:8]}"
        out_requires_grad = self.requires_grad or weight.requires_grad or bias.requires_grad
        
        out_tensor = Tensor(self.context, out_name, out_shape, 0.0, out_requires_grad)
        self.context._engine.fused_linear_relu(self.name, weight.name, bias.name, out_name)
        return out_tensor

    def backward(self, seed_grad=1.0):
        if not self.requires_grad:
            raise RuntimeError("Called .backward() on a tensor that doesn't track gradients.")
        
        self.context._engine.fill_tensor(self.name, seed_grad, True)
        
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                if v.creator is not None:
                    for parent in v.creator.inputs:
                        build_topo(parent)
                    topo.append(v)
                    
        build_topo(self)
        
        for v in reversed(topo):
            if v.creator is not None:
                v.creator.backward(v)

    def item(self, row: int = 0, col: int = 0):
        linear_index = row * self.shape[1] + col
        return self.context._engine.read_value(self.name, linear_index, False)
        
    def grad(self, row: int = 0, col: int = 0):
        if not self.requires_grad:
            raise RuntimeError(f"Tensor {self.name} does not require gradients.")
        linear_index = row * self.shape[1] + col
        return self.context._engine.read_value(self.name, linear_index, True)

    def __del__(self):
        try:
            self.context.delete(self.name)
        except Exception:
            pass

    def __repr__(self):
        return f"FrexTensor '{self.name}' | Shape: {self.shape} | requires_grad: {self.requires_grad}"


# --- GRAPH NODES ---

class MatMulBackward:
    def __init__(self, X, W):
        self.inputs = [X, W]
        self.X = X
        self.W = W

    def backward(self, output_tensor):
        eng = self.X.context._engine
        
        # UNIQUE TEMP NAMES
        dY_name = f"tmp_dY_{uuid.uuid4().hex[:8]}"
        eng.allocate_tensor(dY_name, output_tensor.shape, 4, False)
        eng.extract_grad(output_tensor.name, dY_name)

        if self.X.requires_grad:
            WT_name = f"tmp_WT_{uuid.uuid4().hex[:8]}"
            dX_name = f"tmp_dX_{uuid.uuid4().hex[:8]}"
            
            eng.allocate_tensor(WT_name, [self.W.shape[1], self.W.shape[0]], 4, False)
            eng.allocate_tensor(dX_name, self.X.shape, 4, False)
            
            eng.transpose(self.W.name, WT_name)
            eng.matmul(dY_name, WT_name, dX_name)
            eng.accumulate_grad(dX_name, self.X.name)
            
            eng.free_tensor(WT_name)
            eng.free_tensor(dX_name)

        if self.W.requires_grad:
            XT_name = f"tmp_XT_{uuid.uuid4().hex[:8]}"
            dW_name = f"tmp_dW_{uuid.uuid4().hex[:8]}"
            
            eng.allocate_tensor(XT_name, [self.X.shape[1], self.X.shape[0]], 4, False)
            eng.allocate_tensor(dW_name, self.W.shape, 4, False)
            
            eng.transpose(self.X.name, XT_name)
            eng.matmul(XT_name, dY_name, dW_name)
            eng.accumulate_grad(dW_name, self.W.name)
            
            eng.free_tensor(XT_name)
            eng.free_tensor(dW_name)
            
        eng.free_tensor(dY_name)

class TransposeBackward:
    def __init__(self, in_tensor):
        self.inputs = [in_tensor]
        self.in_tensor = in_tensor

    def backward(self, output_tensor):
        if self.in_tensor.requires_grad:
            eng = self.in_tensor.context._engine
            
            dY_name = f"tmp_dY_trans_{uuid.uuid4().hex[:8]}"
            eng.allocate_tensor(dY_name, output_tensor.shape, 4, False)
            eng.extract_grad(output_tensor.name, dY_name)
            
            dX_name = f"tmp_dX_trans_{uuid.uuid4().hex[:8]}"
            eng.allocate_tensor(dX_name, self.in_tensor.shape, 4, False)
            
            eng.transpose(dY_name, dX_name)
            eng.accumulate_grad(dX_name, self.in_tensor.name)
            
            eng.free_tensor(dY_name)
            eng.free_tensor(dX_name)

class ScaleBackward:
    def __init__(self, in_tensor, scalar):
        self.inputs = [in_tensor]
        self.in_tensor = in_tensor
        self.scalar = scalar

    def backward(self, output_tensor):
        if self.in_tensor.requires_grad:
            eng = self.in_tensor.context._engine
            
            dY_name = f"tmp_dY_scale_{uuid.uuid4().hex[:8]}"
            eng.allocate_tensor(dY_name, output_tensor.shape, 4, False)
            eng.extract_grad(output_tensor.name, dY_name)

            dX_name = f"tmp_dX_scale_{uuid.uuid4().hex[:8]}"
            eng.allocate_tensor(dX_name, self.in_tensor.shape, 4, False)
            
            # The gradient is just the incoming gradient multiplied by the scalar!
            eng.scale(dY_name, self.scalar, dX_name)
            eng.accumulate_grad(dX_name, self.in_tensor.name)
            
            eng.free_tensor(dY_name)
            eng.free_tensor(dX_name)