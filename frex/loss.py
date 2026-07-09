import uuid
from .tensor import Tensor

class CrossEntropyBackward:
    def __init__(self, logits, probs, targets):
        self.inputs = [logits]
        self.logits = logits
        self.probs = probs
        self.targets = targets

    def backward(self, output_tensor):
        if self.logits.requires_grad:
            eng = self.logits.context._engine
            
            # Unique Name Fix
            dX_name = f"tmp_dX_CE_{uuid.uuid4().hex[:8]}"
            eng.allocate_tensor(dX_name, self.logits.shape, 4, False)
            
            eng.cross_entropy_backward(self.probs.name, self.targets.name, dX_name)
            eng.accumulate_grad(dX_name, self.logits.name)
            
            eng.free_tensor(dX_name)

def cross_entropy_loss(logits, targets):
    probs = logits.softmax()
    
    # Unique Name Fix
    loss_name = f"loss_CE_{uuid.uuid4().hex[:8]}"
    loss_tensor = Tensor(logits.context, loss_name, [1, 1], requires_grad=True, creator=CrossEntropyBackward(logits, probs, targets))
    
    logits.context._engine.cross_entropy(probs.name, targets.name, loss_tensor.name)
    return loss_tensor