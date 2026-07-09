class SGD:
    def __init__(self, parameters, lr=0.01):
        """Initializes the Stochastic Gradient Descent optimizer."""
        # Only optimize tensors that actually require gradients (like weights and biases)
        self.parameters = [p for p in parameters if p.requires_grad]
        self.lr = lr

    def step(self):
        """Pushes the weights down the gradient slope to minimize error."""
        for p in self.parameters:
            p.context._engine.sgd_step(p.name, self.lr)

    def zero_grad(self):
        """Clears the gradients for the next training iteration."""
        for p in self.parameters:
            p.context._engine.zero_grad(p.name)