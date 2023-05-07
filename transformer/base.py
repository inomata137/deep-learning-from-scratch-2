class Layer:
    def __init__(self):
        self.params, self.grads = [], []
        return None
    def forward(self, x):
        return x
    def backward(self, dout):
        return dout