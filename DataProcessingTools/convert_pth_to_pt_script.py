"""
This code module is just a example for convert .pth model to ScriptModule, 
you need to find an appropriate place in your source model and add this code into your source model
"""

import torch

# Assume you create a model like below MyModule
class MyModule(torch.nn.Module):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))
        self.linear = torch.nn.Linear(N, M)

    def forward(self, input):
        output = self.weight.mv(input)
        output = self.linear(output)
        return output
        
# This will produce a model.pt file

if __name__ == "__main__":
    scripted_module = torch.jit.script(MyModule(2, 3))
    scripted_module.save("MyModule.pt")