"""
This code module is just a example for convert .pth model to ScriptModule, 
you need to find an appropriate place in your source model and add this code into your source model
"""

import torch

def convert_pth_to_pt_trace(checkpoint_path):
    # First you need create an instance of your model
    model = your_model()
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)
    model.to(torch.device("cuda"))
    model.eval()

    # You need to creare an example input you would normally provide to
    example = torch.rand(1, 3, 256, 256)

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing
    traced_script_module = torch.jit.trace(model, example)

    # This will produce a model.pt file into defined directory
    traced_script_module.save("your_model.pt")
    print("model saved successfully")

if __name__ == "__main__":
    checkpoint_path = " "
    convert_pth_to_pt_trace(checkpoint_path)
