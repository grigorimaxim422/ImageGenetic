import torch
import time
import logging
from torchinfo import summary

def check(net):
    logging.info("----------------------------------------------")
    summary(net, input_size=(1, 3, 32, 32), col_names=("input_size", "output_size", "num_params", "mult_adds"))
    logging.info("----------------------------------------------")
    
    
# Load the TorchScript model
model = torch.jit.load("../saved_model/05-148.pt", map_location="cpu")
check(model)
time.sleep(5)
# Define output files
output_dir = "05-148/"
import os
os.makedirs(output_dir, exist_ok=True)

# 1. Save the TorchScript Python-like code
with open(os.path.join(output_dir, "model_code.txt"), "w") as f:
    f.write(model.code)
print("✔ Model code saved to model_code.txt")

# 2. Save the Inlined Graph
with open(os.path.join(output_dir, "inlined_graph.txt"), "w") as f:
    f.write(str(model.inlined_graph))
print("✔ Inlined graph saved to inlined_graph.txt")

# 3. Save the Complete Graph
with open(os.path.join(output_dir, "graph.txt"), "w") as f:
    f.write(str(model.graph))
print("✔ Model graph saved to graph.txt")

with open(os.path.join(output_dir, "children.txt"), "w") as f:
    f.write(str(model.children))
print("✔ Model children saved to children.txt")

# 4. Save Model Parameters
with open(os.path.join(output_dir, "model_parameters.txt"), "w") as f:
    for name, param in model.named_parameters():
        f.write(f"{name}: {param.shape}\n")
print("✔ Model parameters saved to model_parameters.txt")

# 5. Save Model Architecture (Layers & Details)
with open(os.path.join(output_dir, "model_architecture.txt"), "w") as f:
    for name, module in model.named_modules():
        f.write(f"Module: {name}, Type: {module.__class__.__name__}\n")
print("✔ Model architecture saved to model_architecture.txt")

# # 6. Save the Op names
# with open(os.path.join(output_dir, "opnames.txt"), "w") as f:
#     f.write(str(model.export_opnames))
# print("✔ Model opnames saved to opnames.txt")
import os
def extract_detailed_architecture02():

    # Define output file
    output_file = "detailed_model_architecture02.txt"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)


    with open(output_path, "w") as f:
        f.write("### TorchScript Model Architecture ###\n\n")

        # Iterate through named modules
        for name, module in model.named_modules():
            if name == "":  # Skip the top-level module itself
                continue
            
            # Get the actual class name (instead of RecursiveScriptModule)
            layer_type = module.original_name if hasattr(module, "original_name") else module.__class__.__name__
            
            f.write(f"Module: {name}\n")
            f.write(f"  Type: {layer_type}\n")

            # Get parameter count
            param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            f.write(f"  Trainable Parameters: {param_count}\n")

            # Extract layer attributes if available
            if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                f.write(f"  Input Features: {module.in_features}, Output Features: {module.out_features}\n")
            if hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
                f.write(f"  In Channels: {module.in_channels}, Out Channels: {module.out_channels}\n")
            if hasattr(module, 'kernel_size'):
                f.write(f"  Kernel Size: {module.kernel_size}\n")
            if hasattr(module, 'stride'):
                f.write(f"  Stride: {module.stride}\n")
            if hasattr(module, 'padding'):
                f.write(f"  Padding: {module.padding}\n")
            
            f.write("\n")

    print(f"✔ Detailed model architecture saved to {output_path}")

    
extract_detailed_architecture02()