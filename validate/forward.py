import re
import onnx_tool
import torch.onnx
import torch
from datetime import datetime
import pandas as pd
import requests
import asyncio

import traceback
import plotly.graph_objects as go
import os
from torch.profiler import profile, record_function, ProfilerActivity
import math
import numpy as np
from requests.exceptions import ReadTimeout  
import gc
import onnx

def round_to_nearest_significant(x, n=2):
    if x == 0:
        return 0
    else:
        magnitude = int(math.floor(math.log10(abs(x))))
        factor = 10 ** magnitude
        return round(x / factor, n-1) * factor
    
def round_flops_to_nearest_significant(flops):
    if flops == 0:
        return 0
    else:
        # Determine the number of digits in the integer part of the FLOPs value
        num_digits = len(str(int(abs(flops))))

        # Decide whether to round to 1 or 2 significant digits
        if num_digits <= 7:
            n = 1
        else:
            n = 2

        # Calculate the magnitude and factor for rounding
        magnitude = int(math.floor(math.log10(abs(flops))))
        factor = 10 ** magnitude

        return round(flops / factor, n-1) * factor


def calc_flops_onnx(model):
    fixed_dummy_input = torch.randn(1, 3, 32, 32).cuda()
    onnx_path = "cache/tmp.onnx"
    profile_path = "cache/profile.txt"
    torch.onnx.export(model,
                  fixed_dummy_input,
                  onnx_path,
                  export_params=True,
                  opset_version=12,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes=None)  # No dynamic axes for profiling
    onnx_tool.model_profile(onnx_path,save_profile=profile_path)
    with open(profile_path, 'r') as file:
        profile_content = file.read()

    # Regular expression to find the total Forward MACs
    match = re.search(r'Total\s+_\s+([\d,]+)\s+100%', profile_content)

    if match:
        total_macs = match.group(1)
        total_macs = int(total_macs.replace(',', ''))
        total_macs = round_flops_to_nearest_significant(total_macs)
    return total_macs

