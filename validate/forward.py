import re
import onnx_tool
import torch.onnx
from src.protocol import Dummy
from src.validator.reward import get_rewards
from src.utils.uids import get_random_uids
import torch
from datetime import datetime
import pandas as pd
import requests
import asyncio

import traceback
import plotly.graph_objects as go
import wandb
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
