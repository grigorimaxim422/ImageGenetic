import time
import torch
import asyncio
import threading
import argparse
import traceback
import os
import sys
import logging

import argparse
import requests
import datetime as dt
from model.dummy_trainer import DummyTrainer

from model.offline_vali_trainer import OfflineValiTrainer
from validator.forward import  *

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
args = parser.parse_args()
    
async def main():
    try:   
        upload_dir = "saved_model"
        save_dir = upload_dir
                
        trainer = DummyTrainer(epochs=args.epochs)
        trainer.train()
        model = trainer.get_model()    
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        if not os.path.exists("cache"):
            os.makedirs("cache")
            
            
        save_path = os.path.join(save_dir, 'model.pt')
        scripted_model = torch.jit.script(model)
        scripted_model.save(save_path)
        print(f"model saved into {save_path}")
        
        params = sum(param.numel() for param in model.parameters())
        params = round_to_nearest_significant(params, 1)
        macs = calc_flops_onnx(model)
        print(f"A.🖥️ Params: {params/1000}K")    
        print(f"A.🖥️ Flops: {macs/1000000}M")    
        await asyncio.sleep(5)
        
        model = torch.jit.load(save_path)
        trainer = OfflineValiTrainer(epochs=50, learning_rate=0.025)    
        trainer.initialize_weights(model)
        retrained_model = trainer.train(model)
        accuracy = math.floor(trainer.test(retrained_model))            
        retrained_params = sum(param.numel() for param in retrained_model.parameters())     
        retrained_params = round_to_nearest_significant(retrained_params, 1)
        retrained_macs = calc_flops_onnx(retrained_model)
        
        print(f"B.🖥️ Accuracy: {accuracy}")    
        print(f"B.🖥️ Params: {retrained_params/1000}K")    
        print(f"B.🖥️ Flops: {retrained_macs/1000000}M")    
    except Exception as e:
        logging.error(f"Failed to advertise model on the chain: {e}")

if __name__ == '__main__':        
    asyncio.run(main())