import time
import torch
import asyncio
import threading
import argparse
import traceback
import os
import sys
import logging

import requests
import datetime as dt
from model.dummy_trainer import DummyTrainer

from validate.offline_vali_trainer import OfflineValiTrainer
from validate.forward import  *

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--validate_epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--learning_rate', type=float, default=0.025, help='learning rate')
parser.add_argument('--model_path', type=str, default="saved_model/model.pt", help="path of saved torchscript model")

args = parser.parse_args()
    
async def main():
    try:           
        save_dir = os.path.basename(args.model_path)
                
        trainer = DummyTrainer(epochs=args.epochs)
        trainer.train()
        model = trainer.get_model()    
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        if not os.path.exists("cache"):
            os.makedirs("cache")
                                    
        save_path = args.model_path
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
        trainer = OfflineValiTrainer(epochs=args.validate_epochs, learning_rate=args.learning_rate)    
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