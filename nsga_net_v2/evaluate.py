import os
import json
import torch
import argparse
import numpy as np

class OFAEvaluator:
    def __init__(self):
        print('OFAEvaluator::()')
        
    @staticmethod
    def save_net_config(path, net, config_name='net.config'):
        """ dump run_config and net_config to the model_folder """
        net_save_path = os.path.join(path, config_name)
        json.dump(net.config, open(net_save_path, 'w'), indent=4)
        print('Network configs dump to %s' % net_save_path)

    @staticmethod
    def save_net(path, net, model_name):
        """ dump net weight as checkpoint """
        if isinstance(net, torch.nn.DataParallel):
            checkpoint = {'state_dict': net.module.state_dict()}
        else:
            checkpoint = {'state_dict': net.state_dict()}
        model_path = os.path.join(path, model_name)
        torch.save(checkpoint, model_path)
        print('Network model dump to %s' % model_path)
    
    @staticmethod
    def save_full_net(path, net, model_name):
        """ dump full net weight as checkpoint """                
        # torch.save(net, model_path)
        model_path = os.path.join(path, model_name)                
        torch.save(net, model_path)
        print('Network model dump to %s' % model_path)
        
    @staticmethod
    def save_full_jit_net(path, net, model_name):
        """ dump full jit net weight as checkpoint """                
        # torch.save(net, model_path)
        model_path = os.path.join(path, model_name)        
        scripted_model = torch.jit.script(net)
        scripted_model.save(model_path)
        print('Network model dump to %s' % model_path)
