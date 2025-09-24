r""" Helper functions """
import random

import torch
import numpy as np
import pytorch_lightning as pl


def fix_randseed(seed):
    r""" Set random seeds for reproducibility """
    if seed is None:
        seed = int(random.random() * 1e5)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def to_cpu(tensor):
    return tensor.detach().clone().cpu()

def  print_param_count(model):
    total_number = 0
    learnable_number = 0 
    for para in model.parameters():
        total_number += torch.numel(para)
        if para.requires_grad == True:
            learnable_number+= torch.numel(para)
    print(f"total_params: {total_number:,}\nlearnable params: {learnable_number:,}")

# def print_param_count(model):
#     backbone_param = 0
#     middle_param = 0
#     decoder_param = 0
#     BAM_param = 0

#     s = set()
#     for k in model.state_dict().keys():
#         s.add(k.split('.')[0])
#     print(s)
#     s.pop()
    

#     for k in model.state_dict().keys():
#         n_param = model.state_dict()[k].view(-1).size(0)
#         if k.split('.')[0] in ['PSPNet_']:
#             backbone_param += n_param
#         elif k.split('.')[0] in ['init_merge', 'down_query', 'down_supp']:
#             middle_param += n_param        
#         elif k.split('.')[0] in ['ASPP_meta', 'res1_meta', 'res2_meta', 'cls_meta']:
#             decoder_param += n_param
#         elif k.split('.')[0] in ['learner_base', 'gram_merge', 'cls_merge']:
#             BAM_param += n_param
#         else:
#             raise Exception(f"Wrong layer name {k.split('.')[0]}s")

#     msg = f'Backbone # param.: {backbone_param:,}\n'
#     msg += f'MiddleLayer # param.: {middle_param:,}\n'
#     msg += f'Decoder # param.: {decoder_param:,}\n'
#     msg += f'BAM # param.: {BAM_param:,}\n'
#     msg += f'Learnable # param.: {middle_param + decoder_param + BAM_param:,}\n'
#     print(msg)
