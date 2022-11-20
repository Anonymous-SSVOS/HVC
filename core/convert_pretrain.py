import torch
from collections import OrderedDict

print(f"Loading checkpoint")
state_dict = torch.load('./pretrain_ckpt/hvc.pth', map_location='cpu')

new_ckpt = OrderedDict()


# baseline
for k, v in state_dict['model'].items():
    if k.startswith('online_encoder.'):
        new_v = v
        new_k = k.replace('online_encoder.', '')
        new_ckpt[new_k] = new_v
    
# print(new_ckpt)
torch.save(new_ckpt, './checkpoints/hvc_ytb.pth')

