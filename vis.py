import torch
import matplotlib.pyplot as plt
def draw():
    return 0
def vis():
    loaded_tensor = torch.load('/home/wrl/deit/output/feature/dropinput.pt') 
    after_tensor = torch.load('/home/wrl/deit/output/feature/afterdrop.pt') 
    B,N,C = loaded_tensor.shape
    print(loaded_tensor.shape, after_tensor.shape)
    token_map = torch.mean(loaded_tensor, 2)
    print(token_map.shape)
    cls_token = token_map[:, 0]
    content_token = token_map[:, 1:].reshape((B, 32,32))
    print(content_token.shape)
    return 0

if __name__ == '__main__':
    print('start vis')
    vis()