import torch
from torch import nn

#
def demo01():

    y_true = torch.tensor([1,2,3,4],dtype=torch.float32)
    y_pred = torch.tensor([0.5,1.5,2.5,3.5],dtype=torch.float32)

    # MAE LOSS
    mae_loss_fn = nn.L1Loss()
    mae_loss = mae_loss_fn(y_pred,y_true)
    print(mae_loss)
    # MSE LOSS
    l2_loss_fn = nn.MSELoss()
    l2_loss = l2_loss_fn(y_pred, y_true)
    print(l2_loss)
    # SmoothL1Loss
    smooth_l_loss_fn = nn.SmoothL1Loss()
    smooth_l_loss = smooth_l_loss_fn(y_pred, y_true)
    print(smooth_l_loss)

if __name__ == '__main__':
    demo01()
