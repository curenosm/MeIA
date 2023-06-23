import numpy as np
import torch
import torch.nn as nn

def adversarial_loss(ad_out):
    # Get batch size
    batch_size = ad_out.size(0) // 2
    # Generate labels for source and target domains
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()

    return nn.BCELoss()(ad_out, dc_target)

