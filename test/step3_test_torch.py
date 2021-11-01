import torch
import config
import numpy as np
from reprod_log import ReprodLogger
import os
import pickle
import random
import numpy as np
import csv
from modules.CASENet import CASENet_resnet101
import utils.utils as utils
from torch import sigmoid
#------------------------------------------
def WeightedMultiLabelSigmoidLoss(model_output, target):
    """
    model_output: BS X NUM_CLASSES X H X W
    target: BS X H X W X NUM_CLASSES
    """
    # Calculate weight. (edge pixel and non-edge pixel)
    weight_sum = utils.check_gpu(0, target.sum(dim=1).sum(dim=1).sum(dim=1).float().data)  # BS
    edge_weight = utils.check_gpu(0, weight_sum.data / float(target.size()[1] * target.size()[2]))
    non_edge_weight = utils.check_gpu(0, (target.size()[1] * target.size()[2] - weight_sum.data) / float(
        target.size()[1] * target.size()[2]))
    one_sigmoid_out = sigmoid(model_output)
    zero_sigmoid_out = 1 - one_sigmoid_out
    target = target.transpose(1, 3).transpose(2, 3).float()  # BS X NUM_CLASSES X H X W
    loss = -non_edge_weight.unsqueeze(1).unsqueeze(2).unsqueeze(3) * target * torch.log(
        one_sigmoid_out.clamp(min=1e-10)) - \
           edge_weight.unsqueeze(1).unsqueeze(2).unsqueeze(3) * (1 - target) * torch.log(
        zero_sigmoid_out.clamp(min=1e-10))

    return loss.mean(dim=0).sum()

reprod_logger = ReprodLogger()
args = config.get_args()
model = CASENet_resnet101(pretrained=False, num_classes=args.cls_num)
device = torch.device('cuda:0')
model.to(device)
model.eval()
weight_dict = torch.load('model_casenet.pth.tar')
model.load_state_dict(weight_dict['state_dict'])
fake_data = np.load("fake_data.npy")
fake_data = torch.tensor(fake_data)
fake_data=fake_data.to(device)
fake_label = np.load("fake_label.npy")
fake_label = torch.tensor(fake_label)
fake_label=fake_label.to(device)
score_feats5, fused_feats = model(fake_data)  # BS X NUM_CLASSES X 472 X 472

feats5_loss = WeightedMultiLabelSigmoidLoss(score_feats5, fake_label)
fused_feats_loss = WeightedMultiLabelSigmoidLoss(fused_feats, fake_label)
loss = feats5_loss + fused_feats_loss
print("loss = ",loss)
reprod_logger.add("loss", loss.cpu().detach().numpy())
reprod_logger.save("loss_torch.npy")
