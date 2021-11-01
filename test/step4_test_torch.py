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
from torch.autograd import Variable


def get_model_policy(model):
    score_feats_conv_weight = []
    score_feats_conv_bias = []
    other_pts = []
    for m in model.named_modules():
        if m[0] != '' and m[0] != 'module':
            if ('score' in m[0] or 'fusion' in m[0]) and isinstance(m[1], torch.nn.Conv2d):
                ps = list(m[1].parameters())
                score_feats_conv_weight.append(ps[0])
                if len(ps) == 2:
                    score_feats_conv_bias.append(ps[1])
                print("Totally new layer:{0}".format(m[0]))
            else: # For all the other module that is not totally new layer.
                ps = list(m[1].parameters())
                other_pts.extend(ps)
    return [
            {'params': score_feats_conv_weight, 'lr_mult': 10, 'name': 'score_conv_weight'},
            {'params': score_feats_conv_bias, 'lr_mult': 20, 'name': 'score_conv_bias'}

    ]

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
fake_data = fake_data.to(device)
fake_label = np.load("fake_label.npy")
fake_label = torch.tensor(fake_label)
fake_label = fake_label.to(device)
policies = get_model_policy(model) # Set the lr_mult=10 of new layer
optimizer = torch.optim.SGD(policies, lr=1e-7, momentum=0.9, weight_decay=1e-4)
loss_list = []

for epoch in range(2):
    img = fake_data
    label = fake_label

    score_feats5, fused_feats = model(img)  # BS X NUM_CLASSES X 472 X 472
    feats5_loss = WeightedMultiLabelSigmoidLoss(score_feats5, label)
    fused_feats_loss = WeightedMultiLabelSigmoidLoss(fused_feats, label)
    loss = feats5_loss + fused_feats_loss
    print("loss = ", loss)
    print("loss.grid",loss)
    loss.backward(retain_graph=True)
    del img
    del label
    del score_feats5
    torch.cuda.empty_cache()
    optimizer.step()
    optimizer.zero_grad()
    loss_list.append(loss)

for idx, loss in enumerate(loss_list):
    reprod_logger.add(f"loss_{idx}", loss.cpu().detach().numpy())
reprod_logger.save('bp_align_torch.npy')
