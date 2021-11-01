import paddle
import config
import numpy as np
from reprod_log import ReprodLogger
import os
import pickle
import random
import numpy as np
import csv
from CASENet_paddle import CASENet_resnet101
import utils.utils_paddle as utils

def WeightedMultiLabelSigmoidLoss(model_output, target):
    """
    model_output: BS X NUM_CLASSES X H X W
    target: BS X H X W X NUM_CLASSES
    """
    # Calculate weight. (edge pixel and non-edge pixel)
    weight_sum = utils.check_gpu(0, paddle.to_tensor(paddle.sum(paddle.sum(paddle.sum(target,axis=1),axis=1),axis=1),dtype='float32').data)  # BS
    edge_weight = utils.check_gpu(0, weight_sum.data / float(target.size()[1] * target.size()[2]))
    non_edge_weight = utils.check_gpu(0, (target.size()[1] * target.size()[2] - weight_sum.data) / float(
        target.size()[1] * target.size()[2]))
    # one_sigmoid_out = sigmoid(model_output)
    one_sigmoid_out = paddle.nn.functional.sigmoid(model_output)
    zero_sigmoid_out = 1 - one_sigmoid_out
    #target = paddle.transpose(target,(0, 3, 2, 1).transpose(0, 1, 3, 2).float()  # BS X NUM_CLASSES X H X W
    x1 = paddle.transpose(target,(0,3,2,1))
    x2 = paddle.transpose(x1,(0,1,3,2))
    target = paddle.to_tensor(x2,dtype='float32')
    loss = -non_edge_weight.unsqueeze(1).unsqueeze(2).unsqueeze(3) * target * paddle.log(
        one_sigmoid_out.clip(min=1e-10)) - \
           edge_weight.unsqueeze(1).unsqueeze(2).unsqueeze(3) * (1 - target) * paddle.log(
        zero_sigmoid_out.clip(min=1e-10))

    return paddle.sum(paddle.mean(loss,axis=0))

reprod_logger = ReprodLogger()
args = config.get_args()
model = CASENet_resnet101(pretrained=False, num_classes=args.cls_num)
model.eval()
weight_dict = paddle.load('paddle_weight.pdparams')
model.set_state_dict(weight_dict)
fake_data = np.load("fake_data.npy")
fake_data = paddle.to_tensor(fake_data)
fake_label = np.load("fake_label.npy")
fake_label = paddle.to_tensor(fake_label)
score_feats5, fused_feats = model(fake_data)  # BS X NUM_CLASSES X 472 X 472

feats5_loss = WeightedMultiLabelSigmoidLoss(score_feats5, fake_label)
fused_feats_loss = WeightedMultiLabelSigmoidLoss(fused_feats, fake_label)
loss = feats5_loss + fused_feats_loss
print("loss = ",loss)
reprod_logger.add("loss", loss.cpu().detach().numpy())
reprod_logger.save("loss_paddle.npy")
