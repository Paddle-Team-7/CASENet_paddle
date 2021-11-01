import config
import torch
import numpy as np
from reprod_log import ReprodLogger
from modules.CASENet import CASENet_resnet101
reprod_logger = ReprodLogger()
args = config.get_args()
model = CASENet_resnet101(pretrained=False, num_classes=args.cls_num)
model.eval()

weight_dict = torch.load('model_casenet.pth.tar')
model.load_state_dict(weight_dict['state_dict'])
fake_data = np.load("fake_data.npy")
fake_data = torch.Tensor(fake_data)

out1,out2 = model(fake_data)
print("out1",out1.shape)
print("out2",out2.shape)
reprod_logger.add("logits1", out1.cpu().detach().numpy())
reprod_logger.add("logits2", out2.cpu().detach().numpy())
reprod_logger.save("forward_torch.npy")