import paddle
import numpy as np
import config
from reprod_log import ReprodLogger
from CASENet_paddle import CASENet_resnet101
reprod_logger = ReprodLogger()
args = config.get_args()
model = CASENet_resnet101(pretrained=False, num_classes=args.cls_num)
model.eval()

weight_dict = paddle.load('paddle_weight.pdparams')
model.set_state_dict(weight_dict)
fake_data = np.load("fake_data.npy")
fake_data = paddle.to_tensor(fake_data)

out1,out2 = model(fake_data)
print("out1",out1)
print("out2",out2)
reprod_logger.add("logits1", out1.cpu().detach().numpy())
reprod_logger.add("logits2", out2.cpu().detach().numpy())
reprod_logger.save("forward_paddle.npy")

