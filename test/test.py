import sys
sys.path.append('..')
sys.path.append('..')
from src.nonlocalpooling.pool import NonLocalPool2d, NonLocalPool3d
import torch

img = torch.ones([1, 1, 128, 192, 192])

model = NonLocalPool3d([128,192,192],1,2,squeeze=False)
# parameters = filter(lambda p: p.requires_grad, model.parameters())
# parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
# print('Trainable Parameters: %.3fM' % parameters)

out_img = model(img)

print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]

img = torch.ones([1, 128, 192, 192])

model = NonLocalPool2d(192,128,2)
# parameters = filter(lambda p: p.requires_grad, model.parameters())
# parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
# print('Trainable Parameters: %.3fM' % parameters)

out_img = model(img)

print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]

