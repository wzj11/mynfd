# import torch
# from PIL import Image
# import numpy as np

# test_img = Image.open('test.png')

# image_array = np.array(test_img)

# mask = image_array[..., -1:]

# mask_t = torch.from_numpy(mask)

# H, W, C = mask_t.shape

# print(mask_t.shape)
# # temp = torch.zeros((2*H, 2*W, 3))
# import torch.nn.functional as F

# temp = F.interpolate(mask_t.permute(2, 0, 1), mode='nearest')
# out = temp.repeat(3, -1).numpy()

# image = Image.fromarray(out)
# image.save('wzj.png')

# print(image_array.shape)


    

import torch

class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(3, 3)

    def forward(self, x):
        return self.net(x)

x = torch.randn((1, 3))
x.requires_grad_()

# with torch.no_grad():
l = model()
for param in l.parameters():
    param.requires_grad_(False)
# l = torch.nn.Linear(3, 3)

# with torch.no_grad():
y = l(x)
# y.requires_grad_(False)

# with torch.no_grad():
out = torch.sum(y, dim=-1)

out.backward()

print(l.net.weight, x.grad)