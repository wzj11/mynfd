import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
import os

from axisnetworks import *
device = torch.device('cuda')
from dataset_3d import *

from SDFs.dataset_wzj import wzjData


dataset = wzjData('SDFs/data')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)


num = len(os.listdir('SDFs/data'))
num = 1
model = MultiTriplane(num, input_dim=3, output_dim=1).to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters())

# print('yes')
losses = []

step = 0
# if os.path.exists('decoder_net_ckpt/'):
#     pass
# else:
#     os.makedirs('decoder_net_ckpt/')
current_datetime = datetime.datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H:%M:%S")
os.makedirs(f'decoder_net_ckpt/{formatted_datetime}')
for epoch in range(30000):
    loss_total = 0
    for obj_idx, X, normals, sample in dataloader:
        # X, Y = X.float().cuda(), Y.float().cuda()
        X, normals, sample = X.float().cuda(), normals.float().cuda(), sample.float().cuda()
        X.requires_grad_()
        sample.requires_grad_()
        preds = model(obj_idx, X)
        # loss = nn.BCEWithLogitsLoss()(preds, Y)

        # Done: 修改sdf的loss
        loss = (preds.abs()).mean()

        
        X_grad = gradient(X, preds)


        # normals = normals.view(-1, 3)
        # X.requires_grad_(False)
        normals_loss = ((X_grad - normals).abs()).norm(2, dim=-1).mean()
        # # TODO: 加normals_loss 的系数
        loss = loss + 0.1 * normals_loss

        # TODO: 加随机采样点，使其法向（也就是梯度）的模长唯一作为eikonal loss
        sample_preds = model(obj_idx, sample)
        # sample.requires_grad_()
        sample_grad = gradient(sample, sample_preds)
        eikonal_loss = ((sample_grad.norm(2, dim=-1) - 1)**2).mean()
        loss += 0.1 * eikonal_loss

        
        # # # DENSITY REG
        # rand_coords = torch.rand_like(X) * 2 - 1
        # rand_coords_offset = rand_coords + torch.randn_like(rand_coords) * 1e-2
        # d_rand_coords = model(obj_idx, rand_coords)
        # d_rand_coords_offset = model(obj_idx, rand_coords_offset)
        # loss += nn.functional.mse_loss(d_rand_coords, d_rand_coords_offset) * 3e-1

        loss += model.tvreg() * 1e-2
        loss += model.l2reg() * 1e-3
        
        X.requires_grad_(False)
        sample.requires_grad_(False)
        optimizer.zero_grad()
        loss.backward()

        # print(model.embeddings[0].grad)
        optimizer.step()


        step += 1
        # if step%50 == 0: print(loss.item())

        loss_total += loss
        # print(f'Epoch: {epoch} , idx: {obj_idx} Done')
    print(f"Epoch: {epoch} \t {loss_total.item():01f} \t 'normals_loss:'{normals_loss.item()} \t eikonal_loss:{eikonal_loss.item()} \t ")
    
    if epoch%20 == 0:
        torch.save(model.net.state_dict(), f"decoder_net_ckpt/{formatted_datetime}/{epoch}_decoder.pt")

        torch.save(model.embeddings.state_dict(), f"decoder_net_ckpt/{formatted_datetime}/"+f"triplanes_{epoch}.pt")


