import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
import os
import shutil

from axisnetworks import *
device = torch.device('cuda')
from dataset_3d import *

from SDFs.dataset_wzj import wzjData

# if os.path.exists('decoder_net_ckpt/latest'):
#     pass
# else:
#     os.makedirs('decoder_net_ckpt/latest')

if os.path.exists('decoder_net_ckpt/latest'):
    os.system('rm -rf decoder_net_ckpt/latest')


dataset = wzjData('SDFs/data')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)


# num = len(os.listdir('SDFs/data'))
# num = 1
model = MultiTriplane(1, input_dim=3, output_dim=1, share=False).to(device)
# model.net.load_state_dict(torch.load('/home/wzj/data/project/NFD/nfd/triplane_decoder/decoder_net_ckpt/2024-04-16-20:44:23/600_decoder.pt'))

# model.embeddings.load_state_dict(torch.load('/home/wzj/data/project/NFD/nfd/triplane_decoder/decoder_net_ckpt/2024-04-16-20:44:23/triplanes_600.pt'))
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
os.system(f'ln -s ./{formatted_datetime} decoder_net_ckpt/latest')
t0 = time.time()
for epoch in range(1, 15001):
    model.change_stage() if epoch == 2001 and model.share else None
    loss_total = 0
    optimizer.zero_grad()

    for obj_idx, X, normals, sample in tqdm(dataloader, desc=f'epoch {epoch}'):
    # for obj_idx, X, X_1, truth, normals, sample in dataloader:
        # obj_idx = 0
        # model.grad(obj_idx)
        # X, Y = X.float().cuda(), Y.float().cuda()
        # X, X_n, truth, normals, sample = X.float().cuda(), X_n.float().cuda(), truth.float().cuda(), normals.float().cuda(), sample.float().cuda()
        # T = torch.cat([X, sample], dim=1)
        # X_a, X_b = X[:, :X.shape[1] // 3, :], X[:, X.shape[1] // 3:, :]
        # if epoch == 2001:
            # print('before:', model.embeddings[0]) if obj_idx == 0 else None
            
            # model.update_para(obj_idx)
            # print('after:', model.embeddings[0]) if obj_idx == 0 else None
        T = torch.cat((X, sample), dim=1)
        T.requires_grad_()
        # X_a.requires_grad_()
        # sample.requires_grad_()
        preds = model(obj_idx, T)
        # preds_b = model(obj_idx, X_b)
        # loss = nn.BCEWithLogitsLoss()(preds, Y)

        # Done: 修改sdf的loss
        loss = 2 * ((preds[:, :X.shape[1], :]).abs()).mean()

        
        X_grad = gradient(T, preds)

        T.requires_grad_(False)
        # normals = normals.view(-1, 3)
        # X.requires_grad_(False)
        normals_loss = (X_grad[:, :X.shape[1], :].mul(normals).sum(dim=-1) - 1).abs().mean()
        # # # TODO: 加normals_loss 的系数
        loss = loss + 0.3 * normals_loss



        # sdf reg
        # pred_reg = model(obj_idx, reg_X)
        loss = loss + 0.01 * torch.exp(-1e1 * (preds[:, X.shape[1]:, :]).abs()).mean()

        # TODO: 加随机采样点，使其法向（也就是梯度）的模长唯一作为eikonal loss
        # sample_preds = model(obj_idx, sample)
        # sample.requires_grad_()
        # sample_grad = gradient(sample, sample_preds)
        eikonal_loss = ((X_grad.norm(2, dim=-1) - 1).abs()).mean()
        loss += 0.1 * eikonal_loss

        
        # # DENSITY REG
        # rand_coords = torch.rand_like(X) * 2 - 1
        # rand_coords_offset = rand_coords + torch.randn_like(rand_coords) * 1e-2
        # d_rand_coords = model(obj_idx, rand_coords)
        # d_rand_coords_offset = model(obj_idx, rand_coords_offset)
        # loss += nn.functional.mse_loss(d_rand_coords, d_rand_coords_offset) * 2e-1

        loss += model.tvreg(obj_idx) * 1e-2
        loss += model.l2reg(obj_idx) * 1e-3
        
        # X.requires_grad_(False)
        # sample.requires_grad_(False)
        
        loss.backward()

        # # print(model.embeddings[0].grad)

        # optimizer.zero_grad()

        # step += 1
        # # if step%50 == 0: print(loss.item())

        loss_total += loss
        print(loss) if obj_idx == 0 else None
        # print(obj_idx)
        # print(f'Epoch: {epoch} , idx: {obj_idx} Done')
        # continue
    optimizer.step()
    
    t1 = time.time()
    time1 = t1 - t0
    elapsed_rounded = int(round(time1))
    time1 = str(datetime.timedelta(seconds=elapsed_rounded))
    print(f"Epoch: {epoch} \t {loss_total.item():01f} \t normal_loss:{normals_loss} \t eikonal_loss:{eikonal_loss} \t time:{time1} \t")
    
    if epoch%100 == 0:
        supply = 'share' if epoch < 2001 and model.share else ''
        torch.save(model.net.state_dict(), f"decoder_net_ckpt/{formatted_datetime}/{supply}{epoch}_decoder.pt")
        print(model.embeddings.state_dict().keys())
        torch.save(model.embeddings.state_dict(), f"decoder_net_ckpt/{formatted_datetime}/" + supply + f"triplanes_{epoch}.pt")

        # torch.save(model.net.state_dict(), f"decoder_net_ckpt/latest/{epoch}_decoder.pt")

        # torch.save(model.embeddings.state_dict(), f"decoder_net_ckpt/latest/"+f"triplanes_{epoch}.pt")


