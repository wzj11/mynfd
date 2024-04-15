import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import mcubes
from tqdm import tqdm 

from axisnetworks import *
device = torch.device('cuda')
from dataset_3d import *
from matplotlib import pyplot as plt




def cross_section(model, obj_idx, res=512, max_batch_size=50000, axis='z'):
    # Output a cross section of the fitted volume

    xx = torch.linspace(-1, 1, res)
    yy = torch.linspace(-1, 1, res)
    # zz = torch.linspace(-1, 1, res)
    (x_coords, y_coords) = torch.meshgrid([xx, yy])
    z_coords = torch.zeros_like(x_coords)
    coords = torch.cat([x_coords.unsqueeze(-1), y_coords.unsqueeze(-1), z_coords.unsqueeze(-1)], -1)
    # coords = torch.cat([x_coords.unsqueeze(-1), z_coords.unsqueeze(-1), y_coords.unsqueeze(-1)], -1)

    coords = coords.reshape(res*res, 3)
    prediction = torch.zeros(coords.shape[0], 1)

    with torch.no_grad():
        head = 0
        while head < coords.shape[0]:
            prediction[head:head+max_batch_size] = model(obj_idx, coords[head:head+max_batch_size].to(device).unsqueeze(0)).cpu()
            head += max_batch_size
            
    prediction = (prediction > 0).cpu().numpy().astype(np.uint8)
    prediction = prediction.reshape(res, res)
    plt.figure(figsize=(16, 16))
    plt.imshow(prediction)

def create_obj(model, obj_idx, res=256, max_batch_size=200, output_path='output.obj'):
    # Output a res x res x res x 1 volume prediction. Download ChimeraX to open the files.
    # Set the threshold in ChimeraX to 0.5 if mrc_mode=0, 0 else


    # 先确定要进行预测的空间网格划分
    model.eval()
    xx = torch.linspace(-1, 1, res)
    yy = torch.linspace(-1, 1, res)
    zz = torch.linspace(-1, 1, res)

    # 得到坐标
    (x_coords, y_coords, z_coords) = torch.meshgrid([xx, yy, zz])
    coords = torch.cat([x_coords.unsqueeze(-1), y_coords.unsqueeze(-1), z_coords.unsqueeze(-1)], -1)

    coords = coords.reshape(res*res*res, 3)
    prediction = torch.zeros(coords.shape[0], 1)
    
    with tqdm(total = coords.shape[0]) as pbar:
        with torch.no_grad():
            head = 0
            while head < coords.shape[0]:
                prediction[head:head+max_batch_size] = model(obj_idx, coords[head:head+max_batch_size].to(device).unsqueeze(0)).cpu()
                head += max_batch_size
                pbar.update(min(max_batch_size, coords.shape[0] - head))
    
    # print(prediction)
    prediction = prediction.reshape(res, res, res).cpu().detach().numpy()
    
    smoothed_prediction =  prediction
    vertices, triangles = mcubes.marching_cubes(smoothed_prediction, 0)
    vertices = (vertices - (res // 2)) / (res // 2)
    mcubes.export_obj(vertices, triangles, output_path)

def main(args=None):
    if args is not None:
        args = args
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--input', type=str)
        parser.add_argument('--output', type=str, required=True)
        parser.add_argument('--model_path', type=str, default='/home/wzj/data/project/NFD/nfd/triplane_decoder/decoder_net_ckpt/latest/2980_decoder.pt', required=False)
        parser.add_argument('--res', type=int, default='128', required=False)

        args = parser.parse_args()
	
    # MultiTriplane 将一个三维坐标点的输入转换为一个1位的occupancy value, 其实就是triplane decoder
    
    model = MultiTriplane(1, input_dim=3, output_dim=1).to(device)

    # model.net应该就是将triplane features转为occupancy values的decoder
    model.net.load_state_dict(torch.load(args.model_path))
    model.eval()

    # 每个triplane平面的分辨率为 (128, 128), 可以认为每个triplane feature的维度为32维
    # triplanes = np.load(args.input).reshape(3, 32, 128, 128)
    model.embeddings.load_state_dict(torch.load('/home/wzj/data/project/NFD/nfd/triplane_decoder/decoder_net_ckpt/latest/triplanes_2980.pt'))

    print(model.embeddings)
    
    # with torch.no_grad():
    #     for i in range(3):
    #         model.embeddings[i][0] = torch.tensor(triplanes[i]).to(device)

    print('wzj')
    test = torch.tensor([[[1., 0., 1.]]]).cuda()
    test2 = torch.tensor([[[0.75, 0.75, 0.]]]).cuda()
    test3 = torch.tensor([[[0., 0., 0.]]]).cuda()
    print(model(torch.tensor([0]), test))
    print(model(torch.tensor([0]), test2))
    print(model(torch.tensor([0]), test3))

    # exit()
    create_obj(model, 0, res = args.res, output_path = args.output)  # res = 256
    
if __name__ == "__main__":
    main()
