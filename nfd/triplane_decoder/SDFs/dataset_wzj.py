import torch
from torch.utils.data import Dataset
import torch.utils.data
import numpy as np
import os
# from wzj_mesh import boundary
# TODO:
class wzjData(Dataset):
    def __init__(self, data_dir: str = 'data'):

        # npz_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.npz')]

        npz_files = ['/home/wzj/data/project/NFD/nfd/triplane_decoder/SDFs/right_data/test.npz']
        file_list = []
        for file in npz_files:
            temp = np.load(file, allow_pickle=True)
            # print(type(temp))
            file_list.append(temp)

        self.data = file_list
        # 不应该stack， 维度不一样


        # self.data = np.stack(file_list)
        # self.data = torch.from_numpy(self.data).cuda()
        # self.data = self.data.reshape(-1, *self.data.shape)



    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 一个npz文件里的内容视为一个单独的整体
        # print(self.data[idx].shape)
        return idx, torch.from_numpy(self.data[idx]['P_s'][..., 0:3]), torch.from_numpy(self.data[idx]['P_s'][..., 3:4]), torch.from_numpy(self.data[idx]['P_s'][..., 4:]), torch.from_numpy(self.data[idx]['P_v'])


def main():
    test = wzjData('data')
    l_data = torch.utils.data.DataLoader(test, 1, shuffle=True)

    # l_data 中一共有len(dataset) / batchsize 个元素
    # 假如说dataset 中的 __getitem__返回了a,b,c，也就是返回三个元素
    # 那么l_data[0]就是一个大小为3的列表，第一项关于a，维度为(batchsize, *a.shape), 第二项关于。。。
    for idx, X, truth, normals, Y in l_data:
        print(X.shape, truth.shape, normals.shape, Y.shape)

    print(len(l_data))

    

if __name__ == '__main__':
    main()