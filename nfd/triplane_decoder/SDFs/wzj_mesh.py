import numpy as np
import trimesh
import os
from tqdm import tqdm
from glob import glob
import torch
from typing import Optional, Union

def bbox(vertices: np.ndarray):
    # shape = vertices.shape
    # vertices = vertices.reshape(-1, 3)
    min = np.min(vertices, axis=-2, keepdims=True)
    max = np.max(vertices, axis=-2, keepdims=True)
    
    return np.concatenate([min, max], axis=-2)

def boundary(vertices: np.ndarray):
    min, max = bbox(vertices)
    min = np.abs(np.min(min))
    max = np.abs(np.max(max))
    # print(min, max)

    return np.max((min, max))



file_list = glob('predata/*.obj')
class MeshSDFS:
    def __init__(self, mesh: trimesh.base.Trimesh, num: int=0):
        
        self.mesh = mesh
        self.vertices = mesh.vertices
        self.num = num


    def compute_gt_SDFS(self, points: np.ndarray = None, num: int = None):
        vertex_normals = self.mesh.vertex_normals

        signal = np.sign(np.sum(vertex_normals * self.vertices, axis=-1, keepdims=True))

        vertex_normals = signal * vertex_normals

        # print(np.sum(np.sum(vertex_normals * self.vertices, axis=-1) < 0))

        out = np.concatenate([self.vertices, vertex_normals], axis=-1)
        print(out.shape)

        np.save(f'data/obj_{num}.npy', out)
        # print(range(1, 501))
        
        
        
        
        
        
        # _, distance, _ = self.mesh.nearest.on_surface(points)

        # print(distance.shape)



        # # 算出最短距离
        # num = points.shape[0] // 10000
        # points_list = np.array_split(points, num, axis=0)

        # min_list = []
        # for i in points_list:
        #     distance = np.linalg.norm(i[:, None, :] - self.vertices[None, ...], axis=-1)

        #     min_distance = np.min(distance, axis=-1)


        # # 解决符号的问题, 找到的是距离最近的点的下标
        #     min_index = np.argmin(distance, axis=-1)

        # # vertex_normals = self.mesh.vertex_normals

        # # vertex_normal = vertex_normals[min_index]

        #     min_point = self.vertices[min_index]

        #     sign = np.sum((i - min_point) * min_point, axis=-1)
        #     signal = np.sign(sign)

        # # print(signal)

        #     min_distance *= signal

        #     min_list.append(min_distance)

        # min_distance = np.concatenate(min_list, axis=0)



        # return min_distance
    

    def sample_sdfs(self):

        
        # output = {}
        # TODO: 在测试时用了某一个物体它的boundary，但是在实际处理时，所有数据需要先对齐，所以这里的scale要改为一个对齐后的统一的scale
        scale = boundary(self.vertices)
        self.vertices = self.vertices / (2 * scale)

        vertex_normals = self.mesh.vertex_normals

        signal = np.sign(np.sum(vertex_normals * self.vertices, axis=-1, keepdims=True))

        vertex_normals = signal * vertex_normals

        # print(np.sum(np.sum(vertex_normals * self.vertices, axis=-1) < 0))

        # out = np.concatenate([self.vertices, vertex_normals], axis=-1)
        # print(out.shape)

        # np.save(f'data/obj_{num}.npy', out)
        *shape, _ = self.vertices.shape
        # sdfs = np.zeros((*shape, 1))
        # P_s = np.concatenate((self.vertices, sdfs), axis=-1)
        P_s = self.vertices
        P_s = np.concatenate([P_s, vertex_normals], axis=-1)
        # output['P_s'] = P_s

        *shape_1, points_num, dim = self.vertices.shape
        P_v = np.random.random((*shape_1, 4 * points_num, 3)) * 2 - 1

        # row = np.where((np.abs(P_v) <= 1).any(axis=-1))[0]
        # P_v = P_v[row]

        print(P_v.shape, points_num)
        print(P_s.shape)

        # output['P_v'] = P_v

        np.savez(f'right_data/{self.num}.npz', P_s=P_s, P_v=P_v)

    def test_circle(self):
        np.random.seed(1111)
        phi = np.random.uniform(0, np.pi, 30000)
        theta = np.random.uniform(0, 2*np.pi, 30000)
        
        # 将球面坐标转换为笛卡尔坐标
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        P_v = np.random.random((4 * 30000, 3)) * 2 - 1

        row = np.where((np.abs(P_v) <= 1).any(axis=-1))[0]
        P_v = P_v[row]

        # print(P_v.shape, points_num)
        # print(P_s.shape)
        np.savez('right_data/test.npz', P_s=np.concatenate((np.stack((x / 2, y / 2, z / 2), axis=-1), np.stack((x, y, z), axis=-1)), axis=-1), P_v=P_v)
        
        # 返回笛卡尔坐标系下的点
        return np.stack((x, y, z), axis=-1)

        # sample_nums = int(5e6)
        # minus = int(2.5 * 1e6 // self.vertices.shape[0])
        # print(minus)
        # # shape: (minus, )
        # dis = np.linspace(0, 0.5, minus)
        # vertex_normals = self.mesh.vertex_normals
        # dis = dis[:, None, None]
        # temp_vn = vertex_normals[None, ...]
        # addition = temp_vn * dis
        

        # min_num = np.min(self.vertices, axis=0)
        # max_num = np.max(self.vertices, axis=0)

        # sample_list = []
        # for min, max in zip(min_num, max_num):
        #     sample_list.append(np.random.uniform(min, max, size=(sample_nums, 1)))

        # points = np.concatenate(sample_list, axis=-1)
        # points = np.concatenate([(self.vertices[None, ...] + addition).reshape(-1, 3), (self.vertices[None, ...] - addition).reshape(-1, 3), points], axis=0)
        # print(points.shape)
        # # print(points.shape)

        # # points = np.random.uniform(-1, 1, size=(sample_nums, 3))

        # sample_sdfs = self.compute_gt_SDFS(points)

        # # print(sample_sdfs.shape)
        # sample_sdfs = sample_sdfs[..., None]

        # print(np.sum(sample_sdfs.reshape(-1) < 0))
        # sample = np.concatenate((points, sample_sdfs), axis=-1)
        
        # # gt_sdfs = np.zeros((self.vertices.shape[0], 1))
        # # gt = np.concatenate((self.vertices, gt_sdfs), axis=-1)

        # # all_sample = np.concatenate((sample, gt), axis=0)

        # return sample

def main(file: str = 'mytest.obj', num: int = None):
    # print('wzjok')
    # testobj = trimesh.load(file)
    # test = MeshSDFS(testobj)
    # sample = test.sample_sdfs()

    # # print(sample.shape)

    # if os.path.exists('newdata/'):
    #     pass
    # else:
    #     os.makedirs('newdata/')
    # filename = file.split('/')[-1]
    # fileprefix = filename.split('.')[0]

    # obj = trimesh.load(file)
    # test = MeshSDFS(obj)
    # test.compute_gt_SDFS(num = num)
    # np.save(f'newdata/{fileprefix}_out.npy', sample) 
    # testobj = trimesh.load('/home/wzj/data/project/NFD/nfd/triplane_decoder/SDFs/mytest.obj')
    # test = MeshSDFS(testobj)
    # test.compute_gt_SDFS() 

    wzj = trimesh.load('/home/wzj/data/datasets/facescape/1/models_reg/1_neutral.obj')
    l = MeshSDFS(wzj)
    l.sample_sdfs()

    print(l.test_circle().shape)
    # print(bbox(wzj.vertices))
    # print(boundary(wzj.vertices))

def test():
    testobj = trimesh.load('./mytest.obj')
    test = MeshSDFS(testobj)
    print(test.compute_gt_SDFS(np.concatenate((test.vertices[0:2, ...], np.array([[0, 0, 0]])), axis=0)))


if __name__ == "__main__":

    # file_list = [os.path.join('newpre', file) for file in os.listdir('newpre') if file.endswith('.obj')]

    # print(file_list)

    # list(tqdm(map(main, file_list, range(1, 501)), total=len(file_list)))

    main()

    # main()
    # map(print, '12345')
    # test()




        