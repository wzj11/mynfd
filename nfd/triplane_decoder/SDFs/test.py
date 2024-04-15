from wzj_mesh import MeshSDFS
import trimesh
import numpy as np
import json

if __name__ == "__main__":
    # wzj_obj = trimesh.load("mytest.obj")
    # print(type(wzj_obj))
    # test = MeshSDFS(wzj_obj)

    # po = wzj_obj.vertices
    # test1 = po[0:1] + 0.8 * po[0:1]
    # test1 = np.concatenate((test1, po[0:2]), axis=0)
    # print(test.compute_gt_SDFS(test1))

    # test.sample_sdfs()
    

    wzj = np.load('/home/wzj/data/project/NFD/nfd/triplane_decoder/SDFs/data/obj_female_1_out.npy')

    print(wzj.shape)

    out = wzj[:, 3:]

    print(np.sum(out < 0))
    print(np.sum(out > 0))
    l = {'output': out.tolist()}

    with open('wzj.json', 'w') as f:
        json.dump(l, f, indent=2)