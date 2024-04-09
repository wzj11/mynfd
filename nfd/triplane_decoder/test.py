import trimesh

wzj = trimesh.load('bunny_10k.obj')

print(wzj.vertices[wzj.faces][0])


n_tri = wzj.vertices[wzj.faces].shape[0]
wzjtem = wzj.vertices[wzj.faces].reshape(3 * n_tri, 3)

print(wzjtem[0:3].shape)