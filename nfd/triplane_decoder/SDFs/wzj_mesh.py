import numpy as np


class MeshSDFS:
    def __init__(self, mesh):
        self.mesh = mesh
        self.vertices = mesh.vertices


    def computeSDFS(self, points):
        # points.shape: (N, 3)
        out = np.zeros(points.shape[0], self.vertices.shape[0], 3)
        points.unsqueeze(-1).repeat(self.vertices.shape[0], axis=1) - self.vertices.unsqueeze(0).repeat()

