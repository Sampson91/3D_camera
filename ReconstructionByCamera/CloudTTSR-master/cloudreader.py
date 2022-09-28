import numpy as np
import open3d as o3d
import random


class ReadObj:
    def __init__(self, file_name):
        self.vertices_with_colors = []
        self.file_name = file_name

    def cloud_read(self, file_name):
        self.vertices_with_colors = []
        self.minimum_vertice = []
        count = 0
        for line in open(file_name, "r"):
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'v' and count < 50000:
                vertice = [float(number) for number in values[1:4]]
                self.minimum_vertice.append(min(vertice))
                color = [float(number) * 255 for number in values[4:7]]
                vertice_color = np.hstack((vertice, color))
                self.vertices_with_colors.append(vertice_color)
                count += 1
        number_minimum = min(self.minimum_vertice)
        vertices_with_colors = np.array(self.vertices_with_colors)
        vertices_with_colors[:, 0:3] = vertices_with_colors[:,
                                                            0:3] - number_minimum + 1
        return vertices_with_colors


def cloud_save(save_path, cloud):
    with open(save_path, 'w') as file:
        for line_ in cloud:
            line_[3:6] = line_[3:6] / 256
            file.writelines('v' + ' ' + ' '.join(str(i) for i in line_) + '\n')


def keep_random_point_dropout(point_cloud, keep_number):
    size = int(point_cloud.shape[0])
    index = [index_number for index_number in range(size)]
    drop_index = random.sample(index, size - keep_number)
    point_cloud = np.delete(point_cloud, drop_index, 0)
    return point_cloud


def read_pcd(file_path):
    cloud = []
    pcd = o3d.io.read_point_cloud(file_path)
    colors = np.asarray(pcd.colors) * 255
    points = np.asarray(pcd.points) * 5000 / 20

    # x = points[0,:]
    # y = points[1,:]
    # z = points[2,:]
    # x -= x.min()
    # y -= y.min()
    # z -= z.min()
    # points = np.concatenate([x, y, z], axis=-1)

    cloud = np.concatenate([points, colors], axis=-1)
    cloud = keep_random_point_dropout(cloud, 50000)
    return cloud
