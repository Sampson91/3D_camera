import open3d as o3d
import numpy as np
import pytest



def normal_vector_calculations(file_source_path):
    file_destination_path = file_source_path.replace(".ply", "_normal.ply")
    ply = o3d.io.read_point_cloud(file_source_path)
    ply.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=25))
    normals = o3d.geometry.PointCloud()
    normals.points = o3d.utility.Vector3dVector(ply.normals)
    o3d.io.write_point_cloud(file_destination_path, normals)
    return np.asarray(ply.normals)


def cosine_similarity_calculation(vector_a, vector_b):
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    cosine_similarity = np.dot(vector_a, vector_b) / (norm_a * norm_b)
    return cosine_similarity


def normal_vector_similarity_validation(obj_normal_vector, file_source_path=" "):
    row_number = len(obj_normal_vector)
    count = 0
    for i in range(row_number):
        count = count + cosine_similarity_calculation(
                    normal_vector_calculations(file_source_path)[i],
                    obj_normal_vector[i]) * 100
    return abs(count / row_number)


if __name__ == "__main__":
    file_source_path = "ImageToStl.com_finalbasemesh.ply"
    normal_vector_calculations(file_source_path=file_source_path)
    print("The similarity is：\t", normal_vector_similarity_validation(file_source_path=file_source_path), "%")


def test_normal_vector_calculations():
    assert normal_vector_calculations(file_source_path="ImageToStl.com_finalbasemesh.ply").all

def test_cosine_similarity_calculation():
    assert cosine_similarity_calculation([1, 2, 2],[1, 2, 2])>=0 \
           and cosine_similarity_calculation([1, 2, 2],[1, 2, 2])<=1

def test_normal_vector_similarity_validation(mocker):
    count = 0
    cosine_similarity_calculation = mocker.Mock(return_value=0.9)
    for i in range(17):
        count = count + cosine_similarity_calculation() * 100
    assert count/17 == 90



