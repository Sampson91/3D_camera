import os
import shutil
from plyfile import PlyData
import argparse


def parse_args(file_path1, file_path2):
    """
    argparse.ArgumentParser():  Create a parser
    add_argument():  Add parameters
    parse_args():  Parsing parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--ply_path', default=file_path1)
    parser.add_argument('--obj_path', default=file_path2)
    parsing_parameters = parser.parse_args()
    return parsing_parameters.ply_path, parsing_parameters.obj_path


def convert_file(ply_path, open_func=open):
    """
    Converts the given .ply file to an .obj file
    """
    obj_path = os.path.splitext(ply_path)[0] + '.obj'
    ply_information = PlyData.read(ply_path)

    with open_func(obj_path, 'w') as file:
        file.write("# OBJ file\n")

        verteces = ply_information['vertex']
        for vertex in verteces:
            position = [vertex['x'], vertex['y'], vertex['z']]
            if 'red' in vertex and 'green' in vertex and 'blue' in vertex:
                color = [vertex['red'], vertex['green'], vertex['blue']]

            else:
                color = [0, 0, 0]
            position_color = position + color
            file.write("vertex %.6f %.6f %.6f %.6f %.6f %.6f \n" % tuple(position_color))

        for vertex in verteces:
            if 'nx' in vertex and 'ny' in vertex and 'nz' in vertex:
                normal_vectors = (vertex['nx'], vertex['ny'], vertex['nz'])
                file.write("vn %.6f %.6f %.6f\n" % normal_vectors)
        if 'face' in ply_information:
            for paremeter_1 in ply_information['face']['vertex_indices']:
                file.write("f")
                for parameter_2 in range(paremeter_1.size):
                    value = [paremeter_1[parameter_2] + 1, paremeter_1[parameter_2] + 1, paremeter_1[parameter_2] + 1]
                    file.write(" %d/%d/%d" % tuple(value))
                file.write("\n")
    return obj_path


def batch_process_files(input_path, output_path):

    files_list = os.listdir(input_path)
    for i in range(len(files_list)):
        ply_path, obj_path = parse_args(files_list[i], output_path)
        convert_file(ply_path)

if __name__ == '__main__':
    batch_process_files()
