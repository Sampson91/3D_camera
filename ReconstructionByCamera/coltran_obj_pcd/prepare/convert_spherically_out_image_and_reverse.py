import json
import numpy as np
import operator
import os
from PIL import Image
import cv2
import open3d as o3d


def get_number_of_points(file_direction, file):
    _, file_ext = os.path.splitext(file)
    total_path = os.path.join(file_direction, file)

    if file_ext == '.obj':
        # 打开一个文件只读模式
        with open(total_path, 'r') as file_:
            lines = file_.readlines()
        numbers_of_v_line = 0  # obj中v开头的行数
        for numbers_of_v_ in lines:
            if "v " in numbers_of_v_:
                numbers_of_v_line += 1
        file_.close()

        return numbers_of_v_line

    elif file_ext == '.pcd':
        pcd = o3d.io.read_point_cloud(total_path)
        points = np.asarray(pcd.points)

        return len(points)

    else:
        assert file_ext == '.obj' or file_ext == '.pcd', (
            'the cloud point file format is not supported yet')


def get_xyz(file_direction, file):
    _, file_ext = os.path.splitext(file)
    total_path = os.path.join(file_direction, file)
    if file_ext == '.obj':
        # 打开一个文件只读模式
        with open(total_path, 'r') as file_:
            lines = file_.readlines()

        x = []
        y = []
        z = []

        for line_ in lines:
            if line_.strip().split()[0] == 'v':
                x.append(float(line_.strip().split()[1]))
                y.append(float(line_.strip().split()[2]))
                z.append(float(line_.strip().split()[3]))

        return x, y, z

    elif file_ext == '.pcd':
        pcd = o3d.io.read_point_cloud(total_path)
        lines = np.asarray(pcd.points)

        x = []
        y = []
        z = []

        for line_ in lines:
            x.append(float(line_[0]))
            y.append(float(line_[1]))
            z.append(float(line_[2]))

        return x, y, z

    else:
        assert file_ext == '.obj' or file_ext == '.pcd', (
            'the cloud point file format is not supported yet')


def get_rgb(file_direction, file):
    _, file_ext = os.path.splitext(file)
    total_path = os.path.join(file_direction, file)

    if file_ext == '.obj':
        # 打开一个文件只读模式
        with open(total_path, 'r') as file_:
            lines = file_.readlines()

        red = []
        green = []
        blue = []

        for line_ in lines:
            if line_.strip().split()[0] == 'v':
                red.append(float(line_.strip().split()[4]))
                green.append(float(line_.strip().split()[5]))
                blue.append(float(line_.strip().split()[6]))

        file_.close()

        return red, green, blue

    elif file_ext == '.pcd':
        pcd = o3d.io.read_point_cloud(total_path)
        pcd_color = np.asarray(pcd.colors)

        lines = pcd_color

        red = []
        green = []
        blue = []

        for line_ in lines:
            red.append(float(line_[0]))
            green.append(float(line_[1]))
            blue.append(float(line_[2]))

        return red, green, blue

    else:
        assert file_ext == '.obj' or file_ext == '.pcd', (
            'the cloud point file format is not supported yet')


def xyz_to_spherical_uv(x, y, z, project_height=512,
                        project_width=512, project_fov_up=5.0,
                        project_fov_down=-25.0):
    fov_up = project_fov_up / 180.0 * np.pi
    fov_down = project_fov_down / 180.0 * np.pi
    fov = abs(fov_down) + abs(fov_up)

    points = []
    for i in range(len(x)):
        points_ = [x[i],
                   y[i],
                   z[i]]
        points.append(points_)

    depth = np.linalg.norm(points, 2, axis=1)

    # get scan components(x,y,z coordinates)
    scan_x = []
    scan_y = []
    scan_z = []
    for point_ in points:
        scan_x.append(point_[0])
        scan_y.append(point_[1])
        scan_z.append(point_[2])

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    project_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    project_y = 1.0 - (pitch + abs(fov_up)) / fov  # in [0.0, 1.0]

    # project_x is already normalized, normalize project_y
    y_min = project_y.min()
    y_max = project_y.max()
    ranges = y_max - y_min
    project_y -= y_min
    project_y /= ranges

    # scale to image size using angular resolution
    project_x *= (project_width - 1)  # in [0.0, W]
    project_y *= (project_height - 1)  # in [0.0, H]

    for i in range(len(project_x)):
        project_x[i] = round(project_x[i])
    for i in range(len(project_y)):
        project_y[i] = round(project_y[i])

    return project_x, project_y


"""
process kitt2 dataset 
kitt2 url: https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/
process way use RangeNet++ paper url: https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/milioto2019iros.pdf
process 3D point xyz to 2D image x,y
"""


def kitt2_xyz_to_spherical_uv(x, y, z, project_height=256, proj_width=1024,
                              proj_fov_up=100.0,
                              proj_fov_down=-100.0):
    # laser parameters
    fov_up = proj_fov_up / 180.0 * np.pi  # field of view up in rad
    fov_down = proj_fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    points = []
    # kitt2 5000 scale
    x *= 5000
    y *= 5000
    z *= 5000
    for i in range(len(x)):
        points_ = [z[i], x[i], y[i]]
        points.append(points_)

    # get depth of all points
    depth = np.linalg.norm(points, 2, axis=1)
    # get scan components(x,y,z coordinates)
    scan_x = z
    scan_y = x
    scan_z = y

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    project_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    project_x *= proj_width  # in [0.0, W]
    proj_y *= project_height  # in [0.0, H]

    scale_width = proj_width / (max(project_x) - min(project_x) + 1)
    scale_height = project_height / (max(proj_y) - min(proj_y) + 1)

    project_x = (project_x - min(project_x)) * scale_width
    proj_y = (proj_y - min(proj_y)) * scale_height

    # round and clamp for use as index
    project_x = np.floor(project_x)
    project_x = np.minimum(proj_width - 1, project_x)
    project_x = np.maximum(0, project_x).astype(np.int32)  # in [0,W-1]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(project_height - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

    # order in decreasing depth
    order = np.argsort(depth)[::-1]
    project_y = proj_y[order]
    project_x = project_x[order]

    return project_x, project_y


def create_index_xyz_uv_rgb(x, y, z, u, v, r, g, b, index_output_path,
                            index_file):
    file_name, _ = os.path.splitext(index_file)
    with open(index_output_path + '/' + file_name + '.json',
              'w') as out_file:
        dictionary_append = []
        for i in range(len(x)):
            dictionary = {
                'xyz': [x[i],
                        y[i],
                        z[i]],
                'uv': [u[i],
                       v[i]],
                'rgb': [r[i],
                        g[i],
                        b[i]]
            }
            dictionary_append.append(dictionary)
        information_json = json.dumps(dictionary_append,
                                      separators=(',', ':'))
        out_file.writelines(information_json)

    out_file.close()

    print(index_output_path + '/' + file_name + '.json',
          'dictionary output complete')

    return index_output_path, file_name + '.json'


# default width == 512, length == 512
def get_image(index_direction, file_name_with_ext, image_saving_path, width=512,
              length=512):
    with open(index_direction + '/' + file_name_with_ext, 'r') as index_file:
        index_lines = json.load(index_file)
    image = Image.new("RGB", (width, length))
    for sorted_line_ in index_lines:
        # print(sorted_line_['rgb'][0],sorted_line_['rgb'][1],sorted_line_['rgb'][2])
        i = sorted_line_['uv'][0]
        j = sorted_line_['uv'][1]
        red_ = sorted_line_['rgb'][0]
        green_ = sorted_line_['rgb'][1]
        blue_ = sorted_line_['rgb'][2]

        image.putpixel((int(i), int(j)), (int(red_ * 255), int(green_ * 255),
                                          int(blue_ * 255)))
    file_name, _ = os.path.splitext(file_name_with_ext)
    image.save(image_saving_path + '/' + file_name + '.jpeg')  # 保存
    print(
        image_saving_path + '/' + file_name + '.jpeg' + ", image saved complete")


################################################################################
#####                             ↓reverse↓                               ######
################################################################################


def reverse_image_back_to_xyz(generated_image_direction,
                              generated_image_file_searching_by_index_name,
                              index_path, index_file,
                              generated_image_index_path):
    image = Image.open(
        generated_image_direction + '/' + generated_image_file_searching_by_index_name).convert(
        'RGB')

    image_name, _ = os.path.splitext(
        generated_image_file_searching_by_index_name)
    index_file_name, _ = os.path.splitext(index_file)

    assert image_name == index_file_name, (
        'image_name and index_file_name are not the same, please check')

    with open(index_path + '/' + index_file, 'r') as index_file_open:
        index_lines = json.load(index_file_open)

        after_index = []
        for index_ in index_lines:
            # get uv from index
            uv = index_['uv']
            # get rgb according to the uv obtained above from image

            red, green, blue = image.getpixel((int(uv[0]), int(uv[1])))
            # rewrite rgb information
            index_['rgb'] = [round(red / 255), round(green / 255),
                             round(blue / 255)]
            after_index.append(index_)
    # save reversed index
    with open(generated_image_index_path + '/' + index_file, 'w') as save_json:
        information_json = json.dumps(after_index, separators=(',', ':'))
        save_json.writelines(information_json)
        print(generated_image_index_path + '/' + index_file, 'is saved')

    return generated_image_index_path + '/' + index_file


def obtain_obj_from_reversed_index(image_index_direction_with_file_name,
                                   save_path_obj):
    with open(image_index_direction_with_file_name, 'r') as index_file_open:
        index_lines = json.load(index_file_open)

    _, index_file_name = os.path.split(image_index_direction_with_file_name)
    file_name, _ = os.path.splitext(index_file_name)
    with open(save_path_obj + '/' + file_name + '.obj', 'w') as save_obj:
        obj_information = []
        for index_ in index_lines:
            obj_information.append(
                'v' + ' ' + str(index_['xyz'][0]) + ' ' + str(
                    index_['xyz'][1]) + ' ' + str(index_['xyz'][2]) + ' ' + str(
                    index_['rgb'][0]) + ' ' + str(index_['rgb'][1]) + ' ' + str(
                    index_['rgb'][2]) + '\n')

        save_obj.writelines(obj_information)
        print(save_path_obj + '/' + file_name + '.obj', 'is saved')


def add_face_to_reversed_obj(original_obj_path, original_obj_file,
                             reversed_obj_path, reversed_obj_file,
                             save_path):
    _, ext = os.path.splitext(original_obj_file)
    if ext != '.obj':
        return 0

    with open(reversed_obj_path + '/' + reversed_obj_file,
              'r') as reversed_file:
        reversed_lines = reversed_file.readlines()
        for line_ in reversed_lines:
            if 'f' == line_.strip().split()[0]:
                return 0

    with open(original_obj_path + '/' + original_obj_file,
              'r') as original_file:
        original_lines = original_file.readlines()
        for line_ in original_lines:
            if 'f' != line_.strip().split()[0]:
                return 0

    number_of_original_points = get_number_of_points(
        original_obj_path, original_obj_file)
    number_of_reversed_points = get_number_of_points(
        reversed_obj_path, reversed_obj_file)
    assert number_of_reversed_points == number_of_original_points, (
        'cannot reverse face information back, '
        'because number of points are not the same, '
        'please recalculate the face information')

    with open(save_path + '/' + reversed_obj_file, 'w') as file_with_face:
        for line_ in original_lines:
            if 'f' == line_.strip().split()[0]:
                reversed_lines.append(line_)

        file_with_face.writelines(reversed_lines)
        print(save_path + '/' + reversed_obj_file, 'is saved')
    return 1