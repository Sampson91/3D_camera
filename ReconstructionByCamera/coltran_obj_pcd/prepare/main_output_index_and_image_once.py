import os
from prepare import convert_spherically_out_image_and_reverse


def from_obj_to_index_and_image(obj_direction, obj_file, index_output_path,
                                image_saving_path, square_pixel_size=512):
    x, y, z = convert_spherically_out_image_and_reverse.get_xyz(
        obj_direction,
        obj_file)

    r, g, b = convert_spherically_out_image_and_reverse.get_rgb(
        obj_direction,
        obj_file)

    u, v = convert_spherically_out_image_and_reverse.xyz_to_spherical_uv(x, y,
                                                                         z,
                                                                         project_height=square_pixel_size,
                                                                         project_width=square_pixel_size)

    index_direction, file_name_with_ext = convert_spherically_out_image_and_reverse.create_index_xyz_uv_rgb(
        x, y, z, u, v, r, g, b, index_output_path, index_file=obj_file)

    convert_spherically_out_image_and_reverse.get_image(
        index_direction=index_direction, file_name_with_ext=file_name_with_ext,
        image_saving_path=image_saving_path, width=square_pixel_size,
        length=square_pixel_size)


def convert_generated_image_to_obj_reversely(resized_image_direction,
                                             generated_image_file_by_splitting_from_index,
                                             index_path,
                                             index_file,
                                             generated_image_index_path,
                                             obj_saving_path):
    image_index_direction_with_file_name = convert_spherically_out_image_and_reverse.reverse_image_back_to_xyz(
        generated_image_direction=resized_image_direction,
        generated_image_file_searching_by_index_name=generated_image_file_by_splitting_from_index,
        index_path=index_path,
        index_file=index_file,
        generated_image_index_path=generated_image_index_path)

    convert_spherically_out_image_and_reverse.obtain_obj_from_reversed_index(
        image_index_direction_with_file_name=image_index_direction_with_file_name,
        save_path_obj=obj_saving_path)

# if __name__ == '__main__':
#     convert_generated_image_to_obj_reversely(
#         generated_image_direction='./store_image/stage1',
#         generated_image_file='BFM2017.png', index_path='./index',
#         index_file='BFM2017.json',
#         generated_image_index='./generated_image_index',
#         obj_saving_path='./generated_obj')

# if __name__ == '__main__':
#     from_obj_to_index_and_image(obj_direction='./obj', obj_file='BFM2017.obj',
#                                 index_output_path='./test_combine_index', image_saving_path='./test_combine_image')
