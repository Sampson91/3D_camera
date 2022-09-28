# 3D转2D，2D返回3D

1. 所写的函数都是对单个文件进行处理，如果要批量处理自行编写批量语句后嵌套这里的函数
2. 需要的包都在requirements.txt里

## main_output_index_and_image_once.py
这是对于`convert_spherically_out_image_and_reverse`的封装
### `from_obj_to_index_and_image`
将obj或者pcd转成index和图片，封装了： 
1. `convert_spherically_out_image_and_reverse.get_xyz`
2. `convert_spherically_out_image_and_reverse.get_rgb`
3. `convert_spherically_out_image_and_reverse.xyz_to_spherical_uv`
4. `convert_spherically_out_image_and_reverse.create_index_xyz_uv_rgb`
5. `convert_spherically_out_image_and_reverse.get_image`
6. 无返回值
7. 参数：
   1. obj_direction：obj或者pcd文件的路径，不带文件名
   2. obj_file：obj或者pcd文件名
   3. index_output_path：index输出路径，不带文件名
   4. image_saving_path：图片输出路径，不带图片名
   5. square_pixel_size=512：像素，默认512

### `convert_generated_image_to_obj_reversely`  
将图片通过index转回obj，封装了：
1. `convert_spherically_out_image_and_reverse.reverse_image_back_to_xyz`
2. `convert_spherically_out_image_and_reverse.obtain_obj_from_reversed_index`
3. 无返回值
4. 参数：
   1. resized_image_direction：和生成图像相同尺寸的需要转回obj的图片路径，无文件名
   2. generated_image_file：需要转换的图片的名称
   3. index_path：index路径，不含index名
   4. index_file：index名
   5. generated_image_index_path：读取新渲染的图片的rgb后的index
   6. obj_saving_path：保存新obj的路径，无文件名

## convert_spherically_out_image_and_reverse.py
包含了所有用于3d到2d和2d回到3d的函数

### `get_number_of_points`
1. 针对obj文件，读取点的数量，主要用于在`add_face_to_reversed_obj`的时候检查点数是否完整
2. 返回点的数量
3. 参数：
   1. file_direction：obj或者pcd文件的路径，不含名称
   2. file：obj或者pcd文件名

### `get_xyz`
1. 获取所有的xyz坐标
2. 返回x,y,z
3. 目前可以读取obj和pcd，obj的读取为文本格式（自己写的），pcd读取使用的o3d
4. 参数：
   1. file_direction：obj或者pcd文件的路径，不含名称
   2. file：obj或者pcd文件名

### `get_rgb`
1. 获取所有的rgb信息
2. 返回r,g,b
3. 目前可以读取obj和pcd，obj的读取为文本格式（自己写的），pcd读取使用的o3d
4. 参数：
   1. file_direction：obj或者pcd文件的路径，不含名称
   2. file：obj或者pcd文件名

### `xyz_to_spherical_uv`
1. 使用的是rangenet++的环绕铺平公式，以原点为圆心，fov为高度，将360度的球形视野铺平。
2. 返回u,v
3. 参数：
   1. x, y, z：x, y, z的各自的array
   2. project_height=512：v的最大值（图像高度）默认512
   3. project_width=512：u的最大值（图像宽度）默认512
   4. project_fov_up=30.0：fov向上的角度
   5. project_fov_down=-30.0：fov向下的角度

### `kitt2_xyz_to_spherical_uv`
1. kitti2的铺平方式
2. 返回u,v
3. 参数：与`xyz_to_spherical_uv`相同

### `create_index_xyz_uv_rgb`
1. 将xyz坐标，uv坐标和rgb写成index保存
2. 返回输出路径，带ext的文件名
3. 参数：
   1. x, y, z, u, v, r, g, b：x, y, z, u, v, r, g, b各自的array
   2. index_output_path：index输出路径，无名称
   3. index_file：index名称

### `get_image`
1. 通过index将rgb信息贴到对应的uv位置
2. 无返回值
3. 参数：
   1. index_direction：index路径，无文件名
   2. file_name_with_ext：index文件名
   3. image_saving_path：图片保存路径，无文件名
   4. width=512：图片宽度，默认512
   5. length=512：图片高度，默认512


### `reverse_image_back_to_xyz`
1. 通过`create_index_xyz_uv_rgb`生成的index中的uv坐标将新上色的图片中的rgb读取并生成带有新的rgb信息的index保存
2. 返回输出路径和带ext的文件名
3. 参数：
   1. generated_image_direction：需要转成obj的图像的路径，无文件名
   2. generated_image_file：图像名
   3. index_path：对应的index的路径，无文件名
   4. index_file：对应的index的名称，带ext
   5. generated_image_index_path：根据新的图片的rgb写成的新index，用于写obj

### `obtain_obj_from_reversed_index`
1. 通过`reverse_image_back_to_xyz`生成的index将xyz和rgb信息写成obj文件并保存
2. 无返回值
3. 参数：
   1. image_index_direction_with_file_name：从`reverse_image_back_to_xyz`生成的obj的路径，带文件名和ext
   2. save_path_obj：保存obj的路径，不带文件名

### `add_face_to_reversed_obj`
1. 将原始obj的face信息append到新obj
2. 返回值：
   1. 没添加返回0
   2. 添加了返回1
3. 参数：
   1. original_obj_path：带有面信息的obj的路径，不带名称
   2. original_obj_file：带有面信息的obj的名称带ext
   3. reversed_obj_path：需要贴面信息的obj的路径，不带名称
   4. reversed_obj_file：需要贴面信息的obj的名称带ext
   5. save_path：保存路径，不带名称

## rgb_to_gray.py
只有一个将rgb图片转换成灰度图的函数

### `rgb_to_gray_function`
1. 将所给图片转换成灰度图保存
2. 无返回值
3. 参数：
   1. path：路径不带名字，需要被转换的图片的路径
   2. file：图片名称
   3. out_path：路径不带名字，灰度图的输出路径

## resize_image.py
用于resize图片的模块
### `copy_file`
1. 将文件复制到另一个路径
2. 无返回值
3. 参数：
   1. original_path_with_file：带完整文件名的完整路径，被复制的文件
   2. destination_path：不带文件名的路径，需要复制到的路径

### `resize_image_function`
1. 更改图片尺寸,输出为正方形
2. 无返回值
3. 参数：
   1. image_input：带文件名的完整路径，被resize的图片
   2. image_output：带文件名的完整路径，resize后的保存路径
   3. pixel：宽和高的像素数量

### `argb_convert_to_rgb`
1. 将4通道的argb图像转换成3通道的rgb图像，在源文件处直接更改，可以和`copy_file`一起使用
2. 无返回值
3. 参数：
   1. path_with_file：带文件名的完整路径




