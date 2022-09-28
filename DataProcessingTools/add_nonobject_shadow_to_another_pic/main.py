import obtain_object_than_delete
import get_shadow
import add_shadow_to_blank_model


def main():
    # obtain_object_than_delete
    obtain_input_path = 'C:/Users/Administrator/Desktop/11/kt'
    obtain_output_path = 'C:/Users/Administrator/Desktop/11/semantic'
    obtain_delete_output_path = 'C:/Users/Administrator/Desktop/11/cut'
    obtain_object_only_path = 'C:/Users/Administrator/Desktop/11/object_only'

    # get_shadow
    get_input_path = obtain_delete_output_path
    get_output_path = 'C:/Users/Administrator/Desktop/11/shadow'

    # add_shadow_to_blank_model
    add_input_path_shadow = get_output_path
    add_input_path_rgb = 'C:/Users/Administrator/Desktop/11/blank_model'
    add_output_path = 'C:/Users/Administrator/Desktop/11/added_shadow'

    obtain_object_than_delete.main_delete_objects(
        input_path=obtain_input_path, output_path=obtain_output_path,
        delete_output_path=obtain_delete_output_path,
        object_only_path=obtain_object_only_path)
    print('all objects are deleted from images')

    get_shadow.main_obtain_shadow_keep_name(input_path=get_input_path,
                                            output_path=get_output_path)
    print('all shadows are obtained from non-objects images')

    add_shadow_to_blank_model.main_add_shadow(
        input_path_shadow=add_input_path_shadow,
        input_path_rgb=add_input_path_rgb, output_path=add_output_path)
    print('all shadows are added to images')


if __name__ == '__main__':
    main()
