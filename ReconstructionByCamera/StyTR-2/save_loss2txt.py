
def save_loss2txt_function(output_path_file, i, loss):
    with open(output_path_file, 'a') as file:
        file.write(str(i) + ' ' + str(loss) + '\n')