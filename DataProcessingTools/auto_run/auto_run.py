""" Train several models"""

import subprocess
import os
import time

from send_mail import send_email

# subprocess.call("chmod +x train.sh",shell=True) # get permission if no permission


def auto_run_train(model1_path, model2_path, model3_path, time_1, time_2):
    """This method is to train several model continusly
    You need to write conda activate code in your train.sh
    such as <<eval "$(conda shell.bash hook)" conda activate ttsr>>
    """ 
    # TODO(mushanwei): Need to send error information when training error occured
    os.chdir(model1_path)
    model1_name = model1_path.split("/")[-2]
    print("start trainning {}".format(model1_name))
    # subprocess.call("source ~/anaconda3/etc/profile.d", shell=True)
    # subprocess.call("source activate ttsr", shell=True)
    result1 = subprocess.call("./train.sh", shell=True)
    if not result1:
        os.chdir(model2_path)
        print("wait for {}s...".format(time_1))
        time.sleep(time_1)
        model2_name = model2_path.split("/")[-2]
        print("start trainning {}".format(model2_name))
        # subprocess.call("source activate SR", shell=True)
        result2 = subprocess.call("./train1.sh", shell=True)
        if not result2:
            os.chdir(model3_path)
            print("wait for {}s...".format(time_2))
            time.sleep(time_2)
            model3_name = model3_path.split("/")[-2]
            print("start trainning {}".format(model3_name))
            reslut3 = subprocess.call("./train2.sh", shell=True)
            if not reslut3:
                send_email("Your training are finished")
            else:
                send_email("You got an error, please check")
                # send_email(reslut3)
        else:
            send_email("You got an error, please check")
            # send_email(result2)
    else:
        send_email("You got an error, please check")
        # send_email(result1)


if __name__ == '__main__':
    """Please input your model train.sh path"""

    # path1 = "/media/mushan/HDD/MuShanwei/code/Image/TTSR/"
    # path2 = "/media/mushan/HDD/MuShanwei/code/Image/KAIR/"
    # path3 = "/media/mushan/HDD/MuShanwei/code/utils/auto_run/"
    path1 = "/media/mushan/HDD/MuShanwei/code/utils/auto_run/"
    path2 = "/media/mushan/HDD/MuShanwei/code/utils/auto_run/"
    path3 = "/media/mushan/HDD/MuShanwei/code/utils/auto_run/"
    time_to_sleep1 = 5
    time_to_sleep2 = 5
    auto_run_train(path1, path2, path3, time_to_sleep1, time_to_sleep2)
