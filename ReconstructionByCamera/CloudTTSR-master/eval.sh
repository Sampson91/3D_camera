### evaluation
python main.py --save_dir ./eval/CUFED/TTSR \
               --reset True \
               --log_file_name eval.log \
               --eval True \
               --eval_save_results True \
               --num_workers 4 \
               --dataset CUFED \
               --dataset_dir dataset/CUFED \
               --model_path /home/mushan/Desktop/model_00050.pt