# For inference
python main.py --save_dir infer/TTSR/save_results \
               --reset True \
               --log_file_name eval.log \
               --infer True \
               --infer_save_results True \
               --num_workers 4 \
               --dataset CUFED \
               --dataset_dir dataset/CUFED \
               --model_path /home/mushan/Desktop/model_00050.pt