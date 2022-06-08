python main-2epochs.py     \
        --uncertain_criteria ms    \
        --gpu_number 1              \
        --data_path /data/moon/datasets/Large_Captcha_Dataset \
        --digit_compression mean \
        --maximum_iterations 15 \
        -epochs 2 \
        -K 4000 \
        -pn 2-epochs \
        -wandb