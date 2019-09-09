python3 main.py \
    --train_data_file=./data/puns/val.csv \
    --data_dir=data/emnlp_news \
    --eval_data_file=./data/puns/val.csv \
    --gen_model_type=xlnet \
    --gen_model_name_or_path=xlnet-base-cased \
    --dis_model_type=bert \
    --dis_model_name_or_path=bert-base-uncased \
    --do_train \
    --data_dir=./data/puns \
    --max_seq_length=128 \
    --per_gpu_train_batch_size=30 \
    --learning_rate=2e-5 \
    --num_train_epochs=1.0 \
    --do_lower_case \
    --overwrite_output_dir \
    --gradient_accumulation_steps=24 \
    --output_dir=./gan_results             
    # TODO: change the eval file to not be the train file
