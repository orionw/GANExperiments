python3 main.py \
    --train_data_file=./data/emnlp_news/val.csv \
    --data_dir=data/emnlp_news \
    --model_name_or_path=xlnet-base-cased \
    --eval_data_file=./data/emnlp_news/val.csv \
    --model_type=xlnet \
    --do_train \
    --data_dir=./data/emnlp_news \
    --max_seq_length=128 \
    --per_gpu_train_batch_size=36 \
    --learning_rate=2e-5 \
    --num_train_epochs=1.0 \
    --do_lower_case \
    --overwrite_output_dir \
    --gradient_accumulation_steps=24 \
    --output_dir=./gan_results             
    # TODO: change the eval file to not be the train file
