python3 main.py \
    --train_data_file=./data/puns/shortest.csv \
    --eval_data_file=./data/puns/shortest.csv \
    --gen_model_type=xlnet \
    --gen_model_name_or_path=xlnet-base-cased \
    --dis_model_type=xlnet \
    --dis_model_name_or_path=xlnet-base-cased \
    --do_train \
    --data_dir=./data/puns \
    --max_seq_length=32 \
    --per_gpu_train_batch_size=1 \
    --learning_rate=5e-3 \
    --num_train_epochs=500 \
    --do_lower_case \
    --overwrite_output_dir \
    --gradient_accumulation_steps=4 \
    --output_dir=./gan_results \
    --pretrained_decoder_path \
    --gen_epochs_per_dis=1 \
    --gan_only \
    --loss_type=rsgan \
    --record_run
    # --mle_pretraining 
    # --fp16       
