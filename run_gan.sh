python3 main.py \
    --train_data_file=./data/puns/train.csv \
    --eval_data_file=./data/puns/val.csv \
    --gen_model_type=xlnet \
    --gen_model_name_or_path=xlnet-base-cased \
    --dis_model_type=xlnet \
    --dis_model_name_or_path=xlnet-base-cased \
    --do_train \
    --data_dir=./data/puns \
    --max_seq_length=32 \
    --per_gpu_train_batch_size=16 \
    --learning_rate=5e-4 \
    --num_train_epochs=3.0 \
    --do_lower_case \
    --overwrite_output_dir \
    --gradient_accumulation_steps=12 \
    --output_dir=./gan_results \
    --autoencoder_epochs 500 \
    --autoencoder_only \
    --autoencoder_learning_rate 0.001 
    # --record_run
    # --mle_pretraining 
    # --fp16       
    # TODO: change the eval file to not be the train file
