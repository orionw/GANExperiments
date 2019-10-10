from math import ceil
import numpy as np
import sys
import os
import logging
import pickle
import pandas as pd
import gc
import wandb

from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import (DataLoader, Dataset, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import utils.argparser
from utils.load_data import (DiscriminatorDatasetFromFile, load_and_cache_examples_generator, TextDataset, load_and_cache_examples)
from models.gru import GRUDecoder
import models.generative_transformers as generative_transformers
import models.discriminative_transformers as discriminative_transformers
from models.xlnet import XLNetEmbedder
from models.autoencoder import Autoencoder
from utils.utils_glue import (compute_metrics, convert_examples_to_features, output_modes, processors)
from utils.helpers import convert_to_text, sample_and_record_text, set_seed, save_states
from models.training_functions import (train_generator_mle, prepare_opt_and_scheduler, discriminator_eval, train_autoencoder, 
                                        create_transformer_mapping, adversarial_train)
from models.transformer import make_decoder_model
 
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    args = utils.argparser.parse_all_args(sys.argv)
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
        
    args.device = device

     # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed and log params
    set_seed(args)
    logger.info("Training/evaluation parameters %s", args)

    #### Get Models ####
    dis = discriminative_transformers.PretrainedDiscriminativeTransformer(args)
    tokenizer = dis.tokenizer
    gen = generative_transformers.PretrainedTransformerGenerator(args, dis.tokenizer)
    encoder = generative_transformers.PretrainedTransformerGenerator(args, dis.tokenizer)
    
    word_embedder = encoder._modules["model"]._modules["transformer"]._modules["word_embedding"]
    # decoder = make_decoder_model(tokenizer.vocab_size, d_model=encoder.config.d_model, N=args.decoder_layers, d_ff=2048,
    #                              h=8, dropout=args.decoder_dropout, embedder=word_embedder).to(args.device)
    decoder = GRUDecoder(encoder.config.d_model, tokenizer.vocab_size, encoder.config.d_model, args.decoder_layers, 
                            args.decoder_dropout)
    autoencoder = Autoencoder(encoder, decoder, args.device, tokenizer=tokenizer, model_type=args.gen_model_type)

    if args.record_run:
        wandb.init(project="humorgan", config=args)
        wandb.watch((gen, dis))

    # prepare main datasets
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    loaded_train_dataset = DiscriminatorDatasetFromFile(args.train_data_file, label="1")
    real_train_dataset = load_and_cache_examples(args, "cola", tokenizer, loaded_train_dataset, evaluate=True)
    train_sampler = RandomSampler(real_train_dataset) if args.local_rank == -1 else DistributedSampler(real_train_dataset)
    train_dataloader = DataLoader(real_train_dataset, sampler=train_sampler, batch_size=args.per_gpu_train_batch_size, drop_last=True)

    loaded_val_dataset = DiscriminatorDatasetFromFile(args.eval_data_file, label="1")
    real_val_dataset = load_and_cache_examples(args, "cola", tokenizer, loaded_train_dataset, evaluate=True)
    val_sampler = RandomSampler(real_val_dataset) if args.local_rank == -1 else DistributedSampler(real_val_dataset)
    val_dataloader = DataLoader(real_val_dataset, sampler=val_sampler, batch_size=args.per_gpu_train_batch_size, drop_last=True)

    # prepare optimizers and schedulers
    gen, gen_optimizer = prepare_opt_and_scheduler(args, gen, len(real_train_dataset))
    dis, dis_optimizer = prepare_opt_and_scheduler(args, dis, len(real_train_dataset))

      # Load models if they are pretrained
    if args.pretrained_decoder_path:
            decoder.load_state_dict(torch.load(os.path.join(args.output_dir, "decoder-{}-best.pt".format(args.run_name))))

    prev_epochs = 0
    if args.pretrained_generator_path:
        checkpoint = torch.load(os.path.join(args.output_dir, "checkpoint-gan-{}.tar".format(args.run_name_gan)))
        gen.load_state_dict(checkpoint['gen_state_dict'])
        dis.load_state_dict(checkpoint['dis_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer_state_dict'])
        prev_epochs = checkpoint["epochs"]

    #### AUTOENCODER TRAINING ####
    if args.autoencoder_epochs != 0 and not args.gan_only:
        autoencoder = autoencoder.to(args.device)
        decoder = decoder.to(args.device)
        autoencoder_optimizer = optim.Adam(decoder.parameters(), lr=args.autoencoder_learning_rate)
        criterion = nn.CrossEntropyLoss()
        loss_df = pd.DataFrame(columns=['batch_num', 'loss'])
        autoencoder = train_autoencoder(args, autoencoder, train_dataloader, val_dataloader, autoencoder_optimizer, 
                        criterion, 1, loss_df, args.autoencoder_epochs)
        if args.autoencoder_only:
            exit(1)

    if args.mle_pretraining:
        _, gen_mle_optimizer = prepare_opt_and_scheduler(args, gen, len(real_train_dataset))
        logger.info('### Starting Generator MLE Training... ###')
        train_dataset = load_and_cache_examples_generator(args, tokenizer, evaluate=False)
        eval_dataset = load_and_cache_examples_generator(args, tokenizer, evaluate=True)
        train_generator_mle(args, train_dataset, gen, tokenizer, gen_mle_optimizer, eval_dataset)

    # ADVERSARIAL TRAINING
    best_gen_loss = float("inf")
    gen.to(args.device)
    dis.to(args.device)
    for epoch in range(args.num_train_epochs):
        saved = False
        logger.info('### GAN EPOCH: {} ###'.format(epoch))

        # TRAIN DISCRIMINATOR
        loss, dis_optimizer, gen, dis = adversarial_train(args, gen, dis, encoder, tokenizer, dis_optimizer, val_dataloader, 1)
        print("#### Average disriminator loss : {} ####".format(loss))
        if args.record_run:
            # don't save the disciminator until the generator is saved
            wandb.log({"discriminator loss": loss})
        
        # TRAIN GENERATOR
        for gen_steps in range(args.gen_epochs_per_dis):
            loss, gen_optimizer, gen, dis = adversarial_train(args, gen, dis, encoder, tokenizer, gen_optimizer, 
                                                                val_dataloader, 1, is_discriminator=False)
            print("#### Average generator loss : {} ####".format(loss))
            if args.record_run:
                wandb.log({"generator loss": loss})
            # save if the best loss or if K epochs have passed
            if not saved and loss < best_gen_loss:
                save_states(args, gen, dis, gen_optimizer, dis_optimizer, epochs=epoch + prev_epochs, name="best")
                saved = True
            if not saved and epoch and epoch % 25:
                save_states(args, gen, dis, gen_optimizer, dis_optimizer, epochs=epoch + prev_epochs, name="time")

        if args.record_run and epoch % 2 == 0:
            sample_and_record_text(args, gen, decoder, tokenizer)
        