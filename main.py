from __future__ import print_function
from math import ceil
import numpy as np
import sys
import pdb
import os
import logging
import pickle
import pandas as pd

from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.gru import GRUDecoder
import models.generator as generator
import models.discriminator as discriminator
from models.xlnet import XLNetEmbedder
from models.autoencoder import Autoencoder

from utils.load_data import (DiscriminatorDatasetFromFile, load_and_cache_examples_generator, TextDataset, load_and_cache_examples)


from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

import utils.argparser

from metrics.loss import get_losses

from models.training_functions import set_seed

from utils.utils_glue import (compute_metrics, convert_examples_to_features,
                        output_modes, processors)

from models.training_functions import (train_generator_mle, prepare_opt_and_scheduler, discriminator_eval, train_autoencoder)
 
logger = logging.getLogger(__name__)

ADV_TRAIN_EPOCHS = 50


def adversarial_train(args, gen, dis, tokenizer, optimizer, real_dataset, num_steps, is_discriminator = True):
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    total_loss = 0
    
    # create real and generated dataloaders TODO: turn off evaluate=True after debugging is done
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(real_dataset) if args.local_rank == -1 else DistributedSampler(real_dataset)
    training_dataloader = DataLoader(real_dataset, sampler=train_sampler, batch_size=args.train_batch_size, drop_last=True)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(training_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(training_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
       dis  = torch.nn.DataParallel(dis)
       gen = torch.nn.DataParallel(gen)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    logger.info("***** Running adversarial training *****")
    logger.info("  Num examples = %d", len(training_dataloader))
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)


    preds = None
    train_iterator = trange(int(1), desc="Epoch", disable=args.local_rank not in [-1, 0])
    for _ in train_iterator:
        epoch_iterator = tqdm(training_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, (real_batch) in enumerate(epoch_iterator):
            dis.train()
            gen.train()
            gen_batch = gen.sample(args.train_batch_size)
            # run both real and fake data
            d_out_real = discriminator_eval(args, real_batch, dis, tokenizer)
            d_out_fake = discriminator_eval(args, gen_batch, dis, tokenizer)
            # compute losses and return the relevant one
            assert d_out_real.shape == d_out_fake.shape, "shapes are not aligned, error"
            if is_discriminator:
                _, loss = get_losses(d_out_real, d_out_fake, args.loss_type)
            else:
                loss, _ = get_losses(d_out_real, d_out_fake, args.loss_type)

            optimize(optimizer, loss, dis if is_discriminator else gen)
            total_loss += loss.item()

    average_loss = total_loss / step if step != 0 else 0
    return average_loss, optimizer, gen, dis


def optimize(opt, loss, model=None, retain_graph=False):
    opt.zero_grad()
    loss.backward(retain_graph=retain_graph)
    if model is not None: 
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0) # 5 is from RelGan paper/impl.
    opt.step()


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

    # Set seed
    set_seed(args)

    logger.info("Training/evaluation parameters %s", args)
    # get models
    dis = discriminator.Discriminator(args)
    tokenizer = dis.tokenizer
    gen = generator.Generator(args, dis.tokenizer)
    if args.record_run:
        wandb.init(project="humorgan", config=args)
        wandb.watch((gen, dis))
    gen.to(args.device)
    dis.to(args.device)

    # prepare main dataset
    loaded_val_dataset = DiscriminatorDatasetFromFile(args.eval_data_file, label="1")
    loaded_train_dataset = DiscriminatorDatasetFromFile(args.train_data_file, label="1")
    real_train_dataset = load_and_cache_examples(args, "cola", tokenizer, loaded_train_dataset, evaluate=True)
    real_val_dataset = load_and_cache_examples(args, "cola", tokenizer, loaded_train_dataset, evaluate=True)
    val_sampler = RandomSampler(real_val_dataset) if args.local_rank == -1 else DistributedSampler(real_val_dataset)
    val_dataloader = DataLoader(real_val_dataset, sampler=val_sampler, batch_size=args.per_gpu_train_batch_size, drop_last=True)

    # prepare optimizers and schedulers
    gen, gen_optimizer = prepare_opt_and_scheduler(args, gen, len(real_train_dataset))
    dis, dis_optimizer = prepare_opt_and_scheduler(args, dis, len(real_train_dataset))


   
    decoder = GRUDecoder(768, tokenizer.vocab_size, 768, 1, .2).to(args.device)
    autoencoder = Autoencoder(gen, decoder, args.device, tokenizer=tokenizer).to(args.device)
    if args.pretrained_autoencoder_path is not None:
        autoencoder.load_state_dict(torch.load(os.path.join(args.output_dir, "autoencoder.pt")))
    # TODO add arg for autoencder params
    # Set up autoencoder
    autoencoder_optimizer = optim.Adam(autoencoder.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    loss_df = pd.DataFrame(columns=['batch_num', 'loss'])
    autoencoder = train_autoencoder(args, autoencoder, val_dataloader, val_dataloader, autoencoder_optimizer, 
                      criterion, 1, loss_df, args.autoencoder_epochs)

    if args.mle_pretraining:
        _, gen_mle_optimizer = prepare_opt_and_scheduler(args, gen, len(real_train_dataset))

    if args.mle_pretraining:
        logger.info('### Starting Generator MLE Training... ###')
        train_dataset = load_and_cache_examples_generator(args, tokenizer, evaluate=False)
        eval_dataset = load_and_cache_examples_generator(args, tokenizer, evaluate=True)
        train_generator_mle(args, train_dataset, gen, tokenizer, gen_mle_optimizer, eval_dataset)

    # TODO: add loading model config and test it
    # ADVERSARIAL TRAINING
    for epoch in range(ADV_TRAIN_EPOCHS):
        logger.info('### GAN EPOCH: {} ###'.format(epoch))

        # TRAIN DISCRIMINATOR
        loss, dis_optimizer, gen, dis = adversarial_train(args, gen, dis, tokenizer, dis_optimizer, real_train_dataset, 1)
        print("#### Average disriminator loss : {} ####".format(loss))
        wandb.log({"discriminator loss": loss})
        
        # TRAIN GENERATOR
        for gen_steps in range(3):
            loss, gen_optimizer, gen, dis = adversarial_train(args, gen, dis, tokenizer, gen_optimizer, 
                                                                real_train_dataset, 1, is_discriminator=False)
            print("#### Average generator loss : {} ####".format(loss))
            wandb.log({"generator loss": loss})

        set_seed(args)
        if args.record_run and epoch % 2 == 0:
            generated_samples = decoder.decode(gen.sample(10))
            wandb.log({"examples": wandb.Table(data=generated_samples, columns=["Generated Sequences"])})
            print("\n\n The generated samples are: ", generated_samples, "\n\n")

        # Add saving model config and test it
