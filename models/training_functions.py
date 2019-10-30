import argparse
import glob
import logging
import os
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

import torch
from torch.nn.functional import one_hot
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer,
                                  AdamW, WarmupLinearSchedule)

from utils.utils_glue import (compute_metrics, convert_examples_to_features,
                        output_modes, processors)
from utils.helpers import convert_to_text, set_seed
from metrics.loss import get_losses
from metrics.bleu import *


logger = logging.getLogger(__name__)


def discriminator_eval(args, batch, model, tokenizer, prefix=""):
    inputs =  { 'given_embedding': batch}
    outputs = model(**inputs)
    logits, _ = outputs[:2]
    return logits


def create_transformer_mapping(batch, model_type="xlnet", to_cuda=False):
    if to_cuda:
        batch = tuple(t.cuda() for t in batch)

    inputs =  { 'input_ids':      batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2] if model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                'labels':         batch[3]}
    return inputs


def mask_tokens(inputs, tokenizer, args):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    masked_indices = torch.bernoulli(torch.full(labels.shape, args.mlm_probability)).bool()
    labels[~masked_indices] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def train_generator_mle(args, train_dataset, model, tokenizer, optimizer, eval_dataset):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()
            outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.model.model.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return evaluate_generator_mle(args, model, tokenizer, eval_dataset)


def evaluate_generator_mle(args, model, tokenizer, eval_dataset, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    results = {}

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = batch.to(args.device)

        with torch.no_grad():
            outputs = model(batch, masked_lm_labels=batch) if args.mlm else model(batch, labels=batch)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "perplexity": perplexity
    }

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return results

def prepare_opt_and_scheduler(args, model, data_len):
    if args.max_steps > 0:
        t_total = args.max_steps
    else:
        t_total = data_len // args.gradient_accumulation_steps * args.num_train_epochs
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    return model, optimizer #, scheduler


def train_autoencoder(args, model, train_dataloader, val_dataloader, optimizer, criterion, clip, loss_df, num_epochs):
    """
    The training function for the autoencoder model
    Inputs:
        args: A Namespace of training args given at runtime
        model: the Autoencoder model, containing an encoder and a decoder
        train_dataloader: a dataloader of training data
        val_dataloader: a dataloader of validation data
        optimizer: the optimizer for the decoder (NOTE: the encoder should be pretrained)
        criterion: the loss function to use
        clip: the level to clip the loss at
        loss_df: a Pandas DF to hold and plot loss values
        num_epochs: the number of epochs to run the training

    Returns:
        the trained autoencoder model

    This function also logs to wandb if the options are selected.  It also saves the best model to file.
    """
    # try:
    loss_list = []
    recent_loss = []
    mean_val_loss = float("inf")
    best_loss = float("inf")
    val_bleu = 0
    best_bleu = 0
    for epoch in range(num_epochs):
        loop = tqdm(total=len(train_dataloader), position=0, leave=True)
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            target = batch[0].to(args.device)
            output = model(batch, target)
            # reshape the objects so that we can get the loss
            output = output.permute((1, 2, 0)).to(args.device)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            loss_list.append({'batch_num': model.number_of_batches_seen, 'loss': float(loss.item())})
            recent_loss.append(loss.item())
            loop.set_description('epoch:{}. loss:{:.4f}. last_val_loss:{:.4f}. last_val_bleu:{:.4f}'.format(epoch, float(loss.item()),
                                                                                                            mean_val_loss, val_bleu))
            loop.update(1)
            if model.number_of_batches_seen % 100 == 0:
                # see example of output to see how we're doing
                output_text = convert_to_text(output[0], model.tokenizer)
                input_text = convert_to_text(target[0], model.tokenizer, given_ids=True)
                sample_str = "The autoencoder recieved {} \n but produced {} \n".format(input_text, output_text)
                if args.record_run:
                    wandb.log({"autoencoder_samples": wandb.Table(data=[input_text, output_text], columns=["Input", "Output"])})

            # save losses
            if model.number_of_batches_seen % 100 == 0:
                num_batches = model.number_of_batches_seen
                with torch.no_grad():
                    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
                    val_losses = []
                    for i, val_batch in enumerate(val_dataloader):
                        target = val_batch[0].to(args.device)
                        output = model(val_batch, target) 
                        # reshape the objects so that we can get the loss -> now size (batch_size. logits, seq_len)
                        output = output.permute((1, 2, 0)).to(args.device)
                        val_loss = criterion(output, target)
                        val_losses.append(val_loss.item())
                        _, best_guess = torch.max(output, dim=1)  # the logit dimension
                        # calculates bleu statistics and sum them up
                        stats += get_bleu(best_guess, val_batch[0])
                    val_bleu = bleu(stats)
                    mean_val_loss = np.mean(val_losses)

                    if mean_val_loss < best_loss and val_bleu > best_bleu and args.record_run:
                        # want the best model, no matter what
                        exit(1)
                        torch.save(model.decoder.state_dict(), os.path.join(args.output_dir, "decoder-{}-best.pt".format(args.run_name)))

                if args.record_run:
                    wandb.log({"autoencoder_training_loss": np.mean(recent_loss)})
                    wandb.log({"autoencoder_val_loss": mean_val_loss})
                    wandb.log({"autoencoder_bleu": val_bleu})
                    loss_df = loss_df.append(pd.DataFrame(loss_list), ignore_index=True)
                    loss_list = []
                    recent_loss = []

                # don't include validation batches
                model.number_of_batches_seen = num_batches
                

    if args.record_run and loss_df[0] != 0 and epoch and epoch % 5 == 0:
        fig = loss_df.plot(x="batch_num", y="loss")
        plt.savefig(os.path.join(args.output_dir, "loss.png"))
        torch.save(model.decoder.state_dict(), os.path.join(args.output_dir, "decoder-{}-{}-batches.pt".format(str(time.time()), model.number_of_batches_seen)))

    return model

    # except KeyboardInterrupt:
    #     # save the model and leave
    #     torch.save(model.decoder.state_dict(), os.path.join(args.output_dir, "decoder-{}-{}-batches.pt".format(str(time.time()), model.number_of_batches_seen)))
    #     return model

def adversarial_train(args, gen, dis, encoder, tokenizer, optimizer, training_dataloader, num_steps, is_discriminator = True):
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    total_loss = 0

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

    train_iterator = trange(int(1), desc="Epoch", disable=args.local_rank not in [-1, 0])
    for _ in train_iterator:
        epoch_iterator = tqdm(training_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, (batch) in enumerate(epoch_iterator):
            dis.train() if is_discriminator else gen.train() # only optimize the one
            # Get real embeddings
            batch = tuple(t.to(args.device) for t in batch)
            batch_inputs = create_transformer_mapping(batch, "xlnet")
            real_embedding = encoder(**batch_inputs)
            d_out_real = discriminator_eval(args, real_embedding, dis, tokenizer)
            # get fake embeddings
            gen_batch = gen.sample(args.train_batch_size).cuda() # (1, batch_size, embedding_dim)
            d_out_fake = discriminator_eval(args, gen_batch, dis, tokenizer)
            # compute losses and return the relevant one
            assert d_out_real.shape == d_out_fake.shape, "shapes are not aligned, error"
            if is_discriminator:
                _, loss = get_losses(d_out_real, d_out_fake, args.loss_type)
            else:
                loss, _ = get_losses(d_out_real, d_out_fake, args.loss_type)
            epoch_iterator.set_description('loss:{:.4f}'.format(loss.item()))
            epoch_iterator.update(1)
            optimize(optimizer, loss, dis if is_discriminator else gen)
            total_loss += loss.item()

    average_loss = total_loss / num_steps if num_steps != 0 else 0
    return average_loss, optimizer, gen, dis


def optimize(opt, loss, model=None, retain_graph=False):
    opt.zero_grad()
    loss.backward(retain_graph=retain_graph)
    if model is not None: 
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0) # 5 is from RelGan paper/impl.
    opt.step()
