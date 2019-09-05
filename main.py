from __future__ import print_function
from math import ceil
import numpy as np
import sys
import pdb
import os
import logging
import pickle

from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import models.generator as generator
import models.discriminator as discriminator
import utils.helpers as helpers
from utils.load_data import (get_dataloaders, GenerationDatasetList, DiscriminatorDatasetFromList, get_data, get_data_for_generator,
                                load_and_cache_examples_generator, TextDataset, load_and_cache_examples)

import utils.argparser

from utils.utils_glue import (compute_metrics, convert_examples_to_features,
                        output_modes, processors)

from models.training_functions import evaluate_discriminator, train_discriminator, evaluate_generator, train_generator, prepare_opt_and_scheduler

logger = logging.getLogger(__name__)

MLE_TRAIN_EPOCHS = 100
ADV_TRAIN_EPOCHS = 50

def run_epochs(args, model, tokenizer, optimizer, scheduler, is_discriminator: bool = True, given_dataset_train = None, given_dataset_val = None):
    """
    The default function to train and evaluate for `n_epochs` and return the results
    """
    train = train_discriminator if is_discriminator else train_generator
    evaluate = evaluate_discriminator if is_discriminator else evaluate_generator
    task_type = "cola" if is_discriminator else "gan"
    print("Evaluating on type: {} for {}".format(task_type, "discriminator" if is_discriminator else "generator"))

    # Training
    if args.do_train:
        if is_discriminator:
            train_dataset = load_and_cache_examples(args, task_type, tokenizer, given_dataset_train, evaluate=False)
        else:
            train_dataset = load_and_cache_examples_generator(args, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, optimizer, scheduler)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.model.model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    # Evaluation TODO: add this to the run_gan.sh script
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        if is_discriminator:
            eval_dataset = load_and_cache_examples(args, task_type, tokenizer, given_dataset_val, evaluate=True)
        else:
            eval_dataset = load_and_cache_examples_generator(args, tokenizer, evaluate=True)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model.model = model.model.model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, eval_dataset, prefix=global_step)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == '__main__':
    global args  # for convenience,  I know it's a bad idea
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
    helpers.set_seed(args)

    # get data
    # TODO: these are not used. Remove them?
    training_set_generator = get_data(os.path.join("data", "emnlp_news", "train.csv"))
    true_samples_val = get_data(os.path.join("data", "emnlp_news", "val.csv"))
    train_sample_length = len(true_samples_val)

    logger.info("Training/evaluation parameters %s", args)
    # get models
    gen = generator.Generator(args)
    dis = discriminator.Discriminator(args)
    gen.to(args.device)
    dis.to(args.device)

    # assign tokenizers
    tokenizer_gen = gen.tokenizer
    tokenizer_dis = dis.tokenizer

    gen, gen_optimizer, gen_scheduler = prepare_opt_and_scheduler(args, gen, train_sample_length)
    dis, dis_optimizer, dis_scheduler = prepare_opt_and_scheduler(args, dis, train_sample_length)


    # # GENERATOR MLE TRAINING # TODO add MLE training first for adaptation?
    # print('Starting Generator MLE Training...')
    # gen_optimizer = optim.Adam(gen.parameters(), lr=1e-2)
    # train_generator_MLE(gen, gen_optimizer, oracle, oracle_samples, MLE_TRAIN_EPOCHS)
    # TODO: add loading model config and test it

    # ADVERSARIAL TRAINING
    print('\nStarting Adversarial Training...')
    for epoch in range(ADV_TRAIN_EPOCHS):
        print('\n--------\nGAN EPOCH %d\n--------' % (epoch+1))
        # TRAIN GENERATOR
        print('\nAdversarial Training Generator : ', end='')
        sys.stdout.flush()
        run_epochs(args, gen, tokenizer_gen, gen_optimizer, gen_scheduler, is_discriminator=False)

        # TRAIN DISCRIMINATOR
        # TODO make dataset with generator
        print('Generating the negative samples for the discriminator')
        negative_samples = gen.sample_text(train_sample_length)
        print("The Generators output is: {}".format(" ".join([sentence for sentence in negative_samples])))
        training_set_discriminator = get_data_for_generator(os.path.join("data", "emnlp_news", "val.csv"), negative_samples)
        print('\nAdversarial Training Discriminator : ')    
        run_epochs(args, dis, tokenizer_dis, dis_optimizer, dis_scheduler, given_dataset_train=training_set_discriminator, given_dataset_val=training_set_discriminator)
