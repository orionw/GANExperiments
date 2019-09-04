from __future__ import print_function
from math import ceil
import numpy as np
import sys
import pdb
import os

from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import models.generator as generator
import models.discriminator as discriminator
import utils.helpers as helpers
from utils.load_data import get_dataloaders, GenerationDatasetList, DiscriminatorDatasetFromList
import utils.argparser


CUDA = False
VOCAB_SIZE = 5000
START_LETTER = 0
BATCH_SIZE = 32
MLE_TRAIN_EPOCHS = 100
ADV_TRAIN_EPOCHS = 50
POS_NEG_SAMPLES = 10000

GEN_EMBEDDING_DIM = 32
GEN_HIDDEN_DIM = 32
DIS_EMBEDDING_DIM = 64
DIS_HIDDEN_DIM = 64

oracle_samples_path = './oracle_samples.trc'
oracle_state_dict_path = './oracle_EMBDIM32_HIDDENDIM32_VOCAB5000_MAXSEQLEN20.trc'
pretrained_gen_path = './gen_MLEtrain_EMBDIM32_HIDDENDIM32_VOCAB5000_MAXSEQLEN20.trc'
pretrained_dis_path = './dis_pretrain_EMBDIM_64_HIDDENDIM64_VOCAB5000_MAXSEQLEN20.trc'


def train_generator_MLE(gen, gen_opt, oracle, real_data_samples, epochs):
    """
    Max Likelihood Pretraining for the generator
    """
    for epoch in range(epochs):
        print('epoch %d : ' % (epoch + 1), end='')
        sys.stdout.flush()
        total_loss = 0

        for i in range(0, POS_NEG_SAMPLES, BATCH_SIZE):
            inp, target = helpers.prepare_generator_batch(real_data_samples[i:i + BATCH_SIZE], start_letter=START_LETTER,
                                                          gpu=CUDA)
            gen_opt.zero_grad()
            loss = gen.batchNLLLoss(inp, target)
            loss.backward()
            gen_opt.step()

            total_loss += loss.data.item()

            if (i / BATCH_SIZE) % ceil(
                            ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                print('.', end='')
                sys.stdout.flush()

        # each loss in a batch is loss per sample
        total_loss = total_loss / ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / args.max_seq_length

        # sample from generator and compute oracle NLL
        oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, args.max_seq_length,
                                                   start_letter=START_LETTER, gpu=CUDA)

        print(' average_train_NLL = %.4f, oracle_sample_NLL = %.4f' % (total_loss, oracle_loss))


def train_generator_PG(gen, gen_opt, oracle, dis, num_batches):
    """
    The generator is trained using policy gradients, using the reward from the discriminator.
    Training is done for num_batches batches.
    """

    for batch in range(num_batches):
        s = gen.sample(BATCH_SIZE*2)        # 64 works best
        inp, target = helpers.prepare_generator_batch(s, start_letter=START_LETTER, gpu=CUDA)
        rewards = dis.batchClassify(target)

        gen_opt.zero_grad()
        pg_loss = gen.batchPGLoss(inp, target, rewards)
        pg_loss.backward()
        gen_opt.step()

    # sample from generator and compute oracle NLL
    oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, args.max_seq_length,
                                                   start_letter=START_LETTER, gpu=CUDA)

    print(' oracle_sample_NLL = %.4f' % oracle_loss)


def train_discriminator(discriminator, dis_opt, true_samples_trainloader, true_samples_val, generator, d_steps, epochs: int):
    """
    Training the discriminator on true_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    :param true_samples_trainloader: a dataloader of the true samples
    """

    # generating a small validation set before training (using oracle and generator)
    pos_val = true_samples_val.dataset.data[0].tolist()[:5]
    print("Creating fake samples for the discriminator...")
    neg_val = generator.sample_text(5)
    validation_dataloader = DataLoader(GenerationDatasetList(pos_val + neg_val), batch_size=BATCH_SIZE)

    neg_val_train = generator.sample_text(5)
    pos_val_train = true_samples_trainloader.dataset.data[0].tolist()[:5]
    train_dataloader = DataLoader(DiscriminatorDatasetFromList(neg_val_train, pos_val_train), batch_size=BATCH_SIZE)

    for epoch in range(epochs):
        loop = tqdm(total=len(train_dataloader), position=0, leave=False)
        print('epoch %d : ' % (epoch + 1), end='')
        sys.stdout.flush()
        total_loss = 0
        total_acc = 0

        for index, (text, label) in enumerate(train_dataloader):
            dis_opt.zero_grad()
            out = discriminator.forward(text)
            loss_fn = nn.BCELoss()
            loss = loss_fn(out, target)
            loss.backward()
            dis_opt.step()

            total_loss += loss.data.item()
            total_acc += torch.sum((out>0.5)==(target>0.5)).data.item()

            if (i / BATCH_SIZE) % ceil(ceil(2 * POS_NEG_SAMPLES / float(
                    BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                print('.', end='')
                sys.stdout.flush()

        total_loss /= ceil(2 * POS_NEG_SAMPLES / float(BATCH_SIZE))
        total_acc /= float(2 * POS_NEG_SAMPLES)

        correct = 0
        total = 0
        with torch.no_grad():
            for text in tqdm(validation_dataloader):
                outputs = discriminator.forward(text)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the validation set: %d %%' % (
            100 * correct / total))


if __name__ == '__main__':
    global args  # for convenience,  I know it's a bad idea
    args = utils.argparser.parse_all_args(sys.argv)

    true_samples_train = get_dataloaders(os.path.join("data", "emnlp_news", "train.txt"))
    true_samples_val = get_dataloaders(os.path.join("data", "emnlp_news", "val.txt"))

    gen = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, args.max_seq_length, gpu=CUDA)
    dis = discriminator.Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, args.max_seq_length, gpu=CUDA)

    if torch.cuda.is_available():
        gen = gen.cuda()
        dis = dis.cuda()

    # GENERATOR MLE TRAINING
    print('Starting Generator MLE Training...')
    gen_optimizer = optim.Adam(gen.parameters(), lr=1e-2)
    # train_generator_MLE(gen, gen_optimizer, oracle, oracle_samples, MLE_TRAIN_EPOCHS)

    # torch.save(gen.state_dict(), pretrained_gen_path)
    # gen.load_state_dict(torch.load(pretrained_gen_path))

    # PRETRAIN DISCRIMINATOR
    print('\nStarting Discriminator Training...')
    dis_optimizer = optim.Adagrad(dis.parameters())
    train_discriminator(dis, dis_optimizer, true_samples_train, true_samples_val, gen, 50, 3)

    # torch.save(dis.state_dict(), pretrained_dis_path)
    # dis.load_state_dict(torch.load(pretrained_dis_path))

    # ADVERSARIAL TRAINING
    print('\nStarting Adversarial Training...')
    oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, args.max_seq_length,
                                               start_letter=START_LETTER, gpu=CUDA)
    print('\nInitial Oracle Sample Loss : %.4f' % oracle_loss)

    for epoch in range(ADV_TRAIN_EPOCHS):
        print('\n--------\nEPOCH %d\n--------' % (epoch+1))
        # TRAIN GENERATOR
        print('\nAdversarial Training Generator : ', end='')
        sys.stdout.flush()
        train_generator_PG(gen, gen_optimizer, oracle, dis, 1)

        # TRAIN DISCRIMINATOR
        print('\nAdversarial Training Discriminator : ')
        train_discriminator(dis, dis_optimizer, oracle_samples, gen, oracle, 5, 3)