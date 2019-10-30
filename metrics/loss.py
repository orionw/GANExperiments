import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

latent_dim = 20
categorical_dim = 10 # one-of-K vector


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    sample = -Variable(torch.log(-torch.log(U + eps) + eps)).cuda()
    return sample


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1,latent_dim*categorical_dim)


def get_fixed_temperature(temper, i, N, adapt):
    """A function to set up different temperature control policies"""
    N = 5000

    if adapt == 'no':
        temper_var_np = temper  # no increase
    elif adapt == 'lin':
        temper_var_np = 1 + i / (N - 1) * (temper - 1)  # linear increase
    elif adapt == 'exp':
        temper_var_np = temper ** (i / N)  # exponential increase
    elif adapt == 'log':
        temper_var_np = 1 + (temper - 1) / np.log(N) * np.log(i + 1)  # logarithm increase
    elif adapt == 'sigmoid':
        temper_var_np = (temper - 1) * 1 / (1 + np.exp((N / 2 - i) * 20 / N)) + 1  # sigmoid increase
    elif adapt == 'quad':
        temper_var_np = (temper - 1) / (N - 1) ** 2 * i ** 2 + 1
    elif adapt == 'sqrt':
        temper_var_np = (temper - 1) / np.sqrt(N - 1) * np.sqrt(i) + 1
    else:
        raise Exception("Unknown adapt type!")

    return temper_var_np


def get_losses(d_out_real, d_out_fake, loss_type='JS'):
    """Get different adversarial losses according to given loss_type"""
    bce_loss = nn.BCEWithLogitsLoss()

    if loss_type == 'standard':  # the non-satuating GAN loss
        d_loss_real = bce_loss(d_out_real, torch.ones_like(d_out_real))
        d_loss_fake = bce_loss(d_out_fake, torch.zeros_like(d_out_fake))
        d_loss = d_loss_real + d_loss_fake

        g_loss = bce_loss(d_out_fake, torch.ones_like(d_out_fake))

    elif loss_type == 'JS':  # the vanilla GAN loss
        d_loss_real = bce_loss(d_out_real, torch.ones_like(d_out_real))
        d_loss_fake = bce_loss(d_out_fake, torch.zeros_like(d_out_fake))
        d_loss = d_loss_real + d_loss_fake

        g_loss = -d_loss_fake

    elif loss_type == 'KL':  # the GAN loss implicitly minimizing KL-divergence
        d_loss_real = bce_loss(d_out_real, torch.ones_like(d_out_real))
        d_loss_fake = bce_loss(d_out_fake, torch.zeros_like(d_out_fake))
        d_loss = d_loss_real + d_loss_fake

        g_loss = torch.mean(-d_out_fake)

    elif loss_type == 'hinge':  # the hinge loss
        d_loss_real = torch.mean(F.relu(1.0 - d_out_real))
        d_loss_fake = torch.mean(F.relu(1.0 + d_out_fake))
        d_loss = d_loss_real + d_loss_fake

        g_loss = -torch.mean(d_out_fake)

    elif loss_type == 'tv':  # the total variation distance
        d_loss = torch.mean(F.tanh(d_out_fake) - F.tanh(d_out_real))
        g_loss = torch.mean(-F.tanh(d_out_fake))

    elif loss_type == 'rsgan':  # relativistic standard GAN
        d_loss = bce_loss(d_out_real - d_out_fake, torch.ones_like(d_out_real))
        g_loss = bce_loss(d_out_fake - d_out_real, torch.ones_like(d_out_fake))

    else:
        raise NotImplementedError("Divergence '%s' is not implemented" % loss_type)

    return g_loss, d_loss


def truncated_normal_(tensor, mean=0, std=1):
    """
    Implemented by @ruotianluo
    See https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor
