import torch
import random
import numpy as np
import os
import wandb

def convert_to_text(output_logits, tokenizer, given_ids=False):
    """
    A greedy decoding of text from logits
    param: output_logits: either a tensor of tokens or logits of shape (vocab, seq_len)
    returns a string containing the decoded text
    """
    if not given_ids:
        _, tokens = torch.max(output_logits, dim=0) # is our best guess
    else:
        tokens = output_logits # not really logits, given id tokens
    predicted = tokenizer.convert_ids_to_tokens(tokens.tolist())
    return " ".join([item for item in predicted])


def vectorized_convert_to_text(output_logits, tokenizer):
    """
    Same as the above function, but for a batch of logits
    Inputs:
        output_logits: a tensor of shape (vocab_size, batch_size, seq_len)
    Returns:
        A list of sampled text
    """
    sampled_text = []
    for index in range(output_logits.shape[1]):
        sampled_text.append(convert_to_text(output_logits[:, index, :], tokenizer))
    return sampled_text


def sample_and_record_text(args, gen, decoder, tokenizer):
    """
    A function for sampling text from the generator and decoder it.  It also prints and uploads to wandb, depdending on options
    param: args: a Namespace of run options
    param: gen: the generator network
    param: decoder: the decoder network
    param: tokenizer: the tokenizer for the run (also the same as the tokenizer for the disciminator)
    """
    set_seed(args) # set seed to make sure the generator is generating different content and not just random seed changes
    decoder.to(args.device)
    sampled_embeddings = gen.sample(args.train_batch_size).to(args.device) # size (1, batch_size, embedding_size)
    decoded_logits = decoder(sampled_embeddings, args.max_seq_length, args.train_batch_size, device=args.device)
    decoder.to("cpu:0")  # can't handle that much memory usage
    shaped_logits = decoded_logits.permute(2, 1, 0)  # swap axis for tokenizer to get (logits, batch_size, seq_len)
    text = vectorized_convert_to_text(shaped_logits, tokenizer)
    if args.record_run:
        wandb.log({"examples": wandb.Table(data=text, columns=["Generated Sequences"])})


def set_seed(args):
    """ Sets the seed for the random, numpy, and torch modules using the argument value """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def save_states(args, gen, dis, gen_opt, dis_opt, epochs, name="best"):
    """ Used for saving models, optimizers, and the number of epochs """
    torch.save({
            'epochs': epochs,
            'gen_state_dict': gen.state_dict(),
            "dis_state_dict": dis.state_dict(),
            'gen_optimizer_state_dict': gen_opt.state_dict(),
            'dis_optimizer_state_dict': dis_opt.state_dict(),
            }, os.path.join(args.output_dir, "checkpoint-gan-{}.tar".format(args.run_name)))

