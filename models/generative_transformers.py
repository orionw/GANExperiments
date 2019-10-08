import argparse
import logging
import numpy as np
import math
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

from pytorch_transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig
from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer
from pytorch_transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from pytorch_transformers import XLNetLMHeadModel, XLNetTokenizer
from pytorch_transformers import TransfoXLLMHeadModel, TransfoXLTokenizer

from models.training_functions import set_seed
from models.base_models import GeneratorBase
from models.xlnet import XLNetEmbedder

logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig)), ())

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'xlnet': (XLNetConfig, XLNetEmbedder, XLNetTokenizer),
    'transfo-xl': (TransfoXLConfig, TransfoXLLMHeadModel, TransfoXLTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """ In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


class PretrainedTransformerGenerator(GeneratorBase):

    def __init__(self, args, tokenizer):
        super().__init__(args)
        # Load pretrained model and tokenizer
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

        config_class, self.model_class, tokenizer_class = MODEL_CLASSES[args.gen_model_type]
        self.config = config_class.from_pretrained(args.config_name if args.config_name else args.gen_model_name_or_path)
        self.tokenizer = tokenizer
        if args.block_size <= 0:
            args.block_size = self.tokenizer.max_len  # Our input block size will be the max possible for the model
        args.block_size = min(args.block_size, self.tokenizer.max_len)
        self.model = self.model_class.from_pretrained(args.gen_model_name_or_path, from_tf=bool('.ckpt' in args.gen_model_name_or_path), config=self.config)
        self.model.to(args.device)
        self.args = args

        if args.local_rank == 0:
            torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    def forward(self, **kwargs):
        output = self.model(**kwargs)   # The last hidden-state is the first element of the output tuple
        output_layer = output[0]
        embedding = torch.mean(output_layer, dim=1)  # get sentence embedding from mean of word embeddings
        embedding = embedding.unsqueeze(dim=0) # (1, batch_size, embedding_dim)
        return embedding

    def decode(self, embedding, target=None):
        return self.model.decode((embedding, ), target)

    def sample(self, num_samples: int):
        if self.args.length < 0 and self.config.max_position_embeddings > 0:
            self.length = self.config.max_position_embeddings
        elif 0 < self.config.max_position_embeddings < self.args.length:
            self.args.length = self.config.max_position_embeddings  # No generation bigger than model size 
        elif self.args.length < 0:
            self.args.length = MAX_LENGTH  # avoid infinite loop

        raw_text = "what"
        if self.args.gen_model_type in ["transfo-xl", "xlnet"]:
            # Models with memory likes to have a long prompt for short inputs.
            raw_text = (self.args.padding_text if self.args.padding_text else PADDING_TEXT) + raw_text

        context_tokens = self.tokenizer.encode(raw_text)
        out = self.sample_embeddings(
            model=self.model,
            context=context_tokens,
            temperature=self.args.temperature,
            top_k=self.args.top_k,
            top_p=self.args.top_p,
            num_samples=num_samples,
            device=self.args.device,
            is_xlnet=bool(self.args.gen_model_type == "xlnet"),
        )
        return out

    def sample_text(self, num_samples: int):
        if self.args.length < 0 and self.config.max_position_embeddings > 0:
            self.length = self.config.max_position_embeddings
        elif 0 < self.config.max_position_embeddings < self.args.length:
            self.args.length = self.config.max_position_embeddings  # No generation bigger than model size 
        elif self.args.length < 0:
            self.args.length = MAX_LENGTH  # avoid infinite loop

        raw_text = "what"
        if self.args.gen_model_type in ["transfo-xl", "xlnet"]:
            # Models with memory likes to have a long prompt for short inputs.
            raw_text = (self.args.padding_text if self.args.padding_text else PADDING_TEXT) + raw_text

        list_of_samples = []
        context_tokens = self.tokenizer.encode(raw_text)
        for sample_num in range(num_samples):
            if sample_num % 100 == 0:
                print("On sample number {}".format(sample_num))
            out = self.sample_sequence(
                model=self.model,
                context=context_tokens,
                length=self.args.length,
                temperature=self.args.temperature,
                top_k=self.args.top_k,
                top_p=self.args.top_p,
                device=self.args.device,
                is_xlnet=bool(self.args.gen_model_type == "xlnet"),
            )
            out = out[0, len(context_tokens):].cpu().tolist()
            sequence = self.tokenizer.decode(out, clean_up_tokenization_spaces=True)
            list_of_samples.append(sequence)
        return list_of_samples

    def sample_sequence(self, length, model, context, num_samples=1, temperature=1, top_k=0, top_p=0.0, is_xlnet=False, device='cpu'):
        context = torch.tensor(context, dtype=torch.long, device=device)
        context = context.unsqueeze(0).repeat(1, 1)
        model.train()
        final_generated = None
        for sample in range(num_samples):
            generated = context.clone()
            model.train()
            for _ in range(length):
                
                inputs = {'input_ids': generated}
                if is_xlnet: 
                    # XLNet is a direct (predict same token, not next token) and bi-directional model by default
                    # => need one additional dummy token in the input (will be masked), attention mask and target mapping (see model docstring)
                    input_ids = torch.cat((generated, torch.zeros((1, 1), dtype=torch.long, device=device)), dim=1)
                    perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float, device=device)
                    perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
                    target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float, device=device)
                    target_mapping[0, 0, -1] = 1.0  # predict last token
                    inputs = {'input_ids': input_ids, 'perm_mask': perm_mask, 'target_mapping': target_mapping}

                outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
                next_token = torch.argmax(F.softmax(next_token_logits, dim=-1)).unsqueeze(0)
                generated = torch.cat((generated.float(), next_token.unsqueeze(0).float()), dim=1)

            final_generated = generated if final_generated is None else torch.cat((final_generated, generated), dim=0)
        assert final_generated.requires_grad == True, "outputs do not require grad, error"
        return final_generated

    def sample_embeddings(self, model, context, num_samples=1, temperature=1, top_k=0, top_p=0.0, is_xlnet=False, device='cpu'):
        context = torch.tensor(context, dtype=torch.long, device=device)
        context = context.unsqueeze(0).repeat(1, 1)
        batch_outputs = torch.zeros((1, num_samples, 768))
        for sample in range(num_samples):
            generated = context.clone()
            model.train()
            inputs = {'input_ids': generated}
            if is_xlnet: 
                # XLNet is a direct (predict same token, not next token) and bi-directional model by default
                # => need one additional dummy token in the input (will be masked), attention mask and target mapping (see model docstring)
                input_ids = torch.cat((generated, torch.zeros((1, 1), dtype=torch.long, device=device)), dim=1)
                perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float, device=device)
                perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
                target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float, device=device)
                target_mapping[0, 0, -1] = 1.0  # predict last token
                inputs = {'input_ids': input_ids, 'perm_mask': perm_mask, 'target_mapping': target_mapping}

            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            batch_outputs[0, sample, :] = outputs[0]
        assert batch_outputs.requires_grad == True, "outputs do not require grad, error"
        return batch_outputs

    def create_tokens(self, output,
                      cls_token_at_end=False,
                      cls_token='[CLS]',
                      cls_token_segment_id=1,
                      sep_token='[SEP]',
                      sep_token_extra=False,
                      pad_on_left=False,
                      pad_token=0,
                      pad_token_segment_id=0,
                      sequence_a_segment_id=0, 
                      sequence_b_segment_id=1,
                      mask_padding_with_zero=True):
        """
        Used to pad the output of a sample batch from the generator to match what is made for the discriminator
        Takes into account all the models and token configs
        """
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(output) > self.args.max_seq_length - special_tokens_count:
            raise Exception("too many tokens - should never occur")
            # tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        empty_col = torch.Tensor(output.shape[0]).cuda()
        tokens = empty_col.clone().fill_(self.tokenizer.convert_tokens_to_ids([sep_token])[0]).unsqueeze(1).float()
        tokens = torch.cat((output, tokens), dim=1)
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens = torch.cat((tokens, empty_col.clone().fill_(self.tokenizer.convert_tokens_to_ids([sep_token])[0]).unsqueeze(1)), dim=1)
        segment_ids = torch.zeros(tokens.shape).fill_(sequence_a_segment_id).cuda().float()

        end_segment = empty_col.clone().fill_(cls_token_segment_id).unsqueeze(1).cuda().float()
        if cls_token_at_end:
            cls_tokens = empty_col.clone().fill_(self.tokenizer.convert_tokens_to_ids([cls_token])[0]).unsqueeze(1).cuda().float()
            tokens = torch.cat((tokens, cls_tokens), dim=1)
            segment_ids = torch.cat((segment_ids, end_segment), dim=1)
        else:
            cls_tokens = empty_col.clone().fill_(self.tokenizer.convert_tokens_to_ids([cls_token])[0]).unsqueeze(1).cuda().float()
            tokens = torch.cat((cls_tokens, tokens), dim=1)
            segment_ids = torch.cat((end_segment, segment_ids), dim=1)

        input_ids = tokens

        # Zero-pad up to the sequence length.
        padding_length = self.args.max_seq_length - input_ids.shape[1]
        padding_segment = torch.zeros((output.shape[0], padding_length)).fill_(pad_token_segment_id).cuda().float()
        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = torch.zeros((input_ids.shape[0], input_ids.shape[1] + padding_length)).fill_(1 if mask_padding_with_zero else 0).cuda()
        if pad_on_left:
            input_ids = torch.cat((torch.Tensor(input_ids.shape[0], padding_length).fill_(pad_token).cuda().float(), input_ids), dim=1)
            segment_ids = torch.cat((padding_segment, segment_ids), dim=1)
        else:
            input_ids = torch.cat((input_ids, torch.Tensor(input_ids.shape[0], padding_length).fill_(pad_token).cuda().float()), dim=1)
            segment_ids = torch.cat((segment_ids, padding_segment), dim=1)

        assert input_ids.shape[1] == self.args.max_seq_length
        assert input_mask.shape[1] == self.args.max_seq_length
        assert segment_ids.shape[1] == self.args.max_seq_length

        labels = torch.zeros(input_ids.shape[0]).long() # labels are all zeros
        return [input_ids, input_mask, segment_ids, labels]

    # def top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    #     """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    #         Args:
    #             logits: logits distribution shape (vocabulary size)
    #             top_k > 0: keep only top k tokens with highest probability (top-k filtering).
    #             top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    #                 Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    #         From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    #     """
    #     assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    #     top_k = min(top_k, logits.size(-1))  # Safety check
    #     if top_k > 0:
    #         # Remove all tokens with a probability less than the last token of the top-k
    #         indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    #         logits[indices_to_remove] = filter_value

    #     if top_p > 0.0:
    #         sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    #         cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    #         # Remove tokens with cumulative probability above the threshold
    #         sorted_indices_to_remove = cumulative_probs > top_p
    #         # Shift the indices to the right to keep also the first token above the threshold
    #         sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    #         sorted_indices_to_remove[..., 0] = 0

    #         indices_to_remove = sorted_indices[sorted_indices_to_remove]
    #         logits[indices_to_remove] = filter_value
    #     return logits

    @staticmethod
    def add_gumbel(o_t, eps=1e-10, gpu=True):
        """Add o_t by a vector sampled from Gumbel(0,1)"""
        u = torch.rand(o_t.size())
        if gpu:
            u = u.cuda()
        g_t = -torch.log(-torch.log(u + eps) + eps)
        gumbel_t = o_t + g_t
        return gumbel_t
