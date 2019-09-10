import argparse
import logging
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from pytorch_transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig

from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer
from pytorch_transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from pytorch_transformers import XLNetLMHeadModel, XLNetTokenizer
from pytorch_transformers import TransfoXLLMHeadModel, TransfoXLTokenizer

from metrics.loss import gumbel_softmax

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig)), ())

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'xlnet': (XLNetConfig, XLNetLMHeadModel, XLNetTokenizer),
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


class PretrainedTransformerGenerator(nn.Module):

    def __init__(self, args):
        super().__init__()
        # Load pretrained model and tokenizer
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

        config_class, self.model_class, tokenizer_class = MODEL_CLASSES[args.gen_model_type]
        self.config = config_class.from_pretrained(args.config_name if args.config_name else args.gen_model_name_or_path)
        self.tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.gen_model_name_or_path, do_lower_case=args.do_lower_case)
        if args.block_size <= 0:
            args.block_size = self.tokenizer.max_len  # Our input block size will be the max possible for the model
        args.block_size = min(args.block_size, self.tokenizer.max_len)
        self.model = self.model_class.from_pretrained(args.gen_model_name_or_path, from_tf=bool('.ckpt' in args.gen_model_name_or_path), config=self.config)
        self.model.to(args.device)
        self.args = args

        if args.local_rank == 0:
            torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab


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

        list_of_samples = []
        for sample_num in range(num_samples):
            context_tokens = self.tokenizer.encode(raw_text)
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
            out = out[0, len(context_tokens):].tolist()
            list_of_samples.append(out)

        return list_of_samples

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
        for sample_num in range(num_samples):
            context_tokens = self.tokenizer.encode(raw_text)
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
            out = out[0, len(context_tokens):].tolist()
            sequence = self.tokenizer.decode(out, clean_up_tokenization_spaces=True)
            list_of_samples.append(sequence)
        return list_of_samples

    def forward(self, inputs, **kwargs):
        return self.model(inputs, **kwargs)

    def sample_sequence(self, length, model, context, num_samples=1, temperature=1, top_k=0, top_p=0.0, is_xlnet=False, device='cpu'):
        context = torch.tensor(context, dtype=torch.long, device=device)
        context = context.unsqueeze(0).repeat(num_samples, 1)
        generated = context
        with torch.no_grad():
            for _ in trange(length):

                inputs = {'input_ids': generated}
                if is_xlnet: 
                    # XLNet is a direct (predict same token, not next token) and bi-directional model by default
                    # => need one additional dummy token in the input (will be masked), attention mask and target mapping (see model docstring)
                    input_ids = torch.cat((generated, torch.zeros((num_samples, 1), dtype=torch.long, device=device)), dim=1)
                    perm_mask = torch.zeros((num_samples, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float, device=device)
                    perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
                    target_mapping = torch.zeros((num_samples, 1, input_ids.shape[1]), dtype=torch.float, device=device)
                    target_mapping[0, 0, -1] = 1.0  # predict last token
                    inputs = {'input_ids': input_ids, 'perm_mask': perm_mask, 'target_mapping': target_mapping}

                outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
                next_token_logits = outputs[0][0, -1, :] / temperature
                filtered_logits = self.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=num_samples)
                generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
        return generated

    def top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
        return logits

    def set_seed(self, args):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(seed)
    
    # def sample(self, num_samples: int = 1):
    #     if self.args.length < 0 and self.config.max_position_embeddings > 0:
    #         self.length = self.config.max_position_embeddings
    #     elif 0 < self.config.max_position_embeddings < self.args.length:
    #         self.args.length = self.config.max_position_embeddings  # No generation bigger than model size 
    #     elif self.args.length < 0:
    #         self.args.length = MAX_LENGTH  # avoid infinite loop

    #     raw_text = "what"
    #     if self.args.gen_model_type in ["transfo-xl", "xlnet"]:
    #         # Models with memory likes to have a long prompt for short inputs.
    #         raw_text = (self.args.padding_text if self.args.padding_text else PADDING_TEXT) + raw_text

    #     list_of_samples = []
    #     context_tokens = self.tokenizer.encode(raw_text)
    #     out = self.sample_sequence(
    #         model=self.model,
    #         num_samples=num_samples,
    #         context=context_tokens,
    #         length=self.args.length,
    #         temperature=self.args.temperature,
    #         top_k=self.args.top_k,
    #         top_p=self.args.top_p,
    #         device=self.args.device,
    #         is_xlnet=bool(self.args.gen_model_type == "xlnet"),
    #     )
    #     out = out[0, len(context_tokens):].tolist()
    #     return out

    # def sample_text(self, num_samples: int):
    #     outputs = self.sample(num_samples)
    #     final_outputs = []
    #     for sample in outputs:
    #         decoded_sample = self.tokenizer.decode(sample, clean_up_tokenization_spaces=True)
    #         final_outputs.append(decoded_sample)
    #     return final_outputs

    # def forward(self, inputs, **kwargs):
    #     return self.model(inputs, **kwargs)

    # def sample_sequence(self, length, model, context, num_samples=1, temperature=1, top_k=0, top_p=0.0, is_xlnet=False, device='cpu'):
    #     context = torch.tensor(context, dtype=torch.long, device=device)
    #     list_of_generated = []
    #     with torch.no_grad():
    #         curr_batch_size = min(self.args.per_gpu_train_batch_size, num_samples)
    #         batch_context = context.unsqueeze(0).repeat(curr_batch_size, 1)
    #         for batch in range(math.ceil(num_samples / self.args.per_gpu_train_batch_size)):
    #             generated = batch_context.clone()
    #             for _ in trange(length):
    #                 inputs = {'input_ids': generated}
    #                 if is_xlnet: 
    #                     # XLNet is a direct (predict same token, not next token) and bi-directional model by default
    #                     # => need one additional dummy token in the input (will be masked), attention mask and target mapping (see model docstring)
    #                     input_ids = torch.cat((generated, torch.zeros((curr_batch_size, 1), dtype=torch.long, device=device)), dim=1)
    #                     perm_mask = torch.zeros((curr_batch_size, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float, device=device)
    #                     perm_mask[:, :, -1] = 1.0  # curr_batch_size tokens don't see last token
    #                     target_mapping = torch.zeros((curr_batch_size, 1, input_ids.shape[1]), dtype=torch.float, device=device)
    #                     target_mapping[0, 0, -1] = 1.0  # predict last token
    #                     inputs = {'input_ids': input_ids, 'perm_mask': perm_mask, 'target_mapping': target_mapping}

    #                 outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
    #                 next_token_logits = outputs[0][:, -1, :] / temperature
    #                 filtered_logits = self.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
    #                 next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=curr_batch_size)
    #                 # concatenate the batch size nex tokens with the current sequence
    #                 generated = torch.cat((generated, next_token.transpose(0, 1)), dim=1)

    #             list_of_generated.append(generated.cpu())
        
    #     final_generated = torch.cat(list_of_generated, axis=0)
    #     print("Generated ", final_generated.shape)
    #     return final_generated

    # def top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    #     """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    #         Args:
    #             logits: logits distribution shape (vocabulary size)
    #             top_k > 0: keep only top k tokens with highest probability (top-k filtering).
    #             top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    #                 Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    #         From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    #     """
    #     # I am vectorizing it... ignore next comment
    #     # assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    #     top_k = min(top_k, logits.size(-1))  # Safety check
    #     if top_k > 0:
    #         # Remove all tokens with a probability less than the last token of the top-k
    #         indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    #         logits[indices_to_remove] = filter_value

    #     # if top_p > 0.0:
    #     #     sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    #     #     cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    #     #     # Remove tokens with cumulative probability above the threshold
    #     #     sorted_indices_to_remove = cumulative_probs > top_p
    #     #     # Shift the indices to the right to keep also the first token above the threshold
    #     #     sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    #     #     sorted_indices_to_remove[..., 0] = 0

    #     #     indices_to_remove = sorted_indices[sorted_indices_to_remove]
    #     #     logits[indices_to_remove] = filter_value
    #     return logits

    def set_seed(self, args):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def add_gumbel(o_t, eps=1e-10, gpu=True):
        """Add o_t by a vector sampled from Gumbel(0,1)"""
        u = torch.rand(o_t.size())
        if gpu:
            u = u.cuda()
        g_t = -torch.log(-torch.log(u + eps) + eps)
        gumbel_t = o_t + g_t
        return gumbel_t
