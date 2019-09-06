import argparse
from pytorch_transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig

from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer
from pytorch_transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from pytorch_transformers import XLNetLMHeadModel, XLNetTokenizer
from pytorch_transformers import TransfoXLLMHeadModel, TransfoXLTokenizer

from pytorch_transformers import (WEIGHTS_NAME, AdamW, WarmupLinearSchedule,
                                  BertConfig, BertForMaskedLM, BertTokenizer,
                                  GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                                  OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                                  RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig)), ())

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)
}

def parse_all_args(arglist):
     # use a function to gather all args for all tasks, LM, classification, and generation
     parser = argparse.ArgumentParser()

     ## Required parameters
     parser.add_argument("--train_data_file", default=None, type=str, required=True,
                    help="The input training data file (a text file).")
     parser.add_argument("--data_dir", default=None, type=str, required=True,
                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
     parser.add_argument("--output_dir", default=None, type=str, required=True,
                    help="The output directory where the model predictions and checkpoints will be written.")                    
     parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
               help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))

     ## Other parameters
     parser.add_argument("--eval_data_file", default=None, type=str,
                    help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
     parser.add_argument("--model_type", default="bert", type=str,
                    help="The model architecture to be fine-tuned.")
     parser.add_argument("--loss_type", type=str, default="RSGAN",
               help="What type of loss to use for gan training")
     parser.add_argument("--mlm", action='store_true',
                    help="Train with masked-language modeling loss instead of language modeling.")
     parser.add_argument("--mlm_probability", type=float, default=0.15,
                    help="Ratio of tokens to mask for masked language modeling loss")
     parser.add_argument("--config_name", default="", type=str,
                    help="Optional pretrained config name or path if not the same as model_name_or_path")
     parser.add_argument("--tokenizer_name", default="", type=str,
                    help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
     parser.add_argument("--cache_dir", default="", type=str,
                    help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
     parser.add_argument("--block_size", default=-1, type=int,
                    help="Optional input sequence length after tokenization."
                         "The training dataset will be truncated in block of this size for training."
                         "Default to the model max input length for single sentence inputs (take into account special tokens).")
     parser.add_argument("--do_train", action='store_true',
                    help="Whether to run training.")
     parser.add_argument("--do_eval", action='store_true',
                    help="Whether to run eval on the dev set.")
     parser.add_argument("--evaluate_during_training", action='store_true',
                    help="Run evaluation during training at each logging step.")
     parser.add_argument("--do_lower_case", action='store_true',
                    help="Set this flag if you are using an uncased model.")
     parser.add_argument("--per_gpu_train_batch_size", default=16, type=int,
                    help="Batch size per GPU/CPU for training.")
     parser.add_argument("--per_gpu_eval_batch_size", default=16, type=int,
                    help="Batch size per GPU/CPU for evaluation.")
     parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
     parser.add_argument("--learning_rate", default=5e-5, type=float,
                    help="The initial learning rate for Adam.")
     parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="Weight deay if we apply some.")
     parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
     parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
     parser.add_argument("--num_train_epochs", default=1.0, type=float,
                    help="Total number of training epochs to perform.")
     parser.add_argument("--max_steps", default=-1, type=int,
                    help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
     parser.add_argument("--warmup_steps", default=0, type=int,
                    help="Linear warmup over warmup_steps.")
     parser.add_argument('--logging_steps', type=int, default=50,
                    help="Log every X updates steps.")
     parser.add_argument('--save_steps', type=int, default=50,
                    help="Save checkpoint every X updates steps.")
     parser.add_argument("--eval_all_checkpoints", action='store_true',
                    help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
     parser.add_argument("--no_cuda", action='store_true',
                    help="Avoid using CUDA when available")
     parser.add_argument('--overwrite_output_dir', action='store_true',
                    help="Overwrite the content of the output directory")
     parser.add_argument('--overwrite_cache', action='store_true',
                    help="Overwrite the cached training and evaluation sets")
     parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")
     parser.add_argument('--fp16', action='store_true',
                    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
     parser.add_argument('--fp16_opt_level', type=str, default='O1',
                    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                         "See details at https://nvidia.github.io/apex/amp.html")
     parser.add_argument("--local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")
     parser.add_argument("--max_seq_length", default=128, type=int,
               help="The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded.")
     parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
     parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
     parser.add_argument("--prompt", type=str, default="")
     parser.add_argument("--padding_text", type=str, default="")
     parser.add_argument("--length", type=int, default=20)
     parser.add_argument("--temperature", type=float, default=1.0)
     parser.add_argument("--top_k", type=int, default=0)
     parser.add_argument("--top_p", type=float, default=0.9)
     args = parser.parse_args()

     return args