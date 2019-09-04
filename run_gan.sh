python3 main.py \
    "--train_data_file"="train.csv" \
    "--data_dir"="data/emnlp_news" \
    "--output_dir"="gan_results" \              
    "--model_name_or_path"="xlnet" \
    "--eval_data_file"="test.csv" \
    "--model_type"="xlnet" \
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
     parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                    help="Batch size per GPU/CPU for training.")
     parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                    help="Batch size per GPU/CPU for evaluation.")
     parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
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
     parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
     parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
     parser.add_argument("--prompt", type=str, default="")
     parser.add_argument("--padding_text", type=str, default="")
     parser.add_argument("--length", type=int, default=20)
     parser.add_argument("--temperature", type=float, default=1.0)
     parser.add_argument("--top_k", type=int, default=0)
     parser.add_argument("--top_p", type=float, default=0.9)