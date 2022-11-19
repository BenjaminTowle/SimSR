from transformers import HfArgumentParser
from dataclasses import dataclass, field


@dataclass
class Args:  
    # Training args
    output_dir: str = "none"
    task: str = field(default="personachat", metadata={"choices": ["personachat"]})
    model: str = field(default="bert", metadata={"choices": ["gpt", "bert", "prior"]})
    model_type: str = field(default="biencoder", metadata={"choices": ["biencoder", "paror", "crossencoder", "simulator"]})
    debug: bool = False
    policy: str = "none"

    dialogue_length = 8

    candidate_pool_size = 8192

    max_context_length: int = 32
    max_response_length: int = 32
    max_persona_length: int = 10

    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8

    # General args
    gpt_model_path: str = "personachat/checkpoint-32860" #"distilgpt2"  # "microsoft/DialoGPT-small"
    bert_model_path: str = "distilbert-base-uncased" # "huawei-noah/TinyBERT_General_4L_312D" #"biencoder/checkpoint-49290"   # "bert-base-uncased" # # #

    gpt_tokenizer_path: str = "microsoft/DialoGPT-small"
    bert_tokenizer_path: str = "bert-base-uncased"

    prediction_save_path: str = "single.json"
    dataset_path: str = None  # "processed_dataset3"
    seed: int = 0
    num_return_sequences: int = 10
    batch_size: int = 2
    device: str = "cuda"
    
   
    
    learning_rate: float = 1e-4

    num_samples: int = 20
    num_candidates: int = 3
    num_iterations: int = 10

    # Generation args
    self_bleu_alpha: float = 1.0
    k: int = 3

def parse_args():
    parser = HfArgumentParser([Args])
    args, = parser.parse_args_into_dataclasses()
    return args
