import os
import argparse
import json
from tqdm import trange, tqdm
import torch
from fastchat.model import load_model, get_conversation_template, add_model_args
import numpy as np
import pandas as pd
from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    PeftModel,
)
from transformers import LlamaForCausalLM, LlamaTokenizer  # noqa: F402
import safetensors
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

cur_embed = None
embeds = []


def hook(module, input, output):
    global cur_embed, embeds
    input = input[0].cpu().detach().numpy()
    cur_embed = input


def simple_iter(args):
    if args.dataset == "ml-1m":
        movie_dict = json.load(open(os.path.join('data/movie/ml-1m', "movie_detail.json"), "r"))
        for i in trange(1, 3953):
            key = str(i)
            if key not in movie_dict.keys():
                title, genre = "", ""
            else:
                title, genre = movie_dict[key]
            text = \
                f"Here is a movie. Its title is {title}. The movie's genre is {genre}." 
            yield text

    elif args.dataset == "ml-25m":
        movie_dict = json.load(open(os.path.join('data/movie/ml-25m', "movie_detail.json"), "r"))
        for i in trange(1, 62424):
            key = str(i)
            if key not in movie_dict.keys():
                title, genre = "", ""
                continue
            else:
                title, genre = movie_dict[key]
            text = \
                f"Here is a movie. Its title is {title}. The movie's genre is {genre}."
            yield text

    elif args.dataset == "ml-100k":
        movie_dict = json.load(open(os.path.join('data/movie/ml-100k', "movie_detail.json"), "r"))
        for i in trange(1, 1683):
            key = str(i)
            if key not in movie_dict.keys():
                title, genre = "", ""
            else:
                title, genre = movie_dict[key]
            text = \
                f"Here is a movie. Its title is {title}. The movie's genre is {genre}."
            yield text
    
    elif args.dataset == "BookCrossing":
        id2book = json.load(open(os.path.join('data/book', "id2book.json"), "r"))
        for i in trange(len(id2book)):
            isbn, title, author, year, publisher = id2book[str(i)]
            text = \
                f"Here is a book. Its title is {title}. ISBN of the book is {isbn}. The author of the book is {author}. "\
                f"The publication year of the book is {year}. Its publisher is {publisher}."
            yield text
        
    else:
        assert False, "Unsupported dataset"


@torch.inference_mode()
def main(args):
    # Load model.
    base_model = args.model_path
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = prepare_model_for_int8_training(model)

    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules = ["q_proj", "v_proj"]

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    adapters_weights = safetensors.torch.load_file('lora-alpaca/adapter_model.safetensors')
    set_peft_model_state_dict(model, adapters_weights)
    print("lora-alpaca")
    model.lm_head.register_forward_hook(hook)

    print("Model loaded.")

    os.makedirs(args.embed_dir, exist_ok=True)
    fp = os.path.join(args.embed_dir, '_'.join([args.dataset, args.pooling])+".npy")

    global cur_embed, embeds

    # Start inference.
    for txt in simple_iter(args):
        conv = get_conversation_template(args.model_path)
        conv.append_message(conv.roles[0], txt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        cur_embed = None
        input_ids = tokenizer([prompt]).input_ids


        output_ids = model.generate(
            input_ids=torch.as_tensor(input_ids).to(args.device),
            do_sample=True,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
        )
        
        
        if args.pooling == "last":
            cur_embed = cur_embed[0, len(input_ids[0])-1]
        elif args.pooling == "average":
            cur_embed = cur_embed[0, :len(input_ids[0])].mean(axis=0)

        embeds.append(cur_embed)

    np.save(fp, np.stack(embeds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--pooling", type=str, default="average", help="average/last")
    parser.add_argument("--embed_dir", type=str, default="./embeddings")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="ml-100k")
    parser.add_argument("--model_path", type=str, default="Llama-2-7b-hf")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    # Reset default repetition penalty for T5 models.
    if "t5" in args.model_path and args.repetition_penalty == 1.0:
        args.repetition_penalty = 1.2

    args.data_dir = f"./{args.data_dir}/{args.dataset}/proc_data"
    assert args.pooling in ["average", "last"], "Pooling type error"
    main(args)