import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from load_model import load_model
from transformers import GPT2TokenizerFast
import sampling_inference
from data import get_dataset
import torch
import argparse
from load_model import load_model
from transformers import GPT2TokenizerFast
import sampling_inference
import re
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from itertools import chain
import numpy as np
import torch
import urllib.request
import zipfile
import requests
import json
from datasets import Dataset
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_convert
from sklearn.metrics import classification_report

def cycle_loader(dataloader, sampler=None):
    while 1:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data

def wt_detokenizer(string):
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string

def ptb_detokenizer(x):
    x = x.replace(" 's", "'s")
    x = x.replace("s ' ", "s' ")
    x = x.replace(" n't", "n't")
    x = x.replace(" \n ", "\n")
    x = x.replace("\\/", "/")
    for _ in range(10):
        x = x.replace(" N ", " 1 ")
    x = x.replace("$ 1", "$1")
    x = x.replace("# 1", "#1")
    x = x.replace("<unk>", "?")
    return x

def lm1b_detokenizer(x):
    x = x.replace('http : / / ', 'http://')
    x = x.replace('https : / / ', 'https://')
    x = re.sub(r' \'(\w+)', r"'\1", x)
    x = re.sub(r' (\w+) \. ', r' \1. ', x)
    x = re.sub(r' (\w+) \.$', r' \1.', x)
    x = x.replace(' ? ', '? ')
    x = re.sub(r' \?$', '?', x)
    x = x.replace(' ! ', '! ')
    x = re.sub(r' \!$', '!', x)
    x = x.replace(' , ', ', ')
    x = x.replace(' : ', ': ')
    x = x.replace(' ; ', '; ')
    x = x.replace(' / ', '/')
    x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
    x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
    x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
    x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
    x = x.replace('$ ', '$')
    x = x.replace('£ ', '£')
    return x

def lambada_detokenizer(text):
    text = text.replace(""", '"')
    text = text.replace(""", '"')
    return '\n'+text.strip()

def get_lambada_test_dataset():
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"
    def read_jsonl_to_list(url):
        response = requests.get(url, stream=True)
        data_list = []
        for line in response.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                data_list.append(data)
        return data_list
    lambada_data = read_jsonl_to_list(url)
    dataset = Dataset.from_list(lambada_data)
    return dataset

def get_dataset_agnews(name, mode, cache_dir=None, block_size=1024, num_proc=8):
    if name == "wikitext103":
        dataset = load_dataset("wikitext", name="wikitext-103-raw-v1", cache_dir=cache_dir)
    elif name == "wikitext2":
        dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", cache_dir=cache_dir)
    elif name == "ptb":
        dataset = load_dataset("ptb_text_only", cache_dir=cache_dir)
    elif name == "lambada":
        dataset = get_lambada_test_dataset()
    else:
        dataset = load_dataset(name, cache_dir=cache_dir)

    if name == "lambada":
        data = dataset
    else:
        data = dataset[mode]

    if name.startswith("wikitext"):
        detokenizer = wt_detokenizer
    elif name == "ptb":
        detokenizer = ptb_detokenizer
    elif name == "lm1b":
        detokenizer = lm1b_detokenizer
    elif name == "lambada":
        detokenizer = lambada_detokenizer
    else:
        detokenizer = None

    def _apply_detokenizer(detokenizer):
        def detok(text):
            for i, t in enumerate(text, 0):
                text[i] = detokenizer(t)
            return text
        return detok

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2',truncation_side='left')
    EOS = tokenizer.encode(tokenizer.eos_token)[0]

    def preprocess_and_tokenize(example):
        if name == "ptb":
            text = example['sentence']
        elif name in ["ag_news", "snli", "sst5", "emotion", "imdb", "zeroshot/twitter-financial-news-sentiment"]:
            text_label_pairs = zip(example['text'], example['label'])
            text = [str(text)+'. Label:' for text, label in text_label_pairs]
        else:
            text = example["text"]
        
        if detokenizer is not None:
            text = _apply_detokenizer(detokenizer)(text)
        
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        tokens = tokenizer(text, return_attention_mask=False,
                           padding="max_length", 
                            truncation=True, 
                            max_length=128)
        return tokens
    
    tokenized_dataset = data.map(preprocess_and_tokenize, batched=True, num_proc=num_proc, load_from_cache_file=True)
    if name == "ptb":
        tokenized_dataset = tokenized_dataset.remove_columns('sentence')
    elif name in ["ag_news", "snli", "sst5", "emotion", "imdb", "zeroshot/twitter-financial-news-sentiment"]:
        tokenized_dataset = tokenized_dataset.remove_columns(['text'])
    else:
        tokenized_dataset = tokenized_dataset.remove_columns('text')
    
    return tokenized_dataset

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()



def run_model(rank, world_size, args):
    try:
        print(f"Rank {rank}: Setting up distributed process")
        setup(rank, world_size)
        
        print(f"Rank {rank}: Loading model")
        device = torch.device(f'cuda:{rank}')
        model, graph, noise = load_model(args.model_path, device)
        model = DDP(model, device_ids=[rank])
        
        print(f"Rank {rank}: Setting up tokenizer")
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        EOS = tokenizer.encode(tokenizer.eos_token)[0]
        label_tokens = tokenizer.encode(" Label:", add_special_tokens=False)
        
        print(f"Rank {rank}: Loading dataset")
        print(args.dataset)
        if(args.dataset == 'zeroshot/twitter-financial-news-sentiment'):
            valid_set = get_dataset_agnews(args.dataset, "validation", block_size=1024)
        else:
            valid_set = get_dataset_agnews(args.dataset, "test", block_size=1024)
        train_sampler = DistributedSampler(valid_set, num_replicas=world_size, rank=rank)
        
        valid_loader = DataLoader(
            valid_set,
            batch_size=args.batch_size // world_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            sampler=train_sampler,
            collate_fn=default_convert
        )

        all_predicted_labels = []
        all_true_labels = []

        print(f"Rank {rank}: Starting processing")
        for batch_idx, batch in enumerate(valid_loader):
            print(f"Rank {rank}: Processing batch {batch_idx}")
            
            input_ids = torch.tensor([x['input_ids'] for x in batch], device=device)
            batch_size = input_ids.size(0)
            seq_len = input_ids.size(1)
            
            # Create attention mask (1 for real tokens, 0 for padding)
            # attention_mask = (input_ids != tokenizer.pad_token_id).float().to(device)

            print(f"Rank {rank}: Finding label positions")
            label_positions = []
            for i in range(batch_size):
                for j in range(seq_len):
                    if torch.all(input_ids[i, j:j+len(label_tokens)] == torch.tensor(label_tokens, device=device)):
                        label_positions.append(j + len(label_tokens) - 1)
                        break
                if len(label_positions) <= i:  # If no label position found for this sample
                    label_positions.append(seq_len - 1)  # Use end of sequence
            
            def proj_fun(x):
                attention_mask = torch.ones_like(x, dtype=torch.float32)
                for i in range(batch_size):
                    label_pos = label_positions[i]
                    x[i, :label_pos + 1] = input_ids[i, :label_pos + 1]
                    if label_pos + 2 < seq_len:
                        x[i, label_pos + 2:] = EOS
                        # Update attention mask - set to 0 for EOS tokens
                        attention_mask[i, label_pos + 2:] = 0
                        
                # Create or update attention mask
                # print(type(x), "type of x")
                # attention_mask = (x != tokenizer.pad_token_id).float()
                # print(type(attention_mask), "type of attention_mask")
                return x,attention_mask

            print(f"Rank {rank}: Running sampling")
            try:
                sampling_fn = sampling_inference.get_pc_sampler(
                    graph, 
                    noise, 
                    (batch_size, seq_len), 
                    'analytic', 
                    args.steps, 
                    device=device, 
                    proj_fun=proj_fun
                )

                samples = sampling_fn(model)
                text_samples = tokenizer.batch_decode(samples)
            except Exception as e:
                print(f"Rank {rank}: Error in sampling: {e}")
                continue

            print(f"Rank {rank}: Processing results")
            for i, text_sample in enumerate(text_samples):
                try:
                    # More robust label extraction
                    parts = text_sample.split("Label:")
                    if len(parts) > 1:
                        label_part = parts[-1].strip()
                        if label_part:
                            # Look for the first digit in the label part
                            for char in label_part:
                                if char.isdigit():
                                    predicted_label = int(char)
                                    all_predicted_labels.append(predicted_label)
                                    all_true_labels.append(batch[i]['label'])
                                    print(f"Rank {rank}: Successfully processed sample {i} with label {predicted_label}")
                                    break
                            else:
                                print(f"Rank {rank}: No digit found in label part: '{label_part}'")
                        else:
                            print(f"Rank {rank}: Empty label part for sample {i}")
                    else:
                        print(f"Rank {rank}: No 'Label:' found in text: '{text_sample}'")
                except Exception as e:
                    print(f"Rank {rank}: Error processing sample {i}: {e}")
                    print(f"Rank {rank}: Problem text sample: '{text_sample}'")

            # Synchronize GPUs after each batch
            torch.cuda.synchronize()
            dist.barrier()

            if len(all_predicted_labels) > 0:
                try:
                    print(f"Rank {rank}: Gathering results")
                    pred_tensor = torch.tensor(all_predicted_labels, dtype=torch.long, device=device)
                    label_tensor = torch.tensor(all_true_labels, dtype=torch.long, device=device)

                    # Make sure all tensors are the same size
                    max_size = torch.tensor([pred_tensor.size(0)], device=device)
                    dist.all_reduce(max_size, op=dist.ReduceOp.MAX)
                    
                    if pred_tensor.size(0) < max_size.item():
                        # Pad with -1 if needed
                        pad_size = max_size.item() - pred_tensor.size(0)
                        pred_tensor = torch.cat([pred_tensor, torch.full((pad_size,), -1, device=device)])
                        label_tensor = torch.cat([label_tensor, torch.full((pad_size,), -1, device=device)])

                    gathered_preds = [torch.zeros_like(pred_tensor, device=device) for _ in range(world_size)]
                    gathered_labels = [torch.zeros_like(label_tensor, device=device) for _ in range(world_size)]
                    
                    dist.all_gather(gathered_preds, pred_tensor)
                    dist.all_gather(gathered_labels, label_tensor)
                    
                    if rank == 0:
                        # Filter out padding (-1)
                        all_preds = torch.cat(gathered_preds).cpu().numpy()
                        all_labels = torch.cat(gathered_labels).cpu().numpy()
                        mask = all_preds != -1
                        all_preds = all_preds[mask]
                        all_labels = all_labels[mask]
                        report = classification_report(all_labels, all_preds, digits = 4)
                        print("\nIntermediate Results:")
                        print(report)
                except Exception as e:
                    print(f"Rank {rank}: Error gathering results: {e}")

            print(f"Rank {rank}: Completed batch {batch_idx}")

        print(f"Rank {rank}: Cleaning up")
        cleanup()
        
    except Exception as e:
        print(f"Rank {rank}: Critical error: {e}")
        import traceback
        traceback.print_exc()
        cleanup()

def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    
    parser.add_argument("--model_path", default="give_path_to_checkpoint", type=str)
    parser.add_argument("--dataset", default="imdb", type=str)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=16) #original: 1024; 16 is the best.
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    print(f"Starting distributed training with {world_size} GPUs")
    
    try:
        mp.spawn(
            run_model,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

