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

import yaml

def load_dataset_config(dataset_name):
    with open('dataset_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config[dataset_name]



def cycle_loader(dataloader, sampler=None):
    while 1:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data


def wt_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
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

        # Process each line in the response content
        for line in response.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                data_list.append(data)

        return data_list

    lambada_data = read_jsonl_to_list(url)
    dataset = Dataset.from_list(lambada_data)
    return dataset


def get_dataset(name, mode, cache_dir=None, block_size=1024, num_proc=8, train_size = None, logger = None):
    dataset_config = load_dataset_config(name)
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

    original_size = len(dataset[mode])

    if mode == "train" and train_size is not None:
        # Select subset of training data
        train_size = int(train_size) if isinstance(train_size, str) else train_size
        data = dataset[mode].shuffle().select(range(min(train_size, len(dataset[mode]))))
        if logger:
            logger.info(f"Dataset '{name}' reduced from {original_size} to {len(data)} examples as requested for mode {mode}")
    else:
        data = dataset[mode]
        if logger:
            logger.info(f"Using full dataset '{name}' with {original_size} examples for mode {mode}")


    # # If num_train_examples is specified and this is training data,
    # # select a subset of the data
    # if mode == "train" and dataset_config.get('num_train_examples') is not None:
    #     # Ensure we don't try to select more examples than available
    #     num_examples = min(dataset_config['num_train_examples'], len(data))
    #     # Select random subset if num_examples is less than total
    #     if num_examples < len(data):
    #         data = data.shuffle(seed=42).select(range(num_examples))
    #     print(f"Using {num_examples} examples for training")


    # tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2',truncation_side='left')
    EOS = tokenizer.encode(tokenizer.eos_token)[0]

    def preprocess_and_tokenize(example):
        
        if name == dataset_config['train']:
            text_label_pairs = zip(example['text'], example['label'])
            text = [f"{text}. Label:{label}" for text, label in text_label_pairs]
        
        if detokenizer is not None:
            text = _apply_detokenizer(detokenizer)(text)
        
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        tokens = tokenizer(text, return_attention_mask=False, max_length=dataset_config['max_length'], padding='max_length', truncation=True)
        
        return tokens
        
        
        
        
        
        
        # if name == "ptb":
        #     text = example['sentence']
        # elif name in ["ag_news", "snli", "sst2", "IMDb", "emotion", 'zeroshot/twitter-financial-news-sentiment', 'yelp_review_full','SetFit/sst5', 'SetFit/20_newsgroups', 'SetFit/sst2', 'cardiffnlp/tweet_eval/sentiment', 'Sp1786/multiclass-sentiment-analysis-dataset']:
        #     # CHANGE HERE - CONCATENATE THE LABEL TO THE TEXT
        #     # Concatenate the input sentence and label
        #     if name == "snli":
        #         text = f"{example['premise']} {example['hypothesis']} Label: {example['label']}" 
                
        #     elif name == "ag_news":
        #         text_label_pairs = zip(example['text'], example['label'])

        #         text = [str(text)+'. Label:'+str(label) for text, label in text_label_pairs]
        #     elif name == "IMDb":
        #         text_label_pairs = zip(example['text'], example['label'])

        #         text = [str(text)+'Label:'+str(label) for text, label in text_label_pairs]
                
        #     elif name == "emotion":
        #         text_label_pairs = zip(example['text'], example['label'])

        #         text = [str(text)+'. Label:'+str(label) for text, label in text_label_pairs]
        #     elif name == 'zeroshot/twitter-financial-news-sentiment':
        #         text_label_pairs = zip(example['text'], example['label'])

        #         text = [str(text)+'. Label:'+str(label) for text, label in text_label_pairs]
        #     elif name == 'yelp_review_full':
        #         text_label_pairs = zip(example['text'], example['label'])

        #         text = [str(text)+'. Label:'+str(label) for text, label in text_label_pairs]
        #     elif name == "SetFit/sst5":
        #         text_label_pairs = zip(example['text'], example['label'])

        #         text = [str(text)+'. Label:'+str(label) for text, label in text_label_pairs]
        #     elif name == "SetFit/20_newsgroups":
        #         text_label_pairs = zip(example['text'], example['label'])

        #         text = [str(text)+'. Label:'+str(label) for text, label in text_label_pairs]

        #     elif name == "SetFit/sst2":
        #         text_label_pairs = zip(example['text'], example['label'])

        #         text = [str(text)+'. Label:'+str(label) for text, label in text_label_pairs]
                
        #     elif name == "Sp1786/multiclass-sentiment-analysis-dataset":
        #         text_label_pairs = zip(example['text'], example['label'])

        #         text = [str(text)+'. Label:'+str(label) for text, label in text_label_pairs]
            
        #     elif name == "cardiffnlp/tweet_eval/sentiment":
        #         text_label_pairs = zip(example['text'], example['label'])

        #         text = [str(text)+'. Label:'+str(label) for text, label in text_label_pairs]

        #     else:
        #         # text = f"{example['text']}+ 'Label' +{example['label']}"
        #         # text = example["text"]+example['label']
        #         # text = example["text"]
        #         text_label_pairs = zip(example['text'], example['label'])

        #         text = [str(text)+'. Label:'+str(label) for text, label in text_label_pairs]
        # else:
        #     text = example["text"]
        
        # if detokenizer is not None:
        #     text = _apply_detokenizer(detokenizer)(text)
        # # print("text",text,len(text),"len of text")
        # # exit()
        # # tokens = tokenizer(text, return_attention_mask=False) #- used this for ag_news and imdb
        # # tokens = tokenizer(text, return_attention_mask=False, max_length = 64) # used this for emotion and twitterfin
        # # tokens = tokenizer(text, return_attention_mask=False, max_length = 768) # used this for yelp_reviews 
        # # tokenizer.pad_token = tokenizer.eos_token # for multiclass sentiment analysis 2nd run
        # tokenizer.pad_token = tokenizer.eos_token  # for multiclass sentiment analysis 2nd run
        # # tokenizer.padding_side = "left"
        # tokenizer.padding_side = "right"
        # # tokens = tokenizer(text, return_attention_mask=False, max_length = 50, padding='max_length', truncation=True) # for multiclass sentiment analysis 2nd run
        # # tokens = tokenizer(text, return_attention_mask=False, max_length = 127, padding='max_length', truncation=True) # for multiclass sentiment analysis 2nd run
        # # tokens = tokenizer(text, return_attention_mask=False, max_length = 31, padding='max_length', truncation=True) # for multiclass sentiment analysis 2nd run
        
        # # CHANGE HERE - APPROACH 1  - PURE CHUNKING W/O PADDING
        # # tokens = tokenizer(text, return_attention_mask=False) # using this for multiclass
        
        # # CHANGE HERE - APPROACH 2 - PURE PADDING W/O CHUNKING 
        # tokens = tokenizer(text, return_attention_mask=False, max_length = 128, padding='max_length', truncation=True) # max_length hyperparameter T_k 
        # # from remote_pdb import RemotePdb; RemotePdb('127.0.0.1', 4444).set_trace() #it's working till here
        # # to be chosen as per the avg length of text in input
        # # feel free to play around with it
        
        # # import pdb; pdb.set_trace()
        # # add in EOS token following 
        # # https://github.com/jcpeterson/openwebtext/blob/master/tokenize_text.py#L67
        # # Be careful while appending EOS for approach 2 as this would increase the overall length by 1
        # # for token in tokens['input_ids']:
        # #     token.append(EOS)
        # return tokens
    
    
    tokenized_dataset = data.map(preprocess_and_tokenize, batched=True, num_proc=num_proc, load_from_cache_file=True)
    tokenized_dataset = tokenized_dataset.remove_columns(dataset_config['columns_to_remove'])
    
    
    # tokenized_dataset = data.map(preprocess_and_tokenize, batched=True, num_proc=num_proc, load_from_cache_file=True)
    # if name == "ptb":
    #     tokenized_dataset = tokenized_dataset.remove_columns('sentence')
    # elif name in ["ag_news", "snli", "sst2", "IMDb", "emotion", 'zeroshot/twitter-financial-news-sentiment', 'Sp1786/multiclass-sentiment-analysis-dataset', 'yelp_review_full', 'SetFit/sst5', 'SetFit/20_newsgroups', 'SetFit/sst2', 'cardiffnlp/tweet_eval/sentiment']:
    #     ## CHANGE HERE - REMOVE THOSE COLUMNS NOT BEING USING TRAINING
    #     # tokenized_dataset = tokenized_dataset.remove_columns(['text', 'label', 'premise', 'hypothesis']) # Remove original columns
    #     tokenized_dataset = tokenized_dataset.remove_columns(['text', 'label']) # Remove original columns # for ag_news
    #     # tokenized_dataset = tokenized_dataset.remove_columns(['sentence','label']) # Remove original columns
    #     # tokenized_dataset = tokenized_dataset.remove_columns(['text', 'label','id','sentiment']) #multiclass-sent-analysis
        
    # else:
    #     tokenized_dataset = tokenized_dataset.remove_columns('text')
    

    def group_texts(examples):
        # Concatenate all texts.
        # print("len examples",len(examples))
        # print("examples.keys",examples.keys())
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        # import time; time.sleep(10)
        # print(result) 
        # the above should confirm if the padding is to the left and if the chunking is correct. 
        return result

    chunked_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=num_proc, load_from_cache_file=True)
    # from remote_pdb import RemotePdb; RemotePdb('127.0.0.1', 4444).set_trace() # here we need to check chunked_dataset[3] or [4] as that's when the offset is visible
    chunked_dataset = chunked_dataset.with_format('torch')

    return chunked_dataset

# modified by foobar - added a debug collate function
from torch.utils.data.dataloader import default_collate
def debug_collate(batch):
    print("Sample in batch before collate:", batch[0])
    # from remote_pdb import RemotePdb; RemotePdb('127.0.0.1', 4444).set_trace()
    collated = default_collate(batch)
    print("Sample in batch after collate:", collated['input_ids'][0])
    return collated

def get_dataloaders(config, distributed=True, logger = None):
    if logger:
        logger.info("Initializing data loaders...")
    dataset_config = load_dataset_config(config.dataset_name)
    # train_size = dataset_config.get('train_size', None)
    train_size = config.train_size if hasattr(config, 'train_size') else None
    print(f"train size is {train_size}",type(train_size))
    if config.training.batch_size % (config.ngpus * config.training.accum) != 0:
            raise ValueError(f"Train Batch Size {config.training.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")
    if config.eval.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(f"Eval Batch Size for {config.eval.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")


    train_set = get_dataset(
        dataset_config['train'], 
        "train", 
        cache_dir=config.data.cache_dir, 
        block_size=config.model.length,
        train_size=train_size,
        logger = logger
    )
    
    valid_set = get_dataset(dataset_config['valid'], dataset_config['valid_split'], cache_dir=config.data.cache_dir, block_size=config.model.length, logger = logger)

    
    # ## CHANGE HERE BASED ON WHETHER YOU USE PURE CHUNKING OR PURE PADDING APPROACH
    # ## APPROACH 1 - PURE CHUNKING - SET config.model.length = 1024 or 2048 in the config.yaml file
    # ## APPROACH 2 - PURE PADDING - SET config.model.length = T_k in the config.yaml file
    # ## NOTHING TO CHANGE HERE - CHANGE DIRECTLY IN THE small.yaml file in configs folder
    
    # train_set = get_dataset(config.data.train, "train", cache_dir=config.data.cache_dir, block_size=config.model.length) # used for the second run of ag_news and multiclass sentiment analysis
    
    # ## CHANGE HERE - THE validation/test name accordingly
    # ## FOR NOW, WE ARE USING THE TEST DATASET AS THE VALIDATION DATASET
    # valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != "ag_news" else "test", cache_dir=config.data.cache_dir, block_size=config.model.length)


    if distributed:
        train_sampler = DistributedSampler(train_set) 
        test_sampler = DistributedSampler(valid_set)
    else:
        train_sampler = None
        test_sampler = None
    
    
    print("len of train_dataset", len(train_set))

    train_loader = cycle_loader(DataLoader(
        train_set,
        batch_size=config.training.batch_size // (config.ngpus * config.training.accum),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(train_sampler is None),
        # shuffle = False,
        # shuffle=True,
        persistent_workers=True,
        # collate_fn=debug_collate  # Add this line - for debugging # modified by foobar
    ))
    valid_loader = cycle_loader(DataLoader(
        valid_set,
        batch_size=config.eval.batch_size // (config.ngpus * config.training.accum),
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(test_sampler is None),
    ))
    return train_loader, valid_loader




# ignore these config settings
    # train_set = get_dataset(config.data.train, "train", cache_dir=config.data.cache_dir, block_size=config.model.length) # used this for ag_news and imdb
    # train_set = get_dataset(config.data.train, "train", cache_dir=config.data.cache_dir, block_size=64) # used this for emotion and twitterfin dataset
    #train_set = get_dataset(config.data.train, "train", cache_dir=config.data.cache_dir, block_size=768) # used this for yelp reviews dataset
    # train_set = get_dataset(config.data.train, "train", cache_dir=config.data.cache_dir, block_size=51) # used for the second run of ag_news and multiclass sentiment analysis
    # train_set = get_dataset(config.data.train, "train", cache_dir=config.data.cache_dir, block_size=128) # used for the second run of ag_news and multiclass sentiment analysis
    # train_set = get_dataset(config.data.train, "train", cache_dir=config.data.cache_dir, block_size=32) # used for the second run of ag_news and multiclass sentiment analysis
    # valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != "text8" else "test", cache_dir=config.data.cache_dir, block_size=config.model.length)
    
    #valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != "IMDb" else "test", cache_dir=config.data.cache_dir, block_size=config.model.length)
    # valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != "emotion" else "test", cache_dir=config.data.cache_dir, block_size=config.model.length)
    #valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != "emotion" else "test", cache_dir=config.data.cache_dir, block_size=64)
    # valid_set = get_dataset(config.data.valid, "validation" if config.data.valid == 'zeroshot/twitter-financial-news-sentiment' else "test", cache_dir=config.data.cache_dir, block_size=64)
    # valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != 'yelp_review_full' else "test", cache_dir=config.data.cache_dir, block_size=768)
    # valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != 'SetFit/20_newsgroups' else "test", cache_dir=config.data.cache_dir, block_size=768)
    # valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != "SetFit/sst2" else "test", cache_dir=config.data.cache_dir, block_size=32)
    # valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != "Sp1786/multiclass-sentiment-analysis-dataset" else "test", cache_dir=config.data.cache_dir, block_size=64)
    
    # valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != "ag_news" else "test", cache_dir=config.data.cache_dir, block_size=51) # used for the second run of ag_news
    
    
    
    # valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != "Sp1786/multiclass-sentiment-analysis-dataset" else "test", cache_dir=config.data.cache_dir, block_size=128) # used for the second run of multiclass-sentinment
    # valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != "Sp1786/multiclass-sentiment-analysis-dataset" else "test", cache_dir=config.data.cache_dir, block_size=51) # used for the second run of multiclass sentiment analysis
    # valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != "Sp1786/multiclass-sentiment-analysis-dataset" else "test", cache_dir=config.data.cache_dir, block_size=32) # used for the second run of multiclass-sentinment
    # valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != "Sp1786/multiclass-sentiment-analysis-dataset" else "test", cache_dir=config.data.cache_dir, block_size=config.model.length) # used for the second run of multiclass-sentinment
    # valid_set = train_set
    # making a smaller blocksize for sst5
    # valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != "sst5" else "test", cache_dir=config.data.cache_dir, block_size=50)
