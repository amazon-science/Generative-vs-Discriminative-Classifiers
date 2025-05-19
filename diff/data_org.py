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
    text = text.replace("“", '"')
    text = text.replace("”", '"')
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


def get_dataset(name, mode, cache_dir=None, block_size=1024, num_proc=8):
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

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    EOS = tokenizer.encode(tokenizer.eos_token)[0]

    def preprocess_and_tokenize(example):
        if name == "ptb":
            text = example['sentence']
            
            
        elif name in ["ag_news", "snli", "sst2", "IMDb", "emotion", 'zeroshot/twitter-financial-news-sentiment', 'yelp_review_full','SetFit/sst5', 'SetFit/20_newsgroups', 'SetFit/sst2', 'cardiffnlp/tweet_eval/sentiment', 'Sp1786/multiclass-sentiment-analysis-dataset']:
            # Concatenate the input sentence and label
            if name == "snli":
                text = f"{example['premise']} {example['hypothesis']} Label: {example['label']}" 
            elif name == "IMDb":
                text_label_pairs = zip(example['text'], example['label'])

                text = [str(text)+'Label:'+str(label) for text, label in text_label_pairs]
            elif name == "emotion":
                text_label_pairs = zip(example['text'], example['label'])

                text = [str(text)+'. Label:'+str(label) for text, label in text_label_pairs]
            elif name == 'zeroshot/twitter-financial-news-sentiment':
                text_label_pairs = zip(example['text'], example['label'])

                text = [str(text)+'. Label:'+str(label) for text, label in text_label_pairs]
            elif name == 'yelp_review_full':
                text_label_pairs = zip(example['text'], example['label'])

                text = [str(text)+'. Label:'+str(label) for text, label in text_label_pairs]
            elif name == "SetFit/sst5":
                text_label_pairs = zip(example['text'], example['label'])

                text = [str(text)+'. Label:'+str(label) for text, label in text_label_pairs]
            elif name == "SetFit/20_newsgroups":
                text_label_pairs = zip(example['text'], example['label'])

                text = [str(text)+'. Label:'+str(label) for text, label in text_label_pairs]

            elif name == "SetFit/sst2":
                text_label_pairs = zip(example['text'], example['label'])

                text = [str(text)+'. Label:'+str(label) for text, label in text_label_pairs]
                
            elif name == "Sp1786/multiclass-sentiment-analysis-dataset":
                text_label_pairs = zip(example['text'], example['label'])

                text = [str(text)+'. Label:'+str(label) for text, label in text_label_pairs]
            
            elif name == "cardiffnlp/tweet_eval/sentiment":
                text_label_pairs = zip(example['text'], example['label'])

                text = [str(text)+'. Label:'+str(label) for text, label in text_label_pairs]

            else:
                # text = f"{example['text']}+ 'Label' +{example['label']}"
                # text = example["text"]+example['label']
                # text = example["text"]
                text_label_pairs = zip(example['text'], example['label'])

                text = [str(text)+'Label:'+str(label) for text, label in text_label_pairs]
        else:
            text = example["text"]
        # print(list(example.keys()))
        # exit()
        
        if detokenizer is not None:
            text = _apply_detokenizer(detokenizer)(text)

        tokens = tokenizer(text, return_attention_mask=False)
        # add in EOS token following 
        # https://github.com/jcpeterson/openwebtext/blob/master/tokenize_text.py#L67
        for token in tokens['input_ids']:
            token.append(EOS)
        return tokens
    
    tokenized_dataset = data.map(preprocess_and_tokenize, batched=True, num_proc=num_proc, load_from_cache_file=True)
    if name == "ptb":
        tokenized_dataset = tokenized_dataset.remove_columns('sentence')
    elif name in ["ag_news", "snli", "sst2", "IMDb", "emotion", 'zeroshot/twitter-financial-news-sentiment', 'Sp1786/multiclass-sentiment-analysis-dataset', 'yelp_review_full', 'SetFit/sst5', 'SetFit/20_newsgroups', 'SetFit/sst2', 'cardiffnlp/tweet_eval/sentiment']:
        # tokenized_dataset = tokenized_dataset.remove_columns(['text', 'label', 'premise', 'hypothesis']) # Remove original columns
        tokenized_dataset = tokenized_dataset.remove_columns(['text', 'label']) # Remove original columns # for ag_news
        # tokenized_dataset = tokenized_dataset.remove_columns(['sentence','label']) # Remove original columns
        # tokenized_dataset = tokenized_dataset.remove_columns(['text', 'label','id','sentiment']) #multiclass-sent-analysis
        
    else:
        tokenized_dataset = tokenized_dataset.remove_columns('text')
    

    def group_texts(examples):
        # Concatenate all texts.
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
        return result

    chunked_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=num_proc, load_from_cache_file=True)
    chunked_dataset = chunked_dataset.with_format('torch')

    return chunked_dataset


def get_dataloaders(config, distributed=True):
    if config.training.batch_size % (config.ngpus * config.training.accum) != 0:
            raise ValueError(f"Train Batch Size {config.training.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")
    if config.eval.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(f"Eval Batch Size for {config.eval.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")


    train_set = get_dataset(config.data.train, "train", cache_dir=config.data.cache_dir, block_size=config.model.length)
    # valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != "text8" else "test", cache_dir=config.data.cache_dir, block_size=config.model.length)
    valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != "ag_news" else "test", cache_dir=config.data.cache_dir, block_size=config.model.length)



    if distributed:
        train_sampler = DistributedSampler(train_set) 
        test_sampler = DistributedSampler(valid_set)
    else:
        train_sampler = None
        test_sampler = None
    

    train_loader = cycle_loader(DataLoader(
        train_set,
        batch_size=config.training.batch_size // (config.ngpus * config.training.accum),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(train_sampler is None),
        persistent_workers=True,
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

# import re
# from transformers import GPT2TokenizerFast
# from datasets import load_dataset
# from itertools import chain
# import numpy as np
# import torch

# import urllib.request
# import zipfile
# import requests
# import json
# from datasets import Dataset

# from torch.utils.data import DataLoader, DistributedSampler


# def cycle_loader(dataloader, sampler=None):
#     while 1:
#         if sampler is not None:
#             sampler.set_epoch(np.random.randint(0, 100000))
#         for data in dataloader:
#             yield data


# def wt_detokenizer(string):
#     # contractions
#     string = string.replace("s '", "s'")
#     string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
#     # number separators
#     string = string.replace(" @-@ ", "-")
#     string = string.replace(" @,@ ", ",")
#     string = string.replace(" @.@ ", ".")
#     # punctuation
#     string = string.replace(" : ", ": ")
#     string = string.replace(" ; ", "; ")
#     string = string.replace(" . ", ". ")
#     string = string.replace(" ! ", "! ")
#     string = string.replace(" ? ", "? ")
#     string = string.replace(" , ", ", ")
#     # double brackets
#     string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
#     string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
#     string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
#     string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
#     string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
#     # miscellaneous
#     string = string.replace("= = = =", "====")
#     string = string.replace("= = =", "===")
#     string = string.replace("= =", "==")
#     string = string.replace(" " + chr(176) + " ", chr(176))
#     string = string.replace(" \n", "\n")
#     string = string.replace("\n ", "\n")
#     string = string.replace(" N ", " 1 ")
#     string = string.replace(" 's", "'s")
#     return string

# def ptb_detokenizer(x):
#     x = x.replace(" 's", "'s")
#     x = x.replace("s ' ", "s' ")
#     x = x.replace(" n't", "n't")
#     x = x.replace(" \n ", "\n")
#     x = x.replace("\\/", "/")
#     for _ in range(10):
#         x = x.replace(" N ", " 1 ")
#     x = x.replace("$ 1", "$1")
#     x = x.replace("# 1", "#1")
#     x = x.replace("<unk>", "?")
#     return x

# def lm1b_detokenizer(x):
#     x = x.replace('http : / / ', 'http://')
#     x = x.replace('https : / / ', 'https://')
#     x = re.sub(r' \'(\w+)', r"'\1", x)
#     x = re.sub(r' (\w+) \. ', r' \1. ', x)
#     x = re.sub(r' (\w+) \.$', r' \1.', x)
#     x = x.replace(' ? ', '? ')
#     x = re.sub(r' \?$', '?', x)
#     x = x.replace(' ! ', '! ')
#     x = re.sub(r' \!$', '!', x)
#     x = x.replace(' , ', ', ')
#     x = x.replace(' : ', ': ')
#     x = x.replace(' ; ', '; ')
#     x = x.replace(' / ', '/')
#     x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
#     x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
#     x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
#     x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
#     x = x.replace('$ ', '$')
#     x = x.replace('£ ', '£')
#     return x


# def lambada_detokenizer(text):
#     text = text.replace("“", '"')
#     text = text.replace("”", '"')
#     return '\n'+text.strip()


# def get_lambada_test_dataset():
#     url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

#     def read_jsonl_to_list(url):
#         response = requests.get(url, stream=True)
#         data_list = []

#         # Process each line in the response content
#         for line in response.iter_lines(decode_unicode=True):
#             if line:
#                 data = json.loads(line)
#                 data_list.append(data)

#         return data_list

#     lambada_data = read_jsonl_to_list(url)
#     dataset = Dataset.from_list(lambada_data)
#     return dataset


# def get_dataset(name, mode, cache_dir=None, block_size=1024, num_proc=8):
#     if name == "wikitext103":
#         dataset = load_dataset("wikitext", name="wikitext-103-raw-v1", cache_dir=cache_dir)
#     elif name == "wikitext2":
#         dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", cache_dir=cache_dir)
#     elif name == "ptb":
#         dataset = load_dataset("ptb_text_only", cache_dir=cache_dir)
#     elif name == "lambada":
#         dataset = get_lambada_test_dataset()
#     else:
#         dataset = load_dataset(name, cache_dir=cache_dir)

#     if name == "lambada":
#         data = dataset
#     else:
#         data = dataset[mode]

#     if name.startswith("wikitext"):
#         detokenizer = wt_detokenizer
#     elif name == "ptb":
#         detokenizer = ptb_detokenizer
#     elif name == "lm1b":
#         detokenizer = lm1b_detokenizer
#     elif name == "lambada":
#         detokenizer = lambada_detokenizer
#     else:
#         detokenizer = None

#     def _apply_detokenizer(detokenizer):
#         def detok(text):
#             for i, t in enumerate(text, 0):
#                     text[i] = detokenizer(t)
#             return text
#         return detok

#     # tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
#     tokenizer = GPT2TokenizerFast.from_pretrained('gpt2',truncation_side='left')
#     EOS = tokenizer.encode(tokenizer.eos_token)[0]

#     def preprocess_and_tokenize(example):
#         if name == "ptb":
#             text = example['sentence']
#         elif name in ["ag_news", "snli", "sst2", "IMDb", "emotion", 'zeroshot/twitter-financial-news-sentiment', 'yelp_review_full','SetFit/sst5', 'SetFit/20_newsgroups', 'SetFit/sst2', 'cardiffnlp/tweet_eval/sentiment', 'Sp1786/multiclass-sentiment-analysis-dataset']:
#             # Concatenate the input sentence and label
#             if name == "snli":
#                 text = f"{example['premise']} {example['hypothesis']} Label: {example['label']}" 
#             elif name == "IMDb":
#                 text_label_pairs = zip(example['text'], example['label'])

#                 text = [str(text)+'Label:'+str(label) for text, label in text_label_pairs]
#             elif name == "emotion":
#                 text_label_pairs = zip(example['text'], example['label'])

#                 text = [str(text)+'. Label:'+str(label) for text, label in text_label_pairs]
#             elif name == 'zeroshot/twitter-financial-news-sentiment':
#                 text_label_pairs = zip(example['text'], example['label'])

#                 text = [str(text)+'. Label:'+str(label) for text, label in text_label_pairs]
#             elif name == 'yelp_review_full':
#                 text_label_pairs = zip(example['text'], example['label'])

#                 text = [str(text)+'. Label:'+str(label) for text, label in text_label_pairs]
#             elif name == "SetFit/sst5":
#                 text_label_pairs = zip(example['text'], example['label'])

#                 text = [str(text)+'. Label:'+str(label) for text, label in text_label_pairs]
#             elif name == "SetFit/20_newsgroups":
#                 text_label_pairs = zip(example['text'], example['label'])

#                 text = [str(text)+'. Label:'+str(label) for text, label in text_label_pairs]

#             elif name == "SetFit/sst2":
#                 text_label_pairs = zip(example['text'], example['label'])

#                 text = [str(text)+'. Label:'+str(label) for text, label in text_label_pairs]
                
#             elif name == "Sp1786/multiclass-sentiment-analysis-dataset":
#                 text_label_pairs = zip(example['text'], example['label'])

#                 text = [str(text)+'. Label:'+str(label) for text, label in text_label_pairs]
            
#             elif name == "cardiffnlp/tweet_eval/sentiment":
#                 text_label_pairs = zip(example['text'], example['label'])

#                 text = [str(text)+'. Label:'+str(label) for text, label in text_label_pairs]

#             else:
#                 # text = f"{example['text']}+ 'Label' +{example['label']}"
#                 # text = example["text"]+example['label']
#                 # text = example["text"]
#                 text_label_pairs = zip(example['text'], example['label'])

#                 text = [str(text)+'Label:'+str(label) for text, label in text_label_pairs]
#         else:
#             text = example["text"]
        
#         if detokenizer is not None:
#             text = _apply_detokenizer(detokenizer)(text)
#         # print("text",text,len(text),"len of text")
#         # exit()
#         # tokens = tokenizer(text, return_attention_mask=False) - used this for ag_news and imdb
#         # tokens = tokenizer(text, return_attention_mask=False, max_length = 64) # used this for emotion and twitterfin
#         # tokens = tokenizer(text, return_attention_mask=False, max_length = 768) # used this for yelp_reviews 
#         tokenizer.pad_token = tokenizer.eos_token # for multiclass sentiment analysis 2nd run
#         tokenizer.padding_side = "left"
#         tokens = tokenizer(text, return_attention_mask=False, max_length = 50, padding='max_length', truncation=True) # for multiclass sentiment analysis 2nd run
#         # import pdb; pdb.set_trace()
#         # add in EOS token following 
#         # https://github.com/jcpeterson/openwebtext/blob/master/tokenize_text.py#L67
#         for token in tokens['input_ids']:
#             token.append(EOS)
#         return tokens
    
#     tokenized_dataset = data.map(preprocess_and_tokenize, batched=True, num_proc=num_proc, load_from_cache_file=True)
#     if name == "ptb":
#         tokenized_dataset = tokenized_dataset.remove_columns('sentence')
#     elif name in ["ag_news", "snli", "sst2", "IMDb", "emotion", 'zeroshot/twitter-financial-news-sentiment', 'Sp1786/multiclass-sentiment-analysis-dataset', 'yelp_review_full', 'SetFit/sst5', 'SetFit/20_newsgroups', 'SetFit/sst2', 'cardiffnlp/tweet_eval/sentiment']:
#         # tokenized_dataset = tokenized_dataset.remove_columns(['text', 'label', 'premise', 'hypothesis']) # Remove original columns
#         tokenized_dataset = tokenized_dataset.remove_columns(['text', 'label']) # Remove original columns # for ag_news
#         # tokenized_dataset = tokenized_dataset.remove_columns(['sentence','label']) # Remove original columns
#         # tokenized_dataset = tokenized_dataset.remove_columns(['text', 'label','id','sentiment']) #multiclass-sent-analysis
        
#     else:
#         tokenized_dataset = tokenized_dataset.remove_columns('text')
    

#     def group_texts(examples):
#         # Concatenate all texts.
#         # print("len examples",len(examples))
#         print("examples.keys",examples.keys())
#         concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
#         total_length = len(concatenated_examples[list(examples.keys())[0]])
#         # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
#         # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
#         total_length = (total_length // block_size) * block_size
#         # Split by chunks of max_len.
#         result = {
#             k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
#             for k, t in concatenated_examples.items()
#         }
#         # import time; time.sleep(10)
#         # print(result) 
#         # the above should confirm if the padding is to the left and if the chunking is correct. 
#         return result

#     chunked_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=num_proc, load_from_cache_file=True)
#     chunked_dataset = chunked_dataset.with_format('torch')

#     return chunked_dataset


# def get_dataloaders(config, distributed=True):
#     if config.training.batch_size % (config.ngpus * config.training.accum) != 0:
#             raise ValueError(f"Train Batch Size {config.training.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")
#     if config.eval.batch_size % (config.ngpus * config.training.accum) != 0:
#         raise ValueError(f"Eval Batch Size for {config.eval.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")


#     # train_set = get_dataset(config.data.train, "train", cache_dir=config.data.cache_dir, block_size=config.model.length) # used this for ag_news and imdb
#     # train_set = get_dataset(config.data.train, "train", cache_dir=config.data.cache_dir, block_size=64) # used this for emotion and twitterfin dataset
#     #train_set = get_dataset(config.data.train, "train", cache_dir=config.data.cache_dir, block_size=768) # used this for yelp reviews dataset
#     train_set = get_dataset(config.data.train, "train", cache_dir=config.data.cache_dir, block_size=51) # used for the second run of multiclass sentiment analysis
    
    
    
#     # valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != "text8" else "test", cache_dir=config.data.cache_dir, block_size=config.model.length)
#     valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != "ag_news" else "test", cache_dir=config.data.cache_dir, block_size=config.model.length)
#     #valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != "IMDb" else "test", cache_dir=config.data.cache_dir, block_size=config.model.length)
#     # valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != "emotion" else "test", cache_dir=config.data.cache_dir, block_size=config.model.length)
#     #valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != "emotion" else "test", cache_dir=config.data.cache_dir, block_size=64)
#     # valid_set = get_dataset(config.data.valid, "validation" if config.data.valid == 'zeroshot/twitter-financial-news-sentiment' else "test", cache_dir=config.data.cache_dir, block_size=64)
#     # valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != 'yelp_review_full' else "test", cache_dir=config.data.cache_dir, block_size=768)
#     # valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != 'SetFit/20_newsgroups' else "test", cache_dir=config.data.cache_dir, block_size=768)
#     # valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != "SetFit/sst2" else "test", cache_dir=config.data.cache_dir, block_size=32)
#     # valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != "Sp1786/multiclass-sentiment-analysis-dataset" else "test", cache_dir=config.data.cache_dir, block_size=64)
#     # valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != "Sp1786/multiclass-sentiment-analysis-dataset" else "test", cache_dir=config.data.cache_dir, block_size=51) # used for the second run of multiclass sentiment analysis
#     # valid_set = train_set
#     # making a smaller blocksize for sst5
#     # valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != "sst5" else "test", cache_dir=config.data.cache_dir, block_size=50) 


#     if distributed:
#         train_sampler = DistributedSampler(train_set) 
#         test_sampler = DistributedSampler(valid_set)
#     else:
#         train_sampler = None
#         test_sampler = None
    

#     train_loader = cycle_loader(DataLoader(
#         train_set,
#         batch_size=config.training.batch_size // (config.ngpus * config.training.accum),
#         sampler=train_sampler,
#         num_workers=4,
#         pin_memory=True,
#         shuffle=(train_sampler is None),
#         persistent_workers=True,
#     ))
#     valid_loader = cycle_loader(DataLoader(
#         valid_set,
#         batch_size=config.eval.batch_size // (config.ngpus * config.training.accum),
#         sampler=test_sampler,
#         num_workers=4,
#         pin_memory=True,
#         shuffle=(test_sampler is None),
#     ))
#     return train_loader, valid_loader