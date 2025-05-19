import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM
from datasets import load_dataset
from sklearn.metrics import classification_report
from tqdm import tqdm
import argparse
import pandas as pd
import numpy as np

DATASET_PATH = {
    "rottentomatoes": "cornell-movie-review-data/rotten_tomatoes",
    "twitter": "zeroshot/twitter-financial-news-sentiment",
    "hatespeech": "SetFit/hate_speech_offensive",
    "agnews":"ag_news",
    "emotion":"emotion",
    "sst2":"SetFit/sst2",
    "sst5":"SetFit/sst5",
    "multiclasssentiment":"Sp1786/multiclass-sentiment-analysis-dataset",
    "IMDb":"IMDb"
}
ACCEPTABLE_LABELS = {
    "rottentomatoes": {"0", "1"},
    "twitter": {"0", "1", "2"},
    "hatespeech": {"0", "1", "2"},
    "agnews": {"0", "1", "2", "3"},
    "emotion": {"0", "1", "2", "3","4","5"},
    "sst2":{"0","1"},
    "sst5":{"0","1","2","3","4"},
    "multiclasssentiment":{"0","1","2"},
    "IMDb":{"0","1"}
}

FORMAT = "{} Label:{}"

def extract_k_v(checkpoint_path):
    checkpoint_name = checkpoint_path.split("/")[-1]
    checkpoint_kv = checkpoint_name.split("_")
    result_items = {}
    for kv in checkpoint_kv:
        try:
            k, v = kv.split("=")
            result_items[k] = v
        except:
            continue

    return result_items
    

def main(args):
    args_kv = extract_k_v(args.checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)

    if args_kv['strategy'] == "mlm":
        model = AutoModelForMaskedLM.from_pretrained(args.checkpoint_path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    try:
        dataset_test = load_dataset(DATASET_PATH[args_kv['dataset']])["test"]
    except:
        dataset_test = load_dataset(DATASET_PATH[args_kv['dataset']])["validation"]

    dataset_test = dataset_test.map(lambda x: {'text': str(x['text'])})
    
    true_labels = []
    pred_labels = []
    all_scores = []  # To store probability scores

    batch_size = 16
    texts = []
    labels = []

    if args_kv['strategy'] == "mlm":
        for sample in dataset_test:
            text = sample["text"]
            label = str(sample["label"])
            if label not in ACCEPTABLE_LABELS[args_kv['dataset']]:
                continue
            texts.append(FORMAT.format(text, tokenizer.mask_token))
            labels.append(label)
    else:
        for sample in dataset_test:
            text = sample["text"]
            label = str(sample["label"])
            texts.append(text)
            labels.append(label)

    results = []  # To store all results

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]
        
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs).logits

        if args_kv['strategy'] == "classification":
            # Apply softmax to get probabilities
            scores = F.softmax(outputs, dim=-1)
            predicted_labels = outputs.argmax(dim=-1)
            
            # Convert to numpy for easier handling
            scores = scores.cpu().numpy()
            predicted_labels = predicted_labels.cpu().numpy()
            
            for j in range(len(batch_labels)):
                results.append({
                    'ground_truth': batch_labels[j],
                    'predicted_label': str(predicted_labels[j]),
                    'scores': scores[j].tolist()  # Store all class probabilities
                })

        else:  # MLM strategy
            mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
            acceptable_token_ids = {tokenizer.convert_tokens_to_ids(label) for label in ACCEPTABLE_LABELS[args_kv['dataset']]}
            
            masked_logits = outputs[mask_token_index[0], mask_token_index[1]]
            
            # Create mask for acceptable tokens
            mask = torch.zeros_like(masked_logits, dtype=torch.bool)
            for token_id in acceptable_token_ids:
                mask[:, token_id] = True
            
            # Apply softmax only over acceptable tokens
            masked_logits[~mask] = float('-inf')
            scores = F.softmax(masked_logits, dim=-1)
            
            predicted_tokens = masked_logits.argmax(dim=-1)
            predicted_labels = tokenizer.batch_decode(predicted_tokens, skip_special_tokens=True)
            
            scores = scores.cpu().numpy()
            
            for j in range(len(predicted_labels)):
                if predicted_labels[j] in ACCEPTABLE_LABELS[args_kv['dataset']]:
                    # Get scores only for acceptable labels
                    label_scores = {label: scores[j][tokenizer.convert_tokens_to_ids(label)] 
                                  for label in ACCEPTABLE_LABELS[args_kv['dataset']]}
                    
                    results.append({
                        'ground_truth': batch_labels[j],
                        'predicted_label': predicted_labels[j],
                        'scores': label_scores
                    })

    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv('results.csv', index=False)

    # Print classification report
    print(classification_report(df['ground_truth'], df['predicted_label']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training of BERT models")
    parser.add_argument("-checkpoint_path", "--checkpoint_path", type=str, help="Checkpoint path for inference")
    args = parser.parse_args()
    main(args)
