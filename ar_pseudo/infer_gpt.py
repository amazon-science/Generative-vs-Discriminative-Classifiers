import warnings
warnings.filterwarnings("ignore")
import os
import glob
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from sklearn.metrics import classification_report, f1_score, accuracy_score
import numpy as np
from train_gpt import GPT2Classifier, ClassificationDataset, get_dataset
from torch.nn.parallel import DataParallel
from tqdm import tqdm
import ipdb


def eval_model(args, train_texts, train_labels, val_texts, val_labels, model_fpath):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    all_lbls = list(range(len(set(train_labels))))
    label_ixs = [tokenizer.encode(str(label), add_special_tokens=False)[0] for label in all_lbls]
    label2ix = dict(zip(all_lbls, label_ixs))
    # Create datasets
    val_dataset = ClassificationDataset(
        tokenizer, label2ix, val_texts, val_labels, max_length=args.max_len, is_train=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.bsz,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Predictions using the model path {model_fpath}")
    # Write predictions to file
    output_dir = os.path.dirname(model_fpath)
    output_file = os.path.join(output_dir, 'predictions_format.csv')

    # Initialize model
    model = GPT2Classifier.load_from_checkpoint(model_fpath)
    model.eval()
    model = DataParallel(model)
    preds, gts = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader):
            batch = {k: v.cuda() for k, v in batch.items()}
            batch_preds = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            preds.append(batch_preds.cpu().numpy())
            gts.append(batch['val_label'].cpu().numpy())
            # Process outputs as needed
            
    all_probs = np.concatenate(preds, axis=0)
    all_gts = np.concatenate(gts, axis=0).tolist()
    all_preds = np.argmax(all_probs, axis=1).tolist()
    
    with open(output_file, 'w') as f:
        # Write header
        f.write('ground_truth,predicted_label,scores\n')
        
        # Write each prediction
        for gt, pred, prob in zip(all_gts, all_preds, all_probs):
            # Convert probabilities to string format
            scores_str = f"[{', '.join([str(p) for p in prob])}]"
            scores_str = '"' + scores_str + '"'
            f.write(f"{gt},{pred},{scores_str}\n")
    print("Weighted F1 score")
    print(f1_score(all_gts, all_preds, average="weighted"))
    print(classification_report(all_gts, all_preds, digits=4))
    

def get_model_fpath(base_dpath, data_name):
    pattern = os.path.join(base_dpath, data_name, "gpt2-*")
    # Find all files matching the pattern
    matching_files = glob.glob(pattern)
    # Return the first match if found, otherwise None
    return matching_files[0] if matching_files else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_key", type=str, required=True,
        help="Dataset key (e.g., 'SetFit/sst2', 'emotion', etc.)"
    )
    parser.add_argument(
        "--data_name", type=str, required=True,
        help="Dataset key (e.g., 'SetFit/sst2', 'emotion', etc.)"
    )
    parser.add_argument(
        "--base_dpath", type=str, required=True,
        help="The directory in which the trained models is present"
    )
    parser.add_argument(
        "--n_devices", type=int, default=8,
        help="Number of GPU devices to use"
    )
    parser.add_argument(
        "--max_len", type=int, default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--bsz", type=int, default=1024,
        help="Batch size per GPU"
    )
    

    args = parser.parse_args()
    model_fpath = get_model_fpath(args.base_dpath, args.data_name)
    print(model_fpath)
    torch.set_float32_matmul_precision('medium')
    # Get dataset
    train_texts, train_labels, val_texts, val_labels = get_dataset(args.data_key)
    model = eval_model(args, train_texts, train_labels, val_texts, val_labels, model_fpath)
