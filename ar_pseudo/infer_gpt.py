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


def compute_log_likelihoods(model, text, tokenizer, all_lbls, device):
    log_likelihoods = []

    for label in all_lbls:
        formatted_text = f"Label:{label},Text:{text}"
        inputs = tokenizer(
            formatted_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            # Use the model's lm_model directly for computing loss
            outputs = model.module.lm_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["input_ids"]
            )
            log_likelihood = -outputs.loss.item()
            log_likelihoods.append(log_likelihood)

    # Convert to probabilities
    log_likelihoods = torch.tensor(log_likelihoods)
    probs = torch.softmax(log_likelihoods, dim=0)
    return probs.cpu().numpy()


def eval_model(args, train_texts, train_labels, val_texts, val_labels, model_fpath):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = 'right'
    tokenizer.padding_side = 'right'

    all_lbls = list(range(len(set(train_labels))))
    label_ixs = [tokenizer.encode(str(label), add_special_tokens=False)[0] for label in all_lbls]
    label2ix = dict(zip(all_lbls, label_ixs))

    # Initialize model
    model = GPT2Classifier.load_from_checkpoint(
        model_fpath,
        all_lbls=all_lbls,
        label_ixs=label_ixs
    )
    model.eval()
    model = DataParallel(model)
    model.cuda()

    print(f"Predictions using the model path {model_fpath}")
    output_dir = os.path.dirname(model_fpath)
    output_file = os.path.join(output_dir, 'predictions_siva_format.csv')

    all_probs = []
    all_gts = []

    # Process each validation example
    for text, gt in tqdm(zip(val_texts, val_labels), total=len(val_texts)):
        probs = compute_log_likelihoods(model, text, tokenizer, all_lbls, "cuda")
        all_probs.append(probs)
        all_gts.append(gt)

    all_probs = np.array(all_probs)
    all_gts = np.array(all_gts)
    all_preds = np.argmax(all_probs, axis=1).tolist()
    all_gts = all_gts.tolist()

    # Write predictions to file
    with open(output_file, 'w') as f:
        f.write('ground_truth,predicted_label,scores\n')
        for gt, pred, prob in zip(all_gts, all_preds, all_probs):
            scores_str = f"[{', '.join([str(p) for p in prob])}]"
            scores_str = '"' + scores_str + '"'
            f.write(f"{gt},{pred},{scores_str}\n")

    print("Weighted F1 score")
    print(f1_score(all_gts, all_preds, average="weighted"))
    print(classification_report(all_gts, all_preds, digits=4))


def get_model_fpath(base_dpath, data_name):
    pattern = os.path.join(base_dpath, data_name, "gpt2-*")
    matching_files = glob.glob(pattern)
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
        "--bsz", type=int, default=16,
        help="Batch size per GPU"
    )

    args = parser.parse_args()
    model_fpath = get_model_fpath(args.base_dpath, args.data_name)
    print(model_fpath)
    torch.set_float32_matmul_precision('medium')

    train_texts, train_labels, val_texts, val_labels = get_dataset(args.data_key)
    model = eval_model(args, train_texts, train_labels, val_texts, val_labels, model_fpath)

# import warnings
#
# warnings.filterwarnings("ignore")
# import os
# import glob
# import argparse
# import torch
# from torch.utils.data import DataLoader
# from transformers import GPT2Tokenizer
# from sklearn.metrics import classification_report, f1_score, accuracy_score
# import numpy as np
# from train_gpt import GPT2Classifier, ClassificationDataset, get_dataset
# from torch.nn.parallel import DataParallel
# from tqdm import tqdm
# import ipdb
#
#
# def compute_log_likelihoods(model, text, tokenizer, all_lbls, device):
#     log_likelihoods = []
#     probs = []
#
#     for label in all_lbls:
#         formatted_text = f"Label:{label},Text:{text}"
#         inputs = tokenizer(
#             formatted_text,
#             return_tensors="pt",
#             padding=True,
#             truncation=True
#         )
#         inputs = {k: v.to(device) for k, v in inputs.items()}
#
#         with torch.no_grad():
#             outputs = model.module(
#                 **inputs,
#                 labels=inputs["input_ids"]
#             )
#             log_likelihood = -outputs.loss.item()
#             log_likelihoods.append(log_likelihood)
#
#     # Convert to probabilities
#     log_likelihoods = torch.tensor(log_likelihoods)
#     probs = torch.softmax(log_likelihoods, dim=0)
#     return probs.cpu().numpy()
#
#
# def eval_model(args, train_texts, train_labels, val_texts, val_labels, model_fpath):
#     tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.truncation_side = 'right'
#     tokenizer.padding_side = 'right'
#
#     all_lbls = list(range(len(set(train_labels))))
#     label_ixs = [tokenizer.encode(str(label), add_special_tokens=False)[0] for label in all_lbls]
#     label2ix = dict(zip(all_lbls, label_ixs))
#
#     # Initialize model
#     model = GPT2Classifier.load_from_checkpoint(model_fpath)
#     model.eval()
#     model = DataParallel(model)
#     model.cuda()
#
#     print(f"Predictions using the model path {model_fpath}")
#     output_dir = os.path.dirname(model_fpath)
#     output_file = os.path.join(output_dir, 'predictions_siva_format.csv')
#
#     all_probs = []
#     all_gts = []
#
#     # Process each validation example
#     for text, gt in tqdm(zip(val_texts, val_labels), total=len(val_texts)):
#         probs = compute_log_likelihoods(model, text, tokenizer, all_lbls, "cuda")
#         all_probs.append(probs)
#         all_gts.append(gt)
#
#     all_probs = np.array(all_probs)
#     all_gts = np.array(all_gts)
#     all_preds = np.argmax(all_probs, axis=1).tolist()
#     all_gts = all_gts.tolist()
#
#     # Write predictions to file
#     with open(output_file, 'w') as f:
#         f.write('ground_truth,predicted_label,scores\n')
#         for gt, pred, prob in zip(all_gts, all_preds, all_probs):
#             scores_str = f"[{', '.join([str(p) for p in prob])}]"
#             scores_str = '"' + scores_str + '"'
#             f.write(f"{gt},{pred},{scores_str}\n")
#
#     print("Weighted F1 score")
#     print(f1_score(all_gts, all_preds, average="weighted"))
#     print(classification_report(all_gts, all_preds, digits=4))
#
#
# def get_model_fpath(base_dpath, data_name):
#     pattern = os.path.join(base_dpath, data_name, "gpt2-*")
#     matching_files = glob.glob(pattern)
#     return matching_files[0] if matching_files else None
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--data_key", type=str, required=True,
#         help="Dataset key (e.g., 'SetFit/sst2', 'emotion', etc.)"
#     )
#     parser.add_argument(
#         "--data_name", type=str, required=True,
#         help="Dataset key (e.g., 'SetFit/sst2', 'emotion', etc.)"
#     )
#     parser.add_argument(
#         "--base_dpath", type=str, required=True,
#         help="The directory in which the trained models is present"
#     )
#     parser.add_argument(
#         "--n_devices", type=int, default=8,
#         help="Number of GPU devices to use"
#     )
#     parser.add_argument(
#         "--max_len", type=int, default=512,
#         help="Maximum sequence length"
#     )
#     parser.add_argument(
#         "--bsz", type=int, default=16,
#         help="Batch size per GPU"
#     )
#
#     args = parser.parse_args()
#     model_fpath = get_model_fpath(args.base_dpath, args.data_name)
#     print(model_fpath)
#     torch.set_float32_matmul_precision('medium')
#
#     train_texts, train_labels, val_texts, val_labels = get_dataset(args.data_key)
#     model = eval_model(args, train_texts, train_labels, val_texts, val_labels, model_fpath)
