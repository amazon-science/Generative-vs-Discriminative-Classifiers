import warnings

warnings.filterwarnings("ignore")

import re
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from sklearn.metrics import classification_report, f1_score, accuracy_score
import numpy as np
from datasets import load_dataset
import os
from pytorch_lightning import seed_everything
import sys
import ipdb


def get_dataset(data_key, seed_val=42):
    """Load and split the dataset"""
    dataset = load_dataset(data_key, trust_remote_code=True)
    dataset = dataset.map(lambda x: {'text': str(x['text'])})
    train_dataset = dataset['train']
    train_dataset = train_dataset.shuffle(seed=seed_val)
    # NOTE: Using test split for validation here for illustration purposes only.
    # In proper research setup, this should be changed to use validation split for hyperparameter tuning,
    # with final performance evaluation on the test split to avoid overestimating performance.
    val_dataset = dataset['test'] if 'test' in dataset else dataset['validation']
    val_dataset = val_dataset.shuffle(seed=seed_val)
    train_texts = train_dataset['text']
    train_labels = train_dataset['label']
    val_texts = val_dataset['text']
    val_labels = val_dataset['label']
    return train_texts, train_labels, val_texts, val_labels


class ClassificationDataset(Dataset):
    def __init__(self, tokenizer, lbl2ix, texts, labels, max_length=512, is_train=True):
        self.examples = []
        tokenizer.truncation_side = 'right'
        tokenizer.padding_side = 'right'

        for text, label in zip(texts, labels):
            example_dict = {}
            label_ix = lbl2ix[label]
            example_dict['val_label'] = torch.tensor(label, dtype=torch.int8)
            example_dict['val_label_ix'] = torch.tensor(label_ix, dtype=torch.int8)
            example_dict['text'] = text  # Store the original text

            formatted_text = f"Label:{label},Text:{text}"

            tok_op = tokenizer(
                formatted_text,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )

            input_ids = tok_op['input_ids'].squeeze(0)
            attention_mask = tok_op['attention_mask'].squeeze(0)

            if is_train:
                labels_ten = input_ids.clone()
                labels_ten[attention_mask == 0] = -100
                example_dict['labels'] = labels_ten

            example_dict['input_ids'] = input_ids
            example_dict['attention_mask'] = attention_mask
            self.examples.append(example_dict)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class GPT2Classifier(pl.LightningModule):
    def __init__(self, model_size, learning_rate=1e-5, all_lbls=None, label_ixs=None):
        super().__init__()
        self.save_hyperparameters()
        self.config = GPT2Config.from_pretrained('gpt2')
        if model_size == 'small':
            self.config.n_layer = 1
            self.config.n_head = 1
            self.config.n_embd = int(768 / 12)
        if model_size == 'medium':
            self.config.n_layer = 6
            self.config.n_head = 6
            self.config.n_embd = int((768 * 6) / 12)

        self.lm_model = GPT2LMHeadModel(self.config)
        self.learning_rate = learning_rate
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.all_lbls = all_lbls
        self.label_ixs = label_ixs
        self.val_preds = []
        self.val_gts = []

    def forward(self, input_ids, attention_mask):
        op = self.lm_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = op.logits[:, -1, :]
        logits_selected = logits[:, self.label_ixs]
        probs = torch.softmax(logits_selected, dim=-1)
        return probs

    def inference(self, text):
        log_likelihoods = []
        for label in self.all_lbls:
            formatted_text = f"Label:{label},Text:{text}"
            inputs = self.tokenizer(formatted_text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.lm_model(**inputs, labels=inputs["input_ids"])
                log_likelihood = -outputs.loss.item()

            log_likelihoods.append(log_likelihood)

        predicted_label_index = np.argmax(log_likelihoods)
        return self.all_lbls[predicted_label_index]

    def training_step(self, batch, batch_idx):
        outputs = self.lm_model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        loss = outputs.loss
        if loss is not None:
            perplexity = torch.exp(loss)
            self.log_dict({
                'train_loss': loss.detach(),
                'train_perplexity': perplexity.detach(),
            }, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        texts = batch['text']
        labels = batch['val_label']

        preds = []
        for text in texts:
            pred = self.inference(text)
            preds.append(pred)

        # Convert predictions to tensor if they aren't already
        if not isinstance(preds, torch.Tensor):
            preds = torch.tensor(preds, device=self.device)

        self.val_preds.extend(preds.cpu().tolist())
        self.val_gts.extend(labels.cpu().tolist())


# uncomment for epoch-wise results
    # def on_validation_epoch_end(self):
    #     self.trainer.strategy.barrier()
    #     all_preds = self.all_gather(self.val_preds)
    #     all_gts = self.all_gather(self.val_gts)
    #     self.trainer.strategy.barrier()
    #
    #     flat_preds = [item for sublist in all_preds for item in sublist]
    #     flat_gts = [item for sublist in all_gts for item in sublist]
    #
    #     macro_f1 = f1_score(flat_gts, flat_preds, average='macro')
    #     weighted_f1 = f1_score(flat_gts, flat_preds, average='weighted')
    #     accuracy = accuracy_score(flat_gts, flat_preds)
    #
    #     self.log('macro_f1', macro_f1, on_epoch=True, prog_bar=True, sync_dist=True)
    #     self.log('weighted_f1', weighted_f1, on_epoch=True, prog_bar=True, sync_dist=True)
    #     self.log('accuracy', accuracy, on_epoch=True, prog_bar=True, sync_dist=True)
    #
    #     self.val_preds.clear()
    #     self.val_gts.clear()
    #
    #     self.trainer.strategy.barrier()

    def on_validation_epoch_end(self):
        self.trainer.strategy.barrier()

        # Gather predictions and move to CPU
        all_preds = self.all_gather(self.val_preds)
        all_gts = self.all_gather(self.val_gts)

        # Move to CPU and flatten
        if isinstance(all_preds, torch.Tensor):
            flat_preds = all_preds.cpu().view(-1).tolist()
        else:
            flat_preds = [item.cpu().item() if isinstance(item, torch.Tensor) else item
                          for sublist in all_preds
                          for item in sublist]

        if isinstance(all_gts, torch.Tensor):
            flat_gts = all_gts.cpu().view(-1).tolist()
        else:
            flat_gts = [item.cpu().item() if isinstance(item, torch.Tensor) else item
                        for sublist in all_gts
                        for item in sublist]

        # Compute metrics
        macro_f1 = f1_score(flat_gts, flat_preds, average='macro')
        weighted_f1 = f1_score(flat_gts, flat_preds, average='weighted')
        accuracy = accuracy_score(flat_gts, flat_preds)

        # Log metrics
        self.log('macro_f1', macro_f1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('weighted_f1', weighted_f1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('accuracy', accuracy, on_epoch=True, prog_bar=True, sync_dist=True)

        # Clear lists
        self.val_preds.clear()
        self.val_gts.clear()

        self.trainer.strategy.barrier()

    def on_train_start(self):
        if self.current_epoch == 0:
            self.log('weighted_f1', 0.0, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('macro_f1', 0.0, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('accuracy', 0.0, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        return {"optimizer": optimizer}


def train_model(args, train_texts, train_labels, val_texts, val_labels):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    all_lbls = list(range(len(set(train_labels))))
    label_ixs = [tokenizer.encode(str(label), add_special_tokens=False)[0] for label in all_lbls]
    label2ix = dict(zip(all_lbls, label_ixs))

    train_dataset = ClassificationDataset(
        tokenizer, label2ix, train_texts, train_labels, max_length=args.max_len, is_train=True
    )
    val_dataset = ClassificationDataset(
        tokenizer, label2ix, val_texts, val_labels, max_length=args.max_len, is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bsz,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.bsz,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = GPT2Classifier(args.model_size, all_lbls=all_lbls, label_ixs=label_ixs)
    model.config.pad_token_id = tokenizer.pad_token_id

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename='gpt2-{epoch:02d}-{macro_f1:.4f}-{weighted_f1:.4f}-{accuracy:.4f}',
        save_top_k=1,
        monitor='weighted_f1',
        mode='max',
        save_last=True,
        every_n_epochs=1
    )

    early_stopping_callback = EarlyStopping(
        monitor='weighted_f1',
        patience=100,
        mode='max',
        verbose=True,
        min_delta=0.001
    )

    logger = TensorBoardLogger(
        save_dir=args.ckpt_dir,
        name='logs',
        default_hp_metric=False
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu',
        devices=args.n_devices,
        strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        val_check_interval=1.0,
        enable_model_summary=False,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0
    )

    trainer.fit(model, train_loader, val_loader)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_key", type=str, help="Dataset key (e.g., 'SetFit/sst2', 'emotion', etc.)")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Directory to save checkpoints")
    parser.add_argument("--n_devices", type=int, default=4, help="Number of GPU devices to use")
    parser.add_argument("--max_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--bsz", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--model_size", type=str, default="full", help="model size - small, medium, full")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--n_tr_sub", type=int, default=-1)
    parser.add_argument("--max_epochs", type=int, default=500, help="Maximum number of training epochs")

    args = parser.parse_args()
    seed_everything(args.seed)
    torch.set_float32_matmul_precision('medium')

    train_texts, train_labels, val_texts, val_labels = get_dataset(args.data_key, args.seed)
    if args.n_tr_sub > 0:
        train_texts, train_labels = train_texts[:args.n_tr_sub], train_labels[:args.n_tr_sub]
    val_texts, val_labels = val_texts[:480], val_labels[:480]

    model = train_model(args, train_texts, train_labels, val_texts, val_labels)
