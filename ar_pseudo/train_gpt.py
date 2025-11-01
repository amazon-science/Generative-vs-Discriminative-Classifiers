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
        tokenizer.truncation_side = 'left'  # Important for maintaining recent context
        tokenizer.padding_side = 'left'
        if is_train:
            tokenizer.padding_side = 'right'  # Important for GPT2
        
        for text, label in zip(texts, labels):
            example_dict = {}
            label_ix = lbl2ix[label]
            example_dict['val_label'] = torch.tensor(label, dtype=torch.int8)
            example_dict['val_label_ix'] = torch.tensor(label_ix, dtype=torch.int8)
            formatted_text = f"text: {text}, label:"
            
            tok_op = tokenizer(
                formatted_text,
                padding='max_length',  # Will pad to max_length
                truncation=True,       # Enable truncation
                max_length=max_length - 1,
                return_tensors='pt'    # Return PyTorch tensors
            )
                
            # Remove the extra dimension that return_tensors='pt' adds
            input_ids = tok_op['input_ids'].squeeze(0)
            attention_mask = tok_op['attention_mask'].squeeze(0)
            
            # Create a new tensor with -100 where attention_mask is 0
            if is_train:
                input_ids, attention_mask = self.add_label(input_ids, attention_mask, label_ix)
                labels_ten = input_ids.clone()
                labels_ten[attention_mask == 0] = -100
                example_dict['labels'] = labels_ten

            example_dict['input_ids'] = input_ids
            example_dict['attention_mask'] = attention_mask
            self.examples.append(example_dict)
    
    def add_label(self, input_ids, attention_mask, label_ix):
        new_input_ids = torch.empty(len(input_ids) + 1, dtype=input_ids.dtype)
        new_attention_mask = torch.empty(len(attention_mask) + 1, dtype=attention_mask.dtype)
        zeros = (attention_mask == 0).nonzero()
        first_zero_ix = zeros[0].item() if len(zeros) > 0 else len(attention_mask)
        
        new_input_ids[:first_zero_ix] = input_ids[:first_zero_ix]
        new_input_ids[first_zero_ix] = label_ix
        new_input_ids[first_zero_ix + 1:] = input_ids[first_zero_ix:]

        new_attention_mask[:first_zero_ix] = attention_mask[:first_zero_ix]
        new_attention_mask[first_zero_ix] = 1
        new_attention_mask[first_zero_ix + 1:] = attention_mask[first_zero_ix:]

        return new_input_ids, new_attention_mask

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class GPT2Classifier(pl.LightningModule):
    def __init__(self, model_size, learning_rate=1e-5, all_lbls=None, label_ixs=None):
        super().__init__()
        self.save_hyperparameters()
        # Use the same configuration as the original GPT-2
        self.config = GPT2Config.from_pretrained('gpt2')
        if model_size == 'small':
            self.config.n_layer = 1
            self.config.n_head = 1
            self.config.n_embd = int(768/12)
        if model_size == 'medium':
            self.config.n_layer = 6
            self.config.n_head = 6
            self.config.n_embd = int((768 * 6)/12)
            
        self.lm_model = GPT2LMHeadModel(self.config) # Initialize model with GPT-2 architecture (but with random weights)
        self.learning_rate = learning_rate
        self.tok = GPT2Tokenizer.from_pretrained('gpt2')
        self.all_lbls = all_lbls # This will be in increasing order [0, 1, 2, 3]
        self.label_ixs = label_ixs # The corresponding tokens [23, 46, 36, 22]
        self.val_preds = []
        self.val_gts = []

    def forward(self, input_ids, attention_mask):
        # This has to return the predicted label.
        op = self.lm_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = op.logits[:, -1, :]
        logits_selected = logits[:, self.label_ixs]  # Select logits corresponding to label_ixs
        probs = torch.softmax(logits_selected, dim=-1)  # Apply softmax over the selected logits
        return probs

    def training_step(self, batch, batch_idx):
        outputs = self.lm_model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],  # Provide labels for computing loss            
        )
        loss = outputs.loss
        # Add check for None and detach the loss
        if loss is not None:
            perplexity = torch.exp(loss)
            # Log metrics
            self.log_dict({
                'train_loss': loss.detach(),
                'train_perplexity': perplexity.detach(),
                
            }, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # batch_preds = self(input_ids=batch['val_input_ids'], attention_mask=batch['val_attention_mask'])
        batch_preds = self(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        self.val_preds.append(batch_preds)
        self.val_gts.append(batch['val_label'])

    def on_validation_epoch_end(self):
        # Synchronize across all devices to ensure consistency
        self.trainer.strategy.barrier()
        # Gather all predictions and ground truths from all devices
        val_pred_ten = torch.cat(self.val_preds, dim=0)
        val_gt_ten = torch.cat(self.val_gts, dim=0)
        self.trainer.strategy.barrier()
        all_preds = self.trainer.strategy.all_gather(val_pred_ten)
        all_gts = self.trainer.strategy.all_gather(val_gt_ten)
        self.trainer.strategy.barrier()
        
        # Only process on the main process (rank 0) to avoid redundant computation
        # if self.trainer.global_rank == 0:
        # Flatten nested lists from distributed devices
        flat_preds = all_preds.view(-1, len(self.all_lbls))
        flat_argmax = torch.argmax(flat_preds, dim=1).tolist()
        flat_gts = all_gts.view(-1).tolist()

        # Compute evaluation metrics
        macro_f1 = f1_score(flat_gts, flat_argmax, average='macro')
        weighted_f1 = f1_score(flat_gts, flat_argmax, average='weighted')
        accuracy = accuracy_score(flat_gts, flat_argmax)
        # ipdb.set_trace()
        # Log the computed metrics
        self.log('macro_f1', macro_f1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('weighted_f1', weighted_f1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('accuracy', accuracy, on_epoch=True, prog_bar=True, sync_dist=True)

        # Reset validation data for the next epoch
        self.val_preds.clear()
        self.val_gts.clear()
        
        # Final barrier to ensure all processes are synchronized before the next step
        self.trainer.strategy.barrier()

    def on_train_start(self):
        # Manually set 'weighted_f1' to -inf after the first epoch to avoid early stopping triggering early
        if self.current_epoch == 0:
            self.log('weighted_f1', 0.0, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('macro_f1', 0.0, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('accuracy', 0.0, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)  # Add weight decay)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode='max', factor=0.1, patience=1, verbose=True
        # )
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": scheduler,
            #     "monitor": "weighted_f1"
            # }
        }


def train_model(args, train_texts, train_labels, val_texts, val_labels):
    # Use pretrained GPT-2 tokenizer
    # Create directories
    os.makedirs(args.ckpt_dir, exist_ok=True)
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    all_lbls = list(range(len(set(train_labels))))
    label_ixs = [tokenizer.encode(str(label), add_special_tokens=False)[0] for label in all_lbls]
    label2ix = dict(zip(all_lbls, label_ixs))
    # Create datasets
    train_dataset = ClassificationDataset(
        tokenizer, label2ix, train_texts, train_labels, max_length=args.max_len, is_train=True
    )
    val_dataset = ClassificationDataset(
        tokenizer, label2ix, val_texts, val_labels, max_length=args.max_len, is_train=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bsz,
        shuffle=True,
        # num_workers=4,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.bsz,
        shuffle=False,
        # num_workers=4,
        num_workers=0,
        pin_memory=True
    )

    # Initialize model
    
    
    model = GPT2Classifier(args.model_size, all_lbls=all_lbls, label_ixs=label_ixs)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Setup callbacks
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename='gpt2-{epoch:02d}-{macro_f1:.4f}-{weighted_f1:.4f}-{accuracy:.4f}',
        save_top_k=1,
        monitor='weighted_f1',
        mode='max',
        save_last=True,
        every_n_epochs=1
    )
    
    # TODO: Add early stopping callback
    early_stopping_callback = EarlyStopping(
        monitor='weighted_f1',
        patience=100,
        mode='max',
        verbose=True,
        min_delta=0.001
    )
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=args.ckpt_dir,
        name='logs',
        default_hp_metric=False
    )
    
    # Setup trainer with both callbacks
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
    # Train
    trainer.fit(model, train_loader, val_loader)
    return model
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_key", type=str,
        help="Dataset key (e.g., 'SetFit/sst2', 'emotion', etc.)"
    )
    parser.add_argument(
        "--ckpt_dir", type=str, default=None,
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--n_devices", type=int, default=4,
        help="Number of GPU devices to use"
    )
    parser.add_argument(
        "--max_len", type=int, default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--bsz", type=int, default=8,
        help="Batch size per GPU"
    )
    parser.add_argument(
        "--model_size", type=str, default="full",
        help="model size - small, medium, full"
    )
    parser.add_argument(
        "--seed", type=int, default=100,
    )
    parser.add_argument(
        "--n_tr_sub", type=int, default=-1,
    )
    parser.add_argument(
        "--max_epochs", type=int, default=500,
        help="Maximum number of training epochs"
    )
    

    args = parser.parse_args()
    seed_everything(args.seed)
    torch.set_float32_matmul_precision('medium')
    # Get dataset
    train_texts, train_labels, val_texts, val_labels = get_dataset(args.data_key, args.seed)
    if args.n_tr_sub > 0:
        train_texts,  train_labels = train_texts[:args.n_tr_sub],  train_labels[:args.n_tr_sub]
    val_texts, val_labels = val_texts[:480], val_labels[:480] # 8 x 2 x 24  (n_gpus x n_batches x bsz)
    model = train_model(args, train_texts, train_labels, val_texts, val_labels)
