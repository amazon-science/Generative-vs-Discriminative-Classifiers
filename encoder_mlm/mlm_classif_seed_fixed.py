import argparse
import logging
import time
from pathlib import Path

import datasets
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import (AutoConfig, AutoModelForMaskedLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorForLanguageModeling,
                          EarlyStoppingCallback, Trainer, TrainerCallback,
                          TrainingArguments)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("experiment_log.txt"), logging.StreamHandler()],
)


# SEEDS = [12010,120491,120046,120011, 120011]
SEEDS = [79140,24561,54641]
# SAMPLE_SIZES = [128,512,256,1024,2048,4096]
SAMPLE_SIZES = [-1,4096,2048,1024,512,256,128]
# SAMPLE_SIZES = [-1]
# SAMPLE_SIZES = [2048,1024,512,256,128]
# SAMPLE_SIZES = [-1,4096,2048,1024]
# SAMPLE_SIZES = [4096,2048,1024]
# SAMPLE_SIZES = [128,256,512]
# SAMPLE_SIZES = [128,256,512]
# MODEL_CONFIGS = [(1,1),(12,12),(6,6)]  # (num_layers, num_heads)
# MODEL_CONFIGS = [(12,12),(6,6),(1,1)]  # (num_layers, num_heads)
# MODEL_CONFIGS = [(12,12),(6,6)]  # (num_layers, num_heads)
MODEL_CONFIGS = [(1,1),(6,6)]  # (num_layers, num_heads)
# MODEL_CONFIGS = [(12,12),(1,1),(6,6)]  # (num_layers, num_heads)

DATASET_PATH = {
    # "emotion": "emotion",
                    # "IMDb": "IMDb",
        # "sst2": "SetFit/sst2",
        # "sst5": "SetFit/sst5",
                # "rottentomatoes": "cornell-movie-review-data/rotten_tomatoes",
                     "hatespeech": "SetFit/hate_speech_offensive", 
        # "multiclasssentiment": "Sp1786/multiclass-sentiment-analysis-dataset",
            # "agnews": "ag_news",
    # "twitter": "zeroshot/twitter-financial-news-sentiment",

}

# SEEDS = [94785,193673,180076,792,7051]
# # SEEDS = [65644]
# SAMPLE_SIZES = [128,512,256,1024,2048,4096]
# # SAMPLE_SIZES = [4096,2048,1024,512,256,128]
# # SAMPLE_SIZES = [4096,2048,1024]
# # SAMPLE_SIZES = [128,256,512]
# # SAMPLE_SIZES = [128,256,512]
# MODEL_CONFIGS = [(1,1),(12,12),(6,6)]  # (num_layers, num_heads)

# DATASET_PATH = {
#     # "emotion": "emotion",
#                     # "IMDb": "IMDb",
#         # "sst2": "SetFit/sst2",
#         # "sst5": "SetFit/sst5",
#                 # "rottentomatoes": "cornell-movie-review-data/rotten_tomatoes",
#                     #  "hatespeech": "SetFit/hate_speech_offensive", 
#         "multiclasssentiment": "Sp1786/multiclass-sentiment-analysis-dataset",
#             # "agnews": "ag_news",
#     # "twitter": "zeroshot/twitter-financial-news-sentiment",

# }

def get_dataset_subset(dataset, num_samples, seed):
    if num_samples == "full":
        return dataset

    # Get balanced subset of training data
    train_subset = (
        dataset["train"]
        .shuffle(seed=seed)
        .select(range(min(num_samples, len(dataset["train"]))))
    )
    
    print(f"length of train sample {len(train_subset)}")
    
    # For validation/test, take proportional samples
    val_or_test_size = len(dataset["test" if "test" in dataset else "validation"])
    # NOTE: Using test split for validation here for illustration purposes only.
    # In proper research setup, this should be changed to use validation split for hyperparameter tuning,
    # with final performance evaluation on the test split to avoid overestimating performance.
    val_or_test_subset = (
        dataset["test" if "test" in dataset else "validation"]
        .shuffle(seed=seed)
        .select(range(val_or_test_size))
    )

    return datasets.DatasetDict(
        {
            "train": train_subset,
            "test" if "test" in dataset else "validation": val_or_test_subset,
        }
    )

def mlm(args, sample_size, seed):
    dataset = load_dataset(DATASET_PATH[args.dataset])

    # Apply subsetting if needed
    if sample_size != -1:
        dataset = get_dataset_subset(dataset, sample_size, seed)



    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_function(examples):
        text_label_pairs = zip(examples["text"], examples["label"])
        if args.dataset in [
            "agnews", "emotion", "sst2", "sst5", "multiclasssentiment",
            "rottentomatoes", "twitter", "hatespeech", "IMDb",
        ]:
            texts_with_labels = [
                f"{text} Label:{label}" for text, label in text_label_pairs
            ]
            return tokenizer(
                texts_with_labels, padding="max_length", truncation=True, max_length=512
            )
        else:
            raise NotImplementedError

    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=["label", "text"]
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    return tokenized_datasets, tokenizer, data_collator

def classification(args, sample_size, seed):
    dataset = load_dataset(DATASET_PATH[args.dataset])
    dataset = dataset.map(lambda x: {'text': str(x['text'])})
    print("reaching here")
    
    # Apply subsetting if needed
    if sample_size != -1:
        dataset = get_dataset_subset(dataset, sample_size, seed)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    num_labels = len(set(dataset["train"]["label"]))



    def tokenize_function(examples):
        if args.dataset in [
            "agnews", "emotion", "sst2", "sst5", "multiclasssentiment",
            "rottentomatoes", "twitter", "hatespeech", "IMDb",
        ]:
            return tokenizer(
                examples["text"], padding="max_length", truncation=True, max_length=512
            )
        else:
            raise NotImplementedError

    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    return tokenized_datasets, tokenizer, None

def get_custom_config(num_attention_layers, num_attention_heads):
    config = AutoConfig.from_pretrained("bert-base-uncased")
    config.num_hidden_layers = num_attention_layers
    config.num_attention_heads = num_attention_heads
    config.hidden_size = int((768 * num_attention_heads) // 12)
    return config

def run_experiments():
    total_experiments = (
        len(DATASET_PATH)
        * (len(SAMPLE_SIZES) - 1)
        * len(MODEL_CONFIGS)
        * len(SEEDS)
        * 2
    ) + len(DATASET_PATH) * len(MODEL_CONFIGS) * 2

    main_pbar = tqdm(total=total_experiments, desc="Total Progress", position=0)
    experiment_count = 0

    for dataset_name in DATASET_PATH.keys():
        for num_layers, num_heads in MODEL_CONFIGS:
            # for training_strategy in ["mlm"]:
            for training_strategy in ["mlm", "classification"]:
            # for training_strategy in ["classification"]:
                for sample_size in SAMPLE_SIZES:
                    for seed in SEEDS:
                        experiment_count += 1

                        experiment_name = f"bert_dataset={dataset_name}_strategy={training_strategy}_samples={sample_size}_layers={num_layers}_heads={num_heads}_seed={seed}"

                        logging.info(
                            f"Starting experiment {experiment_count}/{total_experiments}: {experiment_name}"
                        )

                        try:
                            output_dir = Path(f"./models/{experiment_name}")
                            final_model_dir = Path(f"./models-trained/{experiment_name}")
                            output_dir.mkdir(parents=True, exist_ok=True)
                            final_model_dir.mkdir(parents=True, exist_ok=True)

                            config = get_custom_config(num_layers, num_heads)

                            if training_strategy == "mlm":
                                model = AutoModelForMaskedLM.from_config(config)
                                tokenized_datasets, tokenizer, data_collator = mlm(
                                    args=argparse.Namespace(
                                        dataset=dataset_name, 
                                        use_pretrained=False
                                    ),
                                    sample_size=sample_size,
                                    seed=seed
                                )
                            else:
                                dataset = load_dataset(DATASET_PATH[dataset_name])
                                if sample_size != -1:
                                    dataset = get_dataset_subset(dataset, sample_size, seed)
                                num_labels = len(set(dataset["train"]["label"]))
                                config.num_labels = num_labels
                                model = AutoModelForSequenceClassification.from_config(config)
                                tokenized_datasets, tokenizer, data_collator = classification(
                                    args=argparse.Namespace(
                                        dataset=dataset_name, 
                                        use_pretrained=False
                                    ),
                                    sample_size=sample_size,
                                    seed=seed
                                )

                            training_args = TrainingArguments(
                                output_dir=str(output_dir),
                                evaluation_strategy="epoch",
                                save_strategy="epoch",
                                per_device_train_batch_size=16,
                                per_device_eval_batch_size=16,
                                num_train_epochs=200,
                                logging_dir=f"./logs/{experiment_name}",
                                save_total_limit=5,
                                report_to="tensorboard",
                                metric_for_best_model="eval_loss",
                                load_best_model_at_end=True,
                                greater_is_better=False,
                                logging_first_step=True,
                                logging_steps=10,
                            )

                            class ProgressCallback(TrainerCallback):
                                def __init__(self, total_steps, experiment_name):
                                    super().__init__()
                                    self.total_steps = total_steps
                                    self.current_step = 0
                                    self.progress_bar = tqdm(
                                        total=total_steps,
                                        desc=f"Training {experiment_name}",
                                        position=1,
                                        leave=False,
                                    )

                                def on_step_begin(self, args, state, control, **kwargs):
                                    if not hasattr(self, "current_step"):
                                        self.current_step = 0

                                def on_step_end(self, args, state, control, **kwargs):
                                    if hasattr(self, "current_step"):
                                        self.current_step += 1
                                        self.progress_bar.update(1)

                                    if state.log_history:
                                        try:
                                            current_loss = state.log_history[-1].get("loss", 0)
                                            self.progress_bar.set_postfix(
                                                {"loss": f"{current_loss:.4f}"}
                                            )
                                        except:
                                            self.progress_bar.set_postfix({"loss": "N/A"})

                                def on_train_end(self, args, state, control, **kwargs):
                                    self.progress_bar.close()

                            total_steps = (
                                len(tokenized_datasets["train"])
                                // training_args.per_device_train_batch_size
                                * training_args.num_train_epochs
                            )

                            trainer = Trainer(
                                model=model,
                                args=training_args,
                                train_dataset=tokenized_datasets["train"],
                                eval_dataset=(
                                    tokenized_datasets["validation"]
                                    if "validation" in tokenized_datasets
                                    else tokenized_datasets["test"]
                                ),
                                tokenizer=tokenizer,
                                data_collator=data_collator,
                                callbacks=[
                                    EarlyStoppingCallback(
                                        early_stopping_patience=10 if training_strategy == "classification" else 10
                                    ),
                                    ProgressCallback(int(total_steps), experiment_name),
                                ],
                            )

                            trainer.train()
                            trainer.save_model(str(final_model_dir))
                            logging.info(f"Successfully completed experiment: {experiment_name}")

                            if sample_size == -1:
                                break

                        except Exception as e:
                            logging.error(f"Failed experiment {experiment_name}: {str(e)}")
                            continue

                        finally:
                            main_pbar.update(1)
                            print("\n" + "=" * 80 + "\n")

    main_pbar.close()
    logging.info("All experiments completed!")

if __name__ == "__main__":
    start_time = time.time()
    run_experiments()
    end_time = time.time()

    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    logging.info(f"Total execution time: {hours}h {minutes}m {seconds}s")
