#!/bin/bash

# Comprehensive Experiment Runner for Generative vs Discriminative Text Classification
# This script runs experiments across all approaches with various configurations

set -e  # Exit on any error

# Configuration
LOG_DIR="./experiment_logs"
RESULTS_DIR="./experiment_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

# Dataset configurations
DATASETS=(
    "SetFit/sst2"
    "SetFit/sst5"
    "emotion"
    "ag_news"
    "imdb"
    "SetFit/hate_speech_offensive"
    "cornell-movie-review-data/rotten_tomatoes"
    "zeroshot/twitter-financial-news-sentiment"
    "Sp1786/multiclass-sentiment-analysis-dataset"
)

# Training sizes for sample efficiency analysis
TRAIN_SIZES=(
    "128"
    "512"
    "1024"
    "2048"
    "4096"
    "null"  # Full dataset
)

# Model configurations
AR_MODEL_SIZES=("small" "medium" "full")
DIFF_MODEL_SIZES=("small" "medium")
ENCODER_LAYER_CONFIGS=("1,1" "6,6" "12,12")  # layers,heads

# Seeds for statistical significance
SEEDS=(42 123 456)

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/main_${TIMESTAMP}.log"
}

# Function to run autoregressive experiments
run_ar_experiments() {
    log_message "Starting Autoregressive Experiments"
    
    cd ar/
    
    for dataset in "${DATASETS[@]}"; do
        for size in "${TRAIN_SIZES[@]}"; do
            for model_size in "${AR_MODEL_SIZES[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    
                    experiment_name="ar_${dataset//\//_}_${size}_${model_size}_seed${seed}"
                    log_file="$LOG_DIR/${experiment_name}_${TIMESTAMP}.log"
                    
                    log_message "Running AR experiment: $experiment_name"
                    
                    # Prepare arguments
                    args=(
                        --data_key "$dataset"
                        --ckpt_dir "../$RESULTS_DIR/ar/$experiment_name"
                        --model_size "$model_size"
                        --seed "$seed"
                        --max_epochs 100
                        --bsz 8
                        --n_devices 1
                        --max_len 512
                    )
                    
                    # Add training size if not full dataset
                    if [ "$size" != "null" ]; then
                        args+=(--n_tr_sub "$size")
                    fi
                    
                    # Run experiment
                    {
                        echo "Starting AR experiment: $experiment_name"
                        echo "Dataset: $dataset, Size: $size, Model: $model_size, Seed: $seed"
                        echo "Command: python train_gpt.py ${args[*]}"
                        echo "----------------------------------------"
                        
                        python train_gpt.py "${args[@]}"
                        
                        echo "Completed AR experiment: $experiment_name"
                    } 2>&1 | tee "$log_file"
                    
                done
            done
        done
    done
    
    cd ..
    log_message "Completed Autoregressive Experiments"
}

# Function to run diffusion experiments
run_diffusion_experiments() {
    log_message "Starting Diffusion Experiments"
    
    cd diff/
    
    for dataset in "${DATASETS[@]}"; do
        for size in "${TRAIN_SIZES[@]}"; do
            for model_size in "${DIFF_MODEL_SIZES[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    
                    experiment_name="diff_${dataset//\//_}_${size}_${model_size}_seed${seed}"
                    log_file="$LOG_DIR/${experiment_name}_${TIMESTAMP}.log"
                    
                    log_message "Running Diffusion experiment: $experiment_name"
                    
                    # Set environment variables
                    export DATASET_NAME="$dataset"
                    export TRAIN_SIZE="$size"
                    export N_ITERS="50000"
                    export SEED="$seed"
                    
                    # Run experiment
                    {
                        echo "Starting Diffusion experiment: $experiment_name"
                        echo "Dataset: $dataset, Size: $size, Model: $model_size, Seed: $seed"
                        echo "Environment: DATASET_NAME=$DATASET_NAME TRAIN_SIZE=$TRAIN_SIZE N_ITERS=$N_ITERS SEED=$SEED"
                        echo "Command: python train.py model=$model_size"
                        echo "----------------------------------------"
                        
                        if [ "$size" = "null" ]; then
                            unset TRAIN_SIZE
                            DATASET_NAME="$dataset" N_ITERS="$N_ITERS" SEED="$seed" python train.py model="$model_size"
                        else
                            DATASET_NAME="$dataset" TRAIN_SIZE="$size" N_ITERS="$N_ITERS" SEED="$seed" python train.py model="$model_size"
                        fi
                        
                        echo "Completed Diffusion experiment: $experiment_name"
                    } 2>&1 | tee "$log_file"
                    
                done
            done
        done
    done
    
    cd ..
    log_message "Completed Diffusion Experiments"
}

# Function to run encoder/MLM experiments
run_encoder_experiments() {
    log_message "Starting Encoder/MLM Experiments"
    
    cd encoder_mlm/
    
    # Create a comprehensive experiment script
    cat > comprehensive_experiments.py << 'EOF'
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
import sys
import os

# Configuration from command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--sample_sizes', nargs='+', type=int, required=True)
    parser.add_argument('--model_configs', nargs='+', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--strategies', nargs='+', choices=['mlm', 'classification'], required=True)
    return parser.parse_args()

def get_dataset_subset(dataset, num_samples, seed):
    if num_samples == -1:
        return dataset
    
    train_subset = (
        dataset["train"]
        .shuffle(seed=seed)
        .select(range(min(num_samples, len(dataset["train"]))))
    )
    
    val_or_test_size = len(dataset["test" if "test" in dataset else "validation"])
    val_or_test_subset = (
        dataset["test" if "test" in dataset else "validation"]
        .shuffle(seed=seed)
        .select(range(min(500, val_or_test_size)))  # Limit validation size
    )

    return datasets.DatasetDict({
        "train": train_subset,
        "test" if "test" in dataset else "validation": val_or_test_subset,
    })

def get_custom_config(num_attention_layers, num_attention_heads):
    config = AutoConfig.from_pretrained("bert-base-uncased")
    config.num_hidden_layers = num_attention_layers
    config.num_attention_heads = num_attention_heads
    config.hidden_size = int((768 * num_attention_heads) // 12)
    return config

def run_experiment(dataset_name, strategy, sample_size, num_layers, num_heads, seed):
    experiment_name = f"bert_dataset={dataset_name}_strategy={strategy}_samples={sample_size}_layers={num_layers}_heads={num_heads}_seed={seed}"
    
    print(f"Running experiment: {experiment_name}")
    
    try:
        output_dir = Path(f"../experiment_results/encoder/{experiment_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        config = get_custom_config(num_layers, num_heads)
        
        # Load and prepare dataset
        dataset = load_dataset(dataset_name)
        dataset = dataset.map(lambda x: {'text': str(x['text'])})
        
        if sample_size != -1:
            dataset = get_dataset_subset(dataset, sample_size, seed)
        
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        if strategy == "mlm":
            model = AutoModelForMaskedLM.from_config(config)
            
            def tokenize_function(examples):
                text_label_pairs = zip(examples["text"], examples["label"])
                texts_with_labels = [f"{text} Label:{label}" for text, label in text_label_pairs]
                return tokenizer(texts_with_labels, padding="max_length", truncation=True, max_length=512)
            
            tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["label", "text"])
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
            
        else:  # classification
            num_labels = len(set(dataset["train"]["label"]))
            config.num_labels = num_labels
            model = AutoModelForSequenceClassification.from_config(config)
            
            def tokenize_function(examples):
                return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
            
            tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
            tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
            data_collator = None
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=50,
            logging_dir=f"../experiment_logs/encoder/{experiment_name}",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_steps=10,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test" if "test" in tokenized_datasets else "validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )
        
        trainer.train()
        trainer.save_model(str(output_dir / "final_model"))
        
        print(f"Successfully completed experiment: {experiment_name}")
        return True
        
    except Exception as e:
        print(f"Failed experiment {experiment_name}: {str(e)}")
        return False

def main():
    args = parse_args()
    
    # Dataset mapping
    dataset_mapping = {
        "sst2": "SetFit/sst2",
        "sst5": "SetFit/sst5", 
        "emotion": "emotion",
        "ag_news": "ag_news",
        "imdb": "imdb",
        "hatespeech": "SetFit/hate_speech_offensive",
        "rottentomatoes": "cornell-movie-review-data/rotten_tomatoes",
        "twitfin": "zeroshot/twitter-financial-news-sentiment",
        "multiclass": "Sp1786/multiclass-sentiment-analysis-dataset"
    }
    
    total_experiments = len(args.datasets) * len(args.sample_sizes) * len(args.model_configs) * len(args.seeds) * len(args.strategies)
    completed = 0
    
    for dataset_key in args.datasets:
        dataset_name = dataset_mapping.get(dataset_key, dataset_key)
        
        for sample_size in args.sample_sizes:
            for model_config in args.model_configs:
                num_layers, num_heads = map(int, model_config.split(','))
                
                for seed in args.seeds:
                    for strategy in args.strategies:
                        success = run_experiment(dataset_name, strategy, sample_size, num_layers, num_heads, seed)
                        completed += 1
                        print(f"Progress: {completed}/{total_experiments} experiments completed")

if __name__ == "__main__":
    main()
EOF
    
    # Run comprehensive experiments
    {
        echo "Starting Encoder/MLM comprehensive experiments"
        echo "----------------------------------------"
        
        python comprehensive_experiments.py \
            --datasets sst2 emotion ag_news \
            --sample_sizes 128 512 1024 2048 4096 -1 \
            --model_configs "1,1" "6,6" "12,12" \
            --seeds 42 123 456 \
            --strategies mlm classification
            
        echo "Completed Encoder/MLM experiments"
    } 2>&1 | tee "$LOG_DIR/encoder_comprehensive_${TIMESTAMP}.log"
    
    cd ..
    log_message "Completed Encoder/MLM Experiments"
}

# Function to run quick demo experiments
run_demo_experiments() {
    log_message "Starting Quick Demo Experiments (reduced scale)"
    
    # Quick AR demo
    cd ar/
    {
        echo "Running quick AR demo..."
        python train_gpt.py \
            --data_key "SetFit/sst2" \
            --ckpt_dir "../$RESULTS_DIR/demo/ar_demo" \
            --model_size "small" \
            --seed 42 \
            --max_epochs 3 \
            --bsz 4 \
            --n_devices 1 \
            --max_len 128 \
            --n_tr_sub 100
    } 2>&1 | tee "$LOG_DIR/demo_ar_${TIMESTAMP}.log"
    cd ..
    
    # Quick diffusion demo
    cd diff/
    {
        echo "Running quick diffusion demo..."
        DATASET_NAME="SetFit/sst2" TRAIN_SIZE="100" N_ITERS="1000" python train.py model=small
    } 2>&1 | tee "$LOG_DIR/demo_diff_${TIMESTAMP}.log"
    cd ..
    
    log_message "Completed Quick Demo Experiments"
}

# Main execution
main() {
    log_message "Starting Comprehensive Experiment Suite"
    log_message "Timestamp: $TIMESTAMP"
    log_message "Log Directory: $LOG_DIR"
    log_message "Results Directory: $RESULTS_DIR"
    
    # Parse command line arguments
    EXPERIMENT_TYPE=${1:-"demo"}
    
    case $EXPERIMENT_TYPE in
        "full")
            log_message "Running full experimental suite"
            run_ar_experiments
            run_diffusion_experiments
            run_encoder_experiments
            ;;
        "ar")
            log_message "Running only autoregressive experiments"
            run_ar_experiments
            ;;
        "diffusion")
            log_message "Running only diffusion experiments"
            run_diffusion_experiments
            ;;
        "encoder")
            log_message "Running only encoder experiments"
            run_encoder_experiments
            ;;
        "demo")
            log_message "Running quick demo experiments"
            run_demo_experiments
            ;;
        *)
            echo "Usage: $0 [full|ar|diffusion|encoder|demo]"
            echo "  full     - Run all experiments (WARNING: Very time consuming!)"
            echo "  ar       - Run only autoregressive experiments"
            echo "  diffusion - Run only diffusion experiments"
            echo "  encoder  - Run only encoder experiments"
            echo "  demo     - Run quick demo experiments (default)"
            exit 1
            ;;
    esac
    
    log_message "Experiment suite completed!"
    log_message "Results saved in: $RESULTS_DIR"
    log_message "Logs saved in: $LOG_DIR"
}

# Execute main function with all arguments
main "$@"
