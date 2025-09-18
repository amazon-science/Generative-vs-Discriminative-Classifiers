#!/bin/bash






datasets=(
    "imdb"
    "zeroshot/twitter-financial-news-sentiment"
    "Sp1786/multiclass-sentiment-analysis-dataset"
    "emotion"
    "SetFit/sst2"
    "SetFit/sst5"
    "SetFit/hate_speech_offensive"
    "imdb"
    "cornell-movie-review-data/rotten_tomatoes"
    "zeroshot/twitter-financial-news-sentiment"
    "Sp1786/multiclass-sentiment-analysis-dataset"
    "emotion"
    "SetFit/sst2"
    "SetFit/sst5"
    "SetFit/hate_speech_offensive"
    "imdb"
    "cornell-movie-review-data/rotten_tomatoes"
    "zeroshot/twitter-financial-news-sentiment"
    "Sp1786/multiclass-sentiment-analysis-dataset"
    "emotion"
    "SetFit/sst2"
    "SetFit/sst5"
    "SetFit/hate_speech_offensive"
    "cornell-movie-review-data/rotten_tomatoes"
)

# datasets=(
#     "ag_news"
#     "emotion"
#     "imdb"
# )

# Array of training sizes (add 'null' for full dataset)
# train_sizes=(
#     "null"
# )


# # Array of training sizes (add 'null' for full dataset)
train_sizes=(
    "128"
    "512"
    "1024"
    "2048"
    "4096"
    "null"
)







# Array of n_iters values (optional - comment out if you want to use default from config)
# n_iters=(
#     "150000"
# )

n_iters=(
    "50000"
)

# n_iters=(
#     "100000"
# )

# n_iters=(
#     "200000"
# )



# Function to run training along with the log file
LOG_FILE="training_log_$(date +%Y%m%d_%H%M%S).txt"

run_training() {
    local dataset=$1
    local size=$2
    local iters=$3
    
    {
        echo "=================================="
        echo "Starting training for:"
        echo "Dataset: $dataset"
        echo "Training size: $size"
        echo "N_iters: $iters"
        echo "Time: $(date)"
        echo "=================================="
        
        if [ "$size" = "null" ]; then
            if [ -z "$iters" ]; then
                DATASET_NAME=$dataset python train.py
            else
                DATASET_NAME=$dataset N_ITERS=$iters python train.py
            fi
        else
            if [ -z "$iters" ]; then
                DATASET_NAME=$dataset TRAIN_SIZE=$size python train.py
            else
                DATASET_NAME=$dataset TRAIN_SIZE=$size N_ITERS=$iters python train.py
            fi
        fi
        
        echo "Finished at: $(date)"
        echo "=================================="
    } 2>&1 | tee -a "$LOG_FILE"
}


# Main execution loop
for dataset in "${datasets[@]}"; do
    for size in "${train_sizes[@]}"; do
        # If you want to use n_iters, uncomment this loop
        for iters in "${n_iters[@]}"; do
            run_training "$dataset" "$size" "$iters"
        done
        
        # If you want to use default n_iters from config, use this instead
        # run_training "$dataset" "$size" ""
    done
done
