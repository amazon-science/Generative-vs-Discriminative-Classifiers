# Quick Start Guide

This guide will help you get started with the Generative vs Discriminative Text Classification experiments quickly.

## 🚀 Quick Setup

### 1. Choose Your Approach

First, decide which modeling approach you want to experiment with:

- **Autoregressive (AR)**: GPT-based generative classification
- **Discrete Diffusion**: Novel generative approach using diffusion models
- **Encoder/MLM**: BERT-based discriminative classification
- **Pseudo-AR**: Hybrid generative-discriminative approach

### 2. Automated Setup

Use our setup script to automatically configure the environment:

```bash
# Check prerequisites
python setup.py --check

# List available approaches
python setup.py --list

# Setup specific approach (example: autoregressive)
python setup.py --approach ar
```

### 3. Manual Setup (Alternative)

If you prefer manual setup:

```bash
# For Autoregressive models
cd ar/
conda env create -f environment.yml
conda activate gendisc

# For Diffusion models
cd diff/
conda env create -f environment.yml
conda activate sedd

# For Encoder models
cd encoder_mlm/
conda env create -f environment.yml
conda activate encoder_mlm
```

## 🧪 Running Experiments

### Quick Demo (Recommended for First Time)

Run a quick demo to verify everything works:

```bash
# Run quick demo experiments (takes ~10-30 minutes)
./examples/run_comprehensive_experiments.sh demo
```

This will run small-scale experiments across all approaches to verify your setup.

### Individual Approach Experiments

#### Autoregressive Classification

```bash
cd ar/
python train_gpt.py \
    --data_key "SetFit/sst2" \
    --ckpt_dir "./checkpoints/sst2_experiment" \
    --model_size "small" \
    --max_epochs 50 \
    --bsz 8 \
    --seed 42
```

#### Diffusion Classification

```bash
cd diff/
DATASET_NAME="SetFit/sst2" TRAIN_SIZE="1024" N_ITERS="50000" python train.py model=small
```

#### Encoder Classification

```bash
cd encoder_mlm/
python mlm_classif_seed_fixed.py
```

### Comprehensive Experiments

For full-scale experiments (WARNING: Very time-consuming):

```bash
# Run all experiments (can take days/weeks)
./examples/run_comprehensive_experiments.sh full

# Run specific approach only
./examples/run_comprehensive_experiments.sh ar        # Autoregressive only
./examples/run_comprehensive_experiments.sh diffusion # Diffusion only
./examples/run_comprehensive_experiments.sh encoder   # Encoder only
```

## 📊 Understanding Results

### Output Structure

Experiments will create the following structure:

```
experiment_results/
├── ar/                 # Autoregressive results
├── diffusion/          # Diffusion results
├── encoder/            # Encoder results
└── demo/              # Demo results

experiment_logs/
├── main_TIMESTAMP.log  # Main experiment log
├── ar_*.log           # AR experiment logs
├── diff_*.log         # Diffusion experiment logs
└── encoder_*.log      # Encoder experiment logs
```

### Key Metrics

Each experiment tracks:
- **Accuracy**: Classification accuracy
- **F1 Score**: Macro and weighted F1 scores
- **Sample Efficiency**: Performance vs training data size
- **Training Time**: Time to convergence
- **Model Size**: Number of parameters

## 🔧 Customization

### Adding New Datasets

1. **For AR models**: Update `get_dataset()` in `ar/train_gpt.py`
2. **For Diffusion**: Modify dataset loading in `diff/data.py`
3. **For Encoders**: Add to `DATASET_PATH` in `encoder_mlm/mlm_classif_seed_fixed.py`

### Modifying Model Architectures

1. **AR models**: Edit `GPT2Classifier` class in `ar/train_gpt.py`
2. **Diffusion**: Use config files in `diff/configs/`
3. **Encoders**: Modify `MODEL_CONFIGS` in the experiment script

### Experiment Configuration

Edit the experiment script variables:

```bash
# In examples/run_comprehensive_experiments.sh
DATASETS=(...)          # Add/remove datasets
TRAIN_SIZES=(...)       # Modify sample sizes
SEEDS=(...)             # Change random seeds
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--bsz 4` or `--bsz 2`
   - Use smaller models: `--model_size small`
   - Reduce sequence length: `--max_len 256`

2. **Environment Issues**
   - Ensure CUDA versions match between PyTorch and flash-attention
   - Try recreating the conda environment
   - Check GPU compatibility

3. **Dataset Loading Errors**
   - Verify internet connection for HuggingFace datasets
   - Check dataset names are correct
   - Some datasets may require authentication

### Getting Help

1. Check the main [README.md](README.md) for detailed documentation
2. Look at experiment logs in `experiment_logs/`
3. Open an issue on GitHub with:
   - Error message
   - System configuration
   - Steps to reproduce

## 📈 Expected Results

### Sample Efficiency Trends

Based on our paper findings:

- **Low Data (128-512 samples)**: Generative models (AR, Diffusion) typically perform better
- **Medium Data (1K-4K samples)**: Performance gap narrows
- **High Data (Full datasets)**: Discriminative models (Encoders) often achieve best performance

### Runtime Expectations

Approximate training times (single GPU):

- **Demo experiments**: 10-30 minutes
- **Single dataset, single approach**: 1-4 hours
- **Full experimental suite**: Days to weeks

### Performance Baselines

Expected accuracy ranges on common datasets:

- **SST-2**: 80-95% (depending on model and data size)
- **AG News**: 85-95%
- **Emotion**: 75-90%
- **IMDb**: 85-95%

## 🎯 Next Steps

After running experiments:

1. **Analyze Results**: Compare performance across approaches
2. **Visualize Trends**: Plot sample efficiency curves
3. **Statistical Testing**: Use multiple seeds for significance testing
4. **Extend Research**: Try new datasets or model architectures

## 📚 Further Reading

- [Main README](README.md) - Comprehensive documentation
- [Paper](https://arxiv.org/abs/2506.12181) - Theoretical background
- [Diffusion README](diff/README.md) - Detailed diffusion model documentation

---

Happy experimenting! 🚀
