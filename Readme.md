# Generative vs Discriminative Text Classification: A Comprehensive Comparison

[![arXiv](https://img.shields.io/badge/arXiv-2506.12181-b31b1b.svg)](https://arxiv.org/abs/2506.12181)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

This repository contains the official implementation for the paper **"Generative or Discriminative? Revisiting Text Classification in the Era of Transformers"** by Siva Rajesh Kasa et al.

## 📖 Abstract

The comparison between discriminative and generative classifiers has intrigued researchers since Efron's seminal analysis of logistic regression versus discriminant analysis. While early theoretical work established that generative classifiers exhibit lower sample complexity but higher asymptotic error in simple linear settings, these trade-offs remain unexplored in the transformer era. We present the first comprehensive evaluation of modern generative and discriminative architectures - Auto-regressive modeling, Masked Language Modeling, Discrete Diffusion, and Encoders for text classification.

## 🏗️ Repository Structure

```
├── README.md                    # This file
├── environment.yml             # Shared conda environment for AR, AR-Pseudo, and Encoder/MLM
├── ar/                         # Autoregressive classifier models
│   ├── train_gpt.py           # Training script for GPT-based classifiers
│   └── infer_gpt.py           # Inference script for GPT-based classifiers
├── ar_pseudo/                  # Pseudo-autoregressive variant classifiers
│   ├── train_gpt.py           # Training script for pseudo-AR classifiers
│   └── infer_gpt.py           # Inference script for pseudo-AR classifiers
├── diff/                       # Discrete diffusion classifier models
│   ├── README.md              # Detailed documentation for diffusion models
│   ├── environment.yml        # Conda environment for diffusion models
│   ├── run_train.py          # Training script
│   ├── run_sample.py         # Sampling script
│   ├── parallel_inference.py # Parallel inference for classification
│   ├── model/                # Model architectures
│   ├── configs/              # Configuration files
│   └── ...                   # Additional diffusion-related files
└── encoder_mlm/               # Encoder and MLM classifier models
    ├── mlm_classif_seed_fixed.py  # Training script with fixed seeds
    └── inference.py           # Inference script
```

## 🚀 Quick Start

### Automated Setup

Use our setup script for easy environment configuration:

```bash
# Check prerequisites and list approaches
python setup.py --check
python setup.py --list

# Setup your chosen approach
python setup.py --approach ar          # Autoregressive models
python setup.py --approach diffusion  # Diffusion models  
python setup.py --approach encoder    # Encoder models
```

### Quick Demo

Run a quick demo to verify your setup:

```bash
# Automated comprehensive demo (recommended)
./examples/run_comprehensive_experiments.sh demo
```

### Manual Installation

If you prefer manual setup:

#### 1. AR, AR-Pseudo, and Encoder/MLM Models
```bash
# Shared environment for AR, AR-Pseudo, and Encoder/MLM approaches
conda env create -f environment.yml
conda activate gendisc-transformers
```

#### 2. Discrete Diffusion Models
```bash
cd diff/
conda env create -f environment.yml
conda activate sedd
```

## 🔬 Experiments

### Autoregressive Classification

Train GPT-based classifiers using generative modeling:

```bash
cd ar/
python train_gpt.py \
    --data_key "SetFit/sst2" \
    --ckpt_dir "./checkpoints/ar_sst2" \
    --model_size "small" \
    --max_epochs 50 \
    --bsz 8
```

**Key Parameters:**
- `--data_key`: Dataset identifier (e.g., "SetFit/sst2", "emotion", "ag_news")
- `--model_size`: Model size ("small", "medium", "full")
- `--n_devices`: Number of GPUs to use
- `--max_len`: Maximum sequence length
- `--seed`: Random seed for reproducibility

### Discrete Diffusion Classification

Train discrete diffusion models for text classification:

```bash
cd diff/
# Single experiment with environment variables
DATASET_NAME="SetFit/sst2" TRAIN_SIZE="1024" N_ITERS="50000" python train.py model=small

# Or run comprehensive experiments across multiple datasets and sizes
./run_exps.sh
```

For inference:
```bash
python parallel_inference.py \
    --model_path "path/to/trained/model" \
    --dataset "ag_news" \
    --batch_size 32
```

### Encoder/MLM Classification

Run comprehensive experiments with BERT-based models:

```bash
cd encoder_mlm/
python mlm_classif_seed_fixed.py
```

This script runs experiments across:
- Multiple datasets (emotion, sst2, ag_news, etc.)
- Different model sizes (1 layer, 6 layers, 12 layers)
- Various training sample sizes (128, 256, 512, 1024, 2048, 4096, full)
- Multiple random seeds for statistical significance
- Both MLM pretraining and direct classification approaches

## 📊 Supported Datasets

The repository supports various text classification datasets:

- **Sentiment Analysis**: SST-2, SST-5, IMDb, Rotten Tomatoes
- **Topic Classification**: AG News
- **Emotion Detection**: Emotion dataset
- **Hate Speech Detection**: Hate Speech Offensive
- **Multi-class Sentiment**: Multi-class sentiment analysis
- **Financial News**: Twitter Financial News Sentiment

## 🔧 Model Architectures

### 1. Autoregressive (AR) Models
- GPT-2 based architecture
- Generative approach: P(label|text) via likelihood estimation
- Configurable model sizes (small, medium, full)

### 2. Pseudo-Autoregressive Models
- Modified autoregressive approach
- Hybrid generative-discriminative training

### 3. Discrete Diffusion Models
- Score-based discrete diffusion
- Novel application to text classification
- Supports both uniform and absorbing noise schedules
- Three model configurations available:
  - **small**: 1 layer, 1 attention head (1,1) - for quick experiments
  - **medium**: 6 layers, 6 attention heads (6,6) - balanced performance
  - **large**: 12 layers, 12 attention heads (12,12) - best performance

### 4. Encoder Models
- BERT-based discriminative classifiers
- Masked Language Model (MLM) pretraining option
- Standard discriminative approach: direct classification head

## 📈 Key Findings

Our comprehensive evaluation reveals:

1. **Sample Efficiency**: Generative models show superior performance in low-data regimes
2. **Asymptotic Performance**: Discriminative models achieve better performance with abundant data
3. **Calibration**: Different architectures exhibit varying calibration properties
4. **Robustness**: Noise robustness varies significantly across approaches
5. **Computational Trade-offs**: Inference speed vs. accuracy considerations

## 🛠️ Customization

### Adding New Datasets

To add support for new datasets, modify the dataset loading functions in each component:

- **AR models**: Update `get_dataset()` in `train_gpt.py`
- **Diffusion models**: Modify `get_dataset()` in `data.py`
- **Encoder models**: Add dataset path in `DATASET_PATH` dictionary

### Model Configuration

Each component supports extensive configuration:

- **AR models**: Modify model architecture in `GPT2Classifier` class
- **Diffusion models**: Use Hydra configs in `configs/` directory
- **Encoder models**: Adjust `MODEL_CONFIGS` for different architectures

## 📝 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{kasa2025generative,
  title={Generative or Discriminative? Revisiting Text Classification in the Era of Transformers},
  author={Kasa, Siva Rajesh and Gupta, Karan and Roychowdhury, Sumegh and Kumar, Ashutosh and Biruduraju, Yaswanth and Kasa, Santhosh Kumar and Pattisapu, Nikhil Priyatam and Bhattacharya, Arindam and Agarwal, Shailendra and huddar, Vijay},
  journal={arXiv preprint arXiv:2506.12181},
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions! Please feel free to:

1. Report bugs or issues
2. Suggest new features or improvements
3. Submit pull requests with enhancements
4. Add support for new datasets or models

## 📄 License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This work builds upon several excellent open-source projects:
- [Score Entropy Discrete Diffusion](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion) by Aaron Lou et al. - Our discrete diffusion implementation is based on this foundational work
- [Transformers](https://github.com/huggingface/transformers) by Hugging Face
- [PyTorch Lightning](https://github.com/Lightning-AI/lightning) for training infrastructure
- [Score SDE](https://github.com/yang-song/score_sde_pytorch) for diffusion model foundations
- [PLAID](https://github.com/igul222/plaid) for discrete diffusion insights

## 📞 Contact

For questions or issues, please:
1. Open an issue on GitHub
2. Contact the corresponding authors via the paper

---
