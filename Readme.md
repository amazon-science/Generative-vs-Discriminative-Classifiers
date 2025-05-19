
# Code Repo Structure 

The repository is organized into the following main folders:

- `ar/`: Contains autoregressive classifier models and related code
- `ar_pseudo/`: Contains autoregressive-pseudo variant classifier models and related code
- `diff/`: Contains diffusion classifier models implementation
- `encoder_mlm/`: Contains encoder classifier and masked language classifier model implementations


# Conda environment 

The yml file for installing the conda environment is present in the respective folders as `environment.yml` file. 


# Training and inference 

- `ar/` and `ar_pseudo/`: train_gpt.py for training and infer_gpt.py for inference
- `diff/`: run_exps.sh for training and parallel_inference.py for inference
- `encoder_mlm/`: mlm_classif_seed_fixed.py for training and inference.py for inference







