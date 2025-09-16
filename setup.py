#!/usr/bin/env python3
"""
Setup script for Generative vs Discriminative Text Classification
This script helps users set up the appropriate environment for their chosen approach.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

def run_command(command, cwd=None):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True,
            cwd=cwd
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error output: {e.stderr}")
        return None

def check_conda():
    """Check if conda is installed."""
    result = run_command("conda --version")
    if result:
        print(f"✓ Conda found: {result.strip()}")
        return True
    else:
        print("✗ Conda not found. Please install Miniconda or Anaconda first.")
        print("Visit: https://docs.conda.io/en/latest/miniconda.html")
        return False

def setup_environment(approach):
    """Set up the conda environment for the specified approach."""
    approach_dirs = {
        'ar': 'ar',
        'ar_pseudo': 'ar_pseudo', 
        'diffusion': 'diff',
        'encoder': 'encoder_mlm'
    }
    
    if approach not in approach_dirs:
        print(f"Unknown approach: {approach}")
        return False
    
    approach_dir = approach_dirs[approach]
    env_file = Path(approach_dir) / "environment.yml"
    
    if not env_file.exists():
        print(f"Environment file not found: {env_file}")
        return False
    
    print(f"Setting up environment for {approach} approach...")
    print(f"Using environment file: {env_file}")
    
    # Create conda environment
    command = f"conda env create -f {env_file}"
    result = run_command(command)
    
    if result is not None:
        print(f"✓ Environment setup complete for {approach}")
        
        # Get environment name from the yml file
        with open(env_file, 'r') as f:
            for line in f:
                if line.startswith('name:'):
                    env_name = line.split(':')[1].strip()
                    print(f"\nTo activate the environment, run:")
                    print(f"conda activate {env_name}")
                    break
        
        return True
    else:
        print(f"✗ Failed to setup environment for {approach}")
        return False

def list_approaches():
    """List all available approaches."""
    approaches = {
        'ar': 'Autoregressive models (GPT-based generative classification)',
        'ar_pseudo': 'Pseudo-autoregressive models (hybrid approach)',
        'diffusion': 'Discrete diffusion models (novel generative approach)',
        'encoder': 'Encoder/MLM models (BERT-based discriminative classification)'
    }
    
    print("Available approaches:")
    for key, description in approaches.items():
        print(f"  {key}: {description}")

def main():
    parser = argparse.ArgumentParser(
        description="Setup script for Generative vs Discriminative Text Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup.py --approach ar          # Setup autoregressive models
  python setup.py --approach diffusion  # Setup diffusion models
  python setup.py --list               # List all approaches
  python setup.py --check              # Check prerequisites
        """
    )
    
    parser.add_argument(
        '--approach', 
        choices=['ar', 'ar_pseudo', 'diffusion', 'encoder'],
        help='Choose the modeling approach to setup'
    )
    
    parser.add_argument(
        '--list', 
        action='store_true',
        help='List all available approaches'
    )
    
    parser.add_argument(
        '--check', 
        action='store_true',
        help='Check prerequisites (conda installation)'
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_approaches()
        return
    
    if args.check:
        if check_conda():
            print("✓ All prerequisites satisfied")
        else:
            sys.exit(1)
        return
    
    if not args.approach:
        print("Please specify an approach or use --list to see options")
        parser.print_help()
        return
    
    # Check prerequisites
    if not check_conda():
        sys.exit(1)
    
    # Setup the chosen approach
    success = setup_environment(args.approach)
    
    if success:
        print(f"\n🎉 Setup complete for {args.approach} approach!")
        print("\nNext steps:")
        print(f"1. Activate the environment: conda activate <env_name>")
        print(f"2. Navigate to the {args.approach} directory")
        print("3. Follow the instructions in the README for training/inference")
    else:
        print(f"\n❌ Setup failed for {args.approach} approach")
        sys.exit(1)

if __name__ == "__main__":
    main()
