#!/usr/bin/env python3
"""
Validation script for Generative vs Discriminative Text Classification repository
This script validates that all components are properly set up and can run basic operations.
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path
import argparse

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Python 3.9+ required. Current version:", sys.version)
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def check_conda():
    """Check if conda is available."""
    try:
        result = subprocess.run(['conda', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Conda available: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ Conda not found. Please install Miniconda or Anaconda.")
    return False

def check_git():
    """Check if git is available and repository is properly set up."""
    try:
        result = subprocess.run(['git', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Git available: {result.stdout.strip()}")
            
            # Check if we're in a git repository
            result = subprocess.run(['git', 'status'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Git repository properly initialized")
                return True
            else:
                print("⚠️  Not in a git repository (optional)")
                return True
    except FileNotFoundError:
        print("⚠️  Git not found (optional for running experiments)")
        return True
    
    return True

def check_directory_structure():
    """Check if all required directories and files exist."""
    required_paths = [
        'ar/environment.yml',
        'ar/train_gpt.py',
        'ar_pseudo/environment.yml', 
        'ar_pseudo/train_gpt.py',
        'diff/environment.yml',
        'diff/run_train.py',
        'encoder_mlm/environment.yml',
        'encoder_mlm/mlm_classif_seed_fixed.py',
        'setup.py',
        'QUICKSTART.md',
        'LICENSE',
        'examples/run_comprehensive_experiments.sh'
    ]
    
    missing_paths = []
    for path in required_paths:
        if not Path(path).exists():
            missing_paths.append(path)
    
    if missing_paths:
        print("❌ Missing required files:")
        for path in missing_paths:
            print(f"   - {path}")
        return False
    
    print("✅ All required files present")
    return True

def check_environment_files():
    """Check if environment files are valid YAML."""
    env_files = [
        'ar/environment.yml',
        'ar_pseudo/environment.yml',
        'diff/environment.yml',
        'encoder_mlm/environment.yml'
    ]
    
    try:
        import yaml
    except ImportError:
        print("⚠️  PyYAML not available, skipping environment file validation")
        return True
    
    for env_file in env_files:
        try:
            with open(env_file, 'r') as f:
                yaml.safe_load(f)
            print(f"✅ {env_file} is valid")
        except Exception as e:
            print(f"❌ {env_file} is invalid: {e}")
            return False
    
    return True

def check_script_permissions():
    """Check if scripts have proper permissions."""
    scripts = [
        'examples/run_comprehensive_experiments.sh',
        'setup.py'
    ]
    
    for script in scripts:
        path = Path(script)
        if path.exists():
            if os.access(path, os.X_OK):
                print(f"✅ {script} is executable")
            else:
                print(f"⚠️  {script} is not executable (run: chmod +x {script})")
        else:
            print(f"❌ {script} not found")
            return False
    
    return True

def test_basic_imports():
    """Test if basic Python packages can be imported."""
    basic_packages = [
        'pathlib',
        'argparse', 
        'subprocess',
        'os',
        'sys'
    ]
    
    for package in basic_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} import successful")
        except ImportError as e:
            print(f"❌ {package} import failed: {e}")
            return False
    
    return True

def validate_setup_script():
    """Test the setup script functionality."""
    try:
        result = subprocess.run([sys.executable, 'setup.py', '--check'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ Setup script works correctly")
            return True
        else:
            print(f"❌ Setup script failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("⚠️  Setup script timed out")
        return False
    except Exception as e:
        print(f"❌ Setup script error: {e}")
        return False

def run_quick_syntax_check():
    """Run basic syntax checks on Python files."""
    python_files = [
        'setup.py',
        'validate_setup.py',
        'ar/train_gpt.py',
        'ar_pseudo/train_gpt.py',
        'diff/run_train.py',
        'encoder_mlm/mlm_classif_seed_fixed.py'
    ]
    
    for py_file in python_files:
        if Path(py_file).exists():
            try:
                result = subprocess.run([sys.executable, '-m', 'py_compile', py_file],
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✅ {py_file} syntax OK")
                else:
                    print(f"❌ {py_file} syntax error: {result.stderr}")
                    return False
            except Exception as e:
                print(f"❌ Error checking {py_file}: {e}")
                return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Validate repository setup")
    parser.add_argument('--quick', action='store_true', 
                       help='Run only quick validation checks')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output')
    
    args = parser.parse_args()
    
    print("🔍 Validating Generative vs Discriminative Text Classification Repository")
    print("=" * 70)
    
    checks = [
        ("Python Version", check_python_version),
        ("Conda Installation", check_conda),
        ("Git Setup", check_git),
        ("Directory Structure", check_directory_structure),
        ("Environment Files", check_environment_files),
        ("Script Permissions", check_script_permissions),
        ("Basic Imports", test_basic_imports),
    ]
    
    if not args.quick:
        checks.extend([
            ("Setup Script", validate_setup_script),
            ("Syntax Check", run_quick_syntax_check),
        ])
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        print(f"\n📋 {check_name}:")
        try:
            if check_func():
                passed += 1
            else:
                print(f"   Check failed: {check_name}")
        except Exception as e:
            print(f"   Check error: {e}")
    
    print("\n" + "=" * 70)
    print(f"📊 Validation Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All validation checks passed! Repository is ready to use.")
        print("\n🚀 Next steps:")
        print("1. Run: python setup.py --list")
        print("2. Choose an approach: python setup.py --approach <choice>")
        print("3. Try the demo: ./examples/run_comprehensive_experiments.sh demo")
        return 0
    else:
        print("❌ Some validation checks failed. Please address the issues above.")
        print("\n🔧 Common solutions:")
        print("- Install missing dependencies")
        print("- Fix file permissions: chmod +x <script>")
        print("- Check environment file syntax")
        return 1

if __name__ == "__main__":
    sys.exit(main())
