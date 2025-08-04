# Ray Tune Integration for codonOnlyClassificationMLP.py

## Overview
The script has been updated to use Ray Tune with ASHAScheduler and HyperOptSearch for automated hyperparameter optimization. All file paths are automatically converted to absolute paths to ensure compatibility with Ray Tune's multiprocessing.

## Key Changes Made

### 1. Added Ray Tune Training Function
- Created `train_model()` function that serves as the training objective for Ray Tune
- This function receives hyperparameter configurations from Ray Tune and reports metrics back

### 2. DataFrame Serialization for Ray Tune
- DataFrames are converted to serializable dictionaries before passing to Ray Tune workers
- Automatically reconstructed within the `train_model()` function
- Ensures compatibility with Ray Tune's multiprocessing

### 3. Absolute Path Conversion
- All input paths (`data_dir`, `output_dir`, `extraFeatures`, `ray_tune_local_dir`, etc.) are automatically converted to absolute paths
- Prevents path resolution issues in Ray Tune worker processes
- Includes safety checks within the training function

### 4. Hyperparameter Search Space
The script now optimizes these hyperparameters based on your best values:
- **learning_rate**: Search range from 1e-5 to 5e-3 (log-uniform distribution)
- **weight_decay**: Search range from 0.001 to 0.1 (uniform distribution)  
- **dropout_percent**: Search range from 0.1 to 0.9 (uniform distribution)

### 3. Ray Tune Configuration
- **ASHAScheduler**: Early stopping for poor performing trials
- **HyperOptSearch**: Bayesian optimization for efficient hyperparameter search
- **Metric**: Optimizes for maximum Spearman correlation (`spearmanr`)

### 4. Command Line Arguments
The script already had Ray Tune arguments which are now functional:
- `--ray_tune_samples`: Number of trials (default: 20)
- `--ray_tune_max_epochs`: Max epochs per trial (default: 10)
- `--ray_tune_grace_period`: Min epochs before early stopping (default: 1)
- `--ray_tune_reduction_factor`: ASHA reduction factor (default: 2)
- `--ray_tune_cpu_per_trial`: CPUs per trial (default: 2)
- `--ray_tune_gpu_per_trial`: GPUs per trial (default: 1.0)
- `--ray_tune_local_dir`: Results directory (default: "./ray_results")
- `--train_final_model`: Train final model with best hyperparameters

### 5. Output Files
- **Best hyperparameters**: Saved to `{output_dir}/best_hyperparameters.json`
- **Ray Tune results**: Saved to `{ray_tune_local_dir}/tune_{label}/`
- **Final model** (if `--train_final_model` is used): Saved to standard output directories

## Usage Example

### Basic Ray Tune Run (hyperparameter search only)
```bash
python scripts/codonOnlyClassificationMLP.py \
    --params params.yaml \
    --label codonOnly \
    --extraFeatures path/to/extra_features.csv \
    --data_dir path/to/data \
    --output_dir outputs/ray_tune_results \
    --ray_tune_samples 50 \
    --ray_tune_max_epochs 15
```

### Ray Tune + Final Model Training
```bash
python scripts/codonOnlyClassificationMLP.py \
    --params params.yaml \
    --label codonOnly \
    --extraFeatures path/to/extra_features.csv \
    --data_dir path/to/data \
    --output_dir outputs/ray_tune_results \
    --ray_tune_samples 50 \
    --ray_tune_max_epochs 15 \
    --train_final_model
```

## Workflow
1. **Ray Tune Phase**: Explores hyperparameter space using ASHA + HyperOpt
2. **Best Configuration**: Saves the best hyperparameters found
3. **Final Training** (optional): Trains a full model with the best hyperparameters
4. **Results**: Best hyperparameters and optionally a fully trained model

## Performance Benefits
- **Efficient Search**: HyperOptSearch uses Bayesian optimization
- **Early Stopping**: ASHAScheduler stops poor trials early
- **Resource Management**: Configurable CPU/GPU allocation per trial
- **Reproducible**: Best hyperparameters are saved for reuse

## Your Best Hyperparameters as Starting Point
Your known best values were used to set reasonable search ranges:
- learning_rate: 5e-4 (search range: 1e-5 to 1e-3)
- weight_decay: 0.03 (search range: 0.001 to 0.1)
- dropout_percent: 0.8 (search range: 0.1 to 0.9)

Ray Tune will efficiently explore around and beyond these values to find even better combinations.
