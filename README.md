# TransformerTF

A PyTorch Lightning framework for high-frequency physics signal modeling with transformer architectures, designed for sensor data analysis at CERN.

## Overview

TransformerTF provides a comprehensive toolkit for high-frequency physics signal modeling using state-of-the-art transformer models. Originally developed for magnetic field transfer functions in particle accelerators, the framework supports various sensor data analysis tasks with configurable data pipelines and multiple model architectures.

### Key Features

- **Multiple Model Architectures**: Temporal Fusion Transformer (TFT), LSTM variants, TSMixer, standard Transformers, and physics-informed models
- **Physics-Informed Models**: Specialized architectures for physics applications (BWLSTM for Bouc-Wen modeling, PETE)
- **Flexible Data Pipeline**: Configurable preprocessing, windowing, and transformation systems
- **Lightning Integration**: Built on PyTorch Lightning for scalable training with automatic logging and checkpointing
- **Configuration-Driven**: YAML-based configuration system for reproducible experiments

## Installation

### Prerequisites

- Python 3.11 or 3.12
- PyTorch 2.2+

### Basic Installation

```bash
pip install transformertf
```

### Development Installation

```bash
git clone https://gitlab.cern.ch/dsb/hysteresis/transformertf.git
cd transformertf
pip install -e ".[dev,test]"
```

### Optional Dependencies

- `dev`: Development tools (pre-commit, ruff, mypy)
- `test`: Testing framework (pytest, pytest-cov)
- `doc`: Documentation tools (sphinx, myst-parser)

## Available Models

| Model | Description | Use Case |
|-------|-------------|----------|
| **TFT** | Temporal Fusion Transformer | Multi-horizon forecasting with attention |
| **LSTM** | Standard LSTM | Basic sequence modeling |
| **BWLSTM** | Bouc-Wen LSTM | Physics-informed hysteresis modeling |
| **TSMixer** | MLP-based time series model | Efficient mixing of time and feature dimensions |
| **Transformer** | Standard transformer | Sequence-to-sequence prediction |
| **PETE** | Physics-Enhanced Transformer Encoder | Physics-constrained forecasting |

## Quick Start

### 1. Training with Sample Configuration

```bash
# Train a Temporal Fusion Transformer
transformertf fit --config sample_configs/tft_config.yml

# Train with verbose output
transformertf fit -vv --config sample_configs/tft_config.yml

# Custom experiment name
transformertf fit --experiment-name my_experiment --config sample_configs/tft_config.yml
```

### 2. Configuration File Structure

```yaml
# Basic TFT configuration
model:
  class_path: transformertf.models.temporal_fusion_transformer.TemporalFusionTransformer
  init_args:
    d_model: 32
    num_heads: 4
    dropout: 0.2

data:
  class_path: transformertf.data.EncoderDecoderDataModule
  init_args:
    train_df_paths: ["path/to/sensor_data.parquet"]
    val_df_paths: ["path/to/sensor_val.parquet"]
    target_covariate: "magnetic_field"
    ctxt_seq_len: 1000  # 100 seconds at 10 Hz
    tgt_seq_len: 100    # 10 seconds prediction
    batch_size: 16

trainer:
  max_epochs: 100
  gradient_clip_val: 1.0
```

### 3. Prediction

```bash
# Run inference on trained model
transformertf predict --config config.yml --ckpt_path checkpoints/best.ckpt
```

### 4. Programmatic Usage

```python
import transformertf
from transformertf.data import EncoderDecoderDataModule
from transformertf.models.temporal_fusion_transformer import TemporalFusionTransformer

# Initialize data module
data_module = EncoderDecoderDataModule(
    train_df_paths=["train.parquet"],
    val_df_paths=["val.parquet"],
    target_covariate="target",
    ctxt_seq_len=200,
    tgt_seq_len=100
)

# Initialize model
model = TemporalFusionTransformer(
    d_model=32,
    num_heads=4,
    seq_len=200,
    output_dim=1
)

# Train with Lightning
import lightning as L
trainer = L.Trainer(max_epochs=100)
trainer.fit(model, data_module)
```

## Sample Configurations

The `sample_configs/` directory contains example configurations for different models:

- `tft_config.yml` - Temporal Fusion Transformer
- `lstm_config.yml` - Standard LSTM
- `tsmixer_config.yml` - TSMixer model
- `transformer_config.yml` - Standard transformer
- `phylstm_config.yml` - Physics-informed LSTM

## Development

### Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type checking
mypy transformertf/

# Run tests
pytest
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## Documentation

- **Full Documentation**: https://acc-py.web.cern.ch/gitlab/dsb/hysteresis/transformertf/docs/stable/
- **Developer Guide**: See [CLAUDE.md](CLAUDE.md) for detailed development information
- **Repository**: https://gitlab.cern.ch/dsb/hysteresis/transformertf

## Project Structure

```
transformertf/
├── transformertf/           # Main package
│   ├── models/             # Model implementations
│   ├── data/               # Data loading and preprocessing
│   ├── nn/                 # Neural network components
│   └── utils/              # Utilities
├── sample_configs/         # Example configurations
├── tests/                  # Test suite
└── docs/                   # Documentation
```

## License

Other/Proprietary License

## Author

Anton Lu (anton.lu@cern.ch)  
CERN - European Organization for Nuclear Research
