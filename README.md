# Machine Translation Project

A Neural Machine Translation from scratch

## Overview

This repository contains a complete machine translation system built using modern deep learning techniques. The project implements a Transformer-based architecture for translating text between languages, with a flexible and modular design that supports various language pairs.

## Project Structure

```
├── .gitignore
├── data.py
├── requirements.txt
├── test.py
├── architectures/
│   ├── __init__.py
│   ├── decoder.py
│   ├── encoder.py
│   ├── transformer.py
├── training/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── set_up.py
│   ├── train_epoch.py
│   ├── train_evaluate.py
│   ├── train.py
├── utils/
│   ├── __init__.py
│   ├── embedding_input.py
│   ├── positional_encoding.py
│   ├── tokenizer.py
```

### Key Components

#### 🏗️ Architectures
- **encoder.py**: Implements the Transformer encoder with self-attention mechanisms
- **decoder.py**: Implements the Transformer decoder with self-attention and encoder-decoder attention
- **transformer.py**: Combines encoder and decoder into a complete Transformer model

#### 🔄 Training
- **data_loader.py**: Handles loading and batching of parallel text data
- **set_up.py**: Configures training environment and hyperparameters
- **train_epoch.py**: Manages training logic for a single epoch
- **train_evaluate.py**: Evaluates model performance using translation metrics
- **train.py**: Main script for training the complete model

#### 🛠️ Utilities
- **embedding_input.py**: Processes text inputs into embeddings
- **positional_encoding.py**: Adds positional information to token embeddings
- **tokenizer.py**: Handles text tokenization for various languages

#### 📊 Other Files
- **data.py**: Preprocessing scripts for dataset preparation
- **test.py**: Unit tests for validating functionality
- **requirements.txt**: Python dependencies

## Installation

### Prerequisites
- Python 3.7+
- PyTorch 1.7+
- CUDA (optional, for GPU acceleration)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/bachhd75/prj_machine_translation.git
   cd prj_machine_translation
   ```

2. **Create a virtual environment**
   ```bash
   # For Windows
   python -m venv machine_translation
   source machine_translation/Scripts/activate
   
   # For macOS/Linux
   python -m venv machine_translation
   source machine_translation/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the translation model with default settings:

```bash
python training/train.py
```

Custom configuration can be adjusted in `training/set_up.py`.

### Evaluation

Evaluate model performance with:

```bash
python training/train_evaluate.py
```

### Testing

Run unit tests to verify functionality:

```bash
python test.py
```

## Dependencies

Main libraries used in this project:

- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computing
- **Transformers**: Leveraging pre-trained models and components
- **SacreBLEU**: Evaluation of translation quality
- **Underthesea**: Vietnamese language processing (if applicable)
