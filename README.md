# Machine Translation Project

This repository contains a machine translation system built using modern deep learning techniques. The project is structured to include data preprocessing, model architectures, training, and evaluation components.

## Project Structure
├── .gitignore ├── data.py ├── requirements.txt ├── test.py ├── architectures/ │ ├── __init__.py │ ├── decoder.py │ ├── encoder.py │ ├── transformer.py ├── training/ │ ├── __init__.py │ ├── data_loader.py │ ├── set_up.py │ ├── train_epoch.py │ ├── train_evaluate.py │ ├── train.py ├── utils/ │ ├── __init__.py │ ├── embedding_input.py │ ├── positional_encoding.py │ ├── tokenizer.py


### Key Directories and Files

- **`architectures/`**: Contains the model architecture implementations, including encoder, decoder, and transformer models.
- **`training/`**: Includes scripts for data loading, training, evaluation, and setup.
- **`utils/`**: Utility scripts for tokenization, embedding input, and positional encoding.
- **`data.py`**: Script for handling data preprocessing.
- **`test.py`**: Contains unit tests for the project.
- **`requirements.txt`**: Lists the Python dependencies required for the project.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/machine-translation.git
   cd machine-translation

2. Set up a virtual environment
python -m venv machine_translation
source machine_translation/Scripts/activate  # On Windows
source machine_translation/bin/activate      # On macOS/Linux

3. Install dependencies
pip install -r requirements.txt

## Usage
### Training the Model
To train the model, run:
python train.py

Evaluating the Model
To evaluate the model, run: python train.py

Testing
Run the unit tests using:

Features
Transformer Architecture: Implements encoder-decoder models for machine translation.
Custom Tokenizer: Supports tokenization and detokenization for various languages.
Positional Encoding: Adds positional information to input embeddings.
Training Pipeline: Includes data loading, training, and evaluation scripts.
Dependencies
The project requires the following Python libraries:

PyTorch
NumPy
Transformers
SacreBLEU
Underthesea
Install them using:
