Machine Translation Project
This repository contains a machine translation system built using modern deep learning techniques. The project is structured to include data preprocessing, model architectures, training, and evaluation components. It leverages a Transformer-based architecture for translating text between languages, implemented in Python with PyTorch.
Project Structure
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

Key Directories and Files

architectures/: Contains the model architecture implementations, including encoder, decoder, and transformer models.

__init__.py: Initializes the architectures package.
decoder.py: Implements the decoder component of the Transformer.
encoder.py: Implements the encoder component of the Transformer.
transformer.py: Combines encoder and decoder into the full Transformer model.


training/: Includes scripts for data loading, training, evaluation, and setup.

__init__.py: Initializes the training package.
data_loader.py: Manages loading and batching of training data.
set_up.py: Sets up the environment and configurations for training.
train_epoch.py: Defines the training logic for a single epoch.
train_evaluate.py: Evaluates the model during or after training.
train.py: Main script to train the Transformer model.


utils/: Utility scripts for tokenization, embedding input, and positional encoding.

__init__.py: Initializes the utils package.
embedding_input.py: Processes input data into embeddings for the model.
positional_encoding.py: Implements positional encoding for the Transformer.
tokenizer.py: Tokenizes input text for training and inference.


data.py: Script for handling data preprocessing, such as downloading and preparing datasets.

test.py: Contains unit tests for the project to validate functionality.

requirements.txt: Lists the Python dependencies required for the project.

.gitignore: Specifies files to ignore in version control (e.g., __pycache__).


Installation

Clone the Repository:
git clone https://github.com/bachhd75/prj_machine_translation.git
cd prj_machine_translation


Set Up a Virtual Environment:
python -m venv machine_translation
source machine_translation/Scripts/activate  # On Windows
source machine_translation/bin/activate      # On macOS/Linux


Install Dependencies:
pip install -r requirements.txt



Usage
Training the Model
To train the model, run:
python training/train.py


Adjust hyperparameters (e.g., learning rate, epochs) in training/set_up.py as needed.

Evaluating the Model
To evaluate the model, run:
python training/train_evaluate.py

Testing
Run the unit tests using:
python test.py

Features

Transformer Architecture: Implements encoder-decoder models for machine translation, suitable for various language pairs.
Custom Tokenizer: Supports tokenization and detokenization for various languages via utils/tokenizer.py.
Positional Encoding: Adds positional information to input embeddings using utils/positional_encoding.py.
Training Pipeline: Includes data loading, training, and evaluation scripts for a streamlined workflow.

Dependencies
The project requires the following Python libraries:

PyTorch: For building and training the Transformer model.
NumPy: For numerical computations.
Transformers: For leveraging pre-trained models or components.
SacreBLEU: For evaluating translation quality.
Underthesea: For Vietnamese language processing (if applicable).

Install them using:
pip install -r requirements.txt

Notes

This project can be run on Google Colab for free GPU resources. Use the following commands in Colab:!git clone https://github.com/bachhd75/prj_machine_translation.git
%cd /content/prj_machine_translation
!pip install -r requirements.txt
!python training/train.py


Ensure your dataset is properly formatted (e.g., parallel corpora for source and target languages).
If you encounter errors, verify file paths and ensure all dependencies are installed correctly.

Contributing
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.
License
This project is licensed under the MIT License (unless specified otherwise).
