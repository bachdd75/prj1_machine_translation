# machine_translation/__init__.py
# Expose the training, architectures, and utils subpackages
from . import training
from . import architectures
from . import utils
from .data import TranslationDataset, train_df, valid_df, test_df, collate_fn