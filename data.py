import pandas as pd
from datasets import load_dataset # type: ignore[import]
# from datasets import load_dataset  # Ensure you have the datasets library installed
import torch
from torch.utils.data import Dataset

# Load the dataset
dataset = load_dataset("harouzie/vi_en-translation" , download_mode="force_redownload")

train_df = pd.DataFrame(dataset['train'])
valid_df = pd.DataFrame(dataset['valid'])
test_df = pd.DataFrame(dataset['test'])

print("Train set:")
print(train_df.head())

print("\nValidation set:")
print(valid_df.head())

print("\nTest set:")
print(test_df.head())

class TranslationDataset(Dataset):
    def __init__(self, df, src_lang, tgt_lang, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer, max_len):
        self.df = df
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get source and target sentences from the DataFrame
        src_sentence = self.df.iloc[idx]['vi']  # Assuming 'vi' column for Vietnamese
        tgt_sentence = self.df.iloc[idx]['en']  # Assuming 'en' column for English

        # Tokenize sentences
        src_tokens = self.src_tokenizer(src_sentence)
        tgt_tokens = self.tgt_tokenizer(tgt_sentence)

        # Convert tokens to indices, adding <sos> and <eos>
        src_indices = [self.src_vocab["<sos>"]] + [self.src_vocab.get(token, self.src_vocab["<unk>"]) for token in src_tokens] + [self.src_vocab["<eos>"]]
        tgt_indices = [self.tgt_vocab["<sos>"]] + [self.tgt_vocab.get(token, self.tgt_vocab["<unk>"]) for token in tgt_tokens] + [self.tgt_vocab["<eos>"]]

        # Truncate or pad to max_len
        src_indices = src_indices[:self.max_len]
        tgt_indices = tgt_indices[:self.max_len]

        src_indices += [self.src_vocab["<pad>"]] * (self.max_len - len(src_indices))
        tgt_indices += [self.tgt_vocab["<pad>"]] * (self.max_len - len(tgt_indices))

        return torch.tensor(src_indices, dtype=torch.long), torch.tensor(tgt_indices, dtype=torch.long)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = torch.stack(src_batch)
    tgt_batch = torch.stack(tgt_batch)
    return src_batch, tgt_batch