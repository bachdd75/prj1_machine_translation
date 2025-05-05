from torch.utils.data import DataLoader
from ..data import TranslationDataset, train_df, valid_df, test_df, collate_fn

def create_dataloader(dataset, batch_size=512, shuffle=False, collate_fn=collate_fn):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )

def setup_dataloaders(vi_vocab, en_vocab, vi_tokenizer, en_tokenizer, max_len=10, batch_size=512):
    # Create datasets
    train_dataset = TranslationDataset(
        train_df,
        src_lang="Vietnamese",
        tgt_lang="English",
        src_vocab=vi_vocab,
        tgt_vocab=en_vocab,
        src_tokenizer=vi_tokenizer,
        tgt_tokenizer=en_tokenizer,
        max_len=max_len
    )

    valid_dataset = TranslationDataset(
        valid_df,
        src_lang="Vietnamese",
        tgt_lang="English",
        src_vocab=vi_vocab,
        tgt_vocab=en_vocab,
        src_tokenizer=vi_tokenizer,
        tgt_tokenizer=en_tokenizer,
        max_len=max_len
    )

    test_dataset = TranslationDataset(
        test_df,
        src_lang="Vietnamese",
        tgt_lang="English",
        src_vocab=vi_vocab,
        tgt_vocab=en_vocab,
        src_tokenizer=vi_tokenizer,
        tgt_tokenizer=en_tokenizer,
        max_len=max_len
    )

    # Create dataloaders
    train_dataloader = create_dataloader(train_dataset, batch_size, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = create_dataloader(valid_dataset, batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = create_dataloader(test_dataset, batch_size, shuffle=False, collate_fn=collate_fn)

    return train_dataloader, valid_dataloader, test_dataloader