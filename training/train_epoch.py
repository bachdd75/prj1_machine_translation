import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..utils import vi_vocab, en_vocab
from ..architectures.transformer import generate_square_subsequent_mask
import logging

def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                optimizer: torch.optim.Optimizer, device: torch.device, clip: float = 1.0)-> float:
    model.train()
    epoch_loss = 0
    batch_count = len(dataloader)

    for batch_idx, (src, tgt) in enumerate(dataloader):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:,:-1]
        tgt_output = tgt[:, 1:].contiguous().view(-1)  # Flatten for loss computation

        # Geneate masks
        tgt_seq_len = tgt_input.size(1)
        tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(device)

        # Create padding masks
        src_key_padding_mask = (src == vi_vocab["<pad>"])
        tgt_key_padding_mask = (tgt_input == en_vocab["<pad>"])

        # Forward pass
        optimizer.zero_grad()
        logits = model(
            src= src,
            tgt = tgt_input,
            tgt_mask = tgt_mask,
            src_key_padding_mask = src_key_padding_mask,
            tgt_key_padding_mask = tgt_key_padding_mask,
            memory_key_padding_mask = src_key_padding_mask
        )

        # Reshape logits to (batch_size * tgt_seq_len, tgt_vocab_size)
        logits =  logits.view(-1, logits.size(-1))
        loss = criterion(logits, tgt_output)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        batch_loss = loss.item()
        epoch_loss += batch_loss

        # Log batch-level informations
        logging.info(f"Batch {batch_idx + 1}/{batch_count}, Loss: {batch_loss:.4f}")

    avg_epoch_loss = epoch_loss / len(dataloader)
    logging.info(f"Training Epoch Completed, Average Loss: {avg_epoch_loss:.4f}")
    return avg_epoch_loss
