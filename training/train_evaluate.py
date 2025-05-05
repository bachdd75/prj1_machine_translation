import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from architectures.transformer import generate_square_subsequent_mask
import logging

def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device,
             vi_vocab: dict ,  en_vocab: dict) -> float: 
    model.eval()
    epoch_loss = 0
    batch_count = len(dataloader)

    with torch.no_grad():
        for batch_idx, (src, tgt) in enumerate(dataloader):
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:,:-1]
            tgt_output = tgt[:,1:].contiguous().view(-1) # Flatten for loss computation

            # Generate masks
            tgt_seq_len = tgt_input.size(1)
            tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(device)

            # Create padding masks
            src_key_padding_mask = (src == vi_vocab["<pad>"])
            tgt_key_padding_mask = (tgt_input == en_vocab["<pad>"])

            # Forward pass
            logits = model(
                src= src,
                tgt=tgt_input,
                tgt_mask = tgt_mask,
                src_key_padding_mask = src_key_padding_mask,
                tgt_key_padding_mask = tgt_key_padding_mask,
                memory_key_padding_mask = src_key_padding_mask
            )
            logits = logits.view(-1, logits.size(-1))
            loss = criterion(logits, tgt_output)

            batch_loss = loss.item()
            epoch_loss += batch_loss
            
            # Log batch-level information
            logging.info(f"Validation Batch {batch_idx +1}/{batch_count}, Loss: {batch_loss: .4f}")

        avg_epoch_loss = epoch_loss / batch_count
        logging.info(f"Validation Epoch Completed, Average Loss: {avg_epoch_loss: .4f}")
        return avg_epoch_loss
    
    