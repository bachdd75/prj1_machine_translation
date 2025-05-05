import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_epoch import train_epoch
from train_evaluate import evaluate
import logging

def train(model: nn.Module, train_dataloader: DataLoader, valid_dataloader: DataLoader,optimizer: torch.optim.Optimizer, 
          scheduler: torch.optim.lr_scheduler.LRScheduler, criterion: nn.Module, device: torch.device, vi_vocab: dict, en_vocab: dict,
          num_epochs: int = 10, clip: float = 1.0, save_path: str = "best_model.pth"):
    best_valid_loss = float('inf')
    history = {'train_loss': [], 'valid_loss': []}

    for epoch in tqdm(range(1, num_epochs + 1)):
        logging.info(f"Starting Epoch {epoch}/{num_epochs}")

        train_loss = train_epoch(
            model,
            train_dataloader,
            criterion,
            optimizer,
            device,
            vi_vocab,
            en_vocab,
            clip,
        )

        valid_loss = evaluate(
            model,
            valid_dataloader,
            criterion,
            device,
            vi_vocab,
            en_vocab,
        )

        scheduler.step(valid_loss)

        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), save_path)
            saved = True
            logging.info(f"Model saved at {save_path} (Validation Loss improved to {best_valid_loss:.4f})")
        else:
            saved = False

        logging.info(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Saved: {saved}")
        print(f"\tTrain Loss: {train_loss:.3f} \tValid Loss: {valid_loss:.3f} \t(Saved)" if saved else "")

    logging.info("Training completed.")
    return history