import torch
import torch.nn as nn
from .train import train
from .data_loader import setup_dataloaders
from ..architectures.transformer import Transformer
from ..utils.tokenizer import vi_vocab, en_vocab
import logging
import os
from ..utils.tokenizer import vi_tokenizer, en_tokenizer 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), '..', 'training.log')),
        logging.StreamHandler()  # Also print to console
    ]
)

# Setup dataloaders
train_dataloader, valid_dataloader, test_dataloader = setup_dataloaders(
    vi_vocab=vi_vocab,
    en_vocab=en_vocab,
    vi_tokenizer=vi_tokenizer,
    en_tokenizer=en_tokenizer,
    max_len=10,
    batch_size=512
)

# Initialize the Transformer model
model = Transformer(
    src_vocab_size=len(vi_vocab),
    tgt_vocab_size=len(en_vocab),
    embed_dim=512,
    num_encoder_layers=6,
    num_decoder_layers=6,
    num_heads=8,
    ff_dim=2048,
    dropout=0.1,
)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Loss, optimizer, and scheduler
criterion = nn.CrossEntropyLoss(ignore_index=en_vocab["<pad>"])
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Training parameters
num_epochs = 40
save_path = "best_transformer_model.pth"

# Start training
logging.info("Starting training process...")
history = train(
    model,
    train_dataloader,
    valid_dataloader,
    optimizer,
    scheduler,
    criterion,
    device,
    vi_vocab,
    en_vocab
    num_epochs=num_epochs,
    clip=1.0,
    save_path=save_path,
)

# Log final results
logging.info("Training finished. Final results:")
logging.info(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
logging.info(f"Final Valid Loss: {history['valid_loss'][-1]:.4f}")