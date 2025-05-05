import torch
import torch.nn as nn
import logging
import os
from architectures.transformer import Transformer, generate_square_subsequent_mask
from training.data_loader import setup_dataloaders
from utils.tokenizer import vi_vocab, en_vocab, vi_tokenizer, en_tokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), '..', 'training.log')),
        logging.StreamHandler()
    ]
)

def translate_sentence(model: nn.Module, sentence: str, vi_tokenizer, vi_vocab: dict, en_vocab: dict, device: torch.device) -> str:
    model.eval()
    with torch.no_grad():
        # Tokenize the input sentence
        tokenized_input = vi_tokenizer(sentence)
        src_indices = [vi_vocab["<sos>"]] + [vi_vocab.get(token, vi_vocab["<unk>"]) for token in tokenized_input] + [vi_vocab["<eos>"]]
        src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(device)

        # Initialize target sequence with <sos>
        tgt_indices = [en_vocab["<sos>"]]
        max_len = 50
        for _ in range(max_len):
            tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long).unsqueeze(0).to(device)

            # Generate masks
            tgt_seq_len = tgt_tensor.size(1)
            tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(device)

            # Create padding masks
            src_key_padding_mask = (src_tensor == vi_vocab["<pad>"]).to(device)
            tgt_key_padding_mask = (tgt_tensor == en_vocab["<pad>"]).to(device)

            # Forward pass
            output = model(
                src=src_tensor,
                tgt=tgt_tensor,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )
            next_token_index = output.argmax(dim=-1)[:, -1].item()
            tgt_indices.append(next_token_index)
            if next_token_index == en_vocab["<eos>"]:
                break

        # Convert indices to tokens and join into a sentence
        output_tokens = [en_vocab.get_itos()[idx] for idx in tgt_indices[1:]]
        output_sentence = " ".join(output_tokens)

    return output_sentence

def test_model(model: nn.Module, test_dataloader, vi_tokenizer, vi_vocab: dict, en_vocab: dict, device: torch.device):
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=en_vocab["<pad>"])

    with torch.no_grad():
        for batch_idx, (src, tgt) in enumerate(test_dataloader):
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:].contiguous().view(-1)

            # Generate masks
            tgt_seq_len = tgt_input.size(1)
            tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(device)

            # Create padding masks
            src_key_padding_mask = (src == vi_vocab["<pad>"])
            tgt_key_padding_mask = (tgt_input == en_vocab["<pad>"])

            # Forward pass
            logits = model(
                src=src,
                tgt=tgt_input,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )
            logits = logits.view(-1, logits.size(-1))
            loss = criterion(logits, tgt_output)
            total_loss += loss.item()

            # Log batch-level information
            logging.info(f"Test Batch {batch_idx + 1}/{len(test_dataloader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(test_dataloader)
    logging.info(f"Test Completed, Average Loss: {avg_loss:.4f}")
    return avg_loss

def train(model, train_dataloader, valid_dataloader, optimizer, scheduler, criterion, device, num_epochs, clip, save_path):
    # Note: This function is referenced but not defined in the original code
    # Including a placeholder to maintain functionality
    history = {'train_loss': [], 'valid_loss': []}
    for epoch in range(num_epochs):
        # Training loop placeholder
        model.train()
        train_loss = 0
        for src, tgt in train_dataloader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])
            loss = criterion(output.view(-1, output.size(-1)), tgt[:, 1:].contiguous().view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)
        
        # Validation loop placeholder
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for src, tgt in valid_dataloader:
                src, tgt = src.to(device), tgt.to(device)
                output = model(src, tgt[:, :-1])
                loss = criterion(output.view(-1, output.size(-1)), tgt[:, 1:].contiguous().view(-1))
                valid_loss += loss.item()
        valid_loss /= len(valid_dataloader)
        
        scheduler.step(valid_loss)
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        
        # Save best model
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        torch.save(model.state_dict(), save_path)
        
    return history

# Model and device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Transformer(
    src_vocab_size=len(vi_vocab),
    tgt_vocab_size=len(en_vocab),
    embed_dim=512,
    num_encoder_layers=6,
    num_decoder_layers=6,
    num_heads=8,
    ff_dim=2048,
    dropout=0.1,
).to(device)

# Load model weights
weights_path = os.path.join(os.path.dirname(__file__), 'weights', 'best_transformer_model.pth')
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

# Setup dataloaders
train_dataloader, valid_dataloader, test_dataloader = setup_dataloaders(
    vi_vocab=vi_vocab,
    en_vocab=en_vocab,
    vi_tokenizer=vi_tokenizer,
    en_tokenizer=en_tokenizer,
    max_len=10,
    batch_size=512
)

# Training parameters
num_epochs = 40
save_path = os.path.join(os.path.dirname(__file__), '..', 'weights', 'best_transformer_model.pth')
optimizer = torch.optim.Adam(model.parameters())  # Placeholder optimizer
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)  # Placeholder scheduler
criterion = nn.CrossEntropyLoss(ignore_index=en_vocab["<pad>"])  # Placeholder criterion

# Execute testing and training
logging.info("Starting test process...")
test_loss = test_model(model, test_dataloader, vi_tokenizer, vi_vocab, en_vocab, device)

sample_input = "Tôi ăn 1 b"
logging.info(f"Sample Input: {sample_input}")
translated_sentence = translate_sentence(model, sample_input, vi_tokenizer, vi_vocab, en_vocab, device)
logging.info(f"Translated Sentence: {translated_sentence}")
print("Translated Sentence:", translated_sentence)

logging.info("Starting training process...")
history = train(
    model,
    train_dataloader,
    valid_dataloader,
    optimizer,
    scheduler,
    criterion,
    device,
    num_epochs=num_epochs,
    clip=1.0,
    save_path=save_path,
)

# Log final results
logging.info("Training finished. Final results:")
logging.info(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
logging.info(f"Final Valid Loss: {history['valid_loss'][-1]:.4f}")