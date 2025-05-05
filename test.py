import torch
import torch.nn as nn
from architectures.transformer import Transformer, generate_square_subsequent_mask
from training.data_loader import setup_dataloaders
from utils.tokenizer import vi_vocab, en_vocab
from utils.tokenizer import vi_tokenizer, en_tokenizer
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), '..', 'training.log')),
        logging.StreamHandler()  # Also print to console
    ]
)

def translate_sentence(model: nn.Module, sentence: str, vi_tokenizer, vi_vocab: dict, en_vocab: dict, device: torch.device) -> str:
    model.eval()
    with torch.no_grad():
        # Tokenize the input sentence
        tokenized_input = vi_tokenizer(sentence)
        src_indices = [vi_vocab.get(token, vi_vocab["<unk>"]) for token in tokenized_input]  # Handle unknown tokens
        src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(device)

        # Initialize target sequence with <sos>
        tgt_indices = [en_vocab["<sos>"]]
        for _ in range(50):  # Maximum sequence length
            tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long).unsqueeze(0).to(device)
            output = model(src_tensor, tgt_tensor, tgt_mask=None)
            next_token_index = output.argmax(dim=-1)[:, -1].item()
            tgt_indices.append(next_token_index)
            if next_token_index == en_vocab["<eos>"]:
                break

        # Convert indices to tokens and join into a sentence
        output_tokens = [en_vocab.get_itos()[idx] for idx in tgt_indices[1:-1]]
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

# Load the trained model
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load("best_transformer_model.pth"))
model.to(device)
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

# Test the model on the test dataset
logging.info("Starting test process...")
test_loss = test_model(model, test_dataloader, vi_tokenizer, vi_vocab, en_vocab, device)

# Test a sample input
sample_input = "Tôi ăn 1 b"
logging.info(f"Sample Input: {sample_input}")
translated_sentence = translate_sentence(model, sample_input, vi_tokenizer, vi_vocab, en_vocab, device)
logging.info(f"Translated Sentence: {translated_sentence}")
print("Translated Sentence:", translated_sentence)