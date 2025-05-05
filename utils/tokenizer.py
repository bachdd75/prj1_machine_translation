from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from underthesea import word_tokenize # type: ignore
from data import train_df


# Define tokenizers
def vi_tokenizer(text):
    return word_tokenize(text, format = 'text').split()

en_tokenizer = get_tokenizer("basic_english")

# Break down tokens one by one
def yield_tokens(sentences, tokenizer):
    for sentence in sentences:
        yield tokenizer(sentence)

# Special tokens
special_tokens = ["<pad>", "<unk>", "<sos>", "<eos>"]
vocab_size = 10000

# Build Vietnamese vocb
vi_vocab = build_vocab_from_iterator(   
    yield_tokens(train_df['Vietnamese'], vi_tokenizer),
    max_tokens= vocab_size,
    specials = special_tokens
    )
vi_vocab.set_default_index(vi_vocab["<unk>"])

# Build English vocab
en_vocab = build_vocab_from_iterator(
    yield_tokens(train_df['English'], vi_tokenizer),
    max_tokens= vocab_size,
    specials = special_tokens
    )
en_vocab.set_default_index(vi_vocab["<unk>"])

# Combine vocabularies
def combine_vocabs(vocab1, vocab2):
    # Union of tokens
    combined_tokens = set(vocab1.get_itos()) | set(vocab2.get_itos())
    return build_vocab_from_iterator([combined_tokens], specials = special_tokens)

combined_vocab = combine_vocabs(vi_vocab, en_vocab)
# Store mapping from words to indices and indices to words.
combined_vocab.set_default_index(combined_vocab["<unk>"])

print("Combined Vocabulary Size:", len(combined_vocab))
print("Example tokens:", vi_vocab.get_itos()[:20])
print("Example tokens:", en_vocab.get_itos()[:20])