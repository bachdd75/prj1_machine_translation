from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from underthesea import word_tokenize # type: ignore

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

# Initialize vocabs
vi_vocab = None
en_vocab = None

def initialize_vocabs():
    global vi_vocab, en_vocab
    # Only import data when needed
    from data import train_df
    
    # Build Vietnamese vocab
    vi_vocab = build_vocab_from_iterator(   
        yield_tokens(train_df['Vietnamese'], vi_tokenizer),
        max_tokens= vocab_size,
        specials = special_tokens
        )
    vi_vocab.set_default_index(vi_vocab["<unk>"])

    # Build English vocab
    en_vocab = build_vocab_from_iterator(
        yield_tokens(train_df['English'], en_tokenizer),
        max_tokens= vocab_size,
        specials = special_tokens
        )
    en_vocab.set_default_index(en_vocab["<unk>"])

    print("Example tokens:", vi_vocab.get_itos()[:20])
    print("Example tokens:", en_vocab.get_itos()[:20])
    
    return vi_vocab, en_vocab

# Initialize vocabs when this module is imported
initialize_vocabs()