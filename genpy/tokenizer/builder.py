"""Construction of the standalone GenPy Byte-Level BPE tokenizer."""

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from genpy.config import TokenizerConfig


def build_tokenizer(config: TokenizerConfig, vocab_size=None) -> Tokenizer:
    """Build an untrained Byte-Level BPE with stable special-token ordering."""
    effective_vocab_size = vocab_size or config.vocab_size
    model = BPE(unk_token=config.special_tokens.unk_token)
    tokenizer = Tokenizer(model)
    tokenizer.normalizer = NFC()
    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=config.add_prefix_space,
        use_regex=config.use_regex,
    )
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.post_processor = None
    return tokenizer


def build_trainer(config: TokenizerConfig, vocab_size=None, show_progress: bool = True) -> BpeTrainer:
    return BpeTrainer(
        vocab_size=vocab_size or config.vocab_size,
        min_frequency=config.min_frequency,
        max_token_length=config.max_token_length,
        special_tokens=list(config.special_tokens.ordered),
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=show_progress,
    )
