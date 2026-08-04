# Packed Dataset Format

Token files are headerless little-endian unsigned 16-bit arrays. Loss masks are headerless
unsigned 8-bit arrays containing only zero or one. Production shapes are `[samples, 1025]` and
`[samples, 1024]`; smoke shapes are `[samples, 65]` and `[samples, 64]`.

Each shard has token, loss-mask, and JSON metadata files. Metadata records family, split, sample
count, dimensions, dtypes, tokenizer fingerprint, config hash, source checksums, loss policy,
dataset version, counts, output checksums, and resume identity. It contains no raw text or local
absolute source paths.

Validation checks every file, SHA-256 checksum, byte size, token range, mask value, padding
target, fingerprint, config hash, and source family/split isolation. Corruption fails clearly.
No pickle format is used for packed data.

`PackedDataset` opens NumPy memory maps lazily, maps global to shard-local indices, copies only
one requested row, converts IDs to `torch.long`, builds shifted labels, replaces inactive labels
with `-100`, and creates a non-padding attention mask. Mapping handles are dropped during worker
serialization so Windows spawn workers reopen files safely.

The sampler uses a seeded PyTorch permutation per epoch, deterministic rank slicing, and an exact
epoch/cursor state. It does not rely on process hash values. Validation and test datasets remain
separate from training.
