from genpy.tokenizer.config import TokenizerConfig
from genpy.tokenizer.tokenizer import GenPyTokenizer
from genpy.tokenizer.trainer import train_from_documents


def make_tokenizer() -> GenPyTokenizer:
    backend = train_from_documents(TokenizerConfig(vocab_size=320, min_frequency=1), [
        "<BOS>\n" + "\n".join(["def greet(name):", "    if name:", "        return 'café 🙂'", "\treturn None"]) + "\n<EOS>",
        "<BOS>\nprint('hello\\nworld')\nx //= 2\nitems[::-1]\n<EOS>",
    ])
    return GenPyTokenizer(backend, expected_vocab_size=320)


def test_exact_roundtrip_for_python_and_unicode() -> None:
    tokenizer = make_tokenizer()
    samples = [
        "hello, world!",
        "def add(a, b):\n    return a + b\n",
        "\tvalue = [1, 2, 3]\n\n# café 🙂",
        "text = \"\"\"multiple\nlines\nhere\"\"\"",
        "__name__ == \"__main__\"\nx //= 2\na ** b",
    ]
    for sample in samples:
        assert tokenizer.decode(tokenizer.encode(sample)) == sample


def test_bos_eos_and_skip_special_tokens() -> None:
    tokenizer = make_tokenizer()
    ids = tokenizer.encode("print(1)", add_bos=True, add_eos=True)
    assert ids[0] == 1 and ids[-1] == 2
    assert tokenizer.decode(ids) == "<BOS>print(1)<EOS>"
    assert tokenizer.decode(ids, skip_special_tokens=True) == "print(1)"
