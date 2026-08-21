from genpy.tokenizer import GenPyTokenizer
from genpy.training.packing import encode_record, format_training_document


def test_bos_eos_boundaries_and_no_literal_special_text() -> None:
    tokenizer = GenPyTokenizer.load("artifacts/tokenizer/genpy-32k")
    record = {"instruction": "Write code", "input": "", "response": "print(1)"}
    assert "<BOS>" not in format_training_document(record)
    ids = encode_record(record, tokenizer)
    assert ids[0] == 1 and ids[-1] == 2 and all(0 <= value < 32000 for value in ids)
