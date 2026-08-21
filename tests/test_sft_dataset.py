import numpy as np
import torch

from genpy.tokenizer import GenPyTokenizer
from genpy.training.sft_dataset import SFTMemmapDataset, SFTRandomBatcher, encode_sft_record


def test_response_only_mask_includes_eos_and_masks_prompt() -> None:
    tokenizer = GenPyTokenizer.load("artifacts/tokenizer/genpy-32k")
    encoding = encode_sft_record({"instruction": "Write code", "input": "", "response": "print(1)"}, tokenizer, 64)
    assert encoding.input_ids[0] == tokenizer.bos_token_id
    assert all(label == -100 for label in encoding.labels[:encoding.prompt_tokens])
    assert encoding.labels[-1] == tokenizer.eos_token_id
    assert sum(label != -100 for label in encoding.labels) == encoding.assistant_tokens


def test_sft_memmap_shift_and_padding(tmp_path) -> None:
    inputs = np.asarray([1, 10, 11, 2, 3, 12, 13, 2], dtype=np.uint16)
    labels = np.asarray([-100, -100, 11, 2, -100, -100, 13, 2], dtype=np.int32)
    inputs.tofile(tmp_path / "inputs.bin"); labels.tofile(tmp_path / "labels.bin")
    np.save(tmp_path / "offsets.npy", np.asarray([0, 4, 8], dtype=np.int64))
    dataset = SFTMemmapDataset(tmp_path / "inputs.bin", tmp_path / "labels.bin", tmp_path / "offsets.npy", 5)
    x, y = dataset[0]
    assert x.tolist() == [1, 10, 11, 0, 0]
    assert y.tolist() == [-100, 11, 2, -100, -100]
    assert dataset[1][1].tolist() == [-100, 13, 2, -100, -100]
    dataset.close()


def test_sft_batcher_seed_is_deterministic(tmp_path) -> None:
    inputs = np.tile(np.asarray([1, 2, 3, 2], dtype=np.uint16), 4); labels = np.tile(np.asarray([-100, 2, 3, -100], dtype=np.int32), 4)
    inputs.tofile(tmp_path / "inputs.bin"); labels.tofile(tmp_path / "labels.bin"); np.save(tmp_path / "offsets.npy", np.arange(0, 17, 4, dtype=np.int64))
    first = SFTMemmapDataset(tmp_path / "inputs.bin", tmp_path / "labels.bin", tmp_path / "offsets.npy", 3)
    second = SFTMemmapDataset(tmp_path / "inputs.bin", tmp_path / "labels.bin", tmp_path / "offsets.npy", 3)
    a = SFTRandomBatcher(first, 2, 42).next_batch(); b = SFTRandomBatcher(second, 2, 42).next_batch()
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    first.close(); second.close()
