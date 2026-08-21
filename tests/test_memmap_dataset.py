import numpy as np

from genpy.training.dataset import MemmapTokenDataset


def test_memmap_window_has_shifted_x_y(tmp_path) -> None:
    path = tmp_path / "tokens.bin"; np.arange(12, dtype=np.uint16).tofile(path)
    dataset = MemmapTokenDataset(path, 4)
    x, y = dataset[2]
    assert x.tolist() == [2, 3, 4, 5] and y.tolist() == [3, 4, 5, 6]
    assert len(dataset) == 8
    dataset.close()
