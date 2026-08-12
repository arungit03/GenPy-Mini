from genpy.data.split import assign_split


def test_split_is_deterministic():
    assert assign_split("abc", 0.5, 42) == assign_split("abc", 0.5, 42)


def test_split_seed_can_change_assignment():
    assignments = {assign_split(f"hash-{index}", 0.5, seed) for index in range(20) for seed in (1, 2)}
    assert assignments == {"train", "validation"}
