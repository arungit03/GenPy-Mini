"""Original near-duplicate fixture used only by pipeline tests."""


def running_total(values: list[int]) -> list[int]:
    """Return intermediate sums for a sequence of integers."""
    results: list[int] = []
    current = 0
    for value in values:
        current += value
        results.append(current)
    return results
