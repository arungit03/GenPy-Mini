"""Original fixture module used only by pipeline tests."""


def running_total(values: list[int]) -> list[int]:
    """Return each intermediate sum for a list of integers."""
    results: list[int] = []
    current = 0
    for value in values:
        current += value
        results.append(current)
    return results
