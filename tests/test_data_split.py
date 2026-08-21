from genpy.data.normalize import normalize_example
from genpy.data.schema import example_from_mapping
from genpy.data.split import family_overlap, split_examples


def test_split_is_deterministic_and_family_safe() -> None:
    examples = []
    for family in range(30):
        for variant in range(2):
            examples.append(normalize_example(example_from_mapping({
                "id": f"{family}-{variant}", "family_id": f"family_{family}",
                "instruction": f"Solve problem family {family} variant {variant}",
                "response": f"value = {family + variant}\nprint(value)",
            })))
    first = split_examples(examples, seed=42)
    second = split_examples(examples, seed=42)
    assert {name: [item.id for item in rows] for name, rows in first.items()} == {name: [item.id for item in rows] for name, rows in second.items()}
    assert sum(map(len, first.values())) == len(examples)
    assert len({item.id for rows in first.values() for item in rows}) == len(examples)
    assert not family_overlap(first)


def test_odd_even_family_variants_stay_together() -> None:
    examples = [normalize_example(example_from_mapping({
        "id": str(index), "family_id": "odd_even", "instruction": text,
        "response": "print(1)",
    })) for index, text in enumerate(("Check if 10 is even", "Determine whether an integer is odd or even", "Python odd/even checker"))]
    splits = split_examples(examples, seed=42)
    non_empty = [name for name, rows in splits.items() if rows]
    assert len(non_empty) == 1


def test_odd_even_fallback_groups_obvious_variants() -> None:
    examples = [normalize_example(example_from_mapping({
        "id": str(index), "instruction": text, "response": "print(1)",
    })) for index, text in enumerate(("Check if 10 is even", "Determine whether an integer is odd or even"))]
    assert {example.family_id for example in examples} == {"odd_even"}
