from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_checkpoint_one_structure_exists() -> None:
    required = [
        "README.md", "pyproject.toml", "requirements.txt", ".gitignore",
        "configs/model_200m.yaml", "genpy/__init__.py", "genpy/config.py",
        "genpy/constants.py", "genpy/utils/device.py", "genpy/utils/logging.py",
        "genpy/utils/reproducibility.py", "genpy/utils/paths.py",
        "scripts/check_environment.py", "docs/ARCHITECTURE.md",
        "data/raw", "data/processed", "data/instruction", "checkpoints", "logs", "reports",
    ]
    for path in required:
        assert (ROOT / path).exists(), path
