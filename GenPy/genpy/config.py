"""Dataclass-based configuration loading and validation for Step 1."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Union

import yaml


PathLike = Union[str, Path]


def _read_section(path: PathLike, section_name: str) -> Mapping[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or section_name not in raw:
        raise ValueError(f"Configuration must contain a '{section_name}' section")
    section = raw[section_name]
    if not isinstance(section, dict):
        raise ValueError(f"The '{section_name}' section must be a mapping")
    return section


def _required(section: Mapping[str, Any], name: str, section_name: str) -> Any:
    if name not in section:
        raise ValueError(f"Missing required field '{section_name}.{name}'")
    return section[name]


def _positive(section: Mapping[str, Any], name: str, section_name: str) -> Any:
    value = _required(section, name, section_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"'{section_name}.{name}' must be a positive number")
    return value


@dataclass(frozen=True)
class ModelConfig:
    name: str
    vocab_size: int
    max_seq_len: int
    hidden_size: int
    num_layers: int
    num_heads: int
    head_dim: int
    intermediate_size: int
    norm_eps: float
    rope_theta: float
    tie_embeddings: bool

    @classmethod
    def from_mapping(cls, section: Mapping[str, Any]) -> "ModelConfig":
        required_names = (
            "name", "vocab_size", "max_seq_len", "hidden_size", "num_layers",
            "num_heads", "head_dim", "intermediate_size", "norm_eps", "rope_theta",
            "tie_embeddings",
        )
        for name in required_names:
            _required(section, name, "model")
        positive_names = (
            "vocab_size", "max_seq_len", "hidden_size", "num_layers", "num_heads",
            "head_dim", "intermediate_size", "norm_eps", "rope_theta",
        )
        for name in positive_names:
            _positive(section, name, "model")
        integer_names = (
            "vocab_size", "max_seq_len", "hidden_size", "num_layers", "num_heads",
            "head_dim", "intermediate_size",
        )
        for name in integer_names:
            if not isinstance(section[name], int) or isinstance(section[name], bool):
                raise ValueError(f"'model.{name}' must be an integer")
        if section["hidden_size"] % section["num_heads"] != 0:
            raise ValueError("'model.hidden_size' must be divisible by 'model.num_heads'")
        expected_head_dim = section["hidden_size"] // section["num_heads"]
        if section["head_dim"] != expected_head_dim:
            raise ValueError(
                "'model.head_dim' must equal 'model.hidden_size // model.num_heads'"
            )
        if not isinstance(section["name"], str) or not section["name"].strip():
            raise ValueError("'model.name' must be a non-empty string")
        if not isinstance(section["tie_embeddings"], bool):
            raise ValueError("'model.tie_embeddings' must be a boolean")
        return cls(**{name: section[name] for name in required_names})


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    min_learning_rate: float
    weight_decay: float
    warmup_ratio: float
    grad_clip: float
    precision: str
    log_interval: int
    eval_interval: int
    save_interval: int
    checkpoint_dir: str
    log_dir: str

    @classmethod
    def from_mapping(cls, section: Mapping[str, Any]) -> "TrainingConfig":
        required_names = tuple(cls.__dataclass_fields__)
        for name in required_names:
            _required(section, name, "training")
        for name in (
            "micro_batch_size", "gradient_accumulation_steps", "log_interval",
            "eval_interval", "save_interval",
        ):
            value = section[name]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"'training.{name}' must be a positive integer")
        if not isinstance(section["seed"], int) or isinstance(section["seed"], bool):
            raise ValueError("'training.seed' must be an integer")
        for name in ("learning_rate", "min_learning_rate", "weight_decay", "grad_clip"):
            value = section[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"'training.{name}' must be non-negative")
        warmup_ratio = section["warmup_ratio"]
        if isinstance(warmup_ratio, bool) or not isinstance(warmup_ratio, (int, float)) or not 0 <= warmup_ratio <= 1:
            raise ValueError("'training.warmup_ratio' must be between 0 and 1")
        if section["min_learning_rate"] > section["learning_rate"]:
            raise ValueError("'training.min_learning_rate' cannot exceed learning_rate")
        if not isinstance(section["precision"], str) or not section["precision"].strip():
            raise ValueError("'training.precision' must be a non-empty string")
        for name in ("checkpoint_dir", "log_dir"):
            if not isinstance(section[name], str) or not section[name].strip():
                raise ValueError(f"'training.{name}' must be a non-empty string")
        return cls(**{name: section[name] for name in required_names})


def load_model_config(path: PathLike) -> ModelConfig:
    """Load and validate a model YAML configuration."""
    return ModelConfig.from_mapping(_read_section(path, "model"))


def load_training_config(path: PathLike) -> TrainingConfig:
    """Load and validate a training YAML configuration."""
    return TrainingConfig.from_mapping(_read_section(path, "training"))


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    config: str
    split: str
    streaming: bool
    text_field: str


@dataclass(frozen=True)
class ProcessingConfig:
    seed: int
    min_chars: int
    max_chars: int
    normalize_unicode: bool
    normalize_line_endings: bool
    remove_control_characters: bool
    normalize_whitespace: bool
    exact_deduplication: bool


@dataclass(frozen=True)
class DataSplitConfig:
    validation_fraction: float
    seed: int


@dataclass(frozen=True)
class OutputConfig:
    format: str
    shard_max_documents: int
    processed_dir: str
    manifest_dir: str


@dataclass(frozen=True)
class ResumeConfig:
    enabled: bool


@dataclass(frozen=True)
class MetadataConfig:
    preserve_source_metadata: bool


@dataclass(frozen=True)
class DataPipelineConfig:
    dataset: DatasetConfig
    processing: ProcessingConfig
    split: DataSplitConfig
    output: OutputConfig
    resume: ResumeConfig
    metadata: MetadataConfig

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DataPipelineConfig":
        required_sections = ("dataset", "processing", "split", "output", "resume", "metadata")
        for section_name in required_sections:
            if section_name not in raw or not isinstance(raw[section_name], dict):
                raise ValueError(f"Configuration must contain mapping section '{section_name}'")

        dataset = raw["dataset"]
        for name in ("name", "config", "split", "text_field"):
            _required(dataset, name, "dataset")
            if not isinstance(dataset[name], str) or not dataset[name].strip():
                raise ValueError(f"'dataset.{name}' must be a non-empty string")
        if not isinstance(dataset.get("streaming"), bool):
            raise ValueError("'dataset.streaming' must be a boolean")

        processing = raw["processing"]
        for name in ("seed", "min_chars", "max_chars"):
            value = _required(processing, name, "processing")
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"'processing.{name}' must be a non-negative integer")
        if processing["min_chars"] <= 0:
            raise ValueError("'processing.min_chars' must be greater than zero")
        if processing["max_chars"] <= processing["min_chars"]:
            raise ValueError("'processing.max_chars' must be greater than min_chars")
        for name in (
            "normalize_unicode", "normalize_line_endings", "remove_control_characters",
            "normalize_whitespace", "exact_deduplication",
        ):
            if not isinstance(_required(processing, name, "processing"), bool):
                raise ValueError(f"'processing.{name}' must be a boolean")

        split = raw["split"]
        fraction = _required(split, "validation_fraction", "split")
        seed = _required(split, "seed", "split")
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
            raise ValueError("'split.validation_fraction' must be between 0 and 1")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("'split.seed' must be a non-negative integer")

        output = raw["output"]
        for name in ("format", "processed_dir", "manifest_dir"):
            value = _required(output, name, "output")
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"'output.{name}' must be a non-empty string")
        if output["format"] != "jsonl.gz":
            raise ValueError("'output.format' must be 'jsonl.gz' for Step 2")
        shard_size = _required(output, "shard_max_documents", "output")
        if not isinstance(shard_size, int) or isinstance(shard_size, bool) or shard_size <= 0:
            raise ValueError("'output.shard_max_documents' must be a positive integer")

        resume = raw["resume"]
        metadata = raw["metadata"]
        if not isinstance(resume.get("enabled"), bool):
            raise ValueError("'resume.enabled' must be a boolean")
        if not isinstance(metadata.get("preserve_source_metadata"), bool):
            raise ValueError("'metadata.preserve_source_metadata' must be a boolean")

        return cls(
            dataset=DatasetConfig(**{name: dataset[name] for name in DatasetConfig.__dataclass_fields__}),
            processing=ProcessingConfig(**{name: processing[name] for name in ProcessingConfig.__dataclass_fields__}),
            split=DataSplitConfig(validation_fraction=fraction, seed=seed),
            output=OutputConfig(**{name: output[name] for name in OutputConfig.__dataclass_fields__}),
            resume=ResumeConfig(enabled=resume["enabled"]),
            metadata=MetadataConfig(preserve_source_metadata=metadata["preserve_source_metadata"]),
        )


def load_data_config(path: PathLike) -> DataPipelineConfig:
    """Load and validate the Step 2 dataset pipeline configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Dataset configuration must be a YAML mapping")
    return DataPipelineConfig.from_mapping(raw)


@dataclass(frozen=True)
class TokenizerSpecialTokens:
    pad_token: str
    bos_token: str
    eos_token: str
    unk_token: str

    @property
    def ordered(self) -> tuple[str, str, str, str]:
        return (self.pad_token, self.bos_token, self.eos_token, self.unk_token)


@dataclass(frozen=True)
class TokenizerConfig:
    name: str
    algorithm: str
    vocab_size: int
    min_frequency: int
    max_token_length: int
    normalizer: str
    add_prefix_space: bool
    use_regex: bool
    special_tokens: TokenizerSpecialTokens


@dataclass(frozen=True)
class TokenizerTrainingDataConfig:
    input_dir: str
    train_pattern: str
    validation_pattern: str
    text_field: str


@dataclass(frozen=True)
class TokenizerOutputConfig:
    output_dir: str
    tokenizer_file: str
    config_file: str
    manifest_file: str
    evaluation_file: str


@dataclass(frozen=True)
class TokenizerTrainingConfig:
    show_progress: bool
    production_target_bytes: int
    production_minimum_bytes: int


@dataclass(frozen=True)
class TokenizerPipelineConfig:
    tokenizer: TokenizerConfig
    training_data: TokenizerTrainingDataConfig
    output: TokenizerOutputConfig
    training: TokenizerTrainingConfig


def _non_empty_string(section: Mapping[str, Any], name: str, section_name: str) -> str:
    value = _required(section, name, section_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{section_name}.{name}' must be a non-empty string")
    return value


def _positive_integer(section: Mapping[str, Any], name: str, section_name: str, minimum: int = 1) -> int:
    value = _required(section, name, section_name)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"'{section_name}.{name}' must be an integer >= {minimum}")
    return value


def _tokenizer_from_mapping(section: Mapping[str, Any]) -> TokenizerConfig:
    name = _non_empty_string(section, "name", "tokenizer")
    algorithm = _non_empty_string(section, "algorithm", "tokenizer")
    if algorithm != "byte_level_bpe":
        raise ValueError("'tokenizer.algorithm' must be 'byte_level_bpe'")
    vocab_size = _positive_integer(section, "vocab_size", "tokenizer", 257)
    if vocab_size != 32000:
        raise ValueError("'tokenizer.vocab_size' must be exactly 32000 for GenPy")
    min_frequency = _positive_integer(section, "min_frequency", "tokenizer")
    max_token_length = _positive_integer(section, "max_token_length", "tokenizer")
    normalizer = _non_empty_string(section, "normalizer", "tokenizer")
    if normalizer != "nfc":
        raise ValueError("'tokenizer.normalizer' must be 'nfc'")
    byte_level = section.get("byte_level")
    if not isinstance(byte_level, dict):
        raise ValueError("'tokenizer.byte_level' must be a mapping")
    add_prefix_space = byte_level.get("add_prefix_space")
    use_regex = byte_level.get("use_regex")
    if not isinstance(add_prefix_space, bool) or not isinstance(use_regex, bool):
        raise ValueError("ByteLevel settings must be booleans")
    special = section.get("special_tokens")
    if not isinstance(special, dict):
        raise ValueError("'tokenizer.special_tokens' must be a mapping")
    tokens = TokenizerSpecialTokens(
        pad_token=_non_empty_string(special, "pad_token", "tokenizer.special_tokens"),
        bos_token=_non_empty_string(special, "bos_token", "tokenizer.special_tokens"),
        eos_token=_non_empty_string(special, "eos_token", "tokenizer.special_tokens"),
        unk_token=_non_empty_string(special, "unk_token", "tokenizer.special_tokens"),
    )
    if len(set(tokens.ordered)) != 4:
        raise ValueError("Tokenizer special-token strings must be unique")
    return TokenizerConfig(name, algorithm, vocab_size, min_frequency, max_token_length, normalizer, add_prefix_space, use_regex, tokens)


# Keep construction in a named helper so it is easy to test without a framework.
def _build_tokenizer_config(section: Mapping[str, Any]) -> TokenizerConfig:
    return _tokenizer_from_mapping(section)


def load_tokenizer_config(path: PathLike) -> TokenizerPipelineConfig:
    """Load and validate the Step 3 tokenizer configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Tokenizer configuration must be a YAML mapping")
    for section_name in ("tokenizer", "training_data", "output", "training"):
        if not isinstance(raw.get(section_name), dict):
            raise ValueError(f"Configuration must contain mapping section '{section_name}'")
    tokenizer = _build_tokenizer_config(raw["tokenizer"])
    data = raw["training_data"]
    data_config = TokenizerTrainingDataConfig(
        _non_empty_string(data, "input_dir", "training_data"),
        _non_empty_string(data, "train_pattern", "training_data"),
        _non_empty_string(data, "validation_pattern", "training_data"),
        _non_empty_string(data, "text_field", "training_data"),
    )
    output = raw["output"]
    output_config = TokenizerOutputConfig(*[
        _non_empty_string(output, name, "output")
        for name in TokenizerOutputConfig.__dataclass_fields__
    ])
    training = raw["training"]
    show_progress = _required(training, "show_progress", "training")
    if not isinstance(show_progress, bool):
        raise ValueError("'training.show_progress' must be a boolean")
    target = _positive_integer(training, "production_target_bytes", "training")
    minimum = _positive_integer(training, "production_minimum_bytes", "training")
    if minimum > target:
        raise ValueError("production_minimum_bytes cannot exceed production_target_bytes")
    return TokenizerPipelineConfig(tokenizer, data_config, output_config, TokenizerTrainingConfig(show_progress, target, minimum))


def validate_tokenizer_vocab_contract(model_path: PathLike, tokenizer_config: TokenizerPipelineConfig) -> None:
    """Fail loudly if the model and tokenizer vocabularies disagree."""
    model = load_model_config(model_path)
    if model.vocab_size != tokenizer_config.tokenizer.vocab_size:
        raise ValueError(
            f"Model/tokenizer vocabulary mismatch: model={model.vocab_size}, "
            f"tokenizer={tokenizer_config.tokenizer.vocab_size}"
        )
