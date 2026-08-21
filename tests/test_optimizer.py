from training_helpers import tiny_engine


def test_adamw_coverage_and_no_tied_duplicate(tmp_path) -> None:
    engine = tiny_engine(tmp_path)
    audit = engine.optimizer_audit
    assert audit["duplicate_parameters"] == 0 and audit["missing_parameters"] == 0
    assert engine.optimizer.defaults["betas"] == (0.9, 0.95)
    assert engine.optimizer.defaults["eps"] == 1e-8
    engine.train_dataset.close(); engine.validation_dataset.close()
