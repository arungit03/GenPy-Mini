from genpy.training.logger import TrainingLogger
from genpy.training.metrics import Metrics


def test_metrics_summary_and_jsonl_logging(tmp_path):
    metrics = Metrics()
    metrics.update(2.0, tokens=8)
    summary = metrics.summary(global_step=1, micro_step=2, learning_rate=0.1, gradient_norm=1.0)
    assert summary["training_loss"] == 2.0
    assert summary["tokens_seen"] == 8
    logger = TrainingLogger(tmp_path)
    logger.log(summary)
    assert (tmp_path / "training.jsonl").is_file()
