# from .distillation_ori import Trainer as ScoreDistillationTrainer
from .distillation import Trainer as ScoreDistillationTrainer
from .wan_frameconcat import Trainer as FrameConcatTrainer
from .ode_regression import Trainer as ODERegressionTrainer
from .distillation_frameconcat import Trainer as ScoreDistillationFrameConcatTrainer
from .distillation_frameconcat_streaming import Trainer as StreamingScoreDistillationFrameConcatTrainer

__all__ = [
    "ScoreDistillationTrainer"
]
