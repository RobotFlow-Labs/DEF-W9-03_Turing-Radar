"""ANIMA Turing-Radar module."""

from .config import AppConfig, load_config
from .evaluate import aggregate_scores, evaluate_dataset, evaluate_labels
from .io import PulseTrainRecord, iter_pulse_train_files, load_pulse_train_h5, save_predictions_json
from .model import Deinterleaver, EmbeddedClusterDeinterleaver, SklearnClusterDeinterleaver, build_deinterleaver
from .preprocess import PDWPreprocessor

__all__ = (
    "AppConfig",
    "Deinterleaver",
    "EmbeddedClusterDeinterleaver",
    "PDWPreprocessor",
    "PulseTrainRecord",
    "SklearnClusterDeinterleaver",
    "aggregate_scores",
    "build_deinterleaver",
    "evaluate_dataset",
    "evaluate_labels",
    "iter_pulse_train_files",
    "load_config",
    "load_pulse_train_h5",
    "save_predictions_json",
)
