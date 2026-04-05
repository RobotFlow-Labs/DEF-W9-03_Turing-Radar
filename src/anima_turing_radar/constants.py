"""Constants for the Turing-Radar module."""

from __future__ import annotations

PDW_FEATURES: tuple[str, ...] = (
    "time_of_arrival",
    "centre_frequency",
    "pulse_width",
    "angle_of_arrival",
    "amplitude",
)

PDW_INDEX: dict[str, int] = {
    "toa": 0,
    "time_of_arrival": 0,
    "frequency": 1,
    "centre_frequency": 1,
    "pulse_width": 2,
    "pw": 2,
    "angle_of_arrival": 3,
    "aoa": 3,
    "amplitude": 4,
    "a": 4,
}

NUM_PDW_FEATURES = len(PDW_FEATURES)
