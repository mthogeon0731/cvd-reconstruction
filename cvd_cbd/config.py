"""All dimensional defaults are visible in JSON; no calibration is implied."""
from dataclasses import asdict, dataclass
from .json_utils import read_json
import math
from pathlib import Path

FEATURES = ["aspect_ratio", "viscosity_mpa_s", "temperature_c"]
TARGET = "sc_proxy"

@dataclass(frozen=True)
class Config:
    mode: str
    seed: int
    n_samples: int
    width_um: float
    aspect_ratio_range: list[float]
    viscosity_mpa_s_range: list[float]
    temperature_c_range: list[float]
    reference_temperature_c: float
    reference_viscosity_mpa_s: float
    diffusivity_ref_m2_s: float
    surface_rate_ref_m_s: float
    activation_energy_j_mol: float
    bottom_surface_rate_ratio: float
    n_estimators: int
    min_samples_leaf: int
    doe_temperature_levels_c: list[float]
    doe_grid_levels_per_other_factor: int
    doe_runs: int
    fdm_nodes: int
    illustrative_sc_threshold: float

    def __post_init__(self):
        if self.mode != "uncalibrated_reference_demo":
            raise ValueError("This reconstruction supports uncalibrated_reference_demo only.")
        for key in ("width_um", "reference_viscosity_mpa_s", "diffusivity_ref_m2_s"):
            v = getattr(self, key)
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ("surface_rate_ref_m_s", "activation_energy_j_mol", "bottom_surface_rate_ratio"):
            v = getattr(self, key)
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if not math.isfinite(self.reference_temperature_c) or self.reference_temperature_c <= -273.15:
            raise ValueError("Reference temperature must exceed absolute zero")
        for key in ("aspect_ratio_range", "viscosity_mpa_s_range", "temperature_c_range"):
            v = getattr(self, key)
            if len(v) != 2 or not all(math.isfinite(x) for x in v) or v[0] >= v[1]:
                raise ValueError(f"Invalid increasing interval: {key}")
            if v[0] <= (-273.15 if key == "temperature_c_range" else 0):
                raise ValueError(f"Nonphysical interval: {key}")
        for key, minimum in (("seed", 0), ("n_samples", 50), ("n_estimators", 1),
                             ("min_samples_leaf", 1), ("doe_grid_levels_per_other_factor", 3),
                             ("doe_runs", 1), ("fdm_nodes", 3)):
            v = getattr(self, key)
            if isinstance(v, bool) or not isinstance(v, int) or v < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        levels = self.doe_temperature_levels_c
        if not levels or len(set(levels)) != len(levels) or any(
            not math.isfinite(v) or not self.temperature_c_range[0] <= v <= self.temperature_c_range[1]
            for v in levels
        ):
            raise ValueError("DoE temperature levels must be unique and within training range")
        candidates = self.doe_grid_levels_per_other_factor ** 2 * len(levels)
        if self.doe_runs > candidates:
            raise ValueError("Requested more DoE runs than unique candidates")
        if not math.isfinite(self.illustrative_sc_threshold) or not 0 < self.illustrative_sc_threshold < 1:
            raise ValueError("Illustrative threshold must lie strictly between zero and one")

    @property
    def bounds(self):
        return [self.aspect_ratio_range, self.viscosity_mpa_s_range, self.temperature_c_range]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        return cls(**read_json(path, encoding="utf-8"))
