"""Fixed warmup calibration and hybrid state acceptance."""
from dataclasses import dataclass
import math
import numpy as np

@dataclass(frozen=True)
class Calibration:
    energy_center: float
    energy_scale: float
    rtm_center: float
    rtm_scale: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in vars(self).values()):
            raise ValueError('Non-finite calibration')
        if self.energy_scale <= 0 or self.rtm_scale <= 0:
            raise ValueError('Calibration scales must be positive')

    @classmethod
    def fit(cls, energies, scores, floor=1e-3):
        e, s = np.asarray(energies, dtype=float), np.asarray(scores, dtype=float)
        if e.ndim != 1 or s.shape != e.shape or not len(e) or not np.isfinite(e).all() or not np.isfinite(s).all():
            raise ValueError('Invalid calibration population')
        if not math.isfinite(floor) or floor <= 0:
            raise ValueError('Invalid calibration floor')
        def stats(x):
            return float(np.median(x)), max(float(np.percentile(x,75)-np.percentile(x,25)),floor)
        return cls(*stats(e), *stats(s))


def objective(energy, rtm, mode, calibration, weight=1.):
    if mode not in ('hybrid', 'rtm', 'vina') or not math.isfinite(weight) or weight < 0:
        raise ValueError('Invalid objective configuration')
    if not math.isfinite(energy) or not math.isfinite(rtm):
        raise ValueError('Non-finite score')
    e = (energy-calibration.energy_center)/calibration.energy_scale
    s = (rtm-calibration.rtm_center)/calibration.rtm_scale
    return -s if mode == 'rtm' else e if mode == 'vina' else e-weight*s


def metropolis(old, new, temperature, uniform):
    if not all(math.isfinite(x) for x in (old,new,temperature,uniform)) or temperature <= 0 or not 0 <= uniform < 1:
        raise ValueError('Invalid Metropolis inputs')
    return new <= old or uniform == 0 or math.log(uniform) < (old-new)/temperature
