"""Strict configuration for the calibrated workflow; separate from legacy demo."""
from dataclasses import dataclass, asdict, fields
from .json_utils import read_json
from pathlib import Path
import math

@dataclass(frozen=True)
class StudyConfig:
    origin: str = 'SYNTHETIC'
    seed: int = 0
    w_um: float = 30.0
    h_um: float = 30.0
    faces: int = 4
    wall_materials: tuple = ('pdms', 'pdms', 'glass', 'pdms')
    end_material: str = 'pdms'
    end_rate_ratio: float = 1.0
    C0_mol_m3: float = 50.0
    Vm_m3_mol: float = 1.45e-5
    D_ref_m2_s: float = 6e-10
    D_ref_T_C: float = 25.0
    eta_ref_Pa_s: float = 0.00089
    D_ref_source: str = 'SYNTHETIC assumption; not a measured bath value'
    gly_basis: str = 'mass_pct'
    ar_ladder: tuple = (9, 13, 18, 25, 35, 49)
    top_offset_um: float = 50.0
    window_half_um: float = 16.0
    tile_px: int = 32
    wall_margin_px: int = 4
    invalid_pixel_fraction_max: float = 0.01
    t_min: float = 1e-4
    t_max: float = 2.0
    sc_target: float = 0.90
    inner_splits: int = 3
    mc_repeats: int = 4000
    mc_noise_sd: float = 0.05
    mc_noise_kind: str = 'additive'
    map_observable: str = 'endpoints'
    map_ar_limit: float = 1000000.0
    induction_fraction: float = 0.05
    nonlinear_alpha: float = 0.05
    allow_flagged_growth: bool = False
    min_top_nm: float = 1.0

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if f.type in (float, int):
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                    raise ValueError(f'{f.name}: finite numeric value required')
                if f.type is int and not isinstance(v, int):
                    raise ValueError(f'{f.name}: integer required')
            elif f.type is bool and not isinstance(v, bool):
                raise ValueError(f'{f.name}: boolean required')
            elif f.type is str and (not isinstance(v,str) or not v.strip()):
                raise ValueError(f'{f.name}: nonempty string required')
        positive = ['w_um','h_um','C0_mol_m3','Vm_m3_mol','D_ref_m2_s','eta_ref_Pa_s','top_offset_um','t_min','t_max','map_ar_limit','min_top_nm']
        if any(getattr(self,k)<=0 for k in positive): raise ValueError('Positive geometry, chemistry and thresholds required')
        if self.D_ref_T_C <= -273.15 or self.origin not in ('SYNTHETIC','REAL'): raise ValueError('Invalid temperature/origin')
        if self.faces not in (2,4) or self.end_material not in ('pdms','glass','inert'): raise ValueError('Invalid faces/end material')
        if not isinstance(self.wall_materials,(tuple,list)) or len(self.wall_materials)!=4 or any(x not in ('pdms','glass','inert') for x in self.wall_materials): raise ValueError('Four side materials required: left/right/floor/ceiling')
        if not isinstance(self.ar_ladder,(list,tuple)) or not self.ar_ladder or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 for x in self.ar_ladder): raise ValueError('Positive AR list required')
        if len(set(self.ar_ladder))!=len(self.ar_ladder): raise ValueError('Duplicate AR')
        if self.gly_basis not in ('mass_pct','volume_pct'): raise ValueError('gly_basis required')
        if self.seed<0 or self.mc_repeats<1 or self.inner_splits<2 or self.tile_px<4 or self.wall_margin_px<0: raise ValueError('Invalid integer range')
        if self.window_half_um<0 or self.end_rate_ratio<0 or self.mc_noise_sd<0: raise ValueError('Negative range')
        if not 0<=self.invalid_pixel_fraction_max<1 or not 0<self.sc_target<1 or not 0<self.induction_fraction<1 or not 0<self.nonlinear_alpha<1 or self.t_min>=self.t_max: raise ValueError('Invalid fraction/range')
        if self.mc_noise_kind not in ('additive','relative') or self.map_observable not in ('endpoints','windows'): raise ValueError('Unknown observable/noise kind')

    @classmethod
    def load(cls,path):
        data=read_json(path)
        if not isinstance(data,dict): raise ValueError('Config must be an object')
        unknown=set(data)-{f.name for f in fields(cls)}
        if unknown: raise ValueError(f'Unknown config keys: {sorted(unknown)}')
        return cls(**data)

    def save(self,path):
        from .io_utils import write_json
        write_json(path,asdict(self))
