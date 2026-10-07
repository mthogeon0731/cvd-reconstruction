"""Protected, portable outputs and explicit SI conversion."""
from pathlib import Path
import csv
import json
import numpy as np


def validate_columns(data,context='table'):
    """Direct DataFrames must already have unique canonical string headers."""
    names=list(data.columns)
    if any(not isinstance(n,str) or not n.strip() for n in names):
        raise ValueError(f'{context}: nonempty string column names required')
    normalized=[n.strip() for n in names]
    duplicate=sorted({n for n in normalized if normalized.count(n)>1})
    if duplicate:raise ValueError(f'{context}: Duplicate column names after whitespace normalization: {duplicate}')
    if names!=normalized:raise ValueError(f'{context}: column names must have no surrounding whitespace')


def read_csv(path,**kwargs):
    """Read UTF-8 comma CSV after checking the raw, whitespace-stripped header.

    Header validation and pandas parsing use the same open stream. No duplicate
    column may be selected implicitly by pandas' automatic suffix mangling.
    """
    import pandas as pd
    forbidden={'names','header','skiprows','sep','delimiter','comment','encoding'} & set(kwargs)
    if forbidden:raise ValueError(f'CSV schema does not support parser overrides: {sorted(forbidden)}')
    p=Path(path)
    with p.open('r',encoding='utf-8-sig',newline='') as f:
        try:raw=next(csv.reader(f,strict=True))
        except StopIteration:raise ValueError(f'{p}: missing CSV header at row 1') from None
        except csv.Error as exc:raise ValueError(f'{p}: invalid CSV header at row 1: {exc}') from exc
        names=[n.strip() for n in raw]
        if not names or any(not n for n in names):
            raise ValueError(f'{p}: empty CSV header at row 1, columns {[i+1 for i,n in enumerate(names) if not n]}')
        repeated={n:[i+1 for i,v in enumerate(names) if v==n] for n in names if names.count(n)>1}
        if repeated:raise ValueError(f'{p}: Duplicate CSV header at row 1 after whitespace normalization: {repeated}')
        f.seek(0)
        kwargs.setdefault('float_precision','round_trip')
        return pd.read_csv(f,header=0,names=names,**kwargs)


def new_dir(path):
    p=Path(path)
    if p.exists() and (not p.is_dir() or any(p.iterdir())): raise FileExistsError(f'Output must be new or empty: {p}')
    p.mkdir(parents=True,exist_ok=True)
    return p


def write_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)


def write_csv(path,df):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='') as f:df.to_csv(f,index=False)


def resolve_input(root,value):
    root=Path(root).resolve();p=Path(str(value))
    if p.is_absolute() or '..' in p.parts: raise ValueError('Use relative paths inside the data root')
    p=(root/p).resolve()
    if not p.is_relative_to(root) or not p.is_file(): raise ValueError(f'Missing/nonportable input: {value}')
    return p


def to_si(value,unit):
    scales={'um':1e-6,'nm':1e-9,'min':60.,'h':3600.,'mPa_s':1e-3,'Pa_s':1.,'m':1.,'s':1.}
    a=np.asarray(value,dtype=float)
    if not np.isfinite(a).all():raise ValueError('Nonfinite unit input')
    if unit=='C':
        if np.any(a<=-273.15):raise ValueError('Temperature below absolute zero')
        return a+273.15
    if unit not in scales:raise ValueError(f'Unknown unit: {unit}')
    return a*scales[unit]
