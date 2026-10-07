"""Shared contracts for isotropic microscopy calibration in micrometres/pixel."""
import re

import numpy as np


_DECIMAL = re.compile(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z')
_AXIS_FIELDS = {
    'um_per_px_x', 'um_per_px_y', 'x_um_per_px', 'y_um_per_px',
    'um_per_pixel_x', 'um_per_pixel_y', 'pixel_size_um_x', 'pixel_size_um_y',
    'pixel_size_x_um', 'pixel_size_y_um', 'pixel_size_x', 'pixel_size_y',
    'pixel_scale_x', 'pixel_scale_y', 'pixel_width_um', 'pixel_height_um',
}


def validate_pixel_scale(values, context='um_per_px'):
    """Return float ndarray of the same shape; reject missing/nonpositive scales.

    Numeric strings use decimal/scientific notation (optional surrounding space).
    Units are fixed by the field schema: um/px. Boolean values, unit suffixes,
    comma/underscore separators and complex values are not calibration numbers.
    """
    try:
        raw = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{context}: numeric pixel calibration in um/px required') from exc
    if raw.size == 0:
        raise ValueError(f'{context}: pixel calibration is missing')
    for item in raw.flat:
        if isinstance(item, (bool, np.bool_)):
            raise ValueError(f'{context}: boolean is not a pixel calibration')
        if isinstance(item, str) and not _DECIMAL.fullmatch(item.strip()):
            raise ValueError(f'{context}: use a decimal/scientific number in um/px without a unit suffix')
        if isinstance(item, (complex, np.complexfloating)):
            raise ValueError(f'{context}: real numeric pixel calibration required')
    try:
        scales = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{context}: numeric pixel calibration in um/px required') from exc
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError(f'{context}: pixel calibration must be finite and strictly positive (um/px)')
    return scales


def scalar_pixel_scale(value, context='um_per_px'):
    """One isotropic scalar per image/model; axis-specific calibration unsupported."""
    scale = validate_pixel_scale(value, context)
    if scale.ndim != 0:
        raise ValueError(f'{context}: one isotropic scalar in um/px required; axis-specific calibration is unsupported')
    return float(scale)


def reject_axis_pixel_calibration(fields, context='pixel calibration'):
    """Do not silently ignore an unsupported x/y calibration beside the scalar."""
    bad = [str(name) for name in fields if str(name).strip().lower() in _AXIS_FIELDS]
    if bad:
        raise ValueError(f'{context}: axis-specific calibration is unsupported; use one verified isotropic scalar in um/px, fields={bad}')
