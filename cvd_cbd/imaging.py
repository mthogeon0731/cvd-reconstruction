"""Conservative optical-microscopy features from explicitly matched regions.

Tiles are computational subdivisions, not independent FE-SEM observations. This
module produces ONE feature row per matched ROI, never one copied label per tile.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import ndimage
from .input_contracts import scalar_pixel_scale,reject_axis_pixel_calibration
from .io_utils import read_csv

TILE_FEATURE_NAMES = (
    "raw_intensity_mean", "raw_intensity_variance",
    "clahe_intensity_mean", "clahe_intensity_variance",
    "edge_gradient_mean", "edge_gradient_p90",
    "scattering_highpass_std", "texture_neighbor_difference",
)
FEATURE_COLUMNS = tuple(
    f"feature_{name}_{stat}" for name in TILE_FEATURE_NAMES for stat in ("mean", "std")
)
MATCH_BASES = {"registered_roi", "validated_region_average"}


@dataclass(frozen=True)
class ImagingConfig:
    tile_size: int = 128
    min_tile_side: int = 16
    min_mask_fraction: float = 0.8
    clahe_grid: tuple[int, int] = (8, 8)
    clahe_clip_limit: float = 2.0
    clahe_bins: int = 256
    highpass_sigma_px: float = 2.0
    expected_width: int | None = 2048
    expected_height: int | None = 1536

    def validate(self) -> None:
        if self.tile_size < 4 or not 2 <= self.min_tile_side <= self.tile_size:
            raise ValueError("Require tile_size >= min_tile_side >= 2 and tile_size >= 4")
        if not 0 < self.min_mask_fraction <= 1:
            raise ValueError("min_mask_fraction must be in (0, 1]")
        if len(self.clahe_grid) != 2 or min(self.clahe_grid) < 1:
            raise ValueError("clahe_grid must contain two positive cell counts")
        if not np.isfinite(self.clahe_clip_limit) or self.clahe_clip_limit < 1:
            raise ValueError("clahe_clip_limit must be finite and >= 1")
        if not 16 <= self.clahe_bins <= 4096:
            raise ValueError("clahe_bins must be between 16 and 4096")
        if not np.isfinite(self.highpass_sigma_px) or self.highpass_sigma_px <= 0:
            raise ValueError("highpass_sigma_px must be finite and positive")


def to_gray_float(image: np.ndarray) -> np.ndarray:
    """Convert a grayscale/RGB image to [0, 1] without per-image normalization.

    Integers use the dtype's positive range. Floating images must already be in
    [0, 1]. Camera black/white calibration is an upstream scientific requirement.
    """
    image = np.asarray(image)
    if image.dtype.kind in "ui":
        if image.dtype.kind == "i" and np.any(image < 0):
            raise ValueError("Signed image contains negative pixels")
        values = image.astype(float) / np.iinfo(image.dtype).max
    elif image.dtype.kind == "f":
        values = image.astype(float)
    else:
        raise ValueError("Images must have integer or floating-point pixel data")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Image intensities must be finite and normalized to [0, 1]")
    if values.ndim == 3 and values.shape[2] in (3, 4):
        if values.shape[2] == 4 and not np.all(values[:, :, 3] == 1):
            raise ValueError("Transparent microscopy images are unsupported")
        values = values[:, :, :3] @ np.array([0.2126, 0.7152, 0.0722])
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("Expected a nonempty grayscale or RGB image")
    return values


def _load_image(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Image extraction requires Pillow (pip install Pillow)") from exc
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"Multi-frame image must be split explicitly: {path}")
        if image.mode == "P":
            image = image.convert("RGB")
        result = np.array(image)
    return result


def clahe(image: np.ndarray, grid: tuple[int, int] = (8, 8),
          clip_limit: float = 2.0, bins: int = 256) -> np.ndarray:
    """NumPy CLAHE: clipped local histograms and bilinear CDF interpolation.

    This is a documented replacement, not a bitwise OpenCV reproduction.
    The clip limit is a multiple of the mean histogram-bin count. Excess is
    redistributed uniformly once, as in common practical CLAHE implementations.
    """
    values = np.asarray(image, dtype=float)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("CLAHE requires a 2-D image with both sides >= 2")
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise ValueError("CLAHE input must be finite in [0, 1]")
    if len(grid) != 2 or min(grid) < 1 or bins < 16 or clip_limit < 1:
        raise ValueError("Invalid CLAHE grid, bins or clip limit")
    if np.ptp(values) == 0:
        return values.copy()  # Do not manufacture contrast in a constant field.
    ny, nx = min(grid[0], values.shape[0]), min(grid[1], values.shape[1])
    yedges = np.linspace(0, values.shape[0], ny + 1, dtype=int)
    xedges = np.linspace(0, values.shape[1], nx + 1, dtype=int)
    luts = np.zeros((ny, nx, bins), dtype=float)
    quantized = np.minimum((values * bins).astype(int), bins - 1)
    for iy in range(ny):
        for ix in range(nx):
            patch = quantized[yedges[iy]:yedges[iy + 1], xedges[ix]:xedges[ix + 1]]
            histogram = np.bincount(patch.ravel(), minlength=bins).astype(float)
            cap = max(1.0, clip_limit * patch.size / bins)
            excess = np.maximum(histogram - cap, 0).sum()
            histogram = np.minimum(histogram, cap) + excess / bins
            luts[iy, ix] = np.cumsum(histogram) / patch.size
    ycenters = (yedges[:-1] + yedges[1:] - 1) / 2
    xcenters = (xedges[:-1] + xedges[1:] - 1) / 2
    yi = np.interp(np.arange(values.shape[0]), ycenters, np.arange(ny))[:, None]
    xi = np.interp(np.arange(values.shape[1]), xcenters, np.arange(nx))[None, :]
    y0, x0 = np.floor(yi).astype(int), np.floor(xi).astype(int)
    y1, x1 = np.minimum(y0 + 1, ny - 1), np.minimum(x0 + 1, nx - 1)
    wy, wx = yi - y0, xi - x0
    return ((1-wy)*(1-wx)*luts[y0, x0, quantized]
            + (1-wy)*wx*luts[y0, x1, quantized]
            + wy*(1-wx)*luts[y1, x0, quantized]
            + wy*wx*luts[y1, x1, quantized])


def extract_region_features(image: np.ndarray, mask: np.ndarray | None = None,
                            config: ImagingConfig | None = None) -> dict[str, float]:
    """Aggregate valid tiles of one already-selected ROI into one observation.

    Border/invalid-mask neighborhoods are excluded from derivative features.
    No claim of optical scattering calibration or automatic segmentation is made.
    """
    config = config or ImagingConfig()
    config.validate()
    gray = to_gray_float(image)
    if mask is None:
        mask = np.ones(gray.shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != gray.shape:
            raise ValueError("Mask must have the same width/height as the ROI")
    rows, counts = [], []
    radius = max(1, int(np.ceil(4 * config.highpass_sigma_px)))
    for y in range(0, gray.shape[0], config.tile_size):
        for x in range(0, gray.shape[1], config.tile_size):
            patch = gray[y:y+config.tile_size, x:x+config.tile_size]
            valid = mask[y:y+config.tile_size, x:x+config.tile_size]
            if min(patch.shape) < config.min_tile_side or valid.mean() < config.min_mask_fraction:
                continue
            interior = ndimage.binary_erosion(valid, iterations=radius, border_value=0)
            if interior.sum() < 4:
                continue
            # Fill masked pixels from their nearest valid neighbor for histogram
            # preprocessing. Their neighborhoods are excluded from texture stats.
            if not valid.all():
                indices = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
                processing = patch[tuple(indices)]
            else:
                processing = patch
            corrected = clahe(processing, config.clahe_grid, config.clahe_clip_limit, config.clahe_bins)
            gx = ndimage.sobel(corrected, axis=1, mode="reflect") / 8
            gy = ndimage.sobel(corrected, axis=0, mode="reflect") / 8
            gradient = np.hypot(gx, gy)[interior]
            highpass = corrected - ndimage.gaussian_filter(corrected, config.highpass_sigma_px, mode="reflect")
            neighbor = np.abs(corrected - ndimage.uniform_filter(corrected, size=3, mode="reflect"))
            rows.append([
                float(patch[valid].mean()), float(patch[valid].var()),
                float(corrected[valid].mean()), float(corrected[valid].var()),
                float(gradient.mean()), float(np.quantile(gradient, .9)),
                float(highpass[interior].std()), float(neighbor[interior].mean()),
            ])
            counts.append(int(valid.sum()))
    if not rows:
        raise ValueError("No valid tiles; check ROI/mask, tile size and minimum coverage")
    array, weights = np.array(rows), np.array(counts, dtype=float)
    means = np.average(array, axis=0, weights=weights)
    stds = np.sqrt(np.average((array-means)**2, axis=0, weights=weights))
    output = {}
    for i, name in enumerate(TILE_FEATURE_NAMES):
        output[f"feature_{name}_mean"] = float(means[i])
        output[f"feature_{name}_std"] = float(stds[i])
    output.update(n_tiles=len(rows), n_valid_pixels=sum(counts), roi_width=gray.shape[1], roi_height=gray.shape[0])
    return output


def _integer(row: pd.Series, name: str) -> int:
    try:
        value = float(row[name])
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"{name} must be an explicit integer pixel coordinate/size") from exc
    if not np.isfinite(value) or value != int(value):
        raise ValueError(f"{name} must be an explicit integer pixel coordinate/size")
    return int(value)


def extract_manifest(manifest_path: str | Path, output_csv: str | Path | None = None,
                     config: ImagingConfig | None = None) -> pd.DataFrame:
    """Read a CSV of explicit matched ROIs, optionally writing aggregated features.

    Paths are resolved relative to the manifest. Targets may be missing for
    prediction; labeled rows must identify the matching FE-SEM measurement.
    """
    config = config or ImagingConfig()
    config.validate()
    path = Path(manifest_path).resolve()
    if output_csv is not None:
        output = Path(output_csv).resolve()
        if output == path:
            raise ValueError("Output CSV must not overwrite the input manifest")
        for candidate in (output, output.with_suffix(".imaging.json")):
            if candidate.exists():
                raise FileExistsError(f"Refusing to overwrite existing output: {candidate}")
    data = read_csv(path, dtype={c: str for c in ("sample_id", "image_id", "condition_id", "measurement_id")})
    reject_axis_pixel_calibration(data.columns)
    required = {"sample_id", "image_id", "condition_id", "image_path", "roi_x", "roi_y", "roi_width", "roi_height", "pixel_size_um", "data_origin"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("Manifest contains no ROIs")
    for column in ("sample_id", "image_id", "condition_id", "image_path", "data_origin"):
        if data[column].isna().any() or data[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing {column}")
    if data.sample_id.duplicated().any():
        raise ValueError("sample_id must be unique")
    if data.groupby("image_id").condition_id.nunique().max() != 1:
        raise ValueError("An original image cannot belong to multiple condition groups")
    if not data.data_origin.isin(["real_measurement", "synthetic_demo"]).all():
        raise ValueError("data_origin must be real_measurement or synthetic_demo")
    records, cached_path, cached_image = [], None, None
    for _, row in data.iterrows():
        image_path = (path.parent / str(row.image_path)).resolve()
        if image_path != cached_path:
            cached_image, cached_path = _load_image(image_path), image_path
        image = cached_image
        height, width = image.shape[:2]
        if config.expected_width and config.expected_height and (width, height) != (config.expected_width, config.expected_height):
            warnings.warn(f"{row.image_id}: image is {width}x{height}; reference describes 2048x1536. Confirm magnification/calibration.", UserWarning)
        x, y, w, h = [_integer(row, c) for c in ("roi_x", "roi_y", "roi_width", "roi_height")]
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > width or y+h > height:
            raise ValueError(f"Invalid/out-of-bounds ROI for {row.sample_id}")
        pixel_size = scalar_pixel_scale(row.pixel_size_um, 'pixel_size_um')
        label = row.get("thickness_nm", np.nan)
        if pd.notna(label):
            try:
                label = float(label)
            except (ValueError, TypeError) as exc:
                raise ValueError("thickness_nm must be numeric when provided") from exc
            if not np.isfinite(label) or label <= 0:
                raise ValueError("thickness_nm must be positive and finite")
            if pd.isna(row.get("measurement_id")) or not str(row.get("measurement_id", "")).strip():
                raise ValueError("Every thickness label requires a matched measurement_id")
            if row.get("match_basis") not in MATCH_BASES:
                raise ValueError(f"Labeled rows require match_basis in {sorted(MATCH_BASES)}")
        mask = None
        if pd.notna(row.get("mask_path")) and str(row.get("mask_path", "")).strip():
            mask_image = _load_image((path.parent / str(row.mask_path)).resolve())
            if mask_image.ndim != 2 or mask_image.shape != (height, width):
                raise ValueError("Manual mask must be a grayscale full-image mask with matching dimensions")
            mask = mask_image[y:y+h, x:x+w] > 0
        record = row.to_dict()
        record['pixel_size_um'] = pixel_size
        record["image_path"] = str(image_path)
        record.update(extract_region_features(image[y:y+h, x:x+w], mask, config))
        records.append(record)
    output = pd.DataFrame(records)
    labeled = output[output.get("thickness_nm", pd.Series(np.nan, index=output.index)).notna()]
    if not labeled.empty and labeled.measurement_id.duplicated().any():
        raise ValueError("A FE-SEM measurement_id is reused: aggregate matching optical regions instead of copying labels")
    if output_csv is not None:
        target = Path(output_csv)
        target.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(target, index=False)
        target.with_suffix(".imaging.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("output_csv")
    parser.add_argument("--tile-size", type=int, default=128)
    args = parser.parse_args(argv)
    frame = extract_manifest(args.manifest, args.output_csv, ImagingConfig(tile_size=args.tile_size))
    print(f"Extracted {len(frame)} matched-region feature rows into {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
