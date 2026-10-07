"""IMG-001 regressions: actual dependencies, stale content and explicit roles."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from cvd_cbd.study_config import StudyConfig
from cvd_cbd.manifest import build_manifest
from cvd_cbd.dataset import build_features


def input_fixture(base):
    """Independent rational-count fixture; no forward physical/image generator."""
    root = base / 'inputs'
    for name in ('meta', 'raw', 'evidence'):
        (root / name).mkdir(parents=True)
    (root / 'evidence/scale.txt').write_text('SYNTHETIC independent scale: 1 um/px')
    (root / 'evidence/sem.txt').write_text('SYNTHETIC registered ROI label')
    y, x = np.indices((64, 96))
    for name, a in [('REF_S01.tif', np.full((64, 96), 9001, np.uint16)),
                    ('DARK_S01.tif', np.full((64, 96), 1001, np.uint16)),
                    ('R01_C1_T1_top.tif', (3001 + 8*x + 16*y).astype(np.uint16))]:
        Image.fromarray(a).save(root / 'raw' / name)
    acquisition = dict(session_id='S01', exposure_ms=10., gain=1., illumination_id='lamp',
                       magnification=10., bit_depth=16, auto_exposure=False, gamma=1.,
                       white_balance='fixed', color_channel='mono')
    image = dict(image_id='R01_C1_T1_top', path='raw/R01_C1_T1_top.tif', kind='trench',
                 run_id='R01', chip=1, slot=1, pos='top', condition_id='65C_G0_mass_pct',
                 batch_id='batch1', specimen_id='spec1', matched_region_id='roi1',
                 measurement_id='sem1', z_anchor_um=50., x_anchor_px=48., um_per_px=1.,
                 scale_source='SYNTHETIC', scale_record='evidence/scale.txt', orientation='x+',
                 roi_mode='manual', roi_x0=32, roi_x1=64, roi_y0=16, roi_y1=48,
                 mask_path='', ref_id='ref1', dark_id='dark1', origin='SYNTHETIC', **acquisition)
    tables = dict(images=[image], runs=[dict(run_id='R01', T_C=65., gly_pct=0.,
        gly_basis='mass_pct', rep=1, date='2026-10-07', session_id='S01', cycles=1,
        dep_time_h=1., batch_id='batch1')], layout=[dict(slot=1, AR=9.375, L_um=300.,
        w_um=32., h_um=32., faces=4, left_material='pdms', right_material='pdms',
        floor_material='glass', ceiling_material='pdms')], frames=[
        dict(frame_id='ref1', path='raw/REF_S01.tif', kind='REF', **acquisition),
        dict(frame_id='dark1', path='raw/DARK_S01.tif', kind='DARK', **acquisition)],
        fesem=[dict(run_id='R01', chip=1, slot=1, pos='top', thickness_nm=100.,
        thickness_sd_nm=1., n_sites=4, sem_file='evidence/sem.txt', measurement_id='sem1',
        matched_region_id='roi1', z_lo_um=34., z_hi_um=66.,
        match_basis='validated_region_average', source='Independent SYNTHETIC fixture')])
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(root / 'meta' / f'{name}.csv', index=False)
    return root, replace(StudyConfig(), tile_px=16, wall_margin_px=0)


@pytest.mark.parametrize('dependency', ['REF', 'DARK', 'OM', 'mask', 'scale', 'label'])
def test_same_path_content_change_invalidates_exported_manifest(tmp_path, dependency):
    root, cfg = input_fixture(tmp_path)
    paths = dict(REF='raw/REF_S01.tif', DARK='raw/DARK_S01.tif', OM='raw/R01_C1_T1_top.tif',
                 mask='raw/mask.tif', scale='evidence/scale.txt', label='evidence/sem.txt')
    if dependency == 'mask':
        Image.fromarray(np.ones((64, 96), np.uint8)).save(root / paths['mask'])
        images = pd.read_csv(root / 'meta/images.csv')
        images['mask_path'] = paths['mask']
        images.to_csv(root / 'meta/images.csv', index=False)
    manifest = build_manifest(root, cfg)
    manifest.to_csv(tmp_path / 'manifest.csv', index=False)
    # Exercise reimported manifests too: DataFrame attrs are not sufficient.
    manifest = pd.read_csv(tmp_path / 'manifest.csv', float_precision='round_trip')
    path = root / paths[dependency]
    if path.suffix == '.tif':
        pixels = np.array(Image.open(path)); pixels[0, 0] += 1
        Image.fromarray(pixels).save(path)
    else:
        path.write_text(path.read_text() + '\nchanged evidence')
    with pytest.raises(ValueError, match='Stale/edited'):
        build_features(manifest, root, tmp_path / 'out', cfg)
    assert not (tmp_path / 'out').exists()


def test_config_and_consulted_metadata_changes_invalidate_manifest(tmp_path):
    root, cfg = input_fixture(tmp_path)
    manifest = build_manifest(root, cfg)
    with pytest.raises(ValueError, match='Stale/edited'):
        build_features(manifest, root, tmp_path / 'config_changed', replace(cfg, t_max=1.5))
    # Even a source/provenance-only metadata edit changes its file fingerprint.
    sem = pd.read_csv(root / 'meta/fesem.csv')
    sem['source'] = 'updated evidence attribution'
    sem.to_csv(root / 'meta/fesem.csv', index=False)
    with pytest.raises(ValueError, match='Stale/edited'):
        build_features(manifest, root, tmp_path / 'metadata_changed', cfg)


def test_dependency_roles_hashes_and_unused_files(tmp_path):
    root, cfg = input_fixture(tmp_path)
    unused = root / 'raw/REF_UNUSED.tif'
    Image.fromarray(np.zeros((64, 96), np.uint16)).save(unused)
    frames = pd.read_csv(root / 'meta/frames.csv')
    extra = frames.iloc[0].copy(); extra['frame_id'] = 'unused'; extra['path'] = 'raw/REF_UNUSED.tif'
    frames = pd.concat([frames, extra.to_frame().T], ignore_index=True)
    frames.to_csv(root / 'meta/frames.csv', index=False)
    manifest = build_manifest(root, cfg)
    # A file present in a consulted table is not necessarily used by extraction.
    Image.fromarray(np.ones((64, 96), np.uint16)).save(unused)
    features = build_features(manifest, root, tmp_path / 'out', cfg)
    provenance = json.loads((tmp_path / 'out/input_provenance.json').read_text())
    entries = {(r['path'], r['role']): r for r in provenance['files']}
    expected = {'raw/R01_C1_T1_top.tif': 'image_pixels', 'raw/REF_S01.tif': 'reference_correction',
                'raw/DARK_S01.tif': 'dark_correction', 'evidence/scale.txt': 'scale_evidence',
                'evidence/sem.txt': 'label_evidence', 'meta/frames.csv': 'metadata_table_consulted'}
    for path, role in expected.items():
        assert entries[path, role]['sha256'] == hashlib.sha256((root / path).read_bytes()).hexdigest()
    assert all(r['path'] != 'raw/REF_UNUSED.tif' for r in provenance['files'])
    assert all(r['role'] != 'roi_mask' for r in provenance['files'])
    assert entries['evidence/sem.txt', 'label_evidence']['usage'] == 'evidence_only'
    assert provenance['effective_config']['sha256'] == manifest.config_sha256.iloc[0]
    assert features.feature_T_mean.iloc[0] == pytest.approx(.3605, abs=1e-14)


def test_om_byte_change_with_identical_pixels_is_detected(tmp_path):
    root, cfg = input_fixture(tmp_path)
    manifest = build_manifest(root, cfg)
    path = root / 'raw/R01_C1_T1_top.tif'
    # TIFF readers ignore trailing bytes; this changes file bytes, not pixels.
    with path.open('ab') as handle:
        handle.write(b'\nfixture provenance annotation\n')
    current = build_manifest(root, cfg)
    assert manifest.content_sha256.iloc[0] == current.content_sha256.iloc[0]
    with pytest.raises(ValueError, match='Stale/edited'):
        build_features(manifest, root, tmp_path / 'out', cfg)


def test_selected_roi_provenance_does_not_claim_other_images(tmp_path):
    root, cfg = input_fixture(tmp_path)
    images = pd.read_csv(root / 'meta/images.csv')
    second = images.iloc[0].copy()
    second['image_id'] = 'R01_C1_T1_bot'; second['path'] = 'raw/R01_C1_T1_bot.tif'
    second['pos'] = 'bot'; second['matched_region_id'] = 'roi2'; second['measurement_id'] = ''
    second['z_anchor_um'] = 250.
    Image.fromarray(np.full((64, 96), 4500, np.uint16)).save(root / second['path'])
    pd.concat([images, second.to_frame().T], ignore_index=True).to_csv(root / 'meta/images.csv', index=False)
    manifest = build_manifest(root, cfg)
    build_features(manifest.iloc[:1], root, tmp_path / 'out', cfg)
    provenance = json.loads((tmp_path / 'out/input_provenance.json').read_text())
    assert all(r['path'] != second['path'] for r in provenance['files'])
    assert all(r.get('image_ids') == ['R01_C1_T1_top'] for r in provenance['files'])


@pytest.mark.parametrize('dependency', ['reference', 'config_source'])
def test_content_changed_during_extraction_cannot_be_declared_success(tmp_path, monkeypatch, dependency):
    from cvd_cbd import metrology
    root, cfg = input_fixture(tmp_path)
    config_path = tmp_path / 'settings.json'; cfg.save(config_path)
    manifest = build_manifest(root, cfg)
    original_overlay = metrology.overlay
    def overlay_and_mutate(*args, **kwargs):
        original_overlay(*args, **kwargs)
        path = root / 'raw/REF_S01.tif' if dependency == 'reference' else config_path
        with path.open('ab') as handle:
            handle.write(b'\n')
    monkeypatch.setattr(metrology, 'overlay', overlay_and_mutate)
    with pytest.raises(ValueError, match='Stale/edited dependency file'):
        build_features(manifest, root, tmp_path / 'out', cfg, config_path=config_path)
    assert not (tmp_path / 'out/features.csv').exists()
