"""Independent regressions for ML-01 and PHY-001; synthetic software checks only."""
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from cvd_cbd.calibration import fit_growth, calibration_table
from cvd_cbd.infer import pair_sc
from cvd_cbd.io_utils import write_csv, write_json
from cvd_cbd.modeling import COLUMNS, compare, fit_final, predict_final
from cvd_cbd.physics import Geometry, observed_sc
from cvd_cbd.similitude import PropertyGrid, compare_similitude, process_window
from cvd_cbd.study_config import StudyConfig
from cvd_cbd.study_design import run_design
from cvd_cbd.uncertainty import NAMES, propagate


def pair_fixture(cfg=None):
    cfg = cfg or StudyConfig()
    common = dict(trench_id='t', condition_id='c', run_id='R01', chip=1,
        slot=1, AR=9., L_um=270., w_um=30., h_um=30., faces=cfg.faces,
        left_material=cfg.wall_materials[0], right_material=cfg.wall_materials[1],
        floor_material=cfg.wall_materials[2], ceiling_material=cfg.wall_materials[3],
        T_C=65., gly_pct=0., batch_id='b', specimen_id='s', origin='SYNTHETIC',
        kind='trench', prediction_role='cross_fitted', label_sd_nm=1.)
    return pd.DataFrame([
        dict(common, pos='top', pred_nm=100., label_nm=100., measurement_id='m1',
             observed_z_lo_um=34., observed_z_hi_um=66., observed_intervals_um='[[34,66]]'),
        dict(common, pos='bot', pred_nm=50., label_nm=50., measurement_id='m2',
             observed_z_lo_um=204., observed_z_hi_um=236., observed_intervals_um='[[204,236]]')])


def uncertainty_fixture():
    means = dict(zip(NAMES, [30e-6, 30e-6, 270e-6, 1e-11, 1e-11,
                            .001, 6e-10, 50., 1.45e-5, 100., 50.]))
    return dict(means=means, log_covariance=np.zeros((11, 11)).tolist(),
                repeats=12, seed=519, T_C=65., model_discrepancy_sd_fraction=0.,
                assumptions='Independent synthetic regression, supplied zero covariance')


def inactive_config(**kwargs):
    return replace(StudyConfig(), faces=2,
        wall_materials=('pdms', 'pdms', 'inert', 'inert'), end_material='inert', **kwargs)


def growth_fixture():
    return pd.DataFrame([dict(T_C=65., gly_pct=0., substrate='pdms', t_min=t,
        thickness_nm=2*t+3, measurement_id=f'p{i}', specimen_id=f'ps{i}',
        method='synthetic', source='independent fixture', origin='SYNTHETIC')
        for i,t in enumerate((10.,20.,30.,40.))])


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf, -1., None, 'missing'])
def test_present_planar_sd_column_requires_valid_values_even_if_all_missing(bad):
    table=growth_fixture(); table['thickness_sd_nm']=bad
    with pytest.raises(ValueError, match='thickness SD'):
        fit_growth(table,inactive_config())


def test_optional_planar_sd_omission_differs_from_explicit_zero():
    table=growth_fixture()
    omitted=fit_growth(table,inactive_config())
    table['thickness_sd_nm']=0.
    zero=fit_growth(table,inactive_config())
    assert omitted.metrology_uncertainty_status.iloc[0]=='NOT_PROVIDED_RESIDUAL_ESTIMATE_ONLY'
    assert zero.metrology_uncertainty_status.iloc[0]=='SUPPLIED_ZERO_SD'
    table['thickness_sd_nm']=.5
    supplied=fit_growth(table,inactive_config())
    assert supplied.metrology_uncertainty_status.iloc[0]=='SUPPLIED_SD'
    assert supplied.G_se_nm_min.iloc[0]==pytest.approx(.5/np.sqrt(500),rel=1e-12)


def test_design_inactive_default_walls_marks_only_unmeasured_hypotheses(tmp_path):
    cfg=replace(StudyConfig(),faces=2,end_material='inert',mc_repeats=4)
    viscosity=pd.DataFrame([dict(T_C=65.,gly_pct=0.,eta_Pa_s=.001,eta_sd_Pa_s=.00001,
        eta_unit='Pa_s',gly_basis='mass_pct',source='test',origin='SYNTHETIC')])
    cal=calibration_table(fit_growth(growth_fixture(),cfg),viscosity,cfg)
    assert 'ks_glass' not in cal
    table=run_design(cal,tmp_path/'design',cfg)
    assert len(table)==len(cfg.ar_ladder) and np.isfinite(table.SC_observed).all()
    sensitivity=pd.read_csv(tmp_path/'design/sensitivity.csv')
    omitted=sensitivity[sensitivity.faces.eq(4)]
    assert omitted.status.eq('NOT_EVALUATED').all()
    assert omitted.SC_endpoint.isna().all() and omitted.Da.isna().all()
    assert omitted.missing_substrates.eq('glass').all()
    assert sensitivity[sensitivity.faces.eq(2)].status.eq('EVALUATED').all()
    lengths=pd.read_csv(tmp_path/'design/new_dead_end_1pct.csv')
    assert lengths[lengths.faces.eq(4)].status.eq('NOT_EVALUATED').all()
    assert lengths[lengths.faces.eq(4)].depth_um.isna().all()
    report=json.loads((tmp_path/'design/design_report.json').read_text())
    assert report['sensitivity_not_evaluated']==18 and report['dead_end_not_evaluated']==2
    active=replace(cfg,faces=4)
    with pytest.raises(ValueError,match='Missing substrate kinetics'):
        run_design(cal,tmp_path/'invalid_actual_active',active)


def test_inactive_raw_images_full_workflow_without_glass_calibration(tmp_path):
    from cvd_cbd.synthetic import generate
    from cvd_cbd.workflow import run
    cfg=replace(StudyConfig(),faces=2,end_material='inert',ar_ladder=(9,13),mc_repeats=4)
    root=generate(tmp_path/'inputs',cfg)
    cal_path=root/'meta/calibration_thickness.csv';planar=pd.read_csv(cal_path)
    planar[planar.substrate.eq('pdms')].to_csv(cal_path,index=False)
    spec_path=root/'uncertainty.json';spec=json.loads(spec_path.read_text())
    names=[k for k in NAMES if k!='G_glass_m_s'];ix=[NAMES.index(k) for k in names]
    spec['means'].pop('G_glass_m_s');spec['covariance_order']=names
    spec['log_covariance']=np.asarray(spec['log_covariance'])[np.ix_(ix,ix)].tolist()
    spec['repeats']=4;spec_path.write_text(json.dumps(spec))
    result=run(root,tmp_path/'results',cfg)
    assert result['status']=='PASS_SOFTWARE_PIPELINE'
    assert result['real_image_check'].startswith('NOT RUN')
    cal=pd.read_csv(tmp_path/'results/calibration.csv');assert 'ks_glass' not in cal
    report=json.loads((tmp_path/'results/uncertainty_report.json').read_text())
    assert report['parameter_status']['G_glass_m_s']=='NOT_APPLICABLE'
    assert report['property']['n_defined']==4
    sensitivity=pd.read_csv(tmp_path/'results/design/sensitivity.csv')
    assert sensitivity[sensitivity.faces.eq(4)].status.eq('NOT_EVALUATED').all()
    observed=pd.read_csv(tmp_path/'results/trench_sc_fesem.csv')
    assert observed.status.eq('PASS').all() and observed.uncertainty_status.eq('AVAILABLE').all()
    assert (tmp_path/'results/final_model/final_model.joblib').is_file()


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf, -1., None, 'missing'])
@pytest.mark.parametrize('row', [0, 1])
def test_reject_invalid_measurement_sd(bad, row):
    data = pair_fixture(); data['label_sd_nm'] = data.label_sd_nm.astype(object)
    data.loc[row, 'label_sd_nm'] = bad
    with pytest.raises(ValueError, match='label_sd_nm'):
        pair_sc(data, StudyConfig(), 'label_nm')


def test_reject_missing_sd_column():
    with pytest.raises(ValueError, match='label_sd_nm'):
        pair_sc(pair_fixture().drop(columns='label_sd_nm'), StudyConfig(), 'label_nm')


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf, -1.01, 1.01, 'missing'])
@pytest.mark.parametrize('row', [0, 1])
def test_reject_invalid_supplied_correlation_on_either_row(bad, row):
    data = pair_fixture(); data['top_bot_correlation'] = pd.Series([0., 0.], dtype=object)
    data.loc[row, 'top_bot_correlation'] = bad
    with pytest.raises(ValueError, match='correlation'):
        pair_sc(data, StudyConfig(), 'label_nm')


def test_reject_conflicting_pair_correlations():
    data = pair_fixture(); data['top_bot_correlation'] = [0., .8]
    with pytest.raises(ValueError, match='correlation'):
        pair_sc(data, StudyConfig(), 'label_nm')


def test_zero_sd_and_unavailable_predictive_error_remain_distinct(tmp_path):
    data = pair_fixture(); data['label_sd_nm'] = 0.
    measured, _ = pair_sc(data, StudyConfig(), 'label_nm')
    predicted, _ = pair_sc(data, StudyConfig(), 'pred_nm')
    assert measured.SC_sd.iloc[0] == 0.
    assert measured.uncertainty_status.iloc[0] == 'AVAILABLE'
    assert pd.isna(predicted.SC_sd.iloc[0])
    assert predicted.uncertainty_status.iloc[0] == 'NOT_AVAILABLE'
    assert measured.missing_uncertainty_policy.iloc[0] == 'error'
    path = tmp_path / 'pairing.csv'
    write_csv(path, pd.DataFrame([measured.iloc[0].to_dict(), predicted.iloc[0].to_dict()]))
    restored = pd.read_csv(path)
    assert restored.SC_sd.iloc[0] == 0. and pd.isna(restored.SC_sd.iloc[1])
    assert list(restored.uncertainty_status) == ['AVAILABLE', 'NOT_AVAILABLE']


def test_perfect_shared_measurement_sd_cancels_without_erasing_missingness():
    data = pair_fixture(); data['label_sd_nm'] = [10., 5.]
    data['top_bot_correlation'] = 1.
    measured, _ = pair_sc(data, StudyConfig(), 'label_nm')
    assert measured.SC_sd.iloc[0] == pytest.approx(0., abs=1e-14)
    assert measured.uncertainty_status.iloc[0] == 'AVAILABLE'


def test_inactive_legacy_zero_growth_and_active_only_schema_agree():
    spec = uncertainty_fixture(); spec['means']['G_glass_m_s'] = 0.
    legacy, report = propagate(spec, inactive_config())
    names = [x for x in NAMES if x != 'G_glass_m_s']
    modern = {**spec, 'means': {k: spec['means'][k] for k in names},
              'covariance_order': names, 'log_covariance': np.zeros((10, 10)).tolist()}
    active, active_report = propagate(modern, inactive_config())
    pd.testing.assert_frame_equal(legacy, active)
    assert report['parameter_status']['G_glass_m_s'] == 'NOT_APPLICABLE'
    assert active_report['parameter_status']['G_glass_m_s'] == 'NOT_APPLICABLE'
    assert 'G_glass_m_s' not in report['sampled_parameters']
    assert report['missing_uncertainty_policy'] == 'error'
    assert report['measurement_uncertainty_status'] == 'SUPPLIED_ZERO_COVARIANCE'


@pytest.mark.parametrize('growth', [0., -1., np.nan, np.inf, None, True])
def test_active_growth_still_required(growth):
    spec = uncertainty_fixture(); spec['means']['G_glass_m_s'] = growth
    with pytest.raises(ValueError, match='means|G_glass'):
        propagate(spec, StudyConfig())


def test_end_material_is_active_even_if_side_faces_are_inert():
    cfg = replace(inactive_config(), end_material='glass')
    spec = uncertainty_fixture(); del spec['means']['G_glass_m_s']
    spec['log_covariance'] = np.zeros((10, 10)).tolist()
    with pytest.raises(ValueError, match='G_glass'):
        propagate(spec, cfg)


def test_zero_end_rate_does_not_require_unused_glass():
    cfg = replace(inactive_config(), end_material='glass', end_rate_ratio=0.)
    spec = uncertainty_fixture(); del spec['means']['G_glass_m_s']
    spec['covariance_order'] = list(spec['means'])
    spec['log_covariance'] = np.zeros((10, 10)).tolist()
    draws, report = propagate(spec, cfg)
    assert np.isfinite(draws.SC_property).all()
    assert report['parameter_status']['G_glass_m_s'] == 'NOT_APPLICABLE'


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_missing_or_invalid_mc_uncertainty_is_never_zero(bad):
    spec = uncertainty_fixture(); spec['log_covariance'][-1][-1] = bad
    with pytest.raises(ValueError, match='covariance'):
        propagate(spec, StudyConfig())


def test_mc_requires_explicit_covariance():
    spec = uncertainty_fixture(); del spec['log_covariance']
    with pytest.raises(ValueError, match='log_covariance'):
        propagate(spec, StudyConfig())


@pytest.mark.parametrize('defect', ['negative_diagonal', 'asymmetric', 'indefinite_small'])
def test_invalid_mc_covariance_rejected_at_its_own_scale(defect):
    spec = uncertainty_fixture(); cov = np.zeros((11, 11))
    if defect == 'negative_diagonal':
        cov[0, 0] = -1e-16
    elif defect == 'asymmetric':
        cov[0, 0] = cov[1, 1] = 1e-15; cov[0, 1] = 1e-16
    else:
        cov[:2, :2] = [[1e-15, 2e-15], [2e-15, 1e-15]]
    spec['log_covariance'] = cov.tolist()
    with pytest.raises(ValueError, match='covariance'):
        propagate(spec, StudyConfig())


@pytest.mark.parametrize('order', [NAMES[:-1], NAMES[:-1] + [NAMES[0]], NAMES[:-1] + ['unknown']])
def test_named_covariance_order_must_be_complete_unique_and_known(order):
    spec = {**uncertainty_fixture(), 'covariance_order': order}
    with pytest.raises(ValueError, match='covariance_order'):
        propagate(spec, StudyConfig())


def test_named_covariance_reordering_is_normalized_before_sampling():
    spec = uncertainty_fixture(); cov = np.diag(np.linspace(.0001, .0011, 11))
    cov[-2, -1] = cov[-1, -2] = .0003
    spec['log_covariance'] = cov.tolist()
    original, _ = propagate(spec, StudyConfig())
    ix = list(reversed(range(11))); names = [NAMES[i] for i in ix]
    reordered = {**spec, 'covariance_order': names,
                 'log_covariance': cov[np.ix_(ix, ix)].tolist()}
    actual, _ = propagate(reordered, StudyConfig())
    pd.testing.assert_frame_equal(original, actual)


def test_shared_measurement_covariance_policy_reaches_monte_carlo(tmp_path):
    spec = uncertainty_fixture(); cov = np.zeros((11, 11)); cov[-2:, -2:] = .01
    spec.update(log_covariance=cov.tolist(), missing_uncertainty_policy='error')
    draws, report = propagate(spec, StudyConfig())
    np.testing.assert_allclose(draws.SC_measurement, .5, atol=1e-9, rtol=0)
    assert report['missing_uncertainty_policy'] == 'error'
    assert report['measurement_uncertainty_status'] == 'SUPPLIED_COVARIANCE'
    write_json(tmp_path / 'report.json', report)
    assert json.loads((tmp_path / 'report.json').read_text())['missing_uncertainty_policy'] == 'error'
    spec['missing_uncertainty_policy'] = 'zero'
    with pytest.raises(ValueError, match='policy'):
        propagate(spec, StudyConfig())


@pytest.mark.parametrize('inactive', [False, True], ids=['active4', 'inactive2'])
def test_active_and_inactive_calibration_learning_sc_mc_and_serialization(tmp_path, inactive):
    cfg = inactive_config() if inactive else StudyConfig()
    materials = ['pdms'] if inactive else ['glass', 'pdms']
    growth_rows, viscosity_rows, regions = [], [], []
    for ci, (temperature, gly) in enumerate([(65., 0.), (65., 40.), (80., 0.), (80., 40.)]):
        for substrate in materials:
            for time in (10., 20., 30., 40.):
                growth_rows.append(dict(T_C=temperature, gly_pct=gly, substrate=substrate,
                    t_min=time, thickness_nm=(1. + ci * .2) * time + 3., thickness_sd_nm=.1,
                    measurement_id=f'cal_{ci}_{substrate}_{time}', specimen_id=f'cal_s_{ci}_{substrate}_{time}',
                    method='synthetic fixture', source='independent test', origin=cfg.origin))
        viscosity_rows.append(dict(T_C=temperature, gly_pct=gly, eta_Pa_s=.001 + ci * .0001,
            eta_sd_Pa_s=.00001, eta_unit='Pa_s', gly_basis='mass_pct', source='independent test', origin=cfg.origin))
        data = pair_fixture(cfg)
        for col in ('trench_id', 'condition_id', 'run_id', 'batch_id', 'specimen_id'):
            data[col] = data[col].astype(str) + str(ci)
        data['T_C'] = temperature; data['gly_pct'] = gly
        data['measurement_id'] = data.measurement_id + str(ci)
        data['image_id'] = 'image_' + data.measurement_id
        data['content_sha256'] = 'hash_' + data.measurement_id
        data['matched_region_id'] = 'region_' + data.measurement_id
        data['match_basis'] = 'validated_region_average'; data['um_per_px'] = .5
        data['label_nm'] = [100. + 10 * ci, 80. + 8 * ci]
        for j, column in enumerate(COLUMNS):
            data[column] = np.log1p(data.label_nm) / (j + 1.) + .002 * ci * j
        regions.append(data)
    calibration = calibration_table(fit_growth(pd.DataFrame(growth_rows), cfg), pd.DataFrame(viscosity_rows), cfg)
    grid = PropertyGrid(calibration, cfg); props = grid.at(65., 0.)
    assert ('ks_glass' in props) != inactive
    features = pd.concat(regions, ignore_index=True)
    selected, model_report = compare(features, tmp_path / 'evaluation', cfg)
    assert selected.prediction_role.eq('cross_fitted').all()
    fit_final(features, tmp_path / 'final', cfg)
    reloaded = predict_final(tmp_path / 'final/final_model.joblib', features, True)
    assert np.isfinite(reloaded.pred_nm).all()
    measured, _ = pair_sc(features, cfg, 'label_nm')
    predicted, _ = pair_sc(selected, cfg)
    assert measured.uncertainty_status.eq('AVAILABLE').all()
    assert predicted.uncertainty_status.eq('NOT_AVAILABLE').all()
    compare_similitude(measured, calibration, tmp_path / 'sim_measured', cfg)
    compare_similitude(predicted, calibration, tmp_path / 'sim_cross_fitted', cfg)
    process_window(calibration, tmp_path / 'map', cfg, T_grid=[65., 80.], G_grid=[0., 40.])
    spec = uncertainty_fixture()
    if inactive:
        del spec['means']['G_glass_m_s']
        spec.update(covariance_order=list(spec['means']), log_covariance=np.zeros((10, 10)).tolist())
    samples, uncertainty_report = propagate(spec, cfg)
    write_csv(tmp_path / 'samples.csv', samples); write_json(tmp_path / 'uncertainty.json', uncertainty_report)
    restored = json.loads((tmp_path / 'uncertainty.json').read_text())
    assert restored['parameter_status']['G_glass_m_s'] == ('NOT_APPLICABLE' if inactive else 'USED')
    assert restored['property']['n_defined'] == spec['repeats']
    assert list(pd.read_csv(tmp_path / 'samples.csv')) == list(samples)
