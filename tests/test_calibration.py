from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from cvd_cbd.study_config import StudyConfig
from cvd_cbd.calibration import fit_growth,calibration_table
from cvd_cbd.similitude import PropertyGrid,ar_limit

@pytest.fixture
def growth_data():
    rows=[]
    for T in [65.,80.]:
        for gly in [0.,40.]:
            for s in ['glass','pdms']:
                for t in [10.,20.,30.,40.]:
                    mid=f'{T}_{gly}_{s}_{t}'
                    rows.append(dict(T_C=T,gly_pct=gly,substrate=s,t_min=t,thickness_nm=2*(t-5),thickness_sd_nm=.2,measurement_id=mid,specimen_id=mid,method='SYNTHETIC',source='SYNTHETIC',origin='SYNTHETIC'))
    return pd.DataFrame(rows)

@pytest.fixture
def viscosity():
    return pd.DataFrame([dict(T_C=T,gly_pct=G,eta_Pa_s=.001,eta_sd_Pa_s=.00002,eta_unit='Pa_s',gly_basis='mass_pct',source='SYNTHETIC',origin='SYNTHETIC') for T in [65.,80.] for G in [0.,40.]])


def test_growth_induction_and_known_measurement_error(growth_data):
    g=fit_growth(growth_data,StudyConfig())
    np.testing.assert_allclose(g.G_nm_min,2);np.testing.assert_allclose(g.t_ind_min,5)
    assert g.induction.all() and (g.G_se_nm_min>0).all() and (g.df==2).all()
    assert (g.slope_ci_low<2).all() and (g.slope_ci_high>2).all()


def test_C0_Vm_propagation_and_missing_substrate(growth_data,viscosity):
    cfg=StudyConfig();g=fit_growth(growth_data,cfg);a=calibration_table(g,viscosity,cfg)
    b=calibration_table(g,viscosity,replace(cfg,C0_mol_m3=100.,Vm_m3_mol=2*cfg.Vm_m3_mol))
    np.testing.assert_allclose(b.ks_glass,a.ks_glass/4)
    with pytest.raises(ValueError,match='substrate'):calibration_table(g[g.substrate=='glass'],viscosity,cfg)
    with pytest.raises(ValueError,match='viscosity'):calibration_table(g,viscosity.iloc[:-1],cfg)
    viscosity.loc[0,'eta_unit']='mPa_s'
    with pytest.raises(ValueError,match='unit'):calibration_table(g,viscosity,cfg)


def test_missing_times_replicates_and_nonpositive(growth_data,viscosity):
    cfg=StudyConfig();d=growth_data[growth_data.t_min!=20]
    g=fit_growth(d,cfg);assert g.missing_times.str.contains('20').all()
    d=growth_data.copy();d.thickness_nm=10.;g=fit_growth(d,cfg)
    with pytest.raises(ValueError,match='assumptions'):calibration_table(g,viscosity,cfg)
    d=growth_data.copy();d.loc[1,'measurement_id']=d.loc[0,'measurement_id']
    with pytest.raises(ValueError,match='Unique'):fit_growth(d,cfg)


def test_nonlinearity_and_concentration(growth_data,viscosity):
    d=growth_data.copy();d.thickness_nm=d.t_min**2;g=fit_growth(d,StudyConfig())
    assert g['flags'].str.contains('nonlinear').all()
    with pytest.raises(ValueError):calibration_table(g,viscosity,StudyConfig())
    d=growth_data.copy();d['C0_mol_m3']=50.;d.loc[d.t_min==40,'C0_mol_m3']=40
    assert fit_growth(d,StudyConfig())['flags'].str.contains('concentration').all()


def test_composition_preserved_grid_and_no_extrapolation(growth_data,viscosity):
    cfg=StudyConfig();g=fit_growth(growth_data,cfg);c=calibration_table(g,viscosity,cfg)
    c.loc[c.gly_pct==40,'ks_glass']*=4;c.loc[c.gly_pct==40,'ks_pdms']*=9
    grid=PropertyGrid(c,cfg)
    assert grid.at(65,40)['ks_glass']/grid.at(65,0)['ks_glass']==pytest.approx(4)
    assert grid.at(65,20)['ks_glass']/grid.at(65,0)['ks_glass']==pytest.approx(2)
    with pytest.raises(ValueError,match='extrapolation'):grid.at(90,0)
    with pytest.raises(ValueError,match='Incomplete'):PropertyGrid(c.iloc[:-1],cfg)
    assert ar_limit({'glass':0.,'pdms':0.},1e-9,0.,cfg)['value'] is None


def test_two_face_only_requires_active_substrates(growth_data,viscosity):
    cfg=replace(StudyConfig(),faces=2,end_material='glass',end_rate_ratio=0.)
    g=fit_growth(growth_data[growth_data.substrate.eq('pdms')],cfg)
    cal=calibration_table(g,viscosity,cfg);grid=PropertyGrid(cal,cfg)
    p=grid.at(70.,20.)
    assert p['k_end']==0 and p['ks_pdms']>0 and 'ks_glass' not in p
