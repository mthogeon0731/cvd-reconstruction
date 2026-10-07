import json
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from cvd_cbd.study_config import StudyConfig
from cvd_cbd.infer import pair_sc
from cvd_cbd.uncertainty import NAMES,propagate
from cvd_cbd.similitude import independent_property_check,compare_similitude

@pytest.fixture
def paired():
    common=dict(trench_id='t',condition_id='c',run_id='R01',chip=1,slot=1,AR=9,L_um=270,w_um=30,h_um=30,faces=4,left_material='pdms',right_material='pdms',floor_material='glass',ceiling_material='pdms',T_C=65,gly_pct=0,batch_id='b',specimen_id='s',origin='SYNTHETIC',kind='trench',prediction_role='cross_fitted',label_sd_nm=1)
    return pd.DataFrame([dict(common,pos='top',pred_nm=100,label_nm=100,measurement_id='m1',observed_z_lo_um=34,observed_z_hi_um=66,observed_intervals_um='[[34,66]]'),dict(common,pos='bot',pred_nm=50,label_nm=50,measurement_id='m2',observed_z_lo_um=204,observed_z_hi_um=236,observed_intervals_um='[[204,236]]')])


def test_pairing_and_actual_counts(paired):
    t,p=pair_sc(paired,StudyConfig());assert t.SC.iloc[0]==.5 and p.n_valid.iloc[0]==1 and pd.isna(t.SC_sd.iloc[0])
    t,p=pair_sc(paired,StudyConfig(),'label_nm');assert t.SC_sd.iloc[0]==pytest.approx(np.sqrt(.000125))
    t,p=pair_sc(paired.iloc[:1],StudyConfig());assert t.reason.iloc[0]=='missing_top_or_bot' and p.n_excluded.iloc[0]==1
    with pytest.raises(ValueError,match='Duplicate'):pair_sc(pd.concat([paired,paired.iloc[:1]]),StudyConfig())

@pytest.mark.parametrize('a,b,reason',[(0.,50.,'top_below_denominator_threshold'),(100.,-1.,'negative_bot_thickness'),(np.nan,3.,'missing_thickness')])
def test_bad_pair_thickness(paired,a,b,reason):
    d=paired.copy();d.pred_nm=[a,b];t,_=pair_sc(d,StudyConfig());assert t.reason.iloc[0]==reason


def test_overlap_and_unverified_predictions(paired,tmp_path):
    d=paired.copy();d.loc[1,'observed_z_lo_um']=50;t,_=pair_sc(d,StudyConfig());assert t.reason.iloc[0]=='overlapping_observation_windows'
    d=paired.copy();d['prediction_role']='training_reprediction';t,_=pair_sc(d,StudyConfig())
    with pytest.raises(ValueError,match='re-predictions'):compare_similitude(t,pd.DataFrame(),tmp_path/'sim',StudyConfig())


def test_circular_properties_refused(paired):
    t,_=pair_sc(paired,StudyConfig())
    cal=pd.DataFrame([dict(property_basis='independent_planar_and_viscosity',calibration_ids='m1',calibration_specimen_ids='other')])
    with pytest.raises(ValueError,match='Circular'):independent_property_check(cal,t)


def test_joint_uncertainty_shared_measurement_cancels():
    cfg=StudyConfig();means=dict(zip(NAMES,[30e-6,30e-6,270e-6,1e-11,1e-11,.001,6e-10,50,1.45e-5,100,50]))
    cov=np.zeros((11,11));cov[-2:,-2:]=.01
    spec=dict(means=means,log_covariance=cov.tolist(),repeats=30,seed=5,T_C=65,model_discrepancy_sd_fraction=0.,assumptions='SYNTHETIC perfect shared multiplicative metrology error')
    d,r=propagate(spec,cfg);np.testing.assert_allclose(d.SC_measurement,.5,atol=1e-9)
    assert r['measurement']['sd']<1e-8
    d2,_=propagate(spec,cfg);pd.testing.assert_frame_equal(d,d2)
    cov[0,0]=-1;spec['log_covariance']=cov.tolist()
    with pytest.raises(ValueError,match='semidefinite'):propagate(spec,cfg)
