from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
import joblib
from sklearn.svm import SVR
from cvd_cbd.study_config import StudyConfig
from cvd_cbd.modeling import COLUMNS,DEPENDENCIES,ScaledRegressor,validate_features,audit_split,compare,fit_final,predict_final,metrics,manual_comparison
from cvd_cbd.thickness import condition_weights

@pytest.fixture(scope='module')
def regions():
    rows=[]
    rng=np.random.default_rng(41)
    for c in range(4):
        for i in range(4):
            iid=f'c{c}i{i}';x=rng.uniform(.1,.9,7)
            rows.append(dict(image_id=iid,trench_id=f'c{c}t{i//2}',specimen_id=f'c{c}s',batch_id=f'c{c}b',measurement_id=f'm{iid}',content_sha256=f'hash{iid}',matched_region_id=iid,label_nm=10+100*x[0],condition_id=f'c{c}',um_per_px=.5,origin='SYNTHETIC',match_basis='validated_region_average',kind='trench',**dict(zip(COLUMNS,x))))
    return pd.DataFrame(rows)

@pytest.fixture(scope='module')
def fitted(regions,tmp_path_factory):
    root=tmp_path_factory.mktemp('models');cfg=StudyConfig()
    oof,r=compare(regions,root/'evaluation',cfg);meta=fit_final(regions,root/'final',cfg)
    return root,oof,r,meta

@pytest.mark.parametrize('key',DEPENDENCIES)
def test_dependency_cross_condition_blocked(regions,key):
    d=regions.copy();d.loc[4,key]=d.loc[0,key]
    with pytest.raises(ValueError,match='crosses'):validate_features(d,'SYNTHETIC')


def test_duplicates_and_condition_audit(regions):
    d=regions.copy();d.loc[1,'measurement_id']=d.loc[0,'measurement_id']
    with pytest.raises(ValueError,match='Duplicate'):validate_features(d,'SYNTHETIC')
    with pytest.raises(ValueError,match='leakage'):audit_split(regions,[0],[1])


def test_feature_and_target_train_only_fit():
    X=np.array([[0.,10.],[2.,20.],[4.,30.]]);y=np.array([10.,20.,30.]);weights=np.array([1.,1.,2.])
    m=ScaledRegressor(SVR()).fit(X,y,weights)
    np.testing.assert_allclose(m.x_scaler_.mean_,np.average(X,axis=0,weights=weights))
    assert m.y_scaler_.mean_[0]==pytest.approx(np.average(y,weights=weights))
    before=m.y_scaler_.mean_.copy();m.predict([[1e6,1e6]])
    np.testing.assert_array_equal(before,m.y_scaler_.mean_)


def test_nested_selection_and_saved_roles(regions,fitted):
    root,oof,r,meta=fitted
    assert len(oof)==len(regions) and oof.measurement_id.nunique()==len(regions)
    assert set(r['per_model'])=={'RF','KRR','SVR'}
    assert r['nested_selection']['n_conditions']==4
    allp=pd.read_csv(root/'evaluation/all_model_oof.csv')
    assert len(allp)==3*len(regions)
    bundle=joblib.load(root/'evaluation/fold_models/fold_0_SVR.joblib')
    assert set(bundle['train_conditions']).isdisjoint(bundle['held_out_conditions'])
    ix=regions.condition_id.isin(bundle['train_conditions']);w=condition_weights(regions.loc[ix,'condition_id'])
    np.testing.assert_allclose(bundle['model'].x_scaler_.mean_,np.average(regions.loc[ix,COLUMNS],axis=0,weights=w))
    assert meta['role']=='final_all_labels' and len(meta['train_measurement_ids'])==len(regions)
    pred=predict_final(root/'final/final_model.joblib',regions,True)
    assert pred.prediction_role.eq('training_reprediction').all()


def test_untrusted_load_blocked_before_io(regions):
    with patch('cvd_cbd.modeling.joblib.load') as loader:
        with pytest.raises(ValueError,match='arbitrary code'):predict_final('missing.joblib',regions)
        loader.assert_not_called()


def test_range_flags_and_scale(regions,fitted):
    root,_,_,_=fitted;d=regions.copy();d.loc[0,COLUMNS[0]]=100
    p=predict_final(root/'final/final_model.joblib',d,True);assert p.outside_training_feature_range.iloc[0]
    d.loc[0,'um_per_px']=.7
    with pytest.raises(ValueError,match='scale'):predict_final(root/'final/final_model.joblib',d,True)


def test_metric_definitions_and_manual_pairing(fitted):
    assert metrics([1,1],[1,2])['R2'] is None
    m=metrics([0,0],[1,2]);assert m['MAPE_pct'] is None and m['n_relative']==0
    m=metrics([10,20],[11,18]);assert m['relative_1sigma_pct']==pytest.approx(np.std([10,-10],ddof=1))
    _,oof,_,_=fitted
    manual=oof[['image_id','measurement_id']].copy();manual['thickness_nm_manual']=oof.label_nm;manual['method']='SYNTHETIC';manual['source']='SYNTHETIC'
    r=manual_comparison(oof,manual.iloc[:5]);assert r['n_paired']==5 and r['manual']['RMSE_nm']==0


def test_cli_csv_roundtrip_preserves_range_flags(regions,fitted,tmp_path):
    import subprocess,sys
    root,_,_,_=fitted
    feature_path=tmp_path/'features.csv';regions.to_csv(feature_path,index=False)
    config_path=tmp_path/'config.json';StudyConfig().save(config_path)
    output=tmp_path/'predictions.csv'
    subprocess.run([sys.executable,'-m','cvd_cbd','study','predict','--config',str(config_path),'--features',str(feature_path),'--model',str(root/'final/final_model.joblib'),'--output',str(output),'--trust-model'],check=True,capture_output=True,text=True)
    expected=predict_final(root/'final/final_model.joblib',regions,True)
    actual=pd.read_csv(output,float_precision='round_trip')
    np.testing.assert_array_equal(actual[COLUMNS],expected[COLUMNS])
    np.testing.assert_array_equal(actual.outside_training_feature_range,expected.outside_training_feature_range)
    np.testing.assert_allclose(actual.pred_nm,expected.pred_nm,rtol=0,atol=1e-12)
