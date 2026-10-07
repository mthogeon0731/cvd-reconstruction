"""Independent regression contracts for ML-02/03/04, defined before fixes."""
import json
from pathlib import Path
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn
from cvd_cbd import io_utils, modeling


@pytest.mark.parametrize('left,right', [('x','x'), (' x','x '), ('x','\tx')])
def test_raw_csv_duplicate_header_rejected_before_pandas(tmp_path, left, right):
    p=tmp_path/'ambiguous.csv'
    p.write_text(f'"{left}","{right}"\n1,100\n',encoding='utf-8')
    with pytest.raises(ValueError,match=r'(?i)duplicate.*header.*1.*2') as error:
        io_utils.read_csv(p)
    assert p.name in str(error.value)


def test_csv_quoted_bom_header_whitespace_and_roundtrip(tmp_path):
    p=tmp_path/'valid.csv';p.write_text('\ufeff" x, y ", z \n0.10000000000000002,2\n',encoding='utf-8')
    d=io_utils.read_csv(p)
    assert list(d)==['x, y','z']
    assert d.iloc[0,0]==0.10000000000000002


@pytest.mark.parametrize('contents',['a,,b\n1,2,3\n','   \n1\n',''])
def test_csv_empty_headers_rejected(tmp_path, contents):
    p=tmp_path/'empty.csv';p.write_text(contents,encoding='utf-8')
    with pytest.raises(ValueError,match='(?i)header'):io_utils.read_csv(p)


def test_cli_duplicate_header_has_location_and_writes_nothing(tmp_path,capsys):
    from cvd_cbd.study_cli import main
    from cvd_cbd.study_config import StudyConfig
    cfg=tmp_path/'config.json';StudyConfig().save(cfg)
    data=tmp_path/'features.csv';data.write_text('feature_od_mean, feature_od_mean \n1,100\n')
    output=tmp_path/'prediction.csv'
    with pytest.raises(SystemExit) as e:
        main(['predict','--config',str(cfg),'--features',str(data),'--model','unused.joblib','--output',str(output),'--trust-model'])
    assert e.value.code==2 and not output.exists()
    error=capsys.readouterr().err
    assert 'Duplicate CSV header' in error and 'features.csv' in error and '1, 2' in error


@pytest.mark.parametrize('y,p',[([1,2],[[1],[2]]),([[1],[2]],[1,2]),([[1],[2]],[[1],[2]])])
def test_metric_column_vectors_are_paired(y,p):
    r=modeling.metrics(y,p)
    assert r['RMSE_nm']==0 and r['R2']==1


@pytest.mark.parametrize('y,p',[(1,1),([1,2],[1]),([[1,2]],[[1,2]]),([[1,2],[3,4]],[[1,2],[3,4]]),([1,np.nan],[1,2]),([1,2],[1,np.inf])])
def test_metric_invalid_shapes_and_finiteness(y,p):
    with pytest.raises(ValueError):modeling.metrics(y,p)


@pytest.mark.parametrize('groups',[['a'], [['a','b']], ['a',None], ['a',' '], [1,np.inf]])
def test_metric_groups_shape_and_missing(groups):
    with pytest.raises(ValueError):modeling.metrics([1,2],[1,2],groups)


def test_empty_and_constant_metrics_are_unevaluated():
    empty=modeling.metrics([],[],[])
    assert empty['n_measurements']==0 and empty['R2'] is None and empty['n_relative']==0
    assert empty['n_conditions']==0 and empty['condition_balanced_RMSE_nm'] is None
    assert modeling.metrics([3,3],[3,3])['R2'] is None


def test_large_constant_target_still_has_undefined_r2():
    result=modeling.metrics([1e308,1e308],[1e308,1e308])
    assert result['R2'] is None and result['RMSE_nm']==0


def test_finite_inputs_cannot_emit_nonfinite_metrics():
    with pytest.raises(ValueError,match='finite'):
        modeling.metrics([1.,2.],[1e308,-1e308])


@pytest.mark.parametrize('values',[np.array([1+100j,2+100j]),np.array([1+0j,2+0j],dtype=object)])
def test_complex_observations_are_not_silently_projected(values):
    with pytest.raises(ValueError,match='real metric'):modeling.metrics(values,[1.,2.])


def minimal_features():
    return pd.DataFrame([dict(kind='trench',image_id='i',trench_id='t',specimen_id='s',batch_id='b',measurement_id='m',
        content_sha256='h',matched_region_id='r',condition_id='c',label_nm=10.,um_per_px=.5,origin='SYNTHETIC',
        match_basis='validated_region_average',**{x:.1 for x in modeling.COLUMNS})])


@pytest.mark.parametrize('scale',[0,-.5,np.nan,np.inf,-np.inf,True,'0','-5','nan','inf'])
def test_invalid_scale_direct_training_contract(scale):
    d=minimal_features();d['um_per_px']=scale
    with pytest.raises(ValueError,match='(?i)scale|um_per_px'):modeling.validate_features(d,'SYNTHETIC')


def test_numeric_string_scale_is_supported():
    d=minimal_features();d['um_per_px']='5e-1'
    modeling.validate_features(d,'SYNTHETIC')


@pytest.mark.parametrize('scale',[0,-1,float('nan'),float('inf'),True])
def test_saved_metadata_scale_rejected_before_model_load(tmp_path,scale):
    (tmp_path/'model_metadata.json').write_text(json.dumps(dict(sklearn_version=sklearn.__version__,pixel_size_um=scale)))
    with patch('cvd_cbd.modeling.joblib.load') as loader:
        with pytest.raises(ValueError,match='(?i)scale|pixel_size'):modeling.predict_final(tmp_path/'model.joblib',minimal_features(),True)
        loader.assert_not_called()


def test_duplicate_dataframe_columns_are_rejected_directly():
    d=minimal_features();d=pd.concat([d,d[[modeling.COLUMNS[0]]]],axis=1)
    with pytest.raises(ValueError,match='(?i)duplicate'):modeling.validate_features(d,'SYNTHETIC')


@pytest.mark.parametrize('where',['features','bundle'])
@pytest.mark.parametrize('scale',[0,-1,np.nan,np.inf,True])
def test_invalid_scale_prediction_entrypoints(tmp_path,where,scale):
    (tmp_path/'model_metadata.json').write_text(json.dumps(dict(sklearn_version=sklearn.__version__,pixel_size_um=.5)))
    d=minimal_features();bundle=dict(pixel_size_um=.5,role='final_all_labels',origin='SYNTHETIC')
    if where=='features':d['um_per_px']=scale
    else:bundle['pixel_size_um']=scale
    with patch('cvd_cbd.modeling.joblib.load',return_value=bundle):
        with pytest.raises(ValueError,match='(?i)scale|pixel_size|calibration'):modeling.predict_final(tmp_path/'model.joblib',d,True)


@pytest.mark.parametrize('field',['um_per_px_x','pixel_size_um_y','x_um_per_px'])
def test_model_rejects_axis_calibration_instead_of_ignoring(field):
    d=minimal_features();d[field]=.6
    with pytest.raises(ValueError,match='axis-specific'):modeling.validate_features(d,'SYNTHETIC')
