from dataclasses import replace
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from cvd_cbd.study_config import StudyConfig
from cvd_cbd.synthetic import generate
from cvd_cbd.manifest import build_manifest,read_table,count_manifest
from cvd_cbd.dataset import build_features
from cvd_cbd.workflow import run

@pytest.fixture(scope='module')
def fixture_root(tmp_path_factory):
    base=tmp_path_factory.mktemp('synthetic_images');cfg=replace(StudyConfig(),ar_ladder=(9,13),mc_repeats=50)
    root=generate(base/'inputs',cfg)
    # Smaller MC uncertainty run for the integration test, not the deliverable.
    p=root/'uncertainty.json';u=json.loads(p.read_text());u['repeats']=20;p.write_text(json.dumps(u))
    return root,cfg,base


def test_disk_tiff_manifest_and_dark_e2e(fixture_root,tmp_path):
    root,cfg,_=fixture_root;m=build_manifest(root,cfg);subset=pd.concat([m[m.kind.eq('trench')].iloc[:2],m[m.kind.eq('flat')].iloc[:1]])
    features=build_features(subset,root,tmp_path/'features',cfg)
    assert len(features)==3 and features.n_tiles.tolist()==[2,2,32]
    qc=pd.read_csv(tmp_path/'features/image_qc.csv');assert qc.status.eq('PASS').all()
    assert np.max(np.abs(qc.channel_y0.dropna()-34))<=2 and np.max(np.abs(qc.channel_y1.dropna()-94))<=2
    d=np.load(tmp_path/'features/corrected'/f'{features.image_id.iloc[0]}.npz')
    r=m.iloc[0];img=np.array(Image.open(root/r.path),float);ref=np.array(Image.open(root/r.ref_path),float);dark=np.array(Image.open(root/r.dark_path),float)
    np.testing.assert_allclose(d['T'][48:80,96:160],((img-dark)/(ref-dark))[48:80,96:160],rtol=1e-6)
    assert (tmp_path/'features/overlays'/f'{r.image_id}.png').is_file()
    assert all(not Path(p).is_absolute() for p in features.path)
    with pytest.raises(FileExistsError):build_features(subset,root,tmp_path/'features',cfg)

@pytest.mark.parametrize('fault',['duplicate_pixels','filename','session','no_dark','scale','layout','label','exposure','flat_condition','run_count'])
def test_manifest_rejections(fixture_root,tmp_path,fault):
    original,cfg,_=fixture_root;root=tmp_path/'inputs';shutil.copytree(original,root)
    images=read_table(root,'images')
    if fault=='duplicate_pixels':shutil.copyfile(root/images.path.iloc[0],root/images.path.iloc[1])
    elif fault=='filename':images.loc[0,'slot']=999
    elif fault=='session':images.loc[0,'session_id']='different'
    elif fault=='no_dark':images.loc[0,'dark_id']='missing'
    elif fault=='scale':images.loc[0,'scale_source']='nominal_width'
    elif fault=='exposure':images.loc[0,'exposure_ms']=99
    elif fault=='label':images.loc[0,'measurement_id']='missing'
    elif fault=='flat_condition':images.loc[images.kind.eq('flat'),'condition_id']='70C_G0_mass_pct'
    elif fault=='run_count':
        runs=read_table(root,'runs');runs['cycles']=1.5;runs.to_csv(root/'meta/runs.csv',index=False)
    elif fault=='layout':
        layout=read_table(root,'layout');layout.loc[0,'L_um']=1;layout.to_csv(root/'meta/layout.csv',index=False)
    images.to_csv(root/'meta/images.csv',index=False)
    with pytest.raises(ValueError):build_manifest(root,cfg)


def test_full_pipeline_and_experimental_status(fixture_root):
    root,cfg,base=fixture_root;r=run(root,base/'results',cfg)
    assert r['status']=='PASS_SOFTWARE_PIPELINE' and r['counts']['trenches_observed']==24
    assert r['real_image_check'].startswith('NOT RUN')
    points=pd.read_csv(base/'results/sc_points_fesem.csv');assert len(points)==12 and points.n_valid.eq(2).all()
    assert (base/'results/process_window/process_map.png').is_file()
    assert (base/'results/similitude_cross_fitted/SC_phi.png').is_file()
    assert (base/'results/final_model/final_model.joblib').is_file()
    assert (base/'results/evaluation/fold_models/fold_0_RF.joblib').is_file()
    report=json.loads((base/'results/similitude_fesem/report.json').read_text());assert report['fitted_parameters']==0 and 'common physical generator' in report['scientific_status']


def test_explicit_exclusions_and_seed_reproducibility(fixture_root,tmp_path):
    root,cfg,_=fixture_root;copyroot=generate(tmp_path/'repeat',cfg)
    for path in ['meta/images.csv','meta/fesem.csv','raw/om/R01_C1_T1_top.tif']:
        assert (root/path).read_bytes()==(copyroot/path).read_bytes()
    images=read_table(copyroot,'images');images['include']=True;images['exclusion_reason']=''
    images.loc[0,'include']=False;images.loc[0,'exclusion_reason']='SYNTHETIC exclusion test'
    images.to_csv(copyroot/'meta/images.csv',index=False)
    m=build_manifest(copyroot,cfg);assert count_manifest(m)['explicitly_excluded_images']==1
    images.loc[1,'image_id']='../escape';images.to_csv(copyroot/'meta/images.csv',index=False)
    with pytest.raises(ValueError,match='filename-safe'):build_manifest(copyroot,cfg)


def test_stale_exported_manifest_blocked(fixture_root,tmp_path):
    root,cfg,_=fixture_root;m=build_manifest(root,cfg).iloc[:1].copy();m.loc[m.index[0],'exposure_ms']=123
    with pytest.raises(ValueError,match='Stale/edited'):build_features(m,root,tmp_path/'stale',cfg)
