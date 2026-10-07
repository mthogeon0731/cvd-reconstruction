"""Duplicate-key regression: exercise existing public APIs, real models and CLIs."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import pytest

from cvd_cbd.config import Config
from cvd_cbd.study_config import StudyConfig
from cvd_cbd.modeling import fit_final, predict_final
from cvd_cbd.io_utils import read_csv

PROJECT=Path(__file__).resolve().parents[1]
@pytest.fixture(scope='module')
def fresh_data(tmp_path_factory):
    """Regenerate all fixture bytes from reviewed code, never historical outputs."""
    from cvd_cbd.synthetic import generate
    from cvd_cbd.workflow import run
    cfg=StudyConfig.load(PROJECT/'configs/synthetic_study.json')
    base=tmp_path_factory.mktemp('json_synthetic')
    inputs=generate(base/'inputs',cfg)
    results=base/'results'
    run(inputs,results,cfg)
    return inputs,results

def snapshot(root):
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file()}

def cli(*args):
    return subprocess.run([sys.executable,'-B','-m','cvd_cbd',*map(str,args)],capture_output=True,text=True,encoding='utf-8',env=os.environ.copy())

def inject(text,key,value):
    """Insert an earlier occurrence; the baseline silently accepts the valid last one."""
    return text.replace(json.dumps(key)+':',json.dumps(key)+':'+value+','+json.dumps(key)+':',1)

@pytest.mark.parametrize('kind',['study','legacy'])
@pytest.mark.parametrize('escaped',[False,True])
def test_core_config_duplicate_rejected(tmp_path,kind,escaped):
    name='synthetic_study.json' if kind=='study' else 'reference_demo.json'
    data=(Path.cwd()/'configs'/name).read_text(encoding='utf-8')
    key='origin' if kind=='study' else 'mode'
    token='"'+('\\u'+format(ord(key[0]),'04x')+key[1:] if escaped else key)+'"'
    data=data.replace(json.dumps(key)+':',token+':"SECRET_FIRST_VALUE",'+json.dumps(key)+':',1)
    path=tmp_path/'config.json';path.write_text(data,encoding='utf-8');before=snapshot(tmp_path)
    with pytest.raises(ValueError,match='Duplicate JSON key') as error:
        (StudyConfig if kind=='study' else Config).load(path)
    assert 'SECRET_FIRST_VALUE' not in str(error.value)
    assert str(path) in str(error.value)
    assert snapshot(tmp_path)==before

def test_core_uncertainty_cli_rejects_duplicate_without_output(tmp_path,fresh_data):
    text=(fresh_data[0]/'uncertainty.json').read_text(encoding='utf-8')
    path=tmp_path/'uncertainty.json';path.write_text(inject(text,'repeats','1'),encoding='utf-8')
    (tmp_path/'keep.bin').write_bytes(b'preserve existing sibling\x00');before=snapshot(tmp_path)
    r=cli('study','uncertainty','--config','configs/synthetic_study.json','--input',path,'--out',tmp_path/'out')
    assert r.returncode==2,r.stdout+r.stderr
    assert 'Duplicate JSON key' in r.stderr
    assert snapshot(tmp_path)==before
    assert not (tmp_path/'out').exists()

def test_core_workflow_rejects_duplicate_before_any_results(tmp_path,fresh_data):
    root=tmp_path/'inputs';shutil.copytree(fresh_data[0],root)
    path=root/'uncertainty.json';path.write_text(inject(path.read_text(encoding='utf-8'),'repeats','1'),encoding='utf-8')
    (tmp_path/'keep.bin').write_bytes(b'preserve existing sibling\x00');before=snapshot(tmp_path)
    r=cli('study','run','--config','configs/synthetic_study.json','--root',root,'--out',tmp_path/'out')
    assert r.returncode==2,r.stdout+r.stderr
    assert 'Duplicate JSON key' in r.stderr
    assert snapshot(tmp_path)==before
    assert not (tmp_path/'out').exists()

@pytest.fixture(scope='module')
def fresh_model(tmp_path_factory,fresh_data):
    # Fit a real model in this run; never deserialize the historical bundled joblib.
    data=read_csv(fresh_data[1]/'features/features.csv')
    root=tmp_path_factory.mktemp('json_model');fit_final(data,root/'model',StudyConfig())
    return root/'model',data

def test_core_saved_metadata_duplicate_rejected(tmp_path,fresh_model):
    model,data=fresh_model;target=tmp_path/'model';shutil.copytree(model,target)
    path=target/'model_metadata.json';path.write_text(inject(path.read_text(encoding='utf-8'),'pixel_size_um','123'),encoding='utf-8')
    before=snapshot(tmp_path)
    with pytest.raises(ValueError,match='Duplicate JSON key'):
        predict_final(target/'final_model.joblib',data,True)
    assert snapshot(tmp_path)==before

@pytest.mark.parametrize('text',[
    '{"k":1,"k":2}',
    '{"a":{"b":{"k":1,"k":2}}}',
    '{"a":[0,{"k":1,"k":2}]}',
    '{"key":1,"\\u006bey":2}',
    '{"a":[{"b":[{"secret_key":1,"secret_key":2}]}]}',
])
def test_common_rejects_every_object_depth(tmp_path,text):
    from cvd_cbd.json_utils import loads_json,read_json
    path=tmp_path/'input.json';path.write_text(text,encoding='utf-8')
    for parse in (lambda:loads_json(text,source='test input'),lambda:read_json(path)):
        with pytest.raises(ValueError,match='Duplicate JSON key') as error:parse()
        assert 'secret_key' not in str(error.value)
        assert text not in str(error.value)

@pytest.mark.parametrize('text',[
    'null','true','false','123456789012345678901234567890','1.2345678901234567',
    '"한글"','[]','{}','[1,2.5,false,null,"x"]',
    '{"a":{"k":1},"b":{"k":2},"c":[{"k":3},{"k":4}]}',
    '{"A":1,"a":2,"a ":3,"é":4,"e\\u0301":5}',
])
def test_common_preserves_standard_types_and_values(tmp_path,text):
    from cvd_cbd.json_utils import loads_json,read_json
    expected=json.loads(text);path=tmp_path/'normal.json';path.write_text(text,encoding='utf-8')
    for actual in (loads_json(text),read_json(path,encoding='utf-8')):
        assert actual==expected and type(actual) is type(expected)

@pytest.mark.parametrize('payload',[
    '{"secret":1,"secret":2}',
    '{"deep":{"a":{"secret":1,"secret":2}}}',
    '{"array":[{"secret":1,"secret":2}]}',
    '{"secret":1,"\\u0073ecret":2}',
])
def test_metadata_nested_duplicate_precedes_deserialization(tmp_path,fresh_model,payload,monkeypatch):
    model,data=fresh_model;target=tmp_path/'model';shutil.copytree(model,target)
    path=target/'model_metadata.json';path.write_text('{"extra":'+payload+','+path.read_text(encoding='utf-8')[1:],encoding='utf-8')
    def forbidden(*a,**k):pytest.fail('Duplicate metadata must fail before joblib.load')
    monkeypatch.setattr('cvd_cbd.modeling.joblib.load',forbidden)
    with pytest.raises(ValueError,match='Duplicate JSON key'):predict_final(target/'final_model.joblib',data,True)

@pytest.mark.parametrize('state',['new','existing'])
@pytest.mark.parametrize('route',['study','legacy','uncertainty','run','predict','similitude'])
def test_cli_rejection_preserves_files_by_hash(tmp_path,fresh_model,fresh_data,route,state,record_property):
    model,data=fresh_model;out=tmp_path/'out';args=[]
    if state=='existing':out.mkdir();(out/'keep.bin').write_bytes(b'do not change or delete\x00')
    if route in ('study','legacy'):
        name='synthetic_study.json' if route=='study' else 'reference_demo.json'
        path=tmp_path/'config.json';path.write_text(inject((Path.cwd()/'configs'/name).read_text(encoding='utf-8'),'seed','1'),encoding='utf-8')
        args=(['study','generate','--root',out] if route=='study' else ['demo','--out',out,'--no-plots'])+['--config',path]
    elif route in ('uncertainty','run'):
        root=tmp_path/'inputs';shutil.copytree(fresh_data[0],root);path=root/'uncertainty.json'
        path.write_text(inject(path.read_text(encoding='utf-8'),'w_m','1'),encoding='utf-8')
        args=['study',route,'--config','configs/synthetic_study.json','--out',out]+(['--root',root] if route=='run' else ['--input',path])
    elif route=='predict':
        target=tmp_path/'model';shutil.copytree(model,target);path=target/'model_metadata.json'
        path.write_text(inject(path.read_text(encoding='utf-8'),'pixel_size_um','1'),encoding='utf-8')
        features=tmp_path/'features.csv';data.to_csv(features,index=False)
        args=['study','predict','--config','configs/synthetic_study.json','--features',features,'--model',target/'final_model.joblib','--trust-model','--output',out/'pred.csv']
    else:
        d=read_csv(fresh_data[1]/'trench_sc_cross_fitted.csv')
        d.loc[d.status.eq('PASS'),'top_observed_intervals_um']='[{"SECRET_KEY":1,"SECRET_KEY":2}]'
        path=tmp_path/'trenches.csv';d.to_csv(path,index=False)
        args=['study','similitude','--config','configs/synthetic_study.json','--trenches',path,'--calibration',fresh_data[1]/'calibration.csv','--out',out]
    (tmp_path/'sibling.txt').write_text('retain me',encoding='utf-8');before=snapshot(tmp_path)
    r=cli(*args)
    record_property('before_sha256',json.dumps(before,sort_keys=True))
    record_property('after_sha256',json.dumps(snapshot(tmp_path),sort_keys=True))
    record_property('exit_code',r.returncode)
    record_property('stderr',r.stderr)
    assert r.returncode==2,r.stdout+r.stderr
    assert 'Duplicate JSON key' in r.stderr,r.stderr
    assert 'SECRET_KEY' not in r.stderr and 'Traceback' not in r.stderr
    assert snapshot(tmp_path)==before
    if state=='new':assert not out.exists()

def test_normal_existing_loaders_and_real_model(fresh_model):
    cfg=StudyConfig.load('configs/synthetic_study.json');legacy=Config.load('configs/reference_demo.json')
    assert cfg.origin=='SYNTHETIC' and legacy.mode=='uncalibrated_reference_demo'
    model,data=fresh_model
    result=predict_final(model/'final_model.joblib',data,True)
    assert len(result)==len(data) and result.pred_nm.notna().all()

def test_metadata_allows_same_keys_in_separate_objects(tmp_path,fresh_model):
    model,data=fresh_model;target=tmp_path/'model';shutil.copytree(model,target)
    path=target/'model_metadata.json'
    path.write_text('{"extra":{"left":{"k":1},"right":{"k":2},"array":[{"k":3},{"k":4}]},'+path.read_text(encoding='utf-8')[1:],encoding='utf-8')
    pd.testing.assert_frame_equal(predict_final(target/'final_model.joblib',data,True),predict_final(model/'final_model.joblib',data,True))

def script(path,*args):
    return subprocess.run([sys.executable,'-B',str(path),*map(str,args)],capture_output=True,text=True,encoding='utf-8',env=os.environ.copy())

def test_report_builder_rejects_duplicate_before_rewriting_report(tmp_path):
    path=tmp_path/'scripts/build_validation_report.py';path.parent.mkdir()
    shutil.copy2(PROJECT/'scripts/build_validation_report.py',path)
    source=tmp_path/'outputs/SYNTHETIC_delivery/SYNTHETIC_results/run_manifest.json';source.parent.mkdir(parents=True)
    source.write_text('{"SECRET_KEY":1,"SECRET_KEY":2}',encoding='utf-8')
    (tmp_path/'validation_report.md').write_text('preserve prior report',encoding='utf-8')
    before=snapshot(tmp_path);r=script(path)
    assert r.returncode!=0 and 'Duplicate JSON key' in r.stderr
    assert 'SECRET_KEY' not in r.stderr and snapshot(tmp_path)==before

def test_comparison_policy_rejects_duplicate_without_outputs(tmp_path):
    path=tmp_path/'verification/compare_runs.py';path.parent.mkdir()
    shutil.copy2(PROJECT/'verification/compare_runs.py',path)
    policy=path.parent/'reports/comparison_policy.json';policy.parent.mkdir()
    policy.write_text('{"rtol":0,"rtol":1e-8,"atol":1e-10}',encoding='utf-8')
    before=snapshot(tmp_path);r=script(path,'--before',tmp_path/'old','--after',tmp_path/'new','--out',tmp_path/'out')
    assert r.returncode==2 and 'Duplicate JSON key' in r.stderr
    assert snapshot(tmp_path)==before and not (tmp_path/'out').exists()

@pytest.mark.parametrize('side',['before','after'])
def test_comparison_csv_json_cannot_swallow_duplicate(side):
    spec=importlib.util.spec_from_file_location('comparison_under_test',PROJECT/'verification/compare_runs.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    left=pd.Series(['{"k":1,"k":2}' if side=='before' else '{"k":2}'])
    right=pd.Series(['{"k":1,"k":2}' if side=='after' else '{"k":2}'])
    with pytest.raises(ValueError,match='Duplicate JSON key'):
        mod.compare_column('parameters',left,right,left,right,1e-8,1e-10)

@pytest.mark.parametrize('target',['input_provenance.json','features/input_provenance.json','run_manifest.json'])
def test_verify_run_rejects_duplicate_without_output(tmp_path,fresh_model,target):
    model,data=fresh_model;results=tmp_path/'run/SYNTHETIC_results';(results/'features').mkdir(parents=True)
    shutil.copytree(model,results/'final_model')
    data.to_csv(results/'features/features.csv',index=False)
    predict_final(model/'final_model.joblib',data,True).to_csv(results/'predictions_final.csv',index=False)
    from cvd_cbd.manifest import config_snapshot
    roles=['image_pixels','reference_correction','dark_correction','scale_evidence','configuration_source','growth_calibration','diffusion_calibration','manual_comparison','uncertainty_specification']
    marker=tmp_path/'marker.txt';marker.write_text('fixture',encoding='utf-8')
    provenance={'files':[dict(path=str(marker),path_kind='absolute',sha256=hashlib.sha256(marker.read_bytes()).hexdigest(),role=role) for role in roles],'effective_config':config_snapshot(StudyConfig())}
    for name in ('input_provenance.json','features/input_provenance.json'):
        (results/name).write_text(json.dumps(provenance),encoding='utf-8')
    (results/'run_manifest.json').write_text('{}',encoding='utf-8')
    (results/target).write_text('{"SECRET_KEY":1,"SECRET_KEY":2}',encoding='utf-8')
    before=snapshot(tmp_path);r=script(PROJECT/'verification/verify_run.py','--run',tmp_path/'run','--out',tmp_path/'result.json')
    assert r.returncode!=0 and 'Duplicate JSON key' in r.stderr,r.stderr
    assert 'SECRET_KEY' not in r.stderr and snapshot(tmp_path)==before

def test_malformed_json_reports_location_without_content(tmp_path):
    from cvd_cbd.json_utils import read_json
    path=tmp_path/'broken.json';path.write_text('{"secret_value":"PRIVATE"\n missing}',encoding='utf-8')
    with pytest.raises(ValueError,match='line 2, column 2') as error:read_json(path)
    assert str(path) in str(error.value) and 'PRIVATE' not in str(error.value)
