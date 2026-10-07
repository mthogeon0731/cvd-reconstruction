"""End-to-end study orchestration with no external service or sharing actions."""
from pathlib import Path
from .json_utils import read_json
import platform
import importlib.metadata
from dataclasses import asdict
from .io_utils import new_dir,write_json,write_csv
from .manifest import build_manifest,read_table,count_manifest,input_provenance,file_record,verify_provenance
from .calibration import fit_growth,calibration_table
from .dataset import build_features
from .modeling import compare,fit_final,predict_final,manual_comparison
from .infer import pair_sc
from .similitude import compare_similitude,process_window
from .study_design import run_design
from .uncertainty import propagate


def run(root,out,cfg,config_path=None):
    uncertainty_path=Path(root)/'uncertainty.json'
    has_uncertainty=uncertainty_path.is_file()
    uncertainty_record=file_record(root,'uncertainty.json','uncertainty_specification','numerical_input') if has_uncertainty else None
    uncertainty_spec=read_json(uncertainty_path) if has_uncertainty else None
    out=new_dir(out)
    manifest=build_manifest(root,cfg)
    provenance=input_provenance(manifest,cfg,config_path)
    for name,role in [('calibration_thickness','growth_calibration'),('viscosity','diffusion_calibration'),('manual_om','manual_comparison')]:
        provenance['files'].append(file_record(root,f'meta/{name}.csv',role,'numerical_table_input'))
    if has_uncertainty:
        provenance['files'].append(uncertainty_record)
    write_json(out/'input_provenance.json',provenance)
    growth=fit_growth(read_table(root,'calibration_thickness'),cfg)
    cal=calibration_table(growth,read_table(root,'viscosity'),cfg)
    write_csv(out/'growth.csv',growth);write_csv(out/'calibration.csv',cal)
    run_design(cal,out/'design',cfg)
    write_csv(out/'manifest.csv',manifest)
    write_json(out/'excluded_images.json',manifest.attrs.get('excluded_images',[]))
    counts=count_manifest(manifest);write_json(out/'counts.json',counts)
    features=build_features(manifest,root,out/'features',cfg,config_path=config_path)
    oof,report=compare(features,out/'evaluation',cfg)
    write_json(out/'evaluation/manual_comparison.json',manual_comparison(oof,read_table(root,'manual_om')))
    final=fit_final(features,out/'final_model',cfg)
    # Artifact created by this call, explicitly trusted; no uploaded pickle is loaded.
    pred=predict_final(out/'final_model/final_model.joblib',features,trust_model=True);write_csv(out/'predictions_final.csv',pred)
    for name,d,value in [('final',pred,'pred_nm'),('cross_fitted',oof,'pred_nm'),('fesem',features,'label_nm')]:
        trenches,points=pair_sc(d,cfg,value)
        write_csv(out/f'trench_sc_{name}.csv',trenches);write_csv(out/f'sc_points_{name}.csv',points)
        if name!='final':compare_similitude(trenches,cal,out/f'similitude_{name}',cfg)
    _,process=process_window(cal,out/'process_window',cfg)
    if has_uncertainty:
        draws,ur=propagate(uncertainty_spec,cfg)
        write_csv(out/'uncertainty_draws.csv',draws);write_json(out/'uncertainty_report.json',ur)
    manifest_report={'origin':cfg.origin,'seed':cfg.seed,'python':platform.python_version(),'OS':platform.platform(),'versions':{p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','scikit-learn','matplotlib','Pillow','joblib']},'config':asdict(cfg),'counts':counts,'tiles':int(features.n_tiles.sum()),'final_model':final['model_name'],'evaluation':report['nested_selection'],'process_status':process['boundary_status'],'real_image_check':'NOT RUN: no real OM data supplied' if cfg.origin=='SYNTHETIC' else 'See per-image QC; physical evidence requires human verification','status':'PASS_SOFTWARE_PIPELINE'}
    verify_provenance(provenance,root)
    manifest_report['input_provenance']=provenance
    write_json(out/'run_manifest.json',manifest_report)
    return manifest_report
