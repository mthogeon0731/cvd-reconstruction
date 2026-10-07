"""CLI for the calibrated local CVD/CBD workflow."""
import argparse
import json
from pathlib import Path
import pandas as pd
from .study_config import StudyConfig
from .json_utils import read_json
from .io_utils import new_dir,write_csv,write_json,read_csv
from .manifest import build_manifest,read_table,count_manifest,input_provenance,file_sha256,config_snapshot,verify_provenance
from .calibration import fit_growth,calibration_table
from .dataset import build_features
from .modeling import compare,fit_final,predict_final,manual_comparison
from .infer import pair_sc
from .similitude import compare_similitude,process_window
from .study_design import run_design
from .uncertainty import propagate
from .synthetic import generate
from .workflow import run


def _stage_inputs(args,cfg):
    """Snapshot file inputs for standalone stages; persist only after success."""
    cmd=args.command
    paths=[(args.config,'configuration_source','validated CLI settings; effective snapshot recorded')]
    if cmd=='calibrate':
        paths += [(Path(args.root)/'meta/calibration_thickness.csv','growth_calibration','numerical_input'),
                  (Path(args.root)/'meta/viscosity.csv','diffusion_calibration','numerical_input')]
    if cmd in ('compare','fit-final','predict','sc'):paths.append((args.features,'features_table','numerical_input'))
    if cmd=='compare' and args.manual:paths.append((args.manual,'manual_comparison','numerical_input'))
    if cmd in ('similitude','process-window','design'):paths.append((args.calibration,'calibrated_properties','numerical_input'))
    if cmd=='similitude':paths.append((args.trenches,'trench_observations','numerical_input'))
    if cmd=='uncertainty':paths.append((args.input,'uncertainty_specification','numerical_input'))
    if cmd=='predict':
        paths += [(args.model,'trained_model','numerical_input'),(Path(args.model).parent/'model_metadata.json','model_metadata','validation_input')]
    return dict(schema_version=1,stage=cmd,effective_config=config_snapshot(cfg),
        files=[dict(path=str(Path(path).resolve()),path_kind='absolute',role=role,usage=usage,sha256=file_sha256(path)) for path,role,usage in paths],
        note='Only actual stage inputs are listed. Prediction uses saved model calibration; CLI configuration is validated but does not override the model.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='command',required=True)
    commands=['validate','generate','calibrate','manifest','extract','compare','fit-final','predict','sc','similitude','process-window','design','uncertainty','run','demo']
    for name in commands:
        p=subs.add_parser(name);p.add_argument('--config',required=True)
        if name in ['validate','generate','calibrate','manifest','extract','run']:p.add_argument('--root',required=True,help='Data root; paths in CSV are relative to this directory')
        if name not in ['validate','generate','predict']:p.add_argument('--out',required=True,help='New or empty output directory')
        if name=='extract':p.add_argument('--manifest',required=True)
        if name in ['compare','fit-final','predict','sc']:p.add_argument('--features',required=True)
        if name=='compare':p.add_argument('--manual')
        if name=='predict':
            p.add_argument('--model',required=True);p.add_argument('--output',required=True);p.add_argument('--trust-model',action='store_true')
        if name=='sc':p.add_argument('--source',choices=['prediction','fesem'],default='prediction')
        if name in ['similitude','process-window','design']:p.add_argument('--calibration',required=True)
        if name=='similitude':p.add_argument('--trenches',required=True)
        if name=='uncertainty':p.add_argument('--input',required=True)
    args=parser.parse_args(argv)
    try:
        cfg=StudyConfig.load(args.config);cmd=args.command
        read=lambda p:read_csv(p,dtype={'run_id':str,'image_id':str,'measurement_id':str,'condition_id':str,'batch_id':str,'specimen_id':str})
        standalone=cmd in ('calibrate','compare','fit-final','predict','sc','similitude','process-window','design','uncertainty')
        provenance=None;provenance_out=None
        if cmd=='predict':
            if Path(args.output).exists():raise FileExistsError('Output exists; will not overwrite')
            if not args.trust_model:raise ValueError('Untrusted model load blocked: joblib can execute arbitrary code; explicit --trust-model required')
            # Parse first so ambiguous feature headers cannot be hidden by a missing model.
            prediction_data=read(args.features)
        if standalone:
            provenance_out=Path(str(args.output)+'.provenance.json') if cmd=='predict' else Path(args.out)/'input_provenance.json'
            if provenance_out.exists():raise FileExistsError('Input provenance output exists; will not overwrite')
            provenance=_stage_inputs(args,cfg)
        if cmd=='generate':generate(args.root,cfg)
        elif cmd in ('validate','manifest'):
            m=build_manifest(args.root,cfg)
            if cmd=='manifest':
                out=new_dir(args.out);write_csv(out/'manifest.csv',m);write_json(out/'counts.json',count_manifest(m));write_json(out/'excluded_images.json',m.attrs.get('excluded_images',[]))
                write_json(out/'input_provenance.json',input_provenance(m,cfg,args.config))
            print(json.dumps(count_manifest(m),ensure_ascii=False))
        elif cmd=='calibrate':
            growth=fit_growth(read_table(args.root,'calibration_thickness'),cfg);cal=calibration_table(growth,read_table(args.root,'viscosity'),cfg)
            out=new_dir(args.out);write_csv(out/'growth.csv',growth);write_csv(out/'calibration.csv',cal)
        elif cmd=='extract':build_features(read(args.manifest),args.root,args.out,cfg,config_path=args.config)
        elif cmd=='compare':
            oof,_=compare(read(args.features),args.out,cfg)
            if args.manual:write_json(Path(args.out)/'manual_comparison.json',manual_comparison(oof,read(args.manual)))
        elif cmd=='fit-final':fit_final(read(args.features),args.out,cfg)
        elif cmd=='predict':
            if Path(args.output).exists():raise FileExistsError('Output exists; will not overwrite')
            d=predict_final(args.model,prediction_data,args.trust_model);write_csv(args.output,d)
        elif cmd=='sc':
            t,p=pair_sc(read(args.features),cfg,'pred_nm' if args.source=='prediction' else 'label_nm');out=new_dir(args.out);write_csv(out/'trench_sc.csv',t);write_csv(out/'sc_points.csv',p)
        elif cmd=='similitude':compare_similitude(read(args.trenches),read(args.calibration),args.out,cfg)
        elif cmd=='process-window':process_window(read(args.calibration),args.out,cfg)
        elif cmd=='design':run_design(read(args.calibration),args.out,cfg)
        elif cmd=='uncertainty':
            d,r=propagate(read_json(args.input),cfg);out=new_dir(args.out);write_csv(out/'draws.csv',d);write_json(out/'report.json',r)
        elif cmd=='run':print(json.dumps(run(args.root,args.out,cfg,config_path=args.config),ensure_ascii=False,indent=2))
        elif cmd=='demo':
            out=new_dir(args.out);generate(out/'SYNTHETIC_inputs',cfg);print(json.dumps(run(out/'SYNTHETIC_inputs',out/'SYNTHETIC_results',cfg,config_path=args.config),ensure_ascii=False,indent=2))
        if provenance is not None:
            verify_provenance(provenance,Path('.'));write_json(provenance_out,provenance)
        print(f'PASS: {cmd} ({cfg.origin}); see artifacts for scientific interpretation')
        return 0
    except (ValueError,KeyError,OSError,TypeError) as exc:parser.exit(2,f'Input/workflow error: {exc}\n')

if __name__=='__main__':raise SystemExit(main())
