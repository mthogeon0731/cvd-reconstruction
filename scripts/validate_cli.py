"""Exercise public stage CLIs against supplied disk-backed SYNTHETIC data."""
import argparse,json,subprocess,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--features',required=True);p.add_argument('--work',required=True);p.add_argument('--log',required=True);a=p.parse_args()
    work=Path(a.work)
    if work.exists():raise SystemExit('Choose a new CLI smoke directory')
    cfg='configs/synthetic_study.json';prefix=[sys.executable,'-m','cvd_cbd','study'];commands=[]
    def add(name,*rest,expected=0):commands.append((name,['--config',cfg,*rest],expected))
    add('validate','--root',a.data)
    add('calibrate','--root',a.data,'--out',str(work/'01'))
    add('manifest','--root',a.data,'--out',str(work/'02'))
    add('design','--calibration',str(work/'01/calibration.csv'),'--out',str(work/'00'))
    add('extract','--root',a.data,'--manifest',str(work/'02/manifest.csv'),'--out',str(work/'03'))
    add('compare','--features',str(work/'03/features.csv'),'--manual',str(Path(a.data)/'meta/manual_om.csv'),'--out',str(work/'04'))
    add('fit-final','--features',str(work/'03/features.csv'),'--out',str(work/'05model'))
    add('predict','--features',str(work/'03/features.csv'),'--model',str(work/'05model/final_model.joblib'),'--output',str(work/'blocked.csv'),expected=2)
    add('predict','--features',str(work/'03/features.csv'),'--model',str(work/'05model/final_model.joblib'),'--output',str(work/'predictions.csv'),'--trust-model')
    add('sc','--features',str(work/'04/selected_oof.csv'),'--out',str(work/'05sc'))
    add('similitude','--trenches',str(work/'05sc/trench_sc.csv'),'--calibration',str(work/'01/calibration.csv'),'--out',str(work/'06'))
    add('process-window','--calibration',str(work/'01/calibration.csv'),'--out',str(work/'07'))
    add('uncertainty','--input',str(Path(a.data)/'uncertainty.json'),'--out',str(work/'08'))
    add('calibrate','--root',a.data,'--out',str(work/'01'),expected=2)
    result=[]
    for name,args,expected in commands:
        start=time.monotonic();r=subprocess.run([*prefix,name,*args],text=True,capture_output=True)
        result.append(dict(command='python -m cvd_cbd study '+name+' '+' '.join(args),expected_exit=expected,actual_exit=r.returncode,status='PASS' if r.returncode==expected else 'FAIL',seconds=round(time.monotonic()-start,3),stdout=r.stdout,stderr=r.stderr))
        print(name,result[-1]['status'],flush=True)
    path=Path(a.log);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    if any(r['status']!='PASS' for r in result):return 1
    return 0
if __name__=='__main__':raise SystemExit(main())
