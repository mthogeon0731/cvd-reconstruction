"""Nested condition LOGO model comparison at independent labeled ROI level."""
import json
from .json_utils import read_json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator,RegressorMixin,clone
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneGroupOut,GroupKFold,ParameterGrid
from .metrology import FEATURES
from .thickness import condition_weights
from .io_utils import new_dir,write_csv,write_json,validate_columns
from .input_contracts import validate_pixel_scale,scalar_pixel_scale,reject_axis_pixel_calibration

COLUMNS=['feature_'+s for s in FEATURES]
DEPENDENCIES=['image_id','trench_id','specimen_id','batch_id','measurement_id','content_sha256']

class ScaledRegressor(RegressorMixin,BaseEstimator):
    """Both feature and target transformations fitted on this call's train rows only."""
    def __init__(self,estimator):self.estimator=estimator
    def fit(self,X,y,sample_weight=None):
        self.x_scaler_=StandardScaler().fit(X,sample_weight=sample_weight)
        self.y_scaler_=StandardScaler().fit(np.asarray(y).reshape(-1,1),sample_weight=sample_weight)
        self.model_=clone(self.estimator).fit(self.x_scaler_.transform(X),self.y_scaler_.transform(np.asarray(y).reshape(-1,1)).ravel(),sample_weight=sample_weight)
        return self
    def predict(self,X):
        return self.y_scaler_.inverse_transform(self.model_.predict(self.x_scaler_.transform(X)).reshape(-1,1)).ravel()


def model_zoo(seed):
    return {
        'RF':(RandomForestRegressor(n_estimators=60,random_state=seed,n_jobs=1),{'min_samples_leaf':[1,3]}),
        'KRR':(KernelRidge(kernel='rbf'),{'alpha':[.01,.1],'gamma':[.03,.2]}),
        'SVR':(SVR(kernel='rbf'),{'C':[1.,7.],'epsilon':[.02,.1],'gamma':[.03,.2]}),
    }


def _metric_vector(values,name):
    raw=np.asarray(values)
    if np.iscomplexobj(raw) or (raw.dtype.kind=='O' and any(isinstance(v,(complex,np.complexfloating)) for v in raw.flat)):
        raise ValueError(f'{name}: real metric inputs required; complex values are unsupported')
    a=np.asarray(values,dtype=float)
    if a.ndim==2 and a.shape[1]==1:a=a[:,0]
    if a.ndim!=1 or not np.isfinite(a).all():
        raise ValueError(f'{name}: finite vector of shape (n,) or (n,1) required; got {a.shape}')
    return a


def metrics(true,pred,groups=None):
    """Reject arithmetic overflow as well as nonfinite supplied observations."""
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            result=_metrics_validated(true,pred,groups)
    except FloatingPointError as exc:
        raise ValueError('Metric arithmetic exceeds finite floating-point range') from exc
    if any(isinstance(v,(float,np.floating)) and not np.isfinite(v) for v in result.values()):
        raise ValueError('Metric arithmetic produced a nonfinite result')
    return result


def _metrics_validated(true,pred,groups=None):
    """Paired single-output metrics; empty/constant-target scores stay undefined."""
    y,p=_metric_vector(true,'true'),_metric_vector(pred,'pred')
    if y.shape!=p.shape:raise ValueError('Paired metric inputs require equal lengths')
    n=len(y)
    if groups is not None:
        groups=np.asarray(groups,dtype=object)
        if groups.ndim!=1 or len(groups)!=n:raise ValueError('groups: one-dimensional vector matching metric length required')
        if any(pd.isna(g) or not str(g).strip() or (isinstance(g,(float,int,np.number)) and not np.isfinite(g)) for g in groups):
            raise ValueError('groups: finite nonmissing group IDs required')
    if not n:
        result={'n_measurements':0,'R2':None,'RMSE_nm':None,'MAE_nm':None,'MAPE_pct':None,'relative_bias_pct':None,'relative_1sigma_pct':None,'n_relative':0,'ddof':1}
        if groups is not None:result.update(n_conditions=0,condition_balanced_RMSE_nm=None)
        return result
    error=p-y;sst=0. if np.all(y==y[0]) else np.sum((y-y.mean())**2);valid=np.abs(y)>=1e-6;rel=100*error[valid]/y[valid]
    result={'n_measurements':n,'R2':float(1-np.sum(error**2)/sst) if n>=2 and sst>1e-20 else None,'RMSE_nm':float(np.sqrt(np.mean(error**2))),'MAE_nm':float(np.mean(abs(error))),
        'MAPE_pct':float(np.mean(abs(rel))) if len(rel) else None,'relative_bias_pct':float(rel.mean()) if len(rel) else None,'relative_1sigma_pct':float(rel.std(ddof=1)) if len(rel)>=2 else None,'n_relative':int(valid.sum()),'ddof':1}
    if groups is not None:
        weights=condition_weights(groups);result['n_conditions']=len(np.unique(groups));result['condition_balanced_RMSE_nm']=float(np.sqrt(np.average(error**2,weights=weights)))
    return result


def validate_features(data,origin):
    validate_columns(data,'model features');reject_axis_pixel_calibration(data.columns)
    required=set(COLUMNS+DEPENDENCIES+['matched_region_id','label_nm','condition_id','um_per_px','origin','match_basis'])
    if required-set(data):raise ValueError(f'Missing model columns: {sorted(required-set(data))}')
    if data.empty:raise ValueError('No labeled regions')
    if not data.origin.eq(origin).all():raise ValueError('Origin mismatch')
    if not data.match_basis.eq('validated_region_average').all():raise ValueError('ROI label scope must match optical support')
    for k in DEPENDENCIES+['condition_id','matched_region_id']:
        if data[k].isna().any() or data[k].astype(str).str.strip().eq('').any():raise ValueError(f'Missing dependency ID: {k}')
        if data.groupby(k).condition_id.nunique().max()>1:raise ValueError(f'{k} crosses condition groups; independent condition LOGO unavailable')
    for k in ('measurement_id','matched_region_id','image_id','content_sha256'):
        if data[k].duplicated().any():raise ValueError(f'Duplicate {k}: aggregate repeat acquisitions/labels before fitting')
    scales=validate_pixel_scale(data.um_per_px.to_numpy(dtype=object),'um_per_px pixel scale')
    if not np.isfinite(data[COLUMNS+['label_nm']].to_numpy(float)).all() or (data.label_nm<=0).any():raise ValueError('Missing/nonpositive labels or invalid features')
    if not np.allclose(scales,scales[0],rtol=.01,atol=0):raise ValueError('Pixel scale mismatch; resample with verified calibration first')


def audit_split(data,train,test):
    for key in ['condition_id',*DEPENDENCIES]:
        overlap=set(data.iloc[train][key])&set(data.iloc[test][key])
        if overlap:raise ValueError(f'Split leakage in {key}: {sorted(overlap)[:3]}')


def select_inner(data,cfg,scope):
    X=data[COLUMNS].to_numpy(float);y=data.label_nm.to_numpy(float);groups=data.condition_id.to_numpy()
    k=min(cfg.inner_splits,len(np.unique(groups)))
    if k<2:raise ValueError('At least two inner conditions required')
    splits=list(GroupKFold(n_splits=k).split(X,y,groups))
    for tr,va in splits:audit_split(data,tr,va)
    candidates=[];winners={};assign=[]
    for fold,(tr,va) in enumerate(splits):
        for i in va:assign.append(dict(scope=scope,inner_fold=fold,measurement_id=data.iloc[i].measurement_id,condition_id=groups[i],role='validation'))
    for name,(base,grid) in model_zoo(cfg.seed).items():
        best=None
        for params in ParameterGrid(grid):
            pred=np.empty(len(y))
            for tr,va in splits:
                model=ScaledRegressor(clone(base).set_params(**params)).fit(X[tr],y[tr],condition_weights(groups[tr]));pred[va]=model.predict(X[va])
            score=metrics(y,pred,groups)['condition_balanced_RMSE_nm']
            candidates.append(dict(scope=scope,model=name,parameters=json.dumps(params,sort_keys=True),inner_condition_RMSE_nm=score))
            if best is None or score<best['score']:best=dict(name=name,score=score,params=params,estimator=clone(base).set_params(**params))
        winners[name]=best
    chosen=min(winners.values(),key=lambda d:d['score'])
    return winners,chosen,candidates,assign


def bootstrap_groups(data,pred_col='pred_nm',seed=0,repeats=500):
    groups=data.condition_id.unique()
    if len(groups)<2:return {'status':'NOT_RUN_fewer_than_two_conditions'}
    rng=np.random.default_rng(seed);scores=[]
    for _ in range(repeats):
        d=pd.concat([data[data.condition_id.eq(g)] for g in rng.choice(groups,len(groups),replace=True)])
        scores.append(float(np.sqrt(np.mean((d[pred_col]-d.label_nm)**2))))
    return {'status':'PASS','unit':'condition cluster','repeats':repeats,'seed':seed,'RMSE_nm_percentile95':np.quantile(scores,[.025,.975]).tolist(),'note':'Conditional resampling of existing cross-fitted predictions; not nested model refit uncertainty'}


def compare(data,out,cfg):
    validate_columns(data,'model features')
    data=data[data.label_nm.notna() & data.kind.eq('trench')].reset_index(drop=True)
    validate_features(data,cfg.origin)
    if data.condition_id.nunique()<3:raise ValueError('Nested LOGO needs >=3 independent conditions; sample adequacy still requires review')
    out=new_dir(out);(out/'fold_models').mkdir();predictions=[];candidates=[];assign=[];splits=[]
    X=data[COLUMNS].to_numpy(float);y=data.label_nm.to_numpy(float);groups=data.condition_id.to_numpy()
    for fold,(tr,te) in enumerate(LeaveOneGroupOut().split(X,y,groups)):
        audit_split(data,tr,te)
        winners,chosen,cs,ins=select_inner(data.iloc[tr].reset_index(drop=True),cfg,f'outer_{fold}')
        candidates+=cs;assign+=ins
        for i in range(len(data)):splits.append(dict(outer_fold=fold,image_id=data.iloc[i].image_id,measurement_id=data.iloc[i].measurement_id,condition_id=groups[i],role='test' if i in set(te) else 'train'))
        for name,selection in winners.items():
            model=ScaledRegressor(selection['estimator']).fit(X[tr],y[tr],condition_weights(groups[tr]))
            p=model.predict(X[te]);d=data.iloc[te].copy();d['pred_nm']=p;d['model']=name;d['outer_fold']=fold;d['prediction_role']='cross_fitted';d['selected_by_inner']=name==chosen['name'];d['negative_prediction']=p<0
            d['outside_training_feature_range']=((X[te]<X[tr].min(axis=0))|(X[te]>X[tr].max(axis=0))).any(axis=1)
            predictions.append(d)
            joblib.dump({'model':model,'columns':COLUMNS,'role':'outer_evaluation','train_conditions':sorted(set(groups[tr])),'held_out_conditions':sorted(set(groups[te])),'sklearn_version':sklearn.__version__},out/'fold_models'/f'fold_{fold}_{name}.joblib',compress=3)
    allp=pd.concat(predictions,ignore_index=True);selected=allp[allp.selected_by_inner].copy()
    report={'origin':cfg.origin,'evaluation_unit':'one independent registered FE-SEM ROI; tiles are not observations','selection_rule':'condition-balanced inner CV RMSE; model family selected within each outer training fold','SVR_units':'C and epsilon in fold-standardized target units; original legacy grids were in nm','per_model':{name:metrics(d.label_nm,d.pred_nm,d.condition_id) for name,d in allp.groupby('model')},'nested_selection':metrics(selected.label_nm,selected.pred_nm,selected.condition_id),'bootstrap':bootstrap_groups(selected,seed=cfg.seed),'scientific_status':'software synthetic validation only' if cfg.origin=='SYNTHETIC' else 'cross-fitted condition evaluation, subject to acquisition independence'}
    write_csv(out/'all_model_oof.csv',allp);write_csv(out/'selected_oof.csv',selected);write_csv(out/'candidates.csv',pd.DataFrame(candidates));write_csv(out/'outer_splits.csv',pd.DataFrame(splits));write_csv(out/'inner_splits.csv',pd.DataFrame(assign));write_json(out/'metrics.json',report)
    return selected,report


def fit_final(data,out,cfg):
    validate_columns(data,'model features')
    data=data[data.label_nm.notna() & data.kind.eq('trench')].reset_index(drop=True);validate_features(data,cfg.origin)
    winners,chosen,cs,assign=select_inner(data,cfg,'final_selection')
    out=new_dir(out);X=data[COLUMNS].to_numpy(float);y=data.label_nm.to_numpy(float)
    model=ScaledRegressor(chosen['estimator']).fit(X,y,condition_weights(data.condition_id))
    metadata={'role':'final_all_labels','origin':cfg.origin,'sklearn_version':sklearn.__version__,'model_name':chosen['name'],'parameters':chosen['params'],'train_measurement_ids':list(data.measurement_id),'train_conditions':sorted(data.condition_id.unique()),'feature_min':X.min(axis=0).tolist(),'feature_max':X.max(axis=0).tolist(),'pixel_size_um':float(data.um_per_px.iloc[0]),'label_range_nm':[float(y.min()),float(y.max())],'columns':COLUMNS,'performance':'No independent performance is computed from final training re-predictions'}
    joblib.dump({'model':model,**metadata},out/'final_model.joblib',compress=3);write_json(out/'model_metadata.json',metadata);write_csv(out/'selection.csv',pd.DataFrame(cs));write_csv(out/'inner_splits.csv',pd.DataFrame(assign))
    return metadata


def predict_final(model_path,data,trust_model=False):
    if not trust_model:raise ValueError('Untrusted model load blocked: joblib can execute arbitrary code; explicit --trust-model required')
    meta=read_json(Path(model_path).parent/'model_metadata.json')
    reject_axis_pixel_calibration(meta,'saved model metadata')
    metadata_scale=scalar_pixel_scale(meta['pixel_size_um'],'metadata pixel_size_um')
    if meta['sklearn_version']!=sklearn.__version__:raise ValueError('Model sklearn version mismatch; recreate model')
    bundle=joblib.load(model_path)
    reject_axis_pixel_calibration(bundle,'saved model bundle')
    model_scale=scalar_pixel_scale(bundle['pixel_size_um'],'bundle pixel_size_um')
    if metadata_scale!=model_scale:raise ValueError('Saved model pixel scale differs from metadata')
    if bundle['role']!='final_all_labels':raise ValueError('Final inference requires an all-label final model, not a fold model')
    validate_columns(data,'prediction features');reject_axis_pixel_calibration(data.columns)
    scales=validate_pixel_scale(data.um_per_px.to_numpy(dtype=object),'prediction um_per_px scale')
    if not data.origin.eq(bundle['origin']).all():raise ValueError('Model/data origin mismatch')
    if not np.allclose(scales,model_scale,rtol=.01,atol=0):raise ValueError('Pixel scale mismatch')
    X=data[bundle['columns']].to_numpy(float)
    if not np.isfinite(X).all():raise ValueError('Nonfinite prediction features')
    out=data.copy();p=bundle['model'].predict(X);out['pred_nm']=p;out['negative_prediction']=p<0
    out['outside_training_feature_range']=((X<bundle['feature_min'])|(X>bundle['feature_max'])).any(axis=1)
    out['outside_training_target_range']=(p<bundle['label_range_nm'][0])|(p>bundle['label_range_nm'][1])
    out['prediction_role']=np.where(out.measurement_id.isin(bundle['train_measurement_ids']),'training_reprediction','unverified_final_inference')
    return out


def manual_comparison(oof,manual):
    cols=['image_id','measurement_id']
    if manual.duplicated(cols).any():raise ValueError('Duplicate manual measurement match')
    d=oof.merge(manual,on=cols,how='inner',validate='one_to_one')
    if d.empty:return {'status':'NOT_RUN','reason':'No paired manual observations'}
    if d[['method','source']].isna().any().any():raise ValueError('Manual method/source required')
    return {'status':'PASS','n_paired':len(d),'definition':'relative error SD, ddof=1, identical held-out labels','model':metrics(d.label_nm,d.pred_nm,d.condition_id),'manual':metrics(d.label_nm,d.thickness_nm_manual,d.condition_id)}
