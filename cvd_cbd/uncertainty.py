"""Joint log-parameter Monte Carlo; separates property, metrology and MC errors.

Missing required uncertainty is rejected. Explicit zero covariance represents a
supplied deterministic assumption, not an inferred measurement precision. Only
growth rates for reactive sidewalls or a reactive end are sampled.
"""
from dataclasses import replace
import numpy as np
import pandas as pd
from .physics import Geometry, observed_sc, observation_windows
from .similitude import ar_limit

NAMES=['w_m','h_m','L_m','G_glass_m_s','G_pdms_m_s','eta_Pa_s','D_ref_m2_s','C0_mol_m3','Vm_m3_mol','top_nm','bot_nm']


def _validated_inputs(spec, cfg):
    required={'means','log_covariance','repeats','seed','T_C','model_discrepancy_sd_fraction','assumptions'}
    if not isinstance(spec,dict):raise ValueError('Uncertainty specification must be an object')
    if required-set(spec):raise ValueError(f'Missing uncertainty keys: {sorted(required-set(spec))}')
    if set(spec)-required-{'covariance_order','missing_uncertainty_policy'}:raise ValueError('Unknown uncertainty keys')
    if spec.get('missing_uncertainty_policy','error')!='error':raise ValueError('Missing uncertainty policy must be error; zero imputation is unsupported')
    if not isinstance(spec['means'],dict):raise ValueError('Uncertainty means must be an object')
    geometry=Geometry(cfg.w_um*1e-6,cfg.h_um*1e-6,cfg.w_um*1e-6,cfg.faces,tuple(cfg.wall_materials))
    active=set(geometry.perimeters)
    if cfg.end_material!='inert' and cfg.end_rate_ratio>0:active.add(cfg.end_material)
    unused={'G_'+material+'_m_s' for material in ('glass','pdms') if material not in active}
    names=[name for name in NAMES if name not in unused]
    missing=set(names)-set(spec['means']);unknown=set(spec['means'])-set(NAMES)
    if missing or unknown:raise ValueError(f'Uncertainty means missing required {sorted(missing)} or unknown {sorted(unknown)}')
    if any(isinstance(spec['means'][k],(bool,np.bool_)) for k in names):raise ValueError('Numeric means required; booleans are not measurements')
    # Full historical 11-variable order is supported. Compact/reordered inputs
    # name every covariance axis, so omission cannot silently shift axes.
    order=spec.get('covariance_order',NAMES)
    if not isinstance(order,list) or not all(isinstance(k,str) for k in order) or len(set(order))!=len(order) or set(order)!=set(spec['means']):
        raise ValueError('covariance_order must list each supplied mean exactly once; active-only input requires explicit covariance_order')
    try:
        cov=np.asarray(spec['log_covariance'],float)
        means=np.array([spec['means'][k] for k in names],float)
    except (TypeError,ValueError) as exc:
        raise ValueError('Numeric means and log covariance required') from exc
    if not np.isfinite(means).all() or (means<=0).any():raise ValueError(f'Positive finite means required for {names}')
    if cov.shape!=(len(order),len(order)) or not np.isfinite(cov).all():
        raise ValueError('Finite symmetric positive semidefinite log covariance with nonnegative diagonal required')
    if np.max(abs(cov-cov.T))>1e-12*np.max(abs(cov)) or np.any(np.diag(cov)<0):
        raise ValueError('Finite symmetric positive semidefinite log covariance with nonnegative diagonal required')
    cov=(cov+cov.T)*.5
    eigenvalues=np.linalg.eigvalsh(cov)
    if eigenvalues.min() < -np.finfo(float).eps*len(order)*max(0.,float(eigenvalues.max())):
        raise ValueError('Positive semidefinite log covariance required')
    indices=[order.index(k) for k in names]
    # Marginalize unused variables; do not condition on a fabricated rate, and
    # do not claim their supplied values contributed to this computation.
    used_cov=cov[np.ix_(indices,indices)]
    n=spec['repeats'];seed=spec['seed'];sd=spec['model_discrepancy_sd_fraction'];temperature=spec['T_C']
    if isinstance(n,bool) or not isinstance(n,int) or n<2:raise ValueError('Invalid MC repeats: integer >=2 required')
    if isinstance(seed,bool) or not isinstance(seed,int) or seed<0:raise ValueError('Invalid MC seed: nonnegative integer required')
    for value,name in [(sd,'discrepancy'),(temperature,'T_C')]:
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value):raise ValueError(f'Finite numeric {name} required')
    if sd<0 or temperature<=-273.15:raise ValueError('Nonnegative discrepancy and T_C above absolute zero required')
    if not isinstance(spec['assumptions'],str) or not spec['assumptions'].strip():raise ValueError('Documented uncertainty assumptions required')
    return means,used_cov,names,order,active,unused


def propagate(spec,cfg):
    means,cov,names,input_order,active,unused=_validated_inputs(spec,cfg)
    n=spec['repeats'];seed=spec['seed'];sd=spec['model_discrepancy_sd_fraction']
    rng=np.random.default_rng(seed)
    # Canonical ordering retains seeded default 4-face results and named-order
    # invariance. Match supplied arithmetic means under lognormal sampling.
    eigenvalues,eigenvectors=np.linalg.eigh(cov)
    rank_tolerance=np.finfo(float).eps*len(names)*max(0.,float(eigenvalues.max()))
    singular=eigenvalues.min()<=rank_tolerance
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
        if singular:
            # SVD can introduce ~sqrt(eps) noise in an exactly zero mode, so
            # project numerical null modes to zero using a scale-based bound.
            eigenvalues=np.where(eigenvalues>rank_tolerance,eigenvalues,0.)
            factor=eigenvectors*np.sqrt(eigenvalues)
            log_draws=rng.standard_normal((n,len(names)))@factor.T+np.log(means)-.5*np.diag(cov)
        else:
            log_draws=rng.multivariate_normal(np.log(means)-.5*np.diag(cov),cov,size=n)
        draws=np.exp(log_draws)
    if not np.isfinite(draws).all() or (draws<=0).any():raise ValueError('Uncertainty draws overflow/underflow; narrow supplied log covariance')
    rows=[]
    for i,a in enumerate(draws):
        p=dict(zip(names,a));g=Geometry(p['w_m'],p['h_m'],p['L_m'],cfg.faces,tuple(cfg.wall_materials))
        ks={s:p['G_'+s+'_m_s']/(p['C0_mol_m3']*p['Vm_m3_mol']) for s in active}
        D=p['D_ref_m2_s']*(spec['T_C']+273.15)/(cfg.D_ref_T_C+273.15)*cfg.eta_ref_Pa_s/p['eta_Pa_s']
        end=0. if cfg.end_material=='inert' or cfg.end_rate_ratio==0 else ks[cfg.end_material]*cfg.end_rate_ratio
        windows={} if cfg.map_observable=='endpoints' else dict(zip(['top','bot'],observation_windows(g.L_m,cfg.top_offset_um*1e-6,cfg.window_half_um*1e-6)))
        sc=observed_sc(g,ks,D,end,**windows)
        limit=ar_limit(ks,D,end,replace(cfg,w_um=g.w_m*1e6,h_um=g.h_m*1e6))
        rows.append(dict(draw=i,SC_property=sc,SC_measurement=p['bot_nm']/p['top_nm'],AR_max=limit['value'],AR_max_status=limit['status']))
    d=pd.DataFrame(rows);d['SC_model_plus_assumed_discrepancy']=d.SC_property+rng.normal(0,sd,n)
    if not np.isfinite(d[['SC_property','SC_measurement','SC_model_plus_assumed_discrepancy']].to_numpy(float)).all():raise ValueError('Nonfinite propagated SC values')
    def summary(a):
        a=a.dropna().to_numpy(float)
        if not len(a):return {'n_defined':0,'quantile95':None}
        return {'n_defined':len(a),'mean':float(a.mean()),'sd':float(a.std(ddof=1)) if len(a)>1 else None,'quantile95':np.quantile(a,[.025,.975]).tolist(),'MC_standard_error_of_mean':float(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else None}
    measurement_indices=[names.index(k) for k in ('top_nm','bot_nm')]
    measurement_cov=cov[np.ix_(measurement_indices,measurement_indices)]
    report={'origin':cfg.origin,'seed':seed,'repeats':n,'assumptions':spec['assumptions'],
        'covariance_order':names,'input_covariance_order':input_order,'sampled_parameters':names,
        'parameter_status':{name:'NOT_APPLICABLE' if name in unused else 'USED' for name in NAMES},
        'unused_supplied_parameters':sorted(set(spec['means'])&unused),
        'covariance_sampler':'eigenfactor_with_numerical_null_modes' if singular else 'numpy_multivariate_normal',
        'missing_uncertainty_policy':'error',
        'measurement_uncertainty_status':'SUPPLIED_ZERO_COVARIANCE' if not measurement_cov.any() else 'SUPPLIED_COVARIANCE',
        'property':summary(d.SC_property),'measurement':summary(d.SC_measurement),'AR_max':summary(d.AR_max),
        'model_plus_assumed_discrepancy':summary(d.SC_model_plus_assumed_discrepancy),'discrepancy_sd_fraction':sd,
        'note':'Shared C0/Vm/D reference and full active covariance preserve specified correlations. Unused reaction parameters are NOT_APPLICABLE and are marginalized, not assigned zero uncertainty. Required missing uncertainty is an error; explicit zero covariance is a supplied assumption. These distributions are supplied assumptions, not estimated from synthetic fit residuals. MC standard error is not physical uncertainty; intervals are unvalidated propagation intervals.'}
    return d,report
