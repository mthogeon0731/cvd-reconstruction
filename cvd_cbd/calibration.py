"""Independent planar growth and viscosity calibration; never fitted to trench SC."""
import json
import numpy as np
import pandas as pd
from scipy.stats import t as tdist, f as fdist
from .physics import Geometry, rectangular_coefficients


def require_columns(frame,columns):
    missing=set(columns)-set(frame)
    if missing:raise ValueError(f'Missing columns: {sorted(missing)}')
    if frame.empty:raise ValueError('Input table is empty')


def fit_growth(table,cfg):
    require_columns(table,['T_C','gly_pct','substrate','t_min','thickness_nm','measurement_id','specimen_id','method','source','origin'])
    if not table.substrate.isin(['glass','pdms']).all():raise ValueError('Unknown calibration substrate')
    if table.measurement_id.isna().any() or table.measurement_id.duplicated().any():raise ValueError('Unique independent measurement IDs required')
    if table[['specimen_id','method','source']].isna().any().any():raise ValueError('Missing planar provenance')
    if not table.origin.eq(cfg.origin).all():raise ValueError('Calibration origin mismatch')
    if not np.isfinite(table[['T_C','gly_pct','t_min','thickness_nm']].to_numpy(float)).all():raise ValueError('Nonfinite growth input')
    if (table.thickness_nm<0).any() or (table.t_min<=0).any() or (table.T_C<=-273.15).any() or not table.gly_pct.between(0,100).all():raise ValueError('Invalid growth input units/ranges')
    rows=[]
    for key,d in table.groupby(['T_C','gly_pct','substrate'],sort=True):
        x=d.t_min.to_numpy(float);y=d.thickness_nm.to_numpy(float);n=len(d)
        if len(np.unique(x))<3:raise ValueError('Growth regression needs >=3 distinct times')
        X=np.column_stack([np.ones(n),x]);coef=np.linalg.lstsq(X,y,rcond=None)[0];res=y-X@coef
        inv=np.linalg.inv(X.T@X);df=n-2;cov=inv*float(res@res)/df;se_kind='OLS independent specimen'
        if d.specimen_id.duplicated().any():
            groups=d.specimen_id.unique();m=len(groups)
            if m<3:raise ValueError('Repeated-time specimens require >=3 independent specimen clusters')
            meat=np.zeros((2,2))
            for group in groups:
                ix=d.specimen_id.eq(group).to_numpy();score=X[ix].T@res[ix];meat+=np.outer(score,score)
            cov=inv@meat@inv*m/(m-1)*(n-1)/(n-2);df=m-1;se_kind='cluster robust specimen'
        metrology_status='NOT_PROVIDED_RESIDUAL_ESTIMATE_ONLY'
        if 'thickness_sd_nm' in d:
            try:sd=d.thickness_sd_nm.to_numpy(float)
            except (TypeError,ValueError) as exc:raise ValueError('Nonnegative finite supplied thickness SD required') from exc
            if not np.isfinite(sd).all() or (sd<0).any():raise ValueError('Nonnegative supplied thickness SD required')
            metrology_status='SUPPLIED_ZERO_SD' if not sd.any() else 'SUPPLIED_SD'
            known=inv@(X.T@(sd[:,None]**2*X))@inv
            # Conservative PSD envelope avoids a zero residual fit erasing known metrology error.
            ev,vec=np.linalg.eigh(known-cov);cov=cov+(vec*np.maximum(ev,0))@vec.T
            se_kind+='; supplied independent SD covariance floor'
        se=np.sqrt(np.maximum(np.diag(cov),0));tc=tdist.ppf(.975,df)
        slope=float(coef[1]);intercept=float(coef[0]);hi=intercept+tc*se[0]
        longest_mean=float(y[x==x.max()].mean())
        induction=bool(hi<0 and -intercept>cfg.induction_fraction*longest_mean)
        quadratic_p=None
        if len(np.unique(x))>=4 and n>3:
            Q=np.column_stack([X,(x-x.mean())**2]);qr=y-Q@np.linalg.lstsq(Q,y,rcond=None)[0]
            sse=float(res@res);sseq=float(qr@qr)
            if sseq>1e-20:quadratic_p=float(fdist.sf(max(0,(sse-sseq)/(sseq/(n-3))),1,n-3))
            elif sse>1e-15:quadratic_p=0.
            else:quadratic_p=1.
        concentration_changed=False
        if 'C0_mol_m3' in d and d.C0_mol_m3.notna().any():
            vals=d.C0_mol_m3.to_numpy(float)
            if not np.isfinite(vals).all() or (vals<=0).any():raise ValueError('Invalid concentration record')
            concentration_changed=bool(np.ptp(vals)/np.mean(vals)>.02 or not np.allclose(vals,cfg.C0_mol_m3,rtol=.02))
        flags=[]
        if slope<=np.finfo(float).eps*max(1.,np.max(abs(y)))/np.ptp(x)*100:flags.append('nonpositive_growth')
        if slope-tc*se[1]<=0:flags.append('positive_growth_not_resolved_at_95pct')
        if quadratic_p is not None and quadratic_p<cfg.nonlinear_alpha:flags.append('nonlinear_growth')
        if concentration_changed:flags.append('concentration_change')
        missing=sorted(set([10.,20.,30.,40.])-set(x))
        row=dict(zip(['T_C','gly_pct','substrate'],key))
        row.update(G_nm_min=slope,G_se_nm_min=float(se[1]),intercept_nm=intercept,intercept_se_nm=float(se[0]),df=int(df),n=n,n_times=len(np.unique(x)),
            slope_ci_low=slope-tc*float(se[1]),slope_ci_high=slope+tc*float(se[1]),induction=induction,t_ind_min=-intercept/slope if induction and slope>0 else 0.,
            r2=float(1-res@res/np.sum((y-y.mean())**2)) if np.ptp(y)>0 else None,quadratic_p=quadratic_p,missing_times=json.dumps(missing),flags=';'.join(flags),standard_error=se_kind,
            source_ids='|'.join(sorted(d.measurement_id.astype(str))),specimen_ids='|'.join(sorted(d.specimen_id.astype(str).unique())),origin=cfg.origin,
            metrology_uncertainty_status=metrology_status,missing_uncertainty_policy='error_if_supplied_column')
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_table(growth,viscosity,cfg):
    require_columns(viscosity,['T_C','gly_pct','eta_Pa_s','eta_sd_Pa_s','eta_unit','gly_basis','source','origin'])
    if viscosity.duplicated(['T_C','gly_pct']).any():raise ValueError('Viscosity must have one independently summarized value per condition')
    if not viscosity.eta_unit.eq('Pa_s').all() or not viscosity.gly_basis.eq(cfg.gly_basis).all():raise ValueError('Viscosity/composition unit mismatch')
    if viscosity.source.isna().any() or not viscosity.origin.eq(cfg.origin).all():raise ValueError('Missing/mixed viscosity provenance')
    if not np.isfinite(viscosity[['eta_Pa_s','eta_sd_Pa_s']].to_numpy(float)).all() or (viscosity.eta_Pa_s<=0).any() or (viscosity.eta_sd_Pa_s<0).any():raise ValueError('Invalid viscosity')
    g=Geometry(cfg.w_um*1e-6,cfg.h_um*1e-6,cfg.w_um*1e-6,cfg.faces,tuple(cfg.wall_materials))
    required=set(g.perimeters)
    if cfg.end_material!='inert' and cfg.end_rate_ratio>0:required.add(cfg.end_material)
    rows=[]
    for key,d in growth.groupby(['T_C','gly_pct']):
        if d.substrate.duplicated().any() or required-set(d.substrate):raise ValueError(f'Missing/duplicate substrate at {key}: {required-set(d.substrate)}')
        if not cfg.allow_flagged_growth and d['flags'].fillna('').ne('').any():raise ValueError(f'Growth assumptions failed at {key}; inspect growth diagnostics')
        v=viscosity[(viscosity.T_C==key[0])&(viscosity.gly_pct==key[1])]
        if len(v)!=1:raise ValueError(f'Missing viscosity at {key}')
        v=v.iloc[0];rates={};r=dict(T_C=key[0],gly_pct=key[1],gly_basis=cfg.gly_basis,origin=cfg.origin)
        for a in d.itertuples():
            ks=a.G_nm_min*1e-9/60/(cfg.C0_mol_m3*cfg.Vm_m3_mol)
            if ks<0:raise ValueError('Negative k_s cannot be used')
            rates[a.substrate]=ks;r['ks_'+a.substrate]=ks;r['ks_se_'+a.substrate]=a.G_se_nm_min*1e-9/60/(cfg.C0_mol_m3*cfg.Vm_m3_mol)
        D=cfg.D_ref_m2_s*((key[0]+273.15)/(cfg.D_ref_T_C+273.15))*cfg.eta_ref_Pa_s/v.eta_Pa_s
        kend=0. if cfg.end_material=='inert' or cfg.end_rate_ratio==0 else rates[cfg.end_material]*cfg.end_rate_ratio
        r.update(D=D,eta_Pa_s=float(v.eta_Pa_s),eta_sd_Pa_s=float(v.eta_sd_Pa_s),k_end=kend,
            ks_effective=rectangular_coefficients(g,rates,D,kend)['ks_effective'],C0_mol_m3=cfg.C0_mol_m3,Vm_m3_mol=cfg.Vm_m3_mol,
            property_basis='independent_planar_and_viscosity',D_basis='Stokes_Einstein_estimate',D_ref_source=cfg.D_ref_source,
            calibration_ids='|'.join(d.source_ids),calibration_specimen_ids='|'.join(d.specimen_ids),viscosity_source=str(v.source),
            growth_assumptions_flagged=bool(d['flags'].fillna('').ne('').any()))
        rows.append(r)
    return pd.DataFrame(rows)
