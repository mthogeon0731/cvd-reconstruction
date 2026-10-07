"""Fixed independent theory comparison and composition-preserving process maps."""
from .json_utils import loads_json
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.special import logsumexp
from .physics import Geometry,rectangular_coefficients,observed_sc,observation_windows,window_log_mean,bracketed_threshold
from .io_utils import write_csv,write_json,new_dir


def rates(row):return {s:float(row['ks_'+s]) for s in ('glass','pdms') if 'ks_'+s in row}


def make_geometry(ar,cfg):return Geometry(cfg.w_um*1e-6,cfg.h_um*1e-6,ar*cfg.w_um*1e-6,cfg.faces,tuple(cfg.wall_materials))


def ar_limit(ks,D,k_end,cfg):
    lower=0. if cfg.map_observable=='endpoints' else 2*(cfg.top_offset_um+cfg.window_half_um)/cfg.w_um+1e-8
    def f(ar):
        if ar==0:return 1.
        g=make_geometry(ar,cfg)
        windows={} if cfg.map_observable=='endpoints' else dict(zip(['top','bot'],observation_windows(g.L_m,cfg.top_offset_um*1e-6,cfg.window_half_um*1e-6)))
        return observed_sc(g,ks,D,k_end,**windows)
    result=bracketed_threshold(f,cfg.sc_target,lower,cfg.map_ar_limit)
    result['observable']=cfg.map_observable
    return result


def independent_property_check(cal,trench):
    if not cal.property_basis.eq('independent_planar_and_viscosity').all():raise ValueError('Circular/unknown property estimation basis')
    measured=set(trench.top_measurement_id.dropna().astype(str))|set(trench.bot_measurement_id.dropna().astype(str))
    used=set('|'.join(cal.calibration_ids.astype(str)).split('|'))
    specimens=set('|'.join(cal.calibration_specimen_ids.astype(str)).split('|'))
    if used&measured or specimens&set(trench.specimen_id):raise ValueError('Circular validation: calibration and SC share measurement/specimen IDs')


def compare_similitude(trenches,cal,out,cfg):
    d=trenches[trenches.status.eq('PASS')].copy()
    if d.empty:raise ValueError('No valid SC observations')
    if not d.prediction_role.isin(['cross_fitted','independent_measurement','independent_holdout']).all():raise ValueError('Similitude requires cross-fitted or independent observations; final re-predictions are disallowed')
    independent_property_check(cal,d)
    # Decode all used CSV-cell JSON before creating any result directory.
    intervals=[(loads_json(r.top_observed_intervals_um,source=f'top_observed_intervals_um, observation {i+1}'),
                loads_json(r.bot_observed_intervals_um,source=f'bot_observed_intervals_um, observation {i+1}'))
               for i,(_,r) in enumerate(d.iterrows())]
    out=new_dir(out)
    rows=[]
    for (_,r),(top_intervals,bot_intervals) in zip(d.iterrows(),intervals):
        cc=cal[(cal.T_C==r.T_C)&(cal.gly_pct==r.gly_pct)]
        if len(cc)!=1:raise ValueError('Missing/ambiguous independent condition calibration')
        c=cc.iloc[0];g=Geometry(r.w_um*1e-6,r.h_um*1e-6,r.L_um*1e-6,int(r.faces),tuple(r[k] for k in ['left_material','right_material','floor_material','ceiling_material']))
        coef=rectangular_coefficients(g,rates(c),c.D,c.k_end)
        def logmean(intervals):
            lengths=np.array([b-a for a,b in intervals]);v=np.array([window_log_mean(a*1e-6,b*1e-6,g.L_m,coef['da'],coef['bottom_biot']) for a,b in intervals])
            return float(logsumexp(v,b=lengths)-np.log(lengths.sum()))
        theory=float(np.exp(logmean(bot_intervals)-logmean(top_intervals)))
        endpoint=observed_sc(g,rates(c),c.D,c.k_end)
        rows.append({**r.to_dict(),'SC_model':theory,'phi':float(np.sqrt(coef['da'])),'Da':coef['da'],'Bi_end':coef['bottom_biot'],'SC_endpoint':endpoint,'endpoint_relative_error_pct':100*(endpoint-theory)/theory if theory>0 else np.nan})
    table=pd.DataFrame(rows);points=table.groupby(['condition_id','AR'],as_index=False).agg(SC_measured=('SC','mean'),SC_model=('SC_model','mean'),phi=('phi','mean'),n_trenches=('trench_id','size'))
    y=points.SC_measured.to_numpy(float);p=points.SC_model.to_numpy(float);sst=np.sum((y-y.mean())**2)
    report={'origin':cfg.origin,'n_points':len(points),'n_conditions':int(points.condition_id.nunique()),'n_trenches':len(table),'fixed_theory_R2':float(1-np.sum((y-p)**2)/sst) if len(y)>1 and sst>1e-20 else None,'RMSE_fraction':float(np.sqrt(np.mean((y-p)**2))),'fitted_parameters':0,'observation':'equal-area union of actual tile z intervals; same side material assumed at top and bot','scientific_status':'SYNTHETIC common physical generator: integration check, not independent physical validation' if cfg.origin=='SYNTHETIC' else 'requires independent calibration assumptions and metrology checks'}
    write_csv(out/'trench_theory.csv',table);write_csv(out/'points.csv',points);write_json(out/'report.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,4),layout='constrained');ax.scatter(points.phi,points.SC_measured,label='observed SC');ax.scatter(points.phi,points.SC_model,marker='x',label='fixed theory');ax.set(xlabel='phi',ylabel='SC (fraction)',title=f'{cfg.origin} | same observation support');ax.legend();fig.savefig(out/'SC_phi.png',dpi=140);plt.close(fig)
    return report


class PropertyGrid:
    def __init__(self,cal,cfg):
        self.cfg=cfg;self.cal=cal
        self.T=np.sort(cal.T_C.unique());self.G=np.sort(cal.gly_pct.unique())
        if len(self.T)<2 or len(self.G)<2:raise ValueError('Process interpolation needs >=2 temperatures and >=2 compositions')
        if cal.duplicated(['T_C','gly_pct']).any() or len(cal)!=len(self.T)*len(self.G):raise ValueError('Incomplete/duplicate interpolation grid')
        self.interp={};self.zero=set()
        needed=set(make_geometry(1.,cfg).perimeters)
        if cfg.end_material!='inert' and cfg.end_rate_ratio>0:needed.add(cfg.end_material)
        for col in ['ks_'+s for s in sorted(needed)]+['eta_Pa_s']:
            if col not in cal:raise ValueError(f'Missing {col}')
            a=cal.pivot(index='T_C',columns='gly_pct',values=col).reindex(index=self.T,columns=self.G).to_numpy(float)
            if not np.isfinite(a).all() or (a<0).any():raise ValueError('Invalid interpolation grid values')
            if (a==0).all() and col!='eta_Pa_s':self.zero.add(col);continue
            if (a<=0).any():raise ValueError('Log interpolation cannot mix zero and positive properties')
            # Arrhenius interpolation for k, ordinary temperature for viscosity.
            axis=1/(self.T+273.15) if col.startswith('ks_') else self.T
            order=np.argsort(axis);self.interp[col]=RegularGridInterpolator((axis[order],self.G),np.log(a[order]),bounds_error=True)
    def at(self,T,G):
        if not self.T[0]<=T<=self.T[-1] or not self.G[0]<=G<=self.G[-1]:raise ValueError('Property extrapolation is disabled')
        vals={c:0. for c in self.zero}
        for c,f in self.interp.items():vals[c]=float(np.exp(f([[1/(T+273.15) if c.startswith('ks_') else T,G]])[0]))
        vals['D']=self.cfg.D_ref_m2_s*(T+273.15)/(self.cfg.D_ref_T_C+273.15)*self.cfg.eta_ref_Pa_s/vals['eta_Pa_s']
        vals['k_end']=0. if self.cfg.end_material=='inert' or self.cfg.end_rate_ratio==0 else vals['ks_'+self.cfg.end_material]*self.cfg.end_rate_ratio
        return vals


def process_window(cal,out,cfg,T_grid=None,G_grid=None):
    out=new_dir(out);grid=PropertyGrid(cal,cfg)
    Ts=np.asarray(T_grid if T_grid is not None else np.linspace(grid.T[0],grid.T[-1],16));Gs=np.asarray(G_grid if G_grid is not None else np.linspace(grid.G[0],grid.G[-1],21))
    rows=[]
    for gly in Gs:
        for T in Ts:
            c=grid.at(T,gly);g=make_geometry(49,cfg);coef=rectangular_coefficients(g,rates(c),c['D'],c['k_end'])
            windows={} if cfg.map_observable=='endpoints' else dict(zip(['top','bot'],observation_windows(g.L_m,cfg.top_offset_um*1e-6,cfg.window_half_um*1e-6)))
            sc=observed_sc(g,rates(c),c['D'],c['k_end'],**windows);limit=ar_limit(rates(c),c['D'],c['k_end'],cfg)
            rows.append(dict(T_C=float(T),gly_pct=float(gly),**c,SC_AR49=sc,Da_AR49=coef['da'],AR_max=limit['value'],AR_max_status=limit['status'],origin=cfg.origin,observable=cfg.map_observable))
    table=pd.DataFrame(rows);sc=table.SC_AR49.to_numpy();status='boundary_in_domain' if sc.min()<cfg.sc_target<sc.max() else 'no_boundary_in_assumed_domain'
    summary={'origin':cfg.origin,'boundary_status':status,'message':'해당 가정과 영역에서 경계 없음' if status!='boundary_in_domain' else 'Grid spans target; interpolated contour shown','n_grid':len(table),'n_at_or_above_target':int((sc>=cfg.sc_target).sum()),'SC_AR49_range':[float(sc.min()),float(sc.max())],'AR_max_range':[float(table.AR_max.min()),float(table.AR_max.max())] if table.AR_max.notna().any() else None,'observable':cfg.map_observable,'SC_target':cfg.sc_target,'interpolation':'per-substrate log(k) linear in inverse Kelvin and composition; log(eta) in T and composition; no extrapolation','uncertainty_note':'two temperature levels identify an interpolant, not Arrhenius residual variance; use independently supplied errors for propagation','interpretation':'Model-dependent maximum AR, not a manufacturing guarantee'}
    write_csv(out/'process_grid.csv',table);write_json(out/'process_report.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained');tt,gg=np.meshgrid(Ts,Gs)
    for ax,key,title in zip(axs,['SC_AR49','AR_max'],['SC at AR 49','Maximum AR satisfying SC target']):
        z=table[key].to_numpy(float).reshape(len(Gs),len(Ts))
        if np.isfinite(z).any():
            im=ax.pcolormesh(tt,gg,z,shading='auto',cmap='viridis');fig.colorbar(im,ax=ax)
        else:ax.text(.5,.5,'No defined roots',transform=ax.transAxes,ha='center')
        ax.set(xlabel='Temperature (C)',ylabel=f'Glycerol ({cfg.gly_basis})',title=title)
    if status=='boundary_in_domain':axs[0].contour(tt,gg,sc.reshape(tt.shape),levels=[cfg.sc_target],colors='white')
    else:axs[0].text(.5,.02,'No target boundary in assumed domain',transform=axs[0].transAxes,ha='center',fontsize=8,color='white',bbox={'facecolor':'black','alpha':.6})
    fig.suptitle(f'{cfg.origin} | {cfg.map_observable} | model assumptions only');fig.savefig(out/'process_map.png',dpi=140);plt.close(fig)
    return table,summary
