"""New design calculations from explicit inputs; not a port of missing scripts."""
from dataclasses import replace
import numpy as np
import pandas as pd
from .physics import Geometry,rectangular_coefficients,observed_sc,analytic_profile,fdm_profile,observation_windows,bracketed_threshold
from .monte_carlo import noise_stress_test
from .similitude import rates,make_geometry
from .io_utils import new_dir,write_csv,write_json


def run_design(cal,out,cfg):
    out=new_dir(out);rows=[];sensitivity=[]
    for _,c in cal.iterrows():
        for ar in cfg.ar_ladder:
            g=make_geometry(ar,cfg);p=rectangular_coefficients(g,rates(c),c.D,c.k_end)
            top,bot=observation_windows(g.L_m,cfg.top_offset_um*1e-6,cfg.window_half_um*1e-6)
            endpoint=observed_sc(g,rates(c),c.D,c.k_end);sc=observed_sc(g,rates(c),c.D,c.k_end,top,bot)
            rows.append(dict(T_C=c.T_C,gly_pct=c.gly_pct,AR=ar,L_um=ar*cfg.w_um,phi=np.sqrt(p['da']),Da=p['da'],Bi_end=p['bottom_biot'],SC_endpoint=endpoint,SC_observed=sc,endpoint_relative_error_pct=100*(endpoint-sc)/sc,origin=cfg.origin))
    table=pd.DataFrame(rows)
    trials,mc=noise_stress_test(table.SC_observed,cfg.mc_repeats,cfg.mc_noise_sd,cfg.seed,noise_kind=cfg.mc_noise_kind)
    c=cal.iloc[0]
    for h in (30.,50.):
        for faces in (2,4):
            for ratio in (0.,1.,3.):
                for ar in (1.,9.,49.):
                    cc=replace(cfg,h_um=h,faces=faces);g=make_geometry(ar,cc);k=c.ks_effective*ratio
                    missing=sorted(set(g.perimeters)-set(rates(c)))
                    row=dict(h_um=h,faces=faces,AR=ar,k_end_over_reference_effective_side=ratio,
                        SC_endpoint=None,Da=None,Bi_end=None,reference_condition=f'{c.T_C:g}C G{c.gly_pct:g}',origin=cfg.origin,
                        status='NOT_EVALUATED' if missing else 'EVALUATED',missing_substrates='|'.join(missing),
                        reason='hypothetical_geometry_requires_unmeasured_substrate' if missing else '')
                    if not missing:
                        p=rectangular_coefficients(g,rates(c),c.D,k)
                        row.update(SC_endpoint=observed_sc(g,rates(c),c.D,k),Da=p['da'],Bi_end=p['bottom_biot'])
                    sensitivity.append(row)
    # Explicit equal-rate square reactive-end algebra, unrelated to measured rates.
    critical=bracketed_threshold(lambda da:float(analytic_profile(1.,da,da/(4*49))),cfg.sc_target)
    fdm=[]
    for da,bi in [(0.,0.),(0.,2.),(4.,.5),(100.,10.)]:
        for n in (51,101,201,401):
            x,u=fdm_profile(da,bi,n);fdm.append(dict(Da=da,Bi_end=bi,nodes=n,max_abs_error=float(np.max(np.abs(u-analytic_profile(x,da,bi))))))
    lengths=[]
    for _,c in cal.iterrows():
        for faces in (2,4):
            for h in (30.,50.):
                cc=replace(cfg,faces=faces,h_um=h)
                missing=sorted(set(make_geometry(1.,cc).perimeters)-set(rates(c)))
                # These are optional hypothetical face-count comparisons. Main
                # configured geometry was validated above and is never skipped.
                crit={'value':None,'status':'NOT_EVALUATED'} if missing else bracketed_threshold(lambda ar:1. if ar==0 else observed_sc(make_geometry(ar,cc),rates(c),c.D,c.k_end),.01,limit=cfg.map_ar_limit)
                lengths.append(dict(T_C=c.T_C,gly_pct=c.gly_pct,faces=faces,h_um=h,threshold_SC_endpoint=.01,depth_um=None if crit['value'] is None else crit['value']*cfg.w_um,status=crit['status'],origin=cfg.origin,
                    missing_substrates='|'.join(missing),reason='hypothetical_geometry_requires_unmeasured_substrate' if missing else ''))
    write_csv(out/'condition_AR.csv',table);write_csv(out/'sensitivity.csv',pd.DataFrame(sensitivity));write_csv(out/'new_dead_end_1pct.csv',pd.DataFrame(lengths));write_csv(out/'fdm_convergence.csv',pd.DataFrame(fdm));write_csv(out/'mc_trials.csv',trials);write_json(out/'mc_report.json',mc)
    write_json(out/'design_report.json',dict(origin=cfg.origin,implementation='new calculation; sc_sim.py/sc_model.py absent',ar_ladder=list(cfg.ar_ladder),adjacent_ratios=(np.array(cfg.ar_ladder[1:])/np.array(cfg.ar_ladder[:-1])).tolist(),lengths_um=(np.array(cfg.ar_ladder)*cfg.w_um).tolist(),critical_equal_rate_square_AR49=critical,critical_definition='Da=phi^2; endpoint ratio, four equal-rate square sides, k_end=k_side; not universal',n_conditions=len(cal),n_design_points=len(table),mc_repeats=cfg.mc_repeats,synthetic_training_samples='not the Monte Carlo repeat count',
        sensitivity_not_evaluated=sum(row['status']=='NOT_EVALUATED' for row in sensitivity),dead_end_not_evaluated=sum(row['status']=='NOT_EVALUATED' for row in lengths),
        hypothetical_geometry_policy='Unavailable substrate kinetics in optional face-count comparisons are NOT_EVALUATED, with null results and named missing materials. Actual configured active geometry still requires all kinetics.'))
    return table
