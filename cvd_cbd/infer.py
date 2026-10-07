"""Observed top/bot pairing with explicit exclusions and actual sample counts."""
import json
import numpy as np
import pandas as pd


def _measurement_sd(row, trench):
    """Missing metrology is an error, never a deterministic zero."""
    try:
        value = float(row.get('label_sd_nm'))
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{trench}/{row.pos}: label_sd_nm must be finite and nonnegative; missing uncertainty policy=error') from exc
    if not np.isfinite(value) or value < 0:
        raise ValueError(f'{trench}/{row.pos}: label_sd_nm must be finite and nonnegative; missing uncertainty policy=error')
    return value


def _pair_correlation(top, bot, trench):
    # An absent column declares the documented independence assumption. A
    # present but missing value does not declare independence.
    if 'top_bot_correlation' not in top.index:
        return 0., 'assumed independent (correlation column absent)'
    try:
        values = np.array([top.top_bot_correlation, bot.top_bot_correlation], float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{trench}: invalid top/bot correlation') from exc
    if not np.isfinite(values).all() or np.any(abs(values) > 1) or values[0] != values[1]:
        raise ValueError(f'{trench}: top/bot correlation must be finite, identical on both rows and in [-1, 1]')
    return float(values[0]), 'supplied on both observations'


def pair_sc(data,cfg,value='pred_nm'):
    d=data[data.kind.eq('trench')].copy()
    if d.empty:raise ValueError('No trench observations')
    rows=[]
    for trench,g in d.groupby('trench_id',sort=True):
        if g.pos.duplicated().any():raise ValueError(f'Duplicate top/bot observation for {trench}; aggregate repeat captures explicitly')
        common=g.iloc[0].to_dict()
        keep=['trench_id','condition_id','run_id','chip','slot','AR','L_um','w_um','h_um','faces','left_material','right_material','floor_material','ceiling_material','T_C','gly_pct','batch_id','specimen_id','origin']
        row={k:common[k] for k in keep};row.update(status='EXCLUDED',reason='',SC=None,SC_sd=None,
            uncertainty_status='NOT_EVALUATED',uncertainty_basis='Excluded observation',missing_uncertainty_policy='error')
        if set(g.pos)!=set(['top','bot']):row['reason']='missing_top_or_bot';rows.append(row);continue
        top=g[g.pos.eq('top')].iloc[0];bot=g[g.pos.eq('bot')].iloc[0]
        for key in ['condition_id','specimen_id','batch_id','AR','L_um','w_um','h_um','faces']:
            if top[key]!=bot[key]:raise ValueError(f'Pair metadata mismatch: {key}')
        for key in ['measurement_id','observed_intervals_um','observed_z_lo_um','observed_z_hi_um']:
            row['top_'+key]=top.get(key);row['bot_'+key]=bot.get(key)
        row['prediction_role']='independent_measurement' if value=='label_nm' else top.get('prediction_role','unverified_final_inference')
        if value!='label_nm' and top.get('prediction_role')!=bot.get('prediction_role'):row['reason']='mixed_prediction_roles';rows.append(row);continue
        if float(top.observed_z_hi_um)>=float(bot.observed_z_lo_um):row['reason']='overlapping_observation_windows';rows.append(row);continue
        a=float(top[value]);b=float(bot[value]);row.update(top_nm=a if np.isfinite(a) else None,bot_nm=b if np.isfinite(b) else None)
        if not np.isfinite([a,b]).all():row['reason']='missing_thickness'
        elif a<cfg.min_top_nm:row['reason']='top_below_denominator_threshold'
        elif b<0:row['reason']='negative_bot_thickness'
        else:
            row.update(status='PASS',SC=b/a)
            # No invented ML prediction uncertainty. FE-SEM SDs are distinct.
            if value=='label_nm':
                sa=_measurement_sd(top,trench);sb=_measurement_sd(bot,trench)
                rho,correlation_basis=_pair_correlation(top,bot,trench)
                try:
                    variance=(sb/a)**2+(b*sa/a**2)**2-2*rho*b*sa*sb/a**3
                    scale=(sb/a)**2+(b*sa/a**2)**2
                    if rho>0:
                        # Equivalent nonnegative sum avoids subtracting nearly
                        # equal terms for perfectly shared multiplicative SD.
                        u=sb/a;v=b*sa/a**2
                        variance=(u-v)**2+2*(1-rho)*u*v
                except OverflowError as exc:
                    raise ValueError(f'{trench}: nonfinite propagated measurement variance') from exc
                if not np.isfinite([variance,scale]).all() or variance < -32*np.finfo(float).eps*scale:
                    raise ValueError(f'{trench}: invalid propagated measurement variance')
                # Clamp finite floating-point cancellation only, after validating
                # both supplied SDs, correlation and the computed variance.
                row['SC_sd']=float(np.sqrt(max(0.,variance)))
                row['uncertainty_status']='AVAILABLE'
                row['uncertainty_basis']=f'FESEM supplied SD, first-order delta method, top/bot rho={rho:g}; {correlation_basis}'
            else:
                row['uncertainty_status']='NOT_AVAILABLE'
                row['uncertainty_basis']='NOT_AVAILABLE: independent predictive error model required'
            row['out_of_training_range']=bool(top.get('outside_training_feature_range',False) or bot.get('outside_training_feature_range',False))
        rows.append(row)
    trenches=pd.DataFrame(rows);points=[]
    for key,g in trenches.groupby(['condition_id','AR']):
        valid=g[g.status.eq('PASS')];vals=valid.SC.to_numpy(float)
        points.append(dict(condition_id=key[0],AR=key[1],n_available=len(g),n_valid=len(valid),n_excluded=len(g)-len(valid),n_specimens=int(valid.specimen_id.nunique()),n_batches=int(valid.batch_id.nunique()),SC_mean=float(vals.mean()) if len(vals) else np.nan,SC_between_trench_sd=float(vals.std(ddof=1)) if len(vals)>1 else np.nan,origin=cfg.origin))
    return trenches,pd.DataFrame(points)
