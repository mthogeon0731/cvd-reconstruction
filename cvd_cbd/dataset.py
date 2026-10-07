"""Persist tiles for audit, aggregate once per physically labeled ROI for learning."""
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from . import metrology as im
from .imaging import _load_image
from .io_utils import resolve_input,new_dir,write_csv,write_json


def build_features(manifest,root,out,cfg,config_path=None):
    from .manifest import build_manifest,input_provenance,verify_provenance
    # An exported manifest cannot silently outlive changed raw files/acquisition records.
    current=build_manifest(root,cfg).set_index('image_id')
    if manifest.image_id.duplicated().any():raise ValueError('Duplicate requested image IDs')
    for _,given in manifest.iterrows():
        if given.image_id not in current.index:raise ValueError('Requested image absent from current validated metadata')
        authoritative=current.loc[given.image_id]
        # Retain the existing pixel-change diagnostic before stricter byte hashes.
        keys=['content_sha256',*[key for key in authoritative.index if key!='content_sha256']]
        for key in keys:
            value=authoritative[key]
            if key not in given:raise ValueError(f'Manifest missing authoritative column: {key}')
            actual=given[key]
            if pd.isna(value) and pd.isna(actual):continue
            if pd.isna(value) or pd.isna(actual):raise ValueError(f'Stale/edited manifest field: {key}')
            equal=str(actual)==str(value)
            if not equal and not key.endswith('_sha256'):
                try:equal=bool(np.isclose(float(actual),float(value),rtol=1e-9,atol=1e-12))
                except (ValueError,TypeError):equal=False
            if not equal:raise ValueError(f'Stale/edited manifest field: {key}; update source CSV and rebuild manifest')
    provenance=input_provenance(manifest,cfg,config_path)
    verify_provenance(provenance,root)
    out=new_dir(out);(out/'overlays').mkdir();(out/'corrected').mkdir()
    write_json(out/'input_provenance.json',provenance)
    tiles=[];regions=[];qc=[]
    for _,series in manifest.iterrows():
        r=series.to_dict();iid=str(r['image_id'])
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',iid) or iid in ('.','..'):raise ValueError('image_id must be a filename-safe identifier')
        try:
            arrays=[]
            for key in ('path','ref_path','dark_path'):
                a=im.select_channel(_load_image(resolve_input(root,r[key])),r['color_channel'])
                arrays.append(im.canonical(a,r['orientation']))
            image,ref,dark=arrays
            T,od,cl,valid,report=im.correct(image,ref,dark,int(r['bit_depth']),cfg)
            bounds=None;mask=None
            if pd.notna(r.get('mask_path')) and str(r.get('mask_path','')).strip():
                mask=im.canonical(_load_image(resolve_input(root,r['mask_path'])),r['orientation'])>0
                if mask.shape!=T.shape:raise ValueError('Mask/image size mismatch')
            if r['kind']=='trench':
                if r['roi_mode']=='auto':bounds=im.detect_channel(cl,float(r['w_um'])/float(r['um_per_px']))
                else:
                    vals=[float(r[k]) for k in ('roi_y0','roi_y1')]
                    if not np.isfinite(vals).all() or any(v!=int(v) for v in vals):raise ValueError('Manual channel edges must be integer canonical pixels')
                    bounds=tuple(map(int,vals))
                    if not 0<=bounds[0]<bounds[1]<=T.shape[0]:raise ValueError('Manual boundaries outside image')
                    if not .75*r['w_um']/r['um_per_px']<=bounds[1]-bounds[0]<=1.25*r['w_um']/r['um_per_px']:raise ValueError('Manual width incompatible with independent scale/layout')
            boxes,roi=im.tile_boxes(T.shape,r,cfg,bounds,mask)
            accepted=[];tile_rows=[];rejected=0
            for j,(x0,y0,x1,y1) in enumerate(boxes):
                if not valid[y0:y1,x0:x1].all():rejected+=1;continue
                f=im.tile_features(T[y0:y1,x0:x1],od[y0:y1,x0:x1],cl[y0:y1,x0:x1])
                f.update(image_id=iid,matched_region_id=r['matched_region_id'],condition_id=r['condition_id'],tile_id=f'{iid}_{j:03d}',x0=x0,y0=y0,x1=x1,y1=y1,origin=r['origin'])
                tile_rows.append(f);accepted.append((x0,y0,x1,y1))
            if not accepted:raise ValueError('No valid complete tiles')
            # Support changes caused by invalid pixels would misregister the ROI label.
            if rejected and pd.notna(r.get('label_nm')):raise ValueError('Invalid pixels change labeled ROI support; re-register label/exclude image')
            td=pd.DataFrame(tile_rows)
            if r['kind']=='trench':
                intervals=sorted((float(r['z_anchor_um'])+(x0-r['x_anchor_px'])*r['um_per_px'],float(r['z_anchor_um'])+(x1-r['x_anchor_px'])*r['um_per_px']) for x0,y0,x1,y1 in accepted)
                r['observed_intervals_um']=json.dumps(intervals)
                r['observed_z_lo_um']=intervals[0][0];r['observed_z_hi_um']=intervals[-1][1]
                if pd.notna(r.get('label_nm')):
                    if not np.allclose([r['observed_z_lo_um'],r['observed_z_hi_um']],[r['label_z_lo_um'],r['label_z_hi_um']],atol=.51*r['um_per_px'],rtol=0):raise ValueError('Optical tile support differs from FE-SEM label region')
                    unique=sorted(set(intervals))
                    if any(abs(a[1]-b[0])>1e-8 for a,b in zip(unique,unique[1:])):raise ValueError('Label region contains untiled gaps')
            for feature in im.FEATURES:
                r['feature_'+feature]=float(td[feature].mean())
            r['n_tiles']=len(accepted);regions.append(r);tiles.extend(tile_rows)
            report.update(image_id=iid,status='PASS',reason='',tiles=len(accepted),rejected_tiles=rejected,channel_y0=None if bounds is None else bounds[0],channel_y1=None if bounds is None else bounds[1],origin=cfg.origin)
            qc.append(report)
            np.savez_compressed(out/'corrected'/f'{iid}.npz',T=T.astype(np.float32),OD=od.astype(np.float32),valid=valid)
            im.overlay(out/'overlays'/f'{iid}.png',image,T,cl,bounds,roi,accepted,f'{cfg.origin} | {iid} | canonical {r["orientation"]} | {r["um_per_px"]} um/px')
        except (ValueError,OSError,KeyError,TypeError) as exc:
            qc.append(dict(image_id=iid,status='FAIL',reason=str(exc),origin=cfg.origin))
    verify_provenance(provenance,root)
    write_csv(out/'image_qc.csv',pd.DataFrame(qc));write_csv(out/'tiles.csv',pd.DataFrame(tiles));write_csv(out/'features.csv',pd.DataFrame(regions))
    write_json(out/'extraction.json',dict(origin=cfg.origin,ROI_learning=True,quantitative_features='T and OD; no CLAHE intensity',tile_size_px=cfg.tile_px,window_half_um=cfg.window_half_um,n_images=len(manifest),n_pass=len(regions),n_tiles=len(tiles)))
    if len(regions)!=len(manifest):raise ValueError(f'Image QC failed for {len(manifest)-len(regions)} image(s); see image_qc.csv and supply manual ROI/correct metadata')
    return pd.DataFrame(regions)
