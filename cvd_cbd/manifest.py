"""File-name parsing, relative paths, provenance joins and dependency checks."""
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from .io_utils import resolve_input,read_csv
from .input_contracts import scalar_pixel_scale,reject_axis_pixel_calibration
from .calibration import require_columns
from .imaging import _load_image

ACQUISITION=['session_id','exposure_ms','gain','illumination_id','magnification','bit_depth','auto_exposure','gamma','white_balance','color_channel']
ID_COLUMNS=['image_id','run_id','session_id','condition_id','batch_id','specimen_id','measurement_id','matched_region_id','trench_id','acquisition_id']
MANIFEST_TABLES=('images','runs','layout','fesem','frames')
SCHEMAS={
'runs':['run_id','T_C','gly_pct','gly_basis','rep','date','session_id','cycles','dep_time_h','batch_id'],
'layout':['slot','AR','L_um','w_um','h_um','faces','left_material','right_material','floor_material','ceiling_material'],
'viscosity':['T_C','gly_pct','eta_Pa_s','eta_sd_Pa_s','eta_unit','gly_basis','source','origin'],
'calibration_thickness':['T_C','gly_pct','substrate','t_min','thickness_nm','thickness_sd_nm','method','measurement_id','specimen_id','C0_mol_m3','source','origin'],
'fesem':['run_id','chip','slot','pos','thickness_nm','thickness_sd_nm','n_sites','sem_file','measurement_id','matched_region_id','z_lo_um','z_hi_um','match_basis','source'],
'manual_om':['image_id','measurement_id','thickness_nm_manual','method','source'],
'frames':['frame_id','path','kind',*ACQUISITION],
'images':['image_id','path','kind','run_id','chip','slot','pos','condition_id','batch_id','specimen_id','matched_region_id','measurement_id','z_anchor_um','x_anchor_px','um_per_px','scale_source','scale_record','orientation','roi_mode','roi_x0','roi_x1','roi_y0','roi_y1','mask_path','ref_id','dark_id','origin',*ACQUISITION],
}


def read_table(root,name):
    p=Path(root)/'meta'/f'{name}.csv'
    return read_csv(p,dtype={k:str for k in [*ID_COLUMNS,'frame_id','ref_id','dark_id']})


def file_sha256(path):
    """Hash file bytes, separately from the image-pixel duplicate fingerprint."""
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def config_snapshot(cfg):
    values=asdict(cfg)
    encoded=json.dumps(values,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
    return dict(values=values,sha256=hashlib.sha256(encoded.encode('utf-8')).hexdigest(),
                encoding='UTF-8 JSON; sorted keys; compact separators; all effective StudyConfig values')


def file_record(root,path,role,usage):
    resolved=resolve_input(root,path)
    return dict(path=resolved.relative_to(Path(root).resolve()).as_posix(),path_kind='root_relative',
                role=role,usage=usage,sha256=file_sha256(resolved))


def input_provenance(manifest,cfg,config_path=None):
    """Record only selected ROI dependencies, distinguishing evidence and tables.

    Hashing a whole consulted CSV does not claim every row or referenced file was
    used numerically. SEM/scale evidence bytes are retained, not interpreted as
    measured pixels or re-derived calibration values by this software.
    """
    entries={}
    def add(path,digest,role,usage,image_id):
        if pd.isna(path) or not str(path).strip():return
        key=(str(path),role)
        if key not in entries:
            entries[key]=dict(path=str(path),path_kind='root_relative',role=role,usage=usage,
                              sha256=str(digest),image_ids=[])
        if entries[key]['sha256']!=digest:raise ValueError(f'Inconsistent dependency hashes: {path}')
        if image_id not in entries[key]['image_ids']:entries[key]['image_ids'].append(image_id)
    dependencies=[('path','image_file_sha256','image_pixels','numerical_input'),
                  ('ref_path','ref_file_sha256','reference_correction','numerical_input'),
                  ('dark_path','dark_file_sha256','dark_correction','numerical_input'),
                  ('mask_path','mask_file_sha256','roi_mask','numerical_input'),
                  ('scale_record','scale_record_sha256','scale_evidence','evidence_only'),
                  ('label_evidence_path','label_evidence_sha256','label_evidence','evidence_only')]
    for _,row in manifest.iterrows():
        for path_col,hash_col,role,usage in dependencies:
            add(row[path_col],row[hash_col],role,usage,str(row.image_id))
        for name in MANIFEST_TABLES:
            add(f'meta/{name}.csv',row[f'{name}_table_sha256'],'metadata_table_consulted',
                'table_consulted; selected records identified by ROI/measurement/frame IDs',str(row.image_id))
    files=sorted(entries.values(),key=lambda x:(x['path'],x['role']))
    for entry in files:entry['image_ids'].sort()
    if config_path is not None:
        source=Path(config_path).resolve()
        files.append(dict(path=str(source),path_kind='absolute',role='configuration_source',
                          usage='source_file; effective_config is authoritative after API overrides',sha256=file_sha256(source)))
    return dict(schema_version=1,files=files,effective_config=config_snapshot(cfg),
                scope='Selected ROI dependencies only; table hashes cover consulted CSV bytes, not use of every row/file',
                evidence_status='File identity only; does not authenticate experimental measurements')


def verify_provenance(provenance,root):
    """Detect dependencies modified during a run before declaring success."""
    for item in provenance['files']:
        path=Path(item['path']) if item['path_kind']=='absolute' else resolve_input(root,item['path'])
        if file_sha256(path)!=item['sha256']:
            raise ValueError(f'Stale/edited dependency file: {item["path"]} ({item["role"]})')


def parse_filename(name):
    m=re.fullmatch(r'R(\d{2,})_C(\d+)_T(\d+)_(top|bot)\.tiff?',name,re.I)
    if m:return dict(kind='trench',run_id=f'R{int(m[1]):02d}',chip=int(m[2]),slot=int(m[3]),pos=m[4].lower())
    m=re.fullmatch(r'CAL_([\d.]+)C_G([\d.]+)_(glass|pdms)_([\d.]+)min\.tiff?',name,re.I)
    if m:return dict(kind='flat',T_C=float(m[1]),gly_pct=float(m[2]),substrate=m[3].lower(),t_min=float(m[4]))
    m=re.fullmatch(r'(REF|DARK)_(.+)\.tiff?',name,re.I)
    if m:return dict(kind=m[1].upper(),session_id=m[2])
    raise ValueError(f'Filename does not match data contract: {name}')


def condition_id(T,gly,basis):return f'{T:g}C_G{gly:g}_{basis}'


def check_acquisition(row):
    for k in ACQUISITION:
        if k not in row or pd.isna(row[k]):raise ValueError(f'Missing acquisition field: {k}')
    for k in ('exposure_ms','gain','magnification'):
        if not np.isfinite(float(row[k])) or float(row[k])<=0:raise ValueError(f'Invalid acquisition {k}')
    if str(row['auto_exposure']).lower() not in ('false','0') or float(row['gamma'])!=1. or row['white_balance']!='fixed':raise ValueError('Quantitative mode requires fixed exposure, gamma=1 and fixed white balance')
    if float(row['bit_depth']) not in (8,12,16):raise ValueError('Supported bit depth: 8/12/16')
    if row['color_channel'] not in ('mono','red','green','blue'):raise ValueError('Explicit color channel required')


def build_manifest(root,cfg):
    fingerprints={f'{name}_table_sha256':file_sha256(resolve_input(root,f'meta/{name}.csv')) for name in MANIFEST_TABLES}
    fingerprints['config_sha256']=config_snapshot(cfg)['sha256']
    cache={}
    def fingerprint(value):
        path=resolve_input(root,value)
        if path not in cache:cache[path]=file_sha256(path)
        return cache[path]
    images=read_table(root,'images');runs=read_table(root,'runs');layout=read_table(root,'layout');sem=read_table(root,'fesem');frames=read_table(root,'frames')
    reject_axis_pixel_calibration(images.columns,'images pixel calibration')
    for name,d in [('images',images),('runs',runs),('layout',layout),('frames',frames)]:require_columns(d,SCHEMAS[name])
    exclusions=[]
    if 'include' in images:
        normalized=images.include.astype(str).str.lower()
        if not normalized.isin(['true','false','1','0']).all():raise ValueError('include must be explicit true/false')
        excluded=images[normalized.isin(['false','0'])]
        if not excluded.empty:
            if 'exclusion_reason' not in excluded or excluded.exclusion_reason.isna().any():raise ValueError('Explicit exclusion reason required')
            exclusions=excluded[['image_id','path','exclusion_reason']].to_dict('records')
        images=images[normalized.isin(['true','1'])].copy()
        if images.empty:raise ValueError('All images excluded')
    for d,key in [(images,'image_id'),(runs,'run_id'),(layout,'slot'),(frames,'frame_id')]:
        if d[key].isna().any() or d[key].duplicated().any():raise ValueError(f'Unique {key} required')
    if not images.image_id.astype(str).str.fullmatch(r'[A-Za-z0-9_.-]+').all() or images.image_id.isin(['.','..']).any():raise ValueError('image_id must be filename-safe')
    if images.path.duplicated().any() or images.matched_region_id.duplicated().any():raise ValueError('Duplicate image path/ROI')
    for k in ('condition_id','batch_id','specimen_id','matched_region_id','origin'):
        if images[k].isna().any():raise ValueError(f'Missing {k}')
    if not images.origin.eq(cfg.origin).all():raise ValueError('Do not mix real and synthetic images')
    nums=layout[['AR','L_um','w_um','h_um']].to_numpy(float)
    if not np.isfinite(nums).all() or (nums<=0).any() or not np.allclose(layout.L_um,layout.AR*layout.w_um,rtol=1e-6):raise ValueError('Layout requires positive SI-convertible dimensions and L=AR*w')
    if (layout.slot<=0).any() or not np.equal(layout.slot,np.floor(layout.slot)).all():raise ValueError('Layout slot must be a positive integer')
    if not layout.faces.isin([2,4]).all():raise ValueError('Layout faces must be 2/4')
    if not runs.gly_basis.eq(cfg.gly_basis).all():raise ValueError('Run composition basis mismatch')
    rv=runs[['T_C','gly_pct','rep','cycles','dep_time_h']].to_numpy(float)
    if not np.isfinite(rv).all() or not runs.gly_pct.between(0,100).all() or (runs.T_C<=-273.15).any() or (runs[['rep','cycles','dep_time_h']]<=0).any().any():raise ValueError('Invalid run units/ranges')
    if not np.equal(runs[['rep','cycles']],np.floor(runs[['rep','cycles']])).all().all():raise ValueError('Run rep/cycles must be integers')
    if pd.to_datetime(runs.date,errors='coerce').isna().any():raise ValueError('Run date required in ISO form')
    if not sem.empty:
        require_columns(sem,SCHEMAS['fesem'])
        if sem.measurement_id.isna().any() or sem.measurement_id.duplicated().any() or sem.matched_region_id.duplicated().any():raise ValueError('Ambiguous FE-SEM ID/ROI join')
    registry=frames.set_index('frame_id');seen={};rows=[]
    for _,r in images.iterrows():
        r=r.to_dict();r.update(fingerprints);path=resolve_input(root,r['path']);parsed=parse_filename(path.name)
        r['image_file_sha256']=fingerprint(r['path'])
        r['mask_file_sha256']=fingerprint(r['mask_path']) if pd.notna(r.get('mask_path')) and str(r['mask_path']).strip() else np.nan
        r['label_evidence_path']=np.nan;r['label_evidence_sha256']=np.nan
        if parsed['kind']!=r['kind']:raise ValueError('Image kind/name mismatch')
        check_acquisition(r)
        for field in ('ref_id','dark_id'):
            if r[field] not in registry.index:raise ValueError(f'Missing {field}')
            frame=registry.loc[r[field]];check_acquisition(frame)
            expected='REF' if field=='ref_id' else 'DARK'
            fp=resolve_input(root,frame.path);name=parse_filename(fp.name)
            if frame.kind!=expected or name['kind']!=expected or name['session_id']!=r['session_id']:raise ValueError('REF/DARK filename/session mismatch')
            for k in ACQUISITION:
                if str(frame[k]).lower()!=str(r[k]).lower():
                    try:equal=float(frame[k])==float(r[k])
                    except (TypeError,ValueError):equal=False
                    if not equal:raise ValueError(f'{expected} acquisition mismatch: {k}')
            r['ref_path' if field=='ref_id' else 'dark_path']=frame.path
            r['ref_file_sha256' if field=='ref_id' else 'dark_file_sha256']=fingerprint(frame.path)
        r['um_per_px']=scalar_pixel_scale(r['um_per_px'],'um_per_px (isotropic um/px)')
        allowed=('stage_micrometer','calibration_record') if cfg.origin=='REAL' else ('SYNTHETIC',)
        if r['scale_source'] not in allowed:raise ValueError('Pixel scale needs independent evidence, not nominal channel width')
        r['scale_record_sha256']=fingerprint(r['scale_record'])
        if r['orientation'] not in ('x+','x-','y+','y-'):raise ValueError('Unsupported rotation: supply registered image and canonical coordinates')
        if r['roi_mode'] not in ('auto','manual'):raise ValueError('roi_mode must be auto/manual')
        a=_load_image(path)
        digest=hashlib.sha256(str((a.shape,a.dtype)).encode()+a.tobytes()).hexdigest()
        if digest in seen:raise ValueError(f'Duplicate pixel content: {r["image_id"]} and {seen[digest]}')
        seen[digest]=r['image_id'];r['content_sha256']=digest
        if r['kind']=='trench':
            for k in ('run_id','chip','slot','pos'):
                if str(r[k])!=str(parsed[k]):
                    try:equal=float(r[k])==float(parsed[k])
                    except (TypeError,ValueError):equal=False
                    if not equal:raise ValueError(f'Filename/metadata mismatch: {k}; T is slot, not temperature')
            rd=runs[runs.run_id.eq(r['run_id'])];ld=layout[layout.slot.eq(float(r['slot']))]
            if len(rd)!=1 or len(ld)!=1:raise ValueError('Missing run/layout join')
            run=rd.iloc[0];lay=ld.iloc[0]
            for k in ('session_id','batch_id'):
                if run[k]!=r[k]:raise ValueError(f'Run/image mismatch: {k}')
            if condition_id(run.T_C,run.gly_pct,run.gly_basis)!=r['condition_id']:raise ValueError('Condition ID differs from physical run condition')
            for k in ('T_C','gly_pct','gly_basis','rep','date','cycles','dep_time_h'):r[k]=run[k]
            for k in SCHEMAS['layout']:r[k]=lay[k]
            r['trench_id']=f'{r["run_id"]}_C{int(r["chip"])}_T{int(r["slot"])}'
            r['label_nm']=np.nan;r['label_sd_nm']=np.nan
            mid=r.get('measurement_id')
            if pd.notna(mid) and str(mid).strip():
                match=sem[sem.measurement_id.eq(mid)]
                if len(match)!=1:raise ValueError('Missing matched FE-SEM label')
                q=match.iloc[0]
                for k in ('run_id','chip','slot','pos','matched_region_id'):
                    if str(q[k])!=str(r[k]):
                        try:equal=float(q[k])==float(r[k])
                        except (TypeError,ValueError):equal=False
                        if not equal:raise ValueError(f'FE-SEM registration join mismatch: {k}')
                vals=np.asarray([q.thickness_nm,q.thickness_sd_nm,q.n_sites,q.z_lo_um,q.z_hi_um],float)
                if not np.isfinite(vals).all() or q.thickness_nm<=0 or q.thickness_sd_nm<0 or q.n_sites<1 or q.n_sites!=int(q.n_sites) or not 0<=q.z_lo_um<q.z_hi_um<=r['L_um']:raise ValueError('Invalid FE-SEM label/range/uncertainty')
                if q.match_basis!='validated_region_average':raise ValueError('ROI-average learning requires registered region-average label; point labels need matched smaller ROI')
                r['label_evidence_path']=q.sem_file;r['label_evidence_sha256']=fingerprint(q.sem_file)
                r.update(label_nm=float(q.thickness_nm),label_sd_nm=float(q.thickness_sd_nm),label_z_lo_um=float(q.z_lo_um),label_z_hi_um=float(q.z_hi_um),label_source=q.source,match_basis=q.match_basis)
        else:
            if condition_id(parsed['T_C'],parsed['gly_pct'],cfg.gly_basis)!=r['condition_id']:raise ValueError('Flat filename/condition metadata mismatch')
            for k,v in parsed.items():
                if k!='kind':
                    if k in r and pd.notna(r[k]) and str(r[k])!=str(v):
                        try:equal=float(r[k])==float(v)
                        except (TypeError,ValueError):equal=False
                        if not equal:raise ValueError(f'Flat filename metadata mismatch: {k}')
                    r[k]=v
            r['label_nm']=np.nan;r['label_sd_nm']=np.nan
        rows.append(r)
    out=pd.DataFrame(rows)
    labeled=out[out.label_nm.notna()]
    if labeled.measurement_id.duplicated().any():raise ValueError('Shared FE-SEM labels: aggregate repeats explicitly before modeling')
    out.attrs['excluded_images']=exclusions
    verify_provenance(input_provenance(out,cfg),root)
    return out


def count_manifest(d):
    t=d[d.kind.eq('trench')]
    return {'explicitly_excluded_images':len(d.attrs.get('excluded_images',[])),'conditions':int(t.condition_id.nunique()),'runs':int(t.run_id.nunique()),'run_replicates':int(t[['condition_id','rep']].drop_duplicates().shape[0]),'chips':int(t[['run_id','chip']].drop_duplicates().shape[0]),'trenches_observed':int(t.trench_id.nunique()),'images':len(d),'top_images':int(t.pos.eq('top').sum()),'bot_images':int(t.pos.eq('bot').sum()),'fesem_independent_ids':int(t.measurement_id.nunique()),'ROIs':int(d.matched_region_id.nunique())}
