"""Disk-backed 16-bit SYNTHETIC fixture with declared generators and matched ROIs."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from .manifest import SCHEMAS,condition_id
from .io_utils import new_dir,write_csv,write_json
from .physics import Geometry,rectangular_coefficients,analytic_profile,window_log_mean
from .uncertainty import NAMES


def generate(root,cfg):
    if cfg.origin!='SYNTHETIC':raise ValueError('Synthetic generator requires origin=SYNTHETIC')
    root=new_dir(root)
    for d in ('meta','raw/om','raw/reference','raw/calibration','evidence'): (root/d).mkdir(parents=True,exist_ok=True)
    (root/'evidence/SYNTHETIC_scale.txt').write_text('SYNTHETIC generator grid: 0.5 um/px. NOT a real stage-micrometer calibration.\n')
    (root/'evidence/SYNTHETIC_fesem.txt').write_text('SYNTHETIC numerical region thickness values; no FE-SEM image exists.\n')
    rng=np.random.default_rng(cfg.seed);runs=[];frames=[];images=[];sem=[];manual=[];cal=[];visc=[];layout=[]
    scale=.5;H,W=128,256;yy,xx=np.mgrid[:H,:W];x_anchor=W/2;yc=H//2;width=cfg.w_um/scale;y0=int(yc-width/2);y1=int(yc+width/2)
    for slot,ar in enumerate(cfg.ar_ladder,1):layout.append(dict(slot=slot,AR=ar,L_um=ar*cfg.w_um,w_um=cfg.w_um,h_um=cfg.h_um,faces=cfg.faces,**dict(zip(['left_material','right_material','floor_material','ceiling_material'],cfg.wall_materials))))
    idx=0
    for T in (65.,80.):
        for gly in (0.,20.,40.):
            idx+=1;rid=f'R{idx:02d}';session=f'S{idx:02d}';condition=condition_id(T,gly,cfg.gly_basis)
            # New illustrative independent property generator, never tuned to old headlines.
            ks={'glass':2e-8*np.exp(.025*(T-65))*(1+.007*gly),'pdms':1.3e-8*np.exp(.03*(T-65))*(1+.012*gly)}
            eta=.0006*np.exp(-.013*(T-65)+.019*gly)
            D=cfg.D_ref_m2_s*(T+273.15)/(cfg.D_ref_T_C+273.15)*cfg.eta_ref_Pa_s/eta
            kend=ks[cfg.end_material]*cfg.end_rate_ratio if cfg.end_material!='inert' else 0.
            runs.append(dict(run_id=rid,T_C=T,gly_pct=gly,gly_basis=cfg.gly_basis,rep=1,date='2026-10-06',session_id=session,cycles=1,dep_time_h=2.,batch_id='SYNTHETIC_BATCH_'+rid))
            visc.append(dict(T_C=T,gly_pct=gly,eta_Pa_s=eta,eta_sd_Pa_s=.02*eta,eta_unit='Pa_s',gly_basis=cfg.gly_basis,source='SYNTHETIC analytic viscosity generator',origin=cfg.origin))
            for substrate,k in ks.items():
                for t in (10.,20.,30.,40.):
                    mid=f'SYNTHETIC_CAL_{idx}_{substrate}_{int(t)}'
                    thickness=k*cfg.C0_mol_m3*cfg.Vm_m3_mol*t*60*1e9
                    # Exact line isolates numerical calibration from camera and label noise.
                    cal.append(dict(T_C=T,gly_pct=gly,substrate=substrate,t_min=t,thickness_nm=thickness,thickness_sd_nm=.5,method='SYNTHETIC growth law',measurement_id=mid,specimen_id=mid,C0_mol_m3=cfg.C0_mol_m3,source='SYNTHETIC independently generated planar fixture',origin=cfg.origin))
            acquisition=dict(session_id=session,exposure_ms=10.,gain=1.,illumination_id='SYNTHETIC_LED',magnification=20.,bit_depth=16,auto_exposure=False,gamma=1.,white_balance='fixed',color_channel='mono')
            dark=np.round(900+25*np.sin(xx/33)+rng.normal(0,2,(H,W))).astype(np.uint16)
            ref=np.round(dark.astype(float)+48000*(1+.025*np.cos(xx/90)+.015*yy/H)).astype(np.uint16)
            for kind,a in [('REF',ref),('DARK',dark)]:
                path=f'raw/reference/{kind}_{session}.tif';Image.fromarray(a).save(root/path,compression='tiff_deflate')
                frames.append(dict(frame_id=kind+'_'+session,path=path,kind=kind,**acquisition))
            common=dict(condition_id=condition,batch_id='SYNTHETIC_BATCH_'+rid,scale_source='SYNTHETIC',scale_record='evidence/SYNTHETIC_scale.txt',um_per_px=scale,orientation='x+',roi_mode='auto',roi_x0=np.nan,roi_x1=np.nan,roi_y0=np.nan,roi_y1=np.nan,mask_path='',ref_id='REF_'+session,dark_id='DARK_'+session,origin=cfg.origin,**acquisition)
            for chip in (1,2):
                specimen=f'SYNTHETIC_SPEC_{rid}_C{chip}'
                for slot,ar in enumerate(cfg.ar_ladder,1):
                    g=Geometry(cfg.w_um*1e-6,cfg.h_um*1e-6,ar*cfg.w_um*1e-6,cfg.faces,tuple(cfg.wall_materials));coef=rectangular_coefficients(g,ks,D,kend)
                    base_nm=ks['glass']*cfg.C0_mol_m3*cfg.Vm_m3_mol*7200*1e9*(1+.01*(chip-1))
                    for pos,z in [('top',cfg.top_offset_um),('bot',ar*cfg.w_um-cfg.top_offset_um)]:
                        iid=f'{rid}_C{chip}_T{slot}_{pos}';mid='SYNTHETIC_SEM_'+iid
                        zpx=z+(np.arange(W)+.5-x_anchor)*scale
                        concentration=analytic_profile(np.clip(zpx/(ar*cfg.w_um),0,1),coef['da'],coef['bottom_biot'])
                        trans=np.full((H,W),.97)
                        trans[y0:y1]=10**(-(base_nm*concentration[None,:])/450.+rng.normal(0,.001,(y1-y0,W)))
                        img=np.round(dark+(ref.astype(float)-dark)*trans+rng.normal(0,20,(H,W))).clip(0,65534).astype(np.uint16)
                        path=f'raw/om/{iid}.tif';Image.fromarray(img).save(root/path,compression='tiff_deflate')
                        lo=z-cfg.window_half_um;hi=z+cfg.window_half_um
                        truth=base_nm*np.exp(window_log_mean(lo*1e-6,hi*1e-6,g.L_m,coef['da'],coef['bottom_biot']))
                        label=max(.1,truth+rng.normal(0,.4))
                        images.append(dict(image_id=iid,path=path,kind='trench',run_id=rid,chip=chip,slot=slot,pos=pos,specimen_id=specimen,matched_region_id='ROI_'+iid,measurement_id=mid,z_anchor_um=z,x_anchor_px=x_anchor,**common))
                        sem.append(dict(run_id=rid,chip=chip,slot=slot,pos=pos,thickness_nm=label,thickness_sd_nm=.4,n_sites=3,sem_file='evidence/SYNTHETIC_fesem.txt',measurement_id=mid,matched_region_id='ROI_'+iid,z_lo_um=lo,z_hi_um=hi,match_basis='validated_region_average',source='SYNTHETIC numerical label, not FESEM'))
                        manual.append(dict(image_id=iid,measurement_id=mid,thickness_nm_manual=label*(1+rng.normal(0,.12)),method='SYNTHETIC relative normal noise SD=12%',source='SYNTHETIC demonstration baseline'))
            if idx in (1,4):
                iid=f'CAL_{T:g}C_G{gly:g}_glass_10min';nm=ks['glass']*cfg.C0_mol_m3*cfg.Vm_m3_mol*600*1e9
                img=np.round(dark+(ref.astype(float)-dark)*10**(-nm/450)+rng.normal(0,20,(H,W))).astype(np.uint16)
                path=f'raw/calibration/{iid}.tif';Image.fromarray(img).save(root/path,compression='tiff_deflate')
                images.append(dict(image_id=iid,path=path,kind='flat',run_id=rid,chip=np.nan,slot=np.nan,pos='flat',specimen_id='SYNTHETIC_FLAT_'+iid,matched_region_id='ROI_'+iid,measurement_id='',z_anchor_um=np.nan,x_anchor_px=np.nan,**common))
    tables=dict(runs=runs,layout=layout,frames=frames,images=images,fesem=sem,manual_om=manual,calibration_thickness=cal,viscosity=visc)
    for name,records in tables.items():write_csv(root/'meta'/f'{name}.csv',pd.DataFrame(records,columns=SCHEMAS[name]))
    means=dict(zip(NAMES,[30e-6,30e-6,49*30e-6,2e-8*cfg.C0_mol_m3*cfg.Vm_m3_mol,1.3e-8*cfg.C0_mol_m3*cfg.Vm_m3_mol,.0006,cfg.D_ref_m2_s,cfg.C0_mol_m3,cfg.Vm_m3_mol,100.,35.]))
    rel=np.array([.01,.02,.01,.05,.05,.02,.1,.03,.01,.04,.04]);cov=np.diag(rel**2);cov[3,4]=cov[4,3]=.5*rel[3]*rel[4];cov[9,10]=cov[10,9]=.7*rel[9]*rel[10]
    write_json(root/'uncertainty.json',dict(means=means,log_covariance=cov.tolist(),repeats=1000,seed=cfg.seed,T_C=65.,model_discrepancy_sd_fraction=.02,assumptions='SYNTHETIC: geometry, rates, viscosity, D reference, C0, Vm and paired thickness errors; shared planar-batch rate correlation=.5; shared metrology correlation=.7. Model discrepancy SD=.02 is assumed, not measured.'))
    write_json(root/'generator.json',dict(origin='SYNTHETIC',seed=cfg.seed,config=asdict(cfg),n_images=len(images),n_trenches=len(sem)//2,n_fesem_surrogate_labels=len(sem),n_planar_points=len(cal),camera_noise_counts_sd=20,OD_texture_sd=.001,label_noise_sd_nm=.4,manual_relative_noise_sd=.12,OD_law='OD=thickness_nm/450 is a chosen synthetic law, not a measured optical law',boundary_truth_pixels=[y0,y1],planar_noise='zero for exact regression check; supplied SD=0.5 nm handled independently',calibration_and_OM_common_physics=True))
    return root
