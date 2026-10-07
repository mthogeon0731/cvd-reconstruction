"""Quantitative OD/T features, separate CLAHE texture, conservative channel QC."""
import numpy as np
from scipy import ndimage,signal
from .imaging import clahe
from .input_contracts import scalar_pixel_scale,reject_axis_pixel_calibration

FEATURES=['od_mean','od_std','od_p10','od_p90','T_mean','texture_gradient','texture_neighbor']


def canonical(a,orientation):
    if orientation=='x+':return a
    if orientation=='x-':return np.flip(a,axis=1)
    if orientation=='y+':return np.rot90(a,1)
    if orientation=='y-':return np.rot90(a,-1)
    raise ValueError('Unregistered orientation')


def select_channel(a,name):
    if a.dtype.kind!='u' or a.dtype.itemsize not in (1,2) or min(a.shape[:2])<2:raise ValueError('Nonempty unsigned 8/16-bit TIFF image required')
    if a.ndim==2 and name=='mono':return a
    if a.ndim==3 and a.shape[2]==3 and name in ('red','green','blue'):return a[:,:,('red','green','blue').index(name)]
    raise ValueError('RGB/monochrome metadata mismatch; specify a calibrated channel')


def correct(img,ref,dark,bit_depth,cfg):
    if img.shape!=ref.shape or img.shape!=dark.shape:raise ValueError('Image/REF/DARK size mismatch')
    if img.ndim!=2 or img.size==0:raise ValueError('Empty/nonmonochrome image')
    if bit_depth not in (8,12,16):raise ValueError('Invalid bit depth')
    maximum=2**bit_depth-1
    if any(a.dtype.kind!='u' or np.max(a)>maximum for a in (img,ref,dark)):raise ValueError('Stored range inconsistent with camera bit depth')
    i,r,d=[a.astype(float) for a in (img,ref,dark)];den=r-d
    zero=den<=0;sat=(i>=maximum)|(r>=maximum)|(d>=maximum);negative=i-d<=0
    T=np.divide(i-d,den,out=np.full_like(i,np.nan),where=~zero)
    low=T<cfg.t_min;high=T>cfg.t_max;bad=zero|sat|negative|low|high|~np.isfinite(T)
    qc={'zero_denominator_pixels':int(zero.sum()),'saturated_pixels':int(sat.sum()),'nonpositive_signal_pixels':int(negative.sum()),'low_T_pixels':int(low.sum()),'high_T_pixels':int(high.sum()),'invalid_pixels':int(bad.sum()),'total_pixels':img.size,'clipped_quantitative_pixels':0}
    T[bad]=np.nan
    if bad.all() or bad.mean()>cfg.invalid_pixel_fraction_max:raise ValueError(f'Invalid corrected pixels exceed limit: {qc}')
    od=-np.log10(T)
    display=np.nan_to_num(T,nan=float(np.nanmedian(T)))
    # Fixed [0,t_max] texture normalization; never used as quantitative intensity.
    texture=clahe(np.clip(display/cfg.t_max,0,1))
    return T,od,texture,~bad,qc


def detect_channel(texture,width_px):
    if not np.isfinite(width_px) or width_px<=0 or width_px>=texture.shape[0]:raise ValueError('Expected channel width incompatible with image/scale')
    profile=np.abs(np.diff(texture,axis=0)).mean(axis=1)
    profile=ndimage.gaussian_filter1d(profile,.6)
    med=np.median(profile);mad=np.median(np.abs(profile-med))
    threshold=max(.003,med+6*mad)
    peaks,_=signal.find_peaks(profile,height=threshold,distance=3)
    pairs=[(profile[a]+profile[b],a+1,b+1) for a in peaks for b in peaks if .85*width_px<=b-a<=1.15*width_px]
    if not pairs:raise ValueError('Channel detection failed: provide manual ROI/mask; no boundary invented')
    pairs.sort(reverse=True)
    if len(pairs)>1 and pairs[1][0]>.95*pairs[0][0] and abs(pairs[0][1]-pairs[1][1])>3:raise ValueError('Ambiguous channel boundaries: manual ROI required')
    _,y0,y1=pairs[0]
    return int(y0),int(y1)


def tile_boxes(shape,row,cfg,boundaries=None,mask=None):
    reject_axis_pixel_calibration(row)
    H,W=shape;s=cfg.tile_px;scale=scalar_pixel_scale(row['um_per_px']);mode=row['roi_mode']
    def pixel(k):
        v=float(row[k])
        if not np.isfinite(v) or v!=int(v):raise ValueError(f'Integer canonical ROI coordinate required: {k}')
        return int(v)
    if row['kind']=='trench':
        z=float(row['z_anchor_um']);xc=float(row['x_anchor_px']);L=float(row['L_um'])
        if not np.isfinite([z,xc]).all() or not 0<=z<=L or not 0<=xc<=W:raise ValueError('Invalid physical z/image anchor')
        if z-cfg.window_half_um<0 or z+cfg.window_half_um>L:raise ValueError('z-window crosses physical channel boundary')
        x0=int(np.ceil(xc-cfg.window_half_um/scale));x1=int(np.floor(xc+cfg.window_half_um/scale))
        y0,y1=boundaries
        y0+=cfg.wall_margin_px;y1-=cfg.wall_margin_px
        if mode=='manual':
            x0=max(x0,pixel('roi_x0'));x1=min(x1,pixel('roi_x1'))
            y0=max(y0,pixel('roi_y0'));y1=min(y1,pixel('roi_y1'))
        if x0<0 or x1>W or y0<0 or y1>H:raise ValueError('ROI/window exceeds image; no negative slices')
    else:
        x0,y0,x1,y1=0,0,W,H
        if mode=='manual':x0,x1,y0,y1=[pixel(k) for k in ('roi_x0','roi_x1','roi_y0','roi_y1')]
        if not 0<=x0<x1<=W or not 0<=y0<y1<=H:raise ValueError('Flat ROI outside image')
    if min(x1-x0,y1-y0)<s:raise ValueError('Tile exceeds channel/ROI/window after wall margin; choose calibrated tile size')
    nx=(x1-x0)//s;ny=(y1-y0)//s;xstart=x0+(x1-x0-nx*s)//2;ystart=y0+(y1-y0-ny*s)//2
    boxes=[]
    for y in range(ystart,ystart+ny*s,s):
        for x in range(xstart,xstart+nx*s,s):
            if mask is None or mask[y:y+s,x:x+s].all():boxes.append((x,y,x+s,y+s))
    if not boxes:raise ValueError('No complete tiles inside manual mask')
    return boxes,(x0,y0,x1,y1)


def tile_features(T,od,texture):
    if T.size<16 or not np.isfinite(T).all() or not np.isfinite(od).all():raise ValueError('Empty/invalid quantitative tile')
    gy,gx=np.gradient(texture)
    return dict(od_mean=float(od.mean()),od_std=float(od.std()),od_p10=float(np.quantile(od,.1)),od_p90=float(np.quantile(od,.9)),T_mean=float(T.mean()),texture_gradient=float(np.hypot(gx,gy).mean()),texture_neighbor=float(np.abs(texture-ndimage.uniform_filter(texture,size=3)).mean()))


def overlay(path,img,T,texture,boundaries,roi,boxes,title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    for ax,a,label in zip(axes,[img,T,texture],['Original counts','T = (I-DARK)/(REF-DARK)','CLAHE texture + ROI + tiles']):
        maximum=float(np.iinfo(img.dtype).max) if label=='Original counts' else (2. if label.startswith('T =') else 1.)
        handle=ax.imshow(a,cmap='gray',vmin=0,vmax=maximum);ax.set_title(label,fontsize=9);ax.axis('off')
        if label.startswith('T ='):fig.colorbar(handle,ax=ax,fraction=.025,pad=.02)
    ax=axes[-1]
    if boundaries:
        for y in boundaries:ax.axhline(y,color='orange',lw=1)
    x0,y0,x1,y1=roi;ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='cyan',lw=1.5))
    for x0,y0,x1,y1 in boxes:ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='lime',lw=.8))
    fig.suptitle(title,fontsize=10);fig.savefig(path,dpi=100);plt.close(fig)
