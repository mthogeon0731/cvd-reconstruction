from dataclasses import replace
import numpy as np
import pytest
from cvd_cbd.study_config import StudyConfig
from cvd_cbd import metrology as im
from cvd_cbd.manifest import parse_filename,check_acquisition


def test_dark_ref_od_and_no_clipping():
    cfg=StudyConfig();dark=np.full((32,32),1000,dtype=np.uint16);ref=dark+40000;img=dark+20000
    T,od,cl,valid,qc=im.correct(img,ref,dark,16,cfg)
    np.testing.assert_allclose(T,.5);np.testing.assert_allclose(od,np.log10(2));assert valid.all() and qc['clipped_quantitative_pixels']==0
    f=im.tile_features(T,od,cl);assert f['od_mean']==pytest.approx(np.log10(2))

@pytest.mark.parametrize('kind',['shape','saturation','zero','negative','bad_range'])
def test_bad_imaging_inputs(kind):
    cfg=StudyConfig();d=np.full((16,16),1000,dtype=np.uint16);r=d+40000;i=d+10000;bits=16
    if kind=='shape':r=r[:8]
    elif kind=='saturation':i[:]=65535
    elif kind=='zero':r=d
    elif kind=='negative':i[:]=0
    elif kind=='bad_range':bits=12
    with pytest.raises(ValueError):im.correct(i,r,d,bits,cfg)


def test_flat_detection_refuses_and_manual_tiles_bounds():
    with pytest.raises(ValueError):im.detect_channel(np.ones((128,256))*.5,60)
    cfg=StudyConfig();row=dict(kind='trench',um_per_px=.5,roi_mode='auto',z_anchor_um=50.,x_anchor_px=128.,L_um=270.)
    boxes,_=im.tile_boxes((128,256),row,cfg,(34,94));assert len(boxes)==2
    assert all(38<=b[1]<b[3]<=90 for b in boxes)
    with pytest.raises(ValueError):im.tile_boxes((128,256),row,replace(cfg,tile_px=128),(34,94))
    with pytest.raises(ValueError):im.tile_boxes((128,256),row,replace(cfg,window_half_um=100),(34,94))
    row['x_anchor_px']=0
    with pytest.raises(ValueError):im.tile_boxes((128,256),row,cfg,(34,94))
    flat=dict(kind='flat',um_per_px=.5,roi_mode='manual',roi_x0=0,roi_x1=64,roi_y0=0,roi_y1=64)
    boxes,_=im.tile_boxes((64,64),flat,cfg);assert len(boxes)==4
    with pytest.raises(ValueError):im.tile_boxes((64,64),flat,cfg,mask=np.zeros((64,64),bool))


def test_color_bit_depth_orientation_and_filename():
    with pytest.raises(ValueError):im.select_channel(np.ones((3,3,3),np.uint16),'mono')
    a=np.arange(12,dtype=np.uint16).reshape(3,4);assert im.canonical(a,'y+').shape==(4,3)
    assert parse_filename('R07_C2_T5_bot.tif')['slot']==5
    assert parse_filename('CAL_80C_G20_pdms_30min.tif')['t_min']==30
    with pytest.raises(ValueError):parse_filename('T80_random.tif')
    row=dict(session_id='s',exposure_ms=1,gain=1,illumination_id='i',magnification=20,bit_depth=16,auto_exposure=True,gamma=1,white_balance='fixed',color_channel='mono')
    with pytest.raises(ValueError,match='fixed exposure'):check_acquisition(row)
