import numpy as np
import pytest
from scipy.integrate import quad
from cvd_cbd.io_utils import to_si
from cvd_cbd.physics import Geometry,rectangular_coefficients,analytic_profile,log_profile,window_log_mean,observed_sc,observation_windows,bracketed_threshold,fdm_profile

@pytest.mark.parametrize('v,u,expected',[(30,'um',30e-6),(400,'nm',400e-9),(10,'min',600),(2,'h',7200),(1,'mPa_s',.001),(25,'C',298.15)])
def test_units(v,u,expected):assert to_si(v,u)==pytest.approx(expected)

@pytest.mark.parametrize('unit', ['bad','micron?'])
def test_unknown_units(unit):
    with pytest.raises(ValueError):to_si(1,unit)


def test_rectangular_material_perimeters_and_independent_end():
    g=Geometry(30e-6,50e-6,270e-6)
    assert g.perimeters==pytest.approx({'glass':30e-6,'pdms':130e-6})
    ks={'glass':1e-8,'pdms':3e-8};c=rectangular_coefficients(g,ks,1e-9,7e-8)
    assert c['sink_rate_s_inv']==pytest.approx((1e-8*30e-6+3e-8*130e-6)/(30e-6*50e-6))
    assert c['bottom_biot']==pytest.approx(7e-8*270e-6/1e-9)
    c2=rectangular_coefficients(Geometry(30e-6,50e-6,270e-6,2),ks,1e-9)
    assert c2['sink_rate_s_inv']==pytest.approx(2*3e-8/30e-6)
    with pytest.raises(ValueError):rectangular_coefficients(g,{'glass':1e-8},1e-9)

@pytest.mark.parametrize('da,bi',[(0,0),(0,2),(.216,0),(4,.5),(100,10),(1e-18,1e8)])
def test_integrated_observable_independent_quadrature(da,bi):
    expected=quad(lambda x:float(analytic_profile(x,da,bi)),.2,.7,epsabs=1e-12)[0]/.5
    assert np.exp(window_log_mean(.2,.7,1,da,bi))==pytest.approx(expected,rel=1e-10)


def test_large_phi_and_bi_log_and_zero_limits():
    for da,bi in [(1e6,1e300),(0,1e308),(1e-18,1e308)]:
        x=np.linspace(0,1,11);u=analytic_profile(x,da,bi)
        assert np.isfinite(u).all() and u[0]==pytest.approx(1)
        assert np.isfinite(log_profile(x,da,bi)).all()
    g=Geometry(1.,1.,1.)
    assert observed_sc(g,{'glass':2.5e5,'pdms':2.5e5},1.,top=(.8,.8),bot=(.9,.9))==pytest.approx(np.exp(-100),rel=1e-10)


def test_observation_boundaries_and_endpoint_difference():
    g=Geometry(30e-6,30e-6,270e-6);ks={'glass':2e-8,'pdms':3e-8}
    with pytest.raises(ValueError):observation_windows(g.L_m,50e-6,100e-6)
    with pytest.raises(ValueError):observation_windows(120e-6,50e-6,16e-6)
    a,b=observation_windows(g.L_m,50e-6,16e-6)
    assert observed_sc(g,ks,1e-9,3e-8,top=a,bot=b)>observed_sc(g,ks,1e-9,3e-8)
    with pytest.raises(ValueError):observed_sc(g,ks,1e-9,top=(0,200e-6),bot=(100e-6,200e-6))


def test_critical_and_no_solution():
    r=bracketed_threshold(lambda da:float(analytic_profile(1,da)),.9)
    assert r['value']==pytest.approx(np.arccosh(1/.9)**2,abs=1e-10)
    assert bracketed_threshold(lambda da:1.,.9)['status']=='no_boundary_within_domain'
    assert bracketed_threshold(lambda da:.5,.9)['status']=='below_target_at_lower_bound'

@pytest.mark.parametrize('da,bi',[(.2,0),(4,.5),(100,10)])
def test_preserved_second_order_fdm(da,bi):
    errs=[]
    for n in [101,201,401]:
        x,u=fdm_profile(da,bi,n);errs.append(np.max(abs(u-analytic_profile(x,da,bi))))
    assert 3.8<errs[0]/errs[1]<4.2
    assert 3.8<errs[1]/errs[2]<4.2
