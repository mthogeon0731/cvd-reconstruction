"""Steady, dilute, 1-D slit model. See docs/MODEL.md for assumptions and units."""
import numpy as np
from scipy.linalg import solve_banded
from .config import Config

GAS_CONSTANT_J_MOL_K = 8.31446261815324


def _finite_nonnegative(value, name):
    a = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(a)) or np.any(a < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return a


def coefficients(aspect_ratio, viscosity_mpa_s, temperature_c, cfg: Config):
    ar, eta, tc = np.broadcast_arrays(
        np.asarray(aspect_ratio, float), np.asarray(viscosity_mpa_s, float),
        np.asarray(temperature_c, float))
    if any(np.any(~np.isfinite(a)) for a in (ar, eta, tc)):
        raise ValueError("All physical inputs must be finite")
    if np.any(ar <= 0) or np.any(eta <= 0) or np.any(tc <= -273.15):
        raise ValueError("AR/viscosity must be positive; temperature must exceed absolute zero")
    temperature_k = tc + 273.15
    tref = cfg.reference_temperature_c + 273.15
    width_m = cfg.width_um * 1e-6
    depth_m = ar * width_m
    # Relative Stokes-Einstein scaling: viscosity is its actual value at T,
    # NOT a glycerol fraction or a viscosity measured at another temperature.
    diffusion = cfg.diffusivity_ref_m2_s * (temperature_k / tref) * (
        cfg.reference_viscosity_mpa_s / eta)
    exponent = (cfg.activation_energy_j_mol / GAS_CONSTANT_J_MOL_K) * (1 / tref - 1 / temperature_k)
    with np.errstate(over="ignore", under="ignore"):
        surface_rate = cfg.surface_rate_ref_m_s * np.exp(exponent)
    sink_rate = 2 * surface_rate / width_m  # m/s * 1/m = 1/s; two slit sidewalls
    da = sink_rate * depth_m ** 2 / diffusion
    beta = cfg.bottom_surface_rate_ratio * surface_rate * depth_m / diffusion
    for name, a in (("diffusivity", diffusion), ("surface rate", surface_rate), ("Da", da), ("beta", beta)):
        if np.any(~np.isfinite(a)):
            raise ValueError(f"Numerically nonfinite {name}; narrow parameter bounds")
    if np.any(diffusion <= 0):
        raise ValueError("Diffusivity underflow; narrow parameter bounds")
    return {"depth_m": depth_m, "diffusivity_m2_s": diffusion,
            "surface_rate_m_s": surface_rate, "sink_rate_s_inv": sink_rate,
            "da": da, "bottom_biot": beta}


def analytic_profile(x, da, bottom_biot=0.0):
    """u'' - Da*u=0, u(0)=1, -u'(1)=Bi*u(1); x is z/L.

    Broadcastable x, Da, Bi. Numerically stable even for large sqrt(Da).
    Bi=0 is a reflecting/axially no-flux end (deep sidewall proxy).
    """
    x, da, bi = np.broadcast_arrays(np.asarray(x, float),
                                  _finite_nonnegative(da, "Da"),
                                  _finite_nonnegative(bottom_biot, "bottom Biot"))
    if np.any(~np.isfinite(x)) or np.any((x < 0) | (x > 1)):
        raise ValueError("x=z/L must be in [0, 1]")
    phi = np.sqrt(da)
    zero = phi < 1e-10
    safe_phi = np.where(zero, 1.0, phi)
    # Divide numerator and denominator by exp(phi). Equivalent to hyperbolic form.
    # Positive scaled cosh/sinh terms avoid cancellation at large Bi/phi.
    # Rescale Bi as well, so an extreme but finite Robin coefficient cannot overflow.
    scale = np.maximum(1.0, bi)
    y = safe_phi * (1 - x)
    numerator = np.exp(-safe_phi * x) * (
        (1 + np.exp(-2 * y)) / scale
        + (bi / scale) * (-np.expm1(-2 * y) / safe_phi))
    denominator = ((1 + np.exp(-2 * safe_phi)) / scale
                   + (bi / scale) * (-np.expm1(-2 * safe_phi) / safe_phi))
    regular = numerator / denominator
    linear = (1 / scale + (bi / scale) * (1 - x)) / (1 / scale + bi / scale)
    return np.where(zero, linear, regular)


def step_coverage_proxy(aspect_ratio, viscosity_mpa_s, temperature_c, cfg):
    c = coefficients(aspect_ratio, viscosity_mpa_s, temperature_c, cfg)
    return analytic_profile(1.0, c["da"], c["bottom_biot"])


def fdm_profile(da: float, bottom_biot: float = 0.0, nodes: int = 301):
    """Second-order central FDM; Robin end via a ghost node. No explicit-time stability issue."""
    da = float(_finite_nonnegative(da, "Da"))
    bi = float(_finite_nonnegative(bottom_biot, "bottom Biot"))
    if not isinstance(nodes, int) or isinstance(nodes, bool) or nodes < 3:
        raise ValueError("nodes must be an integer >= 3")
    x = np.linspace(0, 1, nodes)
    h = 1 / (nodes - 1)
    # Banded matrix format: A[i,i]=main[i], A[i,i+1]=upper[i], A[i+1,i]=lower[i]
    main = np.full(nodes, 2 + da * h * h)
    upper = np.full(nodes - 1, -1.0)
    lower = np.full(nodes - 1, -1.0)
    main[0], upper[0] = 1.0, 0.0
    main[-1], lower[-1] = 2 + 2 * h * bi + da * h * h, -2.0
    banded = np.zeros((3, nodes))
    banded[0, 1:], banded[1], banded[2, :-1] = upper, main, lower
    rhs = np.zeros(nodes)
    rhs[0] = 1.0
    u = solve_banded((1, 1), banded, rhs)
    return x, u


def illustrative_da_at_sc(sc):
    """For Bi=0 only. An algebraic proxy threshold, NOT empirical void Dacrit."""
    sc = np.asarray(sc, float)
    if np.any(~np.isfinite(sc)) or np.any((sc <= 0) | (sc > 1)):
        raise ValueError("SC must be in (0, 1]")
    return np.arccosh(1 / sc) ** 2

# Generalized study model; reuses the analytic/FDM implementation above.
from dataclasses import dataclass
from scipy.special import logsumexp
from scipy.optimize import brentq

@dataclass(frozen=True)
class Geometry:
    w_m: float
    h_m: float
    L_m: float
    faces: int = 4
    materials: tuple = ('pdms','pdms','glass','pdms')

    def __post_init__(self):
        if any(not np.isfinite(v) or v<=0 for v in (self.w_m,self.h_m,self.L_m)) or self.faces not in (2,4): raise ValueError('Positive SI geometry and 2/4 faces required')
        if len(self.materials)!=4 or any(s not in ('glass','pdms','inert') for s in self.materials): raise ValueError('Invalid side materials')

    @property
    def AR(self):return self.L_m/self.w_m

    @property
    def perimeters(self):
        result={}
        for material,length in list(zip(self.materials,(self.h_m,self.h_m,self.w_m,self.w_m)))[:self.faces]:
            if material!='inert':result[material]=result.get(material,0.)+length
        return result


def rectangular_coefficients(g,ks,D,k_end=0.):
    if not np.isfinite(D) or D<=0:raise ValueError('Positive D required')
    if not np.isfinite(k_end) or k_end<0:raise ValueError('Nonnegative separate k_end required')
    if set(g.perimeters)-set(ks):raise ValueError('Missing substrate kinetics')
    if any(not np.isfinite(k) or k<0 for k in ks.values()):raise ValueError('Nonnegative kinetics required')
    kp=sum(ks[s]*p for s,p in g.perimeters.items())
    sink=kp/(g.w_m*g.h_m)
    if not np.isfinite([sink,sink*g.L_m**2/D,k_end*g.L_m/D]).all():raise ValueError('Nonfinite derived physical coefficients')
    return dict(da=sink*g.L_m**2/D,bottom_biot=k_end*g.L_m/D,
                sink_rate_s_inv=sink,ks_effective=kp/sum(g.perimeters.values()) if g.perimeters else 0.)


def log_profile(x,da,bi=0.):
    """log(u) avoids 0/0 in ratios of deeply attenuated concentrations."""
    x,da,bi=np.broadcast_arrays(np.asarray(x,float),_finite_nonnegative(da,'Da'),_finite_nonnegative(bi,'Bi'))
    if np.any(~np.isfinite(x)) or np.any((x<0)|(x>1)):raise ValueError('x outside [0,1]')
    p=np.sqrt(da);zero=p<1e-10;q=np.where(zero,1.,p);scale=np.maximum(1.,bi)
    def term(y):return (1+np.exp(-2*y))/scale+(bi/scale)*(-np.expm1(-2*y)/q)
    regular=-q*x+np.log(term(q*(1-x)))-np.log(term(q))
    linear=np.log(1/scale+(bi/scale)*(1-x))-np.log(1/scale+bi/scale)
    return np.where(zero,linear,regular)


def window_log_mean(lo,hi,L,da,bi):
    if not np.isfinite([lo,hi,L]).all() or L<=0 or not 0<=lo<=hi<=L:raise ValueError('Observation window outside channel')
    if hi==lo:return float(log_profile(lo/L,da,bi))
    # Closed-form integral: exp(-phi*x)[A+B exp(-2phi(1-x))].
    da=float(_finite_nonnegative(da,'Da'));bi=float(_finite_nonnegative(bi,'Bi'))
    p=np.sqrt(da)
    if p<1e-10:return float(log_profile((lo+hi)/(2*L),da,bi))
    a,b=lo/L,hi/L;d=b-a;scale=max(1.,p,bi)
    A=p/scale+bi/scale;B=p/scale-bi/scale
    # Use logarithmic shift for deep windows and negative B without cancellation.
    inner=(2*p/scale-B*(-np.expm1(-2*p*(1-b)-p*d)))*(-np.expm1(-p*d))/(p*d)
    denom=2*p/scale-B*(-np.expm1(-2*p))
    return float(-p*a+np.log(inner)-np.log(denom))


def observed_sc(g,ks,D,k_end=0.,top=(0.,0.),bot=None,rate_ratio=1.):
    bot=bot if bot is not None else (g.L_m,g.L_m)
    if top[1]>=bot[0] and not (top[0]==top[1]==0 and bot[0]==bot[1]==g.L_m):raise ValueError('Top/bot observation windows overlap')
    if not np.isfinite(rate_ratio) or rate_ratio<0:raise ValueError('Invalid observation reaction-rate ratio')
    c=rectangular_coefficients(g,ks,D,k_end)
    a=window_log_mean(*top,g.L_m,c['da'],c['bottom_biot'])
    b=window_log_mean(*bot,g.L_m,c['da'],c['bottom_biot'])
    return float(rate_ratio*np.exp(b-a))


def observation_windows(L,offset,half):
    top=(offset-half,offset+half);bot=(L-offset-half,L-offset+half)
    if half<0 or offset<0 or top[0]<0 or bot[1]>L or top[1]>=bot[0]:raise ValueError('Observation windows cross boundary or overlap')
    return top,bot


def bracketed_threshold(function,target,lower=0.,limit=1e6):
    """Monotone decreasing SC curve; explicitly report unattained thresholds."""
    if not 0<target<1 or not 0<=lower<limit:raise ValueError('Invalid threshold/search domain')
    f0=function(lower)-target
    if f0<0:return {'status':'below_target_at_lower_bound','value':None}
    if f0==0:return {'status':'root','value':float(lower)}
    upper=max(1.,lower*2)
    upper=min(upper,limit)
    while function(upper)>target and upper<limit:upper=min(upper*2,limit)
    if function(upper)>target:return {'status':'no_boundary_within_domain','value':None}
    root=brentq(lambda v:function(v)-target,lower,upper,xtol=1e-12)
    return {'status':'root','value':float(root)}
