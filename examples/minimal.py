"""Small, deterministic synthetic physics example; no file or model inputs."""
import json
from cvd_cbd.physics import Geometry, rectangular_coefficients, analytic_profile

geometry = Geometry(30e-6, 30e-6, 270e-6)
coefficients = rectangular_coefficients(
    geometry, {'glass': 2e-8, 'pdms': 3e-8}, 1e-9, 3e-8
)
sc = float(analytic_profile(1.0, coefficients['da'], coefficients['bottom_biot']))
assert 0.0 < sc <= 1.0
print(json.dumps({'origin': 'SYNTHETIC', 'observable': 'endpoint_concentration_ratio',
                  'SC': sc, 'experimental_validation': 'NOT RUN'}, indent=2))
