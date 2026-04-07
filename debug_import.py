try:
    from astra.propagator import srp_cylindrical_illumination_factor_njit
    print("Import successful!")
except ImportError as e:
    print(f"Import failed: {e}")

from astra.propagator import _srp_illumination_factor_njit
print(f"Internal function exists: {_srp_illumination_factor_njit}")
