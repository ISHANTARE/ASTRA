import astra
try:
    print(f"NumericalState: {astra.NumericalState}")
except AttributeError as e:
    print(f"FAILED: {e}")

print("Checking globals of astra...")
import astra
if "NumericalState" in dir(astra):
    print("NumericalState is in dir(astra)")
else:
    print("NumericalState is NOT in dir(astra)")
