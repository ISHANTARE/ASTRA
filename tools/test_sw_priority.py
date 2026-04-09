import sys
import logging
from datetime import datetime, timezone
import astra
from astra import data_pipeline

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

print("--- Testing Spacebook Integration in get_space_weather() ---")
jd_today = 2451545.0 + (datetime.now(timezone.utc) - datetime(2000, 1, 1, 12, tzinfo=timezone.utc)).total_seconds() / 86400.0

print("\n1. With Spacebook ENABLED (default):")
sw = data_pipeline.get_space_weather(jd_today)
print(f"Result: {sw}")

print("\n2. With Spacebook DISABLED:")
import os
os.environ["ASTRA_SPACEBOOK_ENABLED"] = "false"
# Need to reload module config
import importlib
importlib.reload(astra.spacebook)

sw_celestrak = data_pipeline.get_space_weather(jd_today)
print(f"Result: {sw_celestrak}")

print("\nCheck complete.")
