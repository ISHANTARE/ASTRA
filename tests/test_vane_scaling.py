import pytest
import numpy as np
from astra.models import SatelliteTLE
from astra.orbit import propagate_many_generator

@pytest.fixture
def dummy_tles():
    line1 = "1 25544U 98067A   21001.00000000  .00001480  00000-0  34282-4 0  9997"
    line2 = "2 25544  51.6442 284.1199 0001364 338.5498  21.5664 15.48922536 12341"
    
    line1b = "1 99999U 98067A   21001.00000000  .00001480  00000-0  34282-4 0  9995"
    line2b = "2 99999  51.6442 284.1199 0001364 338.5498  21.5664 15.48922536 12346"
    
    sat_a = SatelliteTLE(
        norad_id="25544", name="ISS", line1=line1, line2=line2, 
        epoch_jd=2459215.5, object_type="PAYLOAD"
    )
    sat_b = SatelliteTLE(
        norad_id="99999", name="TEST", line1=line1b, line2=line2b, 
        epoch_jd=2459215.5, object_type="PAYLOAD"
    )
    return [sat_a, sat_b]

def test_propagate_many_generator_chunking(dummy_tles):
    times_jd = np.linspace(2459215.5, 2459216.5, 2000) # 2000 steps
    chunk_size = 500
    
    gen = propagate_many_generator(dummy_tles, times_jd, chunk_size=chunk_size)
    chunks_received = 0
    total_times_received = 0
    
    for t_chunk, results_map in gen:
        chunks_received += 1
        chunk_len = len(t_chunk)
        total_times_received += chunk_len
        
        assert chunk_len <= chunk_size
        assert "25544" in results_map
        assert "99999" in results_map
        
        # Verify shape of results
        assert results_map["25544"].shape == (chunk_len, 3)
        assert results_map["99999"].shape == (chunk_len, 3)
        
    assert chunks_received == 4  # 2000 / 500 = 4 chunks exactly
    assert total_times_received == 2000
