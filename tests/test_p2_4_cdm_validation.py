import pytest
from astra.cdm import parse_cdm_xml
from astra.errors import AstraError

def test_cdm_validation_miss_distance_p2_4():
    """Verify that negative miss distance triggers validation failure."""
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <CDM>
        <BODY>
            <RELATIVE_METADATA_BLOCK>
                <TCA>2024-01-01T12:00:00Z</TCA>
                <MISS_DISTANCE>-100.0</MISS_DISTANCE>
                <RELATIVE_SPEED>7000.0</RELATIVE_SPEED>
            </RELATIVE_METADATA_BLOCK>
        </BODY>
    </CDM>"""
    with pytest.raises(AstraError) as excinfo:
        parse_cdm_xml(xml)
    assert "Negative miss distance" in str(excinfo.value)
    print("Negative miss distance test PASSED.")

def test_cdm_validation_rel_vel_p2_4():
    """Verify that negative relative speed triggers validation failure."""
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <CDM>
        <BODY>
            <RELATIVE_METADATA_BLOCK>
                <TCA>2024-01-01T12:00:00Z</TCA>
                <MISS_DISTANCE>500.0</MISS_DISTANCE>
                <RELATIVE_SPEED>-5.0</RELATIVE_SPEED>
            </RELATIVE_METADATA_BLOCK>
        </BODY>
    </CDM>"""
    with pytest.raises(AstraError) as excinfo:
        parse_cdm_xml(xml)
    assert "Negative relative velocity" in str(excinfo.value)
    print("Negative relative velocity test PASSED.")

def test_cdm_validation_pc_range_p2_4():
    """Verify that out-of-range Pc triggers validation failure."""
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <CDM>
        <BODY>
            <RELATIVE_METADATA_BLOCK>
                <TCA>2024-01-01T12:00:00Z</TCA>
                <MISS_DISTANCE>500.0</MISS_DISTANCE>
                <RELATIVE_SPEED>7000.0</RELATIVE_SPEED>
                <COLLISION_PROBABILITY>1.5</COLLISION_PROBABILITY>
            </RELATIVE_METADATA_BLOCK>
        </BODY>
    </CDM>"""
    with pytest.raises(AstraError) as excinfo:
        parse_cdm_xml(xml)
    assert "out of range [0, 1]" in str(excinfo.value)
    print("Out-of-range Pc test PASSED.")

if __name__ == "__main__":
    test_cdm_validation_miss_distance_p2_4()
    test_cdm_validation_rel_vel_p2_4()
    test_cdm_validation_pc_range_p2_4()
