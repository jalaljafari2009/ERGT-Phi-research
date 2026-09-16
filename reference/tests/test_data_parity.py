from ergt_four_seed.data_registry import verify_registered_data


def test_registered_data_parity():
    result = verify_registered_data()
    assert result["pass"]
    assert result["data_seeds"] == [21101, 22307, 23509, 24733]
    assert result["cohort_count"] > 40
