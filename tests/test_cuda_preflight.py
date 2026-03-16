from ttt.research.cuda_preflight import _minimum_driver_for_cuda_runtime, _parse_version_tuple


def test_parse_version_tuple_handles_semver_strings():
    assert _parse_version_tuple("570.195.03") == (570, 195, 3)
    assert _parse_version_tuple("12.9.79") == (12, 9, 79)
    assert _parse_version_tuple(None) is None


def test_minimum_driver_for_cuda_12_9_is_frozen():
    assert _minimum_driver_for_cuda_runtime("12.9.79") == "575.51.03"
    assert _minimum_driver_for_cuda_runtime("12.8.1") is None
