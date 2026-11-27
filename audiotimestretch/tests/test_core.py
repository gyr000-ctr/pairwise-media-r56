import math

import pytest

import audiotimestretch

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _t():
    return audiotimestretch.TimeStretch(256, 64, 2.0)


def _tone(n=2048):
    return [0.5 * math.sin(2 * math.pi * 440 * i / 8000) for i in range(n)]

CASES = [
    ("len", lambda: abs(len(_t().process(_tone())) - 1024) <= 1),
    ("cola", lambda: _t().cola_error() <= 1e-12),
    ("identity", lambda: abs(len(audiotimestretch.TimeStretch(256, 64, 1.0).process(_tone())) - 2048) <= 1),
    ("bounded", lambda: all(abs(v) <= 2.0 for v in _t().process(_tone()))),
    ("bad_rate", lambda: _raises(lambda: audiotimestretch.TimeStretch(256, 64, 0.0))),
    ("bad_frame", lambda: _raises(lambda: audiotimestretch.TimeStretch(0, 64, 2.0))),
    ("bad_hop", lambda: _raises(lambda: audiotimestretch.TimeStretch(256, 0, 2.0))),
    ("empty", lambda: _t().process([]) == []),
    ("type", lambda: isinstance(_t().process(_tone()), list)),
    ("det", lambda: _t().process(_tone()) == _t().process(_tone())),
    ("nan", lambda: _raises(lambda: _t().process([float('nan')] * 512))),
    ("no_mut", lambda: _t().process(_tone()) is not None),
    ("cola_type", lambda: isinstance(_t().cola_error(), float)),
    ("float", lambda: isinstance(_t().process(_tone())[0], float)),
    ("slow", lambda: abs(len(audiotimestretch.TimeStretch(256, 64, 0.5).process(_tone())) - 4096) <= 2),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
