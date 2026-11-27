import math

import pytest

import audiopitch

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _p():
    return audiopitch.Pitch(8000, 80.0, 1000.0)


def _sine(hz):
    return [math.sin(2 * math.pi * hz * i / 8000) for i in range(800)]

CASES = [
    ("detect", lambda: abs(_p().detect(_sine(200.0)) - 200.0) <= 2.0),
    ("confidence", lambda: _p().detect(_sine(200.0)) and 0.0 <= _p().confidence() <= 1.0),
    ("silence", lambda: _p().detect([0.0] * 800) == 0.0),
    ("harmonics", lambda: abs(_p().detect([0.5 * _sine(200.0)[i] + 0.2 * _sine(400.0)[i] for i in range(800)]) - 200.0) <= 4.0),
    ("bad_rate", lambda: _raises(lambda: audiopitch.Pitch(0, 80.0, 1000.0))),
    ("bad_range", lambda: _raises(lambda: audiopitch.Pitch(8000, 1000.0, 80.0))),
    ("bad_samples", lambda: _raises(lambda: _p().detect([]))),
    ("type", lambda: isinstance(_p().detect(_sine(200.0)), float)),
    ("det_det", lambda: _p().detect(_sine(200.0)) == _p().detect(_sine(200.0))),
    ("nan", lambda: _raises(lambda: _p().detect([float('nan')] * 800))),
    ("no_mut", lambda: _p().detect(_sine(300.0)) is not None),
    ("high", lambda: abs(_p().detect(_sine(800.0)) - 800.0) <= 8.0),
    ("conf_type", lambda: _p().detect(_sine(200.0)) and isinstance(_p().confidence(), float)),
    ("low", lambda: abs(_p().detect(_sine(100.0)) - 100.0) <= 1.0),
    ("bounded", lambda: 0.0 <= _p().detect(_sine(500.0)) <= 1000.0),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
