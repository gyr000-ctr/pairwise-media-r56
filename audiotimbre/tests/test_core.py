import math

import pytest

import audiotimbre

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _t():
    return audiotimbre.Timbre(8000, 20, 12)


def _tone(hz=440.0, n=512):
    return [0.5 * math.sin(2 * math.pi * hz * i / 8000) for i in range(n)]

CASES = [
    ("bank_shape", lambda: len(_t().mel_bank(512)) == 20),
    ("bank_sum", lambda: abs(sum(sum(f) for f in _t().mel_bank(512)) - 20.0) < 1e-9),
    ("mfcc_len", lambda: len(_t().mfcc(_tone())) == 12),
    ("mfcc_gain_invariant", lambda: all(abs(a - b) < 1e-9 for a, b in zip(_t().mfcc(_tone())[1:], _t().mfcc([2 * v for v in _tone()])[1:]))),
    ("bad_bands", lambda: _raises(lambda: audiotimbre.Timbre(8000, 0, 12))),
    ("bad_coeffs", lambda: _raises(lambda: audiotimbre.Timbre(8000, 20, 0))),
    ("bad_rate", lambda: _raises(lambda: audiotimbre.Timbre(0, 20, 12))),
    ("empty", lambda: _raises(lambda: _t().mfcc([]))),
    ("type", lambda: isinstance(_t().mfcc(_tone()), list)),
    ("det", lambda: _t().mfcc(_tone()) == _t().mfcc(_tone())),
    ("nan", lambda: _raises(lambda: _t().mfcc([float('nan')] * 512))),
    ("no_mut", lambda: _t().mfcc(_tone()) is not None),
    ("dct", lambda: isinstance(_t().dct([1.0] * 20), list)),
    ("float", lambda: isinstance(_t().mfcc(_tone())[0], float)),
    ("bank_type", lambda: isinstance(_t().mel_bank(512)[0], list)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
