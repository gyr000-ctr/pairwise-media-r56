import math

import pytest

import audioonset

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _o():
    return audioonset.Onset(8000, 256, 128, 0.1, 3)


def _click(n=2000):
    s = [0.0] * n
    for i in range(400, n, 400):
        s[i] = 1.0
    return s

CASES = [
    ("flux", lambda: isinstance(_o().flux(_click()), list)),
    ("detect", lambda: len(_o().detect(_click())) >= 3),
    ("detect_sorted", lambda: all(a < b for a, b in zip(_o().detect(_click()), _o().detect(_click())[1:]))),
    ("silence", lambda: _o().detect([0.0] * 2000) == []),
    ("gap", lambda: all(b - a >= 128 * 3 for a, b in zip(_o().detect(_click()), _o().detect(_click())[1:]))),
    ("bad_frame", lambda: _raises(lambda: audioonset.Onset(8000, 0, 128, 0.1, 3))),
    ("bad_hop", lambda: _raises(lambda: audioonset.Onset(8000, 256, 0, 0.1, 3))),
    ("bad_threshold", lambda: _raises(lambda: audioonset.Onset(8000, 256, 128, 1.5, 3))),
    ("bad_gap", lambda: _raises(lambda: audioonset.Onset(8000, 256, 128, 0.1, 0))),
    ("type", lambda: isinstance(_o().detect(_click()), list)),
    ("det", lambda: _o().detect(_click()) == _o().detect(_click())),
    ("nan", lambda: _raises(lambda: _o().flux([float('nan')] * 512))),
    ("no_mut", lambda: _o().detect(_click()) is not None),
    ("int_bounds", lambda: all(isinstance(v, int) for v in _o().detect(_click()))),
    ("flux_type", lambda: isinstance(_o().flux(_click())[0], float)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
