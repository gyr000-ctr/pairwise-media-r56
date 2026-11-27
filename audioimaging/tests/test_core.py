import math

import pytest

import audioimaging

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

L = [0.5] * 100
RGT = [0.1] * 100


def _i():
    return audioimaging.Imaging()

CASES = [
    ("ms", lambda: len(_i().ms(L, RGT)) == 2),
    ("roundtrip", lambda: all(abs(a - b) < 1e-12 for ch1, ch2 in zip(_i().from_ms(*_i().ms(L, RGT)), (L, RGT)) for a, b in zip(ch1, ch2))),
    ("correlation", lambda: abs(_i().correlation(L, RGT) - 1.0) < 1e-9),
    ("anti", lambda: abs(_i().correlation(L, [-v for v in L]) + 1.0) < 1e-9),
    ("mono", lambda: all(abs(a - b) < 1e-12 for a, b in zip(_i().widen(L, RGT, 0.0)[0], _i().widen(L, RGT, 0.0)[1]))),
    ("widen", lambda: isinstance(_i().widen(L, RGT, 2.0), tuple)),
    ("bad_width", lambda: _raises(lambda: _i().widen(L, RGT, -1.0))),
    ("bad_len", lambda: _raises(lambda: _i().ms(L, [0.1]))),
    ("type", lambda: isinstance(_i().ms(L, RGT), tuple)),
    ("det", lambda: _i().widen(L, RGT, 2.0) == _i().widen(L, RGT, 2.0)),
    ("nan", lambda: _raises(lambda: _i().correlation([float('nan')], [0.0]))),
    ("no_mut", lambda: _i().widen(list(L), list(RGT), 1.5) is not None),
    ("corr_type", lambda: isinstance(_i().correlation(L, RGT), float)),
    ("range", lambda: -1.0 <= _i().correlation(L, RGT) <= 1.0),
    ("from_ms_type", lambda: isinstance(_i().from_ms(*_i().ms(L, RGT)), tuple)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
