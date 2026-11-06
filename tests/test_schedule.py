from mpc_execution.schedule import twap, almgren_chriss, from_volume_profile
import numpy as np

import numpy as np
import pytest
from mpc_execution.schedule import twap, almgren_chriss, from_volume_profile

Q, N = 100, 30
CASES = {
    "twap": twap(Q, N),
    "ac": almgren_chriss(Q, N, psi=0.1),
    "profile": from_volume_profile(Q, np.arange(N + 1)),
}

@pytest.mark.parametrize("name,s", CASES.items(), ids=CASES.keys())
def test_schedule_is_a_valid_path_from_0_to_Q(name, s):
    assert len(s) == N + 1
    assert s.dtype.kind == "i"
    assert s[0] == 0 and s[-1] == Q
    assert np.all(np.diff(s) >= 0)

def test_twap_slices_differ_by_at_most_one():
    assert np.ptp(np.diff(twap(Q, N))) <= 1

def test_almgren_chriss_tends_to_twap_for_small_psi():
    assert np.array_equal(almgren_chriss(Q, N, 1e-6), twap(Q, N))

def test_almgren_chriss_is_front_loaded():
    assert np.all(almgren_chriss(Q, N, 0.1) >= twap(Q, N))
    assert almgren_chriss(Q, N, 0.1)[N // 2] > twap(Q, N)[N // 2]

def test_flat_profile_is_twap():
    assert np.array_equal(from_volume_profile(Q, np.arange(N + 1)), twap(Q, N))