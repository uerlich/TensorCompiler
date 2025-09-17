
import math, random, pytest
from tensor_offline_tool.tool.core import simulator as sim
from tensor_offline_tool.tool.core.constants import BANKS, LANES, ATOM_B

HAS_HELPER = hasattr(sim, "banks_for_commit")
assert HAS_HELPER, "banks_for_commit helper must exist"

def banks_for_commit(base_atom_mod32, delta_b, lifts_atoms, lanes_active):
    return sim.banks_for_commit(base_atom_mod32, delta_b, lifts_atoms, lanes_active)

def make_distinct_lifts(delta_b, rng):
    if math.gcd(delta_b, BANKS) <= 2:
        lifts_atoms = [0] * LANES
    else:
        off = rng.randrange(BANKS)
        targets = [(off + 17 * l) % BANKS for l in range(LANES)]
        lifts_atoms = [(targets[l] - (delta_b * l)) % BANKS for l in range(LANES)]
    lifts_B = [k * ATOM_B for k in lifts_atoms]
    return lifts_atoms, lifts_B

@pytest.mark.parametrize("seed", [0,1])
def test_banks_helper_pairwise_distinct(seed):
    rng = random.Random(seed)
    for _ in range(200):
        base_atom = rng.randrange(BANKS)
        delta_b = rng.randrange(BANKS)
        lifts_atoms, _ = make_distinct_lifts(delta_b, rng)
        banks = banks_for_commit(base_atom, delta_b, lifts_atoms, LANES)
        assert len(banks) == LANES
        assert len(set(banks)) == LANES, f"collision: base={base_atom}, Δb={delta_b}, lifts={lifts_atoms}, banks={banks}"

@pytest.mark.parametrize("delta_b", [1,3,5,7,2,6,10,14])
def test_fast_path_matches_natural(delta_b):
    base_atom = 7
    lifts_atoms = [0]*LANES
    natural = [(base_atom + delta_b*l) % BANKS for l in range(LANES)]
    banks = banks_for_commit(base_atom, delta_b, lifts_atoms, LANES)
    assert banks == natural
