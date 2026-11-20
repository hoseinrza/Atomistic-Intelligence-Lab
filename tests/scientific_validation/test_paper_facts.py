"""Validation of the paper-derived facts recorded in ``constants``.

These tests cross-check the numbers transcribed from the paper against
independent arithmetic:

* parameter counts recomputed from the architectures must reproduce Table 1
  (this confirms that "60 x 15 x 15 x 8" means fully connected layers with
  biases);
* the energy shift must map the DFT cohesive energy (Table 2) onto the
  experimental cohesive energy quoted in the Methods;
* dataset sizes must match the Results text.

They guard the documented facts that later sprints build on; they do not
validate physics (that is the job of the Sprint 1-4 scientific tests).
"""

import pytest

from atomistic_pinn import constants as c

pytestmark = pytest.mark.scientific_validation


def _fc_param_count(architecture: tuple[int, ...]) -> int:
    """Parameter count of a fully connected network, biases included."""
    return sum(
        architecture[i] * architecture[i + 1] + architecture[i + 1]
        for i in range(len(architecture) - 1)
    )


def test_table1_parameter_counts_recomputed_from_architectures() -> None:
    assert _fc_param_count(c.PINN_ARCHITECTURE) == 1283
    assert _fc_param_count(c.NN_ARCHITECTURE) == 1265
    assert _fc_param_count(c.NN_PRIME_ARCHITECTURE) == 1225


def test_table1_reference_rows_agree_with_architectures() -> None:
    assert c.TABLE1_REFERENCE_RMSE["PINN"]["params"] == _fc_param_count(c.PINN_ARCHITECTURE)
    assert c.TABLE1_REFERENCE_RMSE["NN"]["params"] == _fc_param_count(c.NN_ARCHITECTURE)
    params = c.TABLE1_REFERENCE_RMSE["NN_prime"]["params"]
    assert params == _fc_param_count(c.NN_PRIME_ARCHITECTURE)


def test_energy_shift_maps_dft_onto_experimental_cohesive_energy() -> None:
    shifted = c.DFT_COHESIVE_ENERGY_EV_PER_ATOM - c.ENERGY_SHIFT_EV_PER_ATOM
    assert shifted == pytest.approx(c.EXPERIMENTAL_COHESIVE_ENERGY_EV_PER_ATOM, abs=5e-3)


def test_network_io_dimensions_match_descriptor_and_bop_contracts() -> None:
    assert c.PINN_ARCHITECTURE[0] == c.N_DESCRIPTORS_PER_ATOM == 60
    assert c.NN_ARCHITECTURE[0] == 60
    assert c.PINN_ARCHITECTURE[-1] == c.N_BOP_PARAMETERS == 8


def test_dataset_sizes_match_paper() -> None:
    assert (c.TRAINING_SUPERCELLS, c.TRAINING_ATOMS) == (3649, 127592)
    assert (c.TEST_SUPERCELLS, c.TEST_ATOMS) == (26425, 2376388)


def test_pinn_outperforms_nn_in_reference_metrics() -> None:
    # Sanity check on the recorded Table 1 / force RMSE ordering quoted in
    # the paper text (PINN validation energy RMSE and validation force RMSE
    # are both better than the NN baseline's).
    assert c.TABLE1_REFERENCE_RMSE["PINN"]["val"] < c.TABLE1_REFERENCE_RMSE["NN"]["val"]
    assert (
        c.FORCE_RMSE_REFERENCE_EV_PER_ANGSTROM["pinn_validation"]
        < c.FORCE_RMSE_REFERENCE_EV_PER_ANGSTROM["nn_validation"]
    )
