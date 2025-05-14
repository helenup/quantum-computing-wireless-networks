# tests/test_quantum_ed.py

import math
import numpy as np
import pytest
from types import SimpleNamespace

from quantum_euclidean_distance.quantum_ed import (
    _compute_classical_distance,
    _build_phi_psi_states,
    quantum_euclidean_distance,
)


class DummyBackend:
    """A fake AerSimulator-like backend for testing swap-test logic."""

    def __init__(self, counts: dict, num_qubits: int):
        # counts: e.g. {'0': 850, '1': 150}
        self._counts = counts
        self.num_qubits = num_qubits
        # Provide a dummy target with dt so transpile() won't crash
        self.target = SimpleNamespace(dt=1e-9)

    def run(self, qobj, shots=None):
        class Job:
            def __init__(self, counts):
                self._counts = counts

            def result(self):
                return self

            def get_counts(self):
                return self._counts

        return Job(self._counts)


def test_compute_classical_distance():
    A = [3.0, 4.0]
    B = [0.0, 0.0]
    dist, normA, normB, Z = _compute_classical_distance(A, B)
    assert math.isclose(dist, 5.0, rel_tol=1e-9)
    assert math.isclose(normA, 5.0, rel_tol=1e-9)
    assert math.isclose(normB, 0.0, rel_tol=1e-9)
    assert math.isclose(Z, 25.0, rel_tol=1e-9)


def test_build_phi_psi_states_shapes_and_norms():
    A = [1.0, 1.0]
    B = [1.0, 0.0]
    _, normA, normB, Z = _compute_classical_distance(A, B)
    phi, psi = _build_phi_psi_states(A, B, Z)

    # φ should be length 2 and normalized
    assert phi.shape == (2,)
    assert pytest.approx(1.0, abs=1e-9) == np.linalg.norm(phi)

    # ψ should be length 8 and normalized
    assert psi.shape == (8,)
    assert pytest.approx(1.0, abs=1e-9) == np.linalg.norm(psi)


@pytest.mark.parametrize("p0,expected_overlap", [
    (1.0, 1.0),   # always measure 0 → full overlap
    (0.5, 0.0),   # p0=0.5 → zero overlap
    (0.0, 1.0),   # p0=0 → |(0-.5)/.5| = 1
])
def test_overlap_calculation(p0, expected_overlap, monkeypatch):
    """
    Mock the backend to return a fixed p0 and verify that
    quantum_euclidean_distance computes the correct overlap→distance.
    """
    A = [2.0, 0.0]
    B = [0.0, 2.0]
    _, normA, normB, Z = _compute_classical_distance(A, B)

    shots = 1000
    counts = {
        '0': int(p0 * shots),
        '1': shots - int(p0 * shots)
    }

    # Circuit uses 1 ancilla + 1 φ-qubit + 3 ψ-qubits = 5 total qubits
    dummy_backend = DummyBackend(counts, num_qubits=5)

    # Monkey-patch AerSimulator inside quantum_ed to return our dummy backend
    import quantum_euclidean_distance.quantum_ed as qmod
    monkeypatch.setitem(qmod.__dict__, 'AerSimulator', lambda: dummy_backend)

    dc, dq = quantum_euclidean_distance(A, B, shots=shots)

    # Classical distance = sqrt(8)
    assert math.isclose(dc, math.sqrt(8), rel_tol=1e-9)

    # Quantum distance should be sqrt(expected_overlap * 2 * Z)
    expected_dist = math.sqrt(expected_overlap * 2 * Z)
    assert math.isclose(dq, expected_dist, rel_tol=1e-9)


def test_invalid_length_raises():
    with pytest.raises(ValueError):
        quantum_euclidean_distance([1, 2], [1, 2, 3])