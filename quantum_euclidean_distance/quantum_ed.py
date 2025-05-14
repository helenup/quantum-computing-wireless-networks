

import math
from typing import Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import Initialize

# Import AerSimulator from the standalone qiskit-aer package:
from qiskit_aer import AerSimulator


def _compute_classical_distance(
        A: Sequence[float], B: Sequence[float]
) -> Tuple[float, float, float, float]:
    A_arr = np.array(A, float)
    B_arr = np.array(B, float)
    if A_arr.shape != B_arr.shape:
        raise ValueError("A and B must have the same length.")
    dist = math.dist(A_arr, B_arr)
    norm_A = np.linalg.norm(A_arr)
    norm_B = np.linalg.norm(B_arr)
    Z = norm_A ** 2 + norm_B ** 2
    return dist, norm_A, norm_B, Z


def _build_phi_psi_states(
        A: Sequence[float], B: Sequence[float], Z: float
) -> Tuple[np.ndarray, np.ndarray]:
    A_arr = np.array(A, float)
    B_arr = np.array(B, float)
    norm_A = np.linalg.norm(A_arr)
    norm_B = np.linalg.norm(B_arr)

    # φ on 1 qubit
    phi = np.array([norm_A / math.sqrt(Z), -norm_B / math.sqrt(Z)], float)

    # ψ on 3 qubits → 8 amplitudes
    m = A_arr.size
    psi = np.zeros(8, float)
    psi[0::2][:m] = A_arr / norm_A
    psi[1::2][:m] = B_arr / norm_B
    psi /= math.sqrt(2)

    return phi, psi


def quantum_euclidean_distance(
        A: Sequence[float], B: Sequence[float], shots: int = 10_000
) -> Tuple[float, float]:
    """
    Returns (classical_distance, quantum_estimate) for vectors A and B.
    """
    # 1) Classical
    dist_classical, norm_A, norm_B, Z = _compute_classical_distance(A, B)

    # 2) State prep
    phi, psi = _build_phi_psi_states(A, B, Z)

    # 3) Circuit: 1 ancilla + φ-qubit + 3-qubit ψ register
    qc = QuantumCircuit(1 + 4, 1, name="qED_phi_psi")
    qc.initialize(phi, [1])
    qc.initialize(psi, [2, 3, 4])

    # SWAP test between φ-qubit (1) and ψ-qubit (2)
    qc.h(0)
    qc.cswap(0, 1, 2)
    qc.h(0)
    qc.measure(0, 0)

    # 4) Execute on the standalone AerSimulator
    backend = AerSimulator()
    qobj = transpile(qc, backend)
    job = backend.run(qobj, shots=shots)
    result = job.result()
    counts = result.get_counts()
    p0 = counts.get("0", 0) / shots

    # 5) Post-process → quantum distance
    overlap = abs((p0 - 0.5) / 0.5)
    x = overlap * 2 * Z
    dist_quantum = math.sqrt(x)

    return dist_classical, dist_quantum


def main():
    A = [2, 9, 8, 5]
    B = [7, 5, 10, 3]
    dc, dq = quantum_euclidean_distance(A, B, shots=50000)
    print(f"Classical distance: {dc:.4f}")
    print(f"Quantum estimate:   {dq:.4f}")


if __name__ == "__main__":
    main()
