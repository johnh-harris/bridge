"""
stim_to_syndrilla.py

Extracts Hx, Hz, Lx, Lz parity check matrices from a stim surface code
circuit and saves them as .alist files for use with syndrilla.

Works by:
1. Reading qubit coordinates from the stim circuit
2. Identifying data qubits (x+y even) vs ancilla qubits (x+y odd)
3. Identifying X-type ancillas (x odd) vs Z-type ancillas (x even)
4. Building CSS matrices from circuit connectivity (orthogonal neighbors)
5. Computing logical operators from the code geometry

Usage:
    python stim_to_syndrilla.py
"""

import stim
import numpy as np


# ── alist writer ──────────────────────────────────────────────────────────────

def matrix_to_alist(mat: np.ndarray, path: str):
    """
    Save a binary matrix as a .alist file matching syndrilla's expected format.

    Header line 1:  num_cols  num_rows
    Header line 2:  max_col_weight  max_row_weight
    Line 3:         weight of each column (space separated)
    Line 4:         weight of each row (space separated, 0 for empty rows)
    Then per column: 1-indexed row positions of 1s (empty line if weight 0)
    Then per row:    1-indexed col positions of 1s (empty line if weight 0)
    """
    if hasattr(mat, "toarray"):
        mat = mat.toarray()
    mat = mat.astype(np.uint8)

    num_rows, num_cols = mat.shape
    col_weights = mat.sum(axis=0)
    row_weights = mat.sum(axis=1)
    max_col_weight = int(col_weights.max()) if num_cols > 0 and col_weights.max() > 0 else 0
    max_row_weight = int(row_weights.max()) if num_rows > 0 and row_weights.max() > 0 else 0

    with open(path, "w") as f:
        f.write(f"{num_cols} {num_rows}\n")
        f.write(f"{max_col_weight} {max_row_weight}\n")
        f.write(" ".join(str(int(w)) for w in col_weights) + " \n")
        f.write(" ".join(str(int(w)) for w in row_weights) + " \n")
        # Column adjacency: 1-indexed row positions of 1s
        for c in range(num_cols):
            indices = (np.where(mat[:, c] == 1)[0] + 1).tolist()
            f.write(" ".join(str(x) for x in indices) + " \n" if indices else "\n")
        # Row adjacency: 1-indexed col positions of 1s
        for r in range(num_rows):
            indices = (np.where(mat[r, :] == 1)[0] + 1).tolist()
            f.write(" ".join(str(x) for x in indices) + " \n" if indices else "\n")

    print(f"Saved {path}  ({num_rows} rows x {num_cols} cols)")


# ── matrix extraction ─────────────────────────────────────────────────────────

def extract_matrices(circuit: stim.Circuit, out_prefix: str):
    """
    Extract Hx, Hz, Lx, Lz from an unrotated stim surface code circuit
    and save as .alist files.

    The circuit must use integer qubit coordinates (as stim's unrotated
    surface code does). Data qubits sit at positions where x+y is even,
    ancilla qubits where x+y is odd.

    Matrix layout (matching syndrilla's examples):
      Hx:  (num_data_qubits rows) x (num_x_stabilizers cols)  -- transposed
      Hz:  (num_data_qubits rows) x (num_z_stabilizers cols)
      Lx:  (num_data_qubits rows) x (1 col)
      Lz:  (num_data_qubits rows) x (1 col)
    """
    coords = circuit.get_final_qubit_coordinates()

    # Separate data qubits (x+y even) from ancillas (x+y odd)
    data_coords = sorted(
        [tuple(map(int, c)) for c in coords.values() if (int(c[0]) + int(c[1])) % 2 == 0]
    )
    ancilla_coords = [
        tuple(map(int, c)) for c in coords.values() if (int(c[0]) + int(c[1])) % 2 == 1
    ]

    # X-type ancillas: x coordinate is odd
    # Z-type ancillas: x coordinate is even
    x_ancillas = sorted([(x, y) for x, y in ancilla_coords if x % 2 == 1])
    z_ancillas = sorted([(x, y) for x, y in ancilla_coords if x % 2 == 0])

    data_idx = {c: i for i, c in enumerate(data_coords)}
    n_data = len(data_coords)

    print(f"Data qubits:    {n_data}")
    print(f"X-stabilizers:  {len(x_ancillas)}")
    print(f"Z-stabilizers:  {len(z_ancillas)}")

    def orthogonal_neighbors(x, y):
        return sorted([
            data_idx[(x + dx, y + dy)]
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if (x + dx, y + dy) in data_idx
        ])

    # Build Hx: rows=X-stabilizers, cols=data qubits
    Hx = np.zeros((len(x_ancillas), n_data), dtype=np.uint8)
    for i, (x, y) in enumerate(x_ancillas):
        for n in orthogonal_neighbors(x, y):
            Hx[i, n] = 1

    # Build Hz: rows=Z-stabilizers, cols=data qubits
    Hz = np.zeros((len(z_ancillas), n_data), dtype=np.uint8)
    for i, (x, y) in enumerate(z_ancillas):
        for n in orthogonal_neighbors(x, y):
            Hz[i, n] = 1

    # Logical X: horizontal chain across the top row of data qubits
    # (qubits with y=0, sorted by x)
    lx_qubits = sorted([data_idx[(x, y)] for (x, y) in data_coords if y == 0])
    Lx = np.zeros((1, n_data), dtype=np.uint8)
    for q in lx_qubits:
        Lx[0, q] = 1

    # Logical Z: vertical chain down the left column of data qubits
    # (qubits with x=0, sorted by y)
    lz_qubits = sorted([data_idx[(x, y)] for (x, y) in data_coords if x == 0])
    Lz = np.zeros((1, n_data), dtype=np.uint8)
    for q in lz_qubits:
        Lz[0, q] = 1

    print(f"Lx support: qubits {lx_qubits}")
    print(f"Lz support: qubits {lz_qubits}")

    # syndrilla format: num_cols=checks, num_rows=data qubits -> transpose Hx/Hz
    matrix_to_alist(Hx.T, f"{out_prefix}_hx.alist")
    matrix_to_alist(Hz.T, f"{out_prefix}_hz.alist")
    matrix_to_alist(Lx.T, f"{out_prefix}_lx.alist")
    matrix_to_alist(Lz.T, f"{out_prefix}_lz.alist")

    return Hx, Hz, Lx, Lz


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for distance in [3, 5, 7]:
        print(f"\n=== Distance {distance} ===")
        circuit = stim.Circuit.generated(
            "surface_code:unrotated_memory_z",
            distance=distance,
            rounds=1,
        )
        extract_matrices(circuit, out_prefix=f"surface_{distance}")