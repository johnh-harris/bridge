import stim
import beliefmatching
import numpy as np
from scipy import sparse

'''
https://github.com/oscarhiggott/PyMatching/issues/73
'''

def write_alist(M: sparse.spmatrix, path: str) -> None:
    """
    Write a binary sparse matrix M (shape m x n) to MacKay AList format.

    n = number of columns (variables)
    m = number of rows (checks)
    """
    # Ensure CSR for row-wise ops, CSC for col-wise ops
    M_csr = M.tocsr()
    M_csc = M.tocsc()

    m, n = M_csr.shape

    # Column and row weights
    col_weights = np.diff(M_csc.indptr)  # length n
    row_weights = np.diff(M_csr.indptr)  # length m

    max_col_w = int(col_weights.max()) if n > 0 else 0
    max_row_w = int(row_weights.max()) if m > 0 else 0

    with open(path, "w") as f:
        # 1) n m
        f.write(f"{n} {m}\n")
        # 2) max_col_weight max_row_weight
        f.write(f"{max_col_w} {max_row_w}\n")

        # 3) column weights (n ints)
        if n > 0:
            f.write(" ".join(str(int(w)) for w in col_weights) + "\n")
        else:
            f.write("\n")

        # 4) row weights (m ints)
        if m > 0:
            f.write(" ".join(str(int(w)) for w in row_weights) + "\n")
        else:
            f.write("\n")

        # 5) For each column: list of row indices with 1s (1-based), padded to max_col_w
        for j in range(n):
            start, end = M_csc.indptr[j], M_csc.indptr[j + 1]
            rows = M_csc.indices[start:end] + 1  # 1-based
            rows = list(map(int, rows))
            padded = rows + [0] * (max_col_w - len(rows))
            f.write(" ".join(str(r) for r in padded) + "\n")

        # 6) For each row: list of column indices with 1s (1-based), padded to max_row_w
        for i in range(m):
            start, end = M_csr.indptr[i], M_csr.indptr[i + 1]
            cols = M_csr.indices[start:end] + 1  # 1-based
            cols = list(map(int, cols))
            padded = cols + [0] * (max_row_w - len(cols))
            f.write(" ".join(str(c) for c in padded) + "\n")


def main():
    p = 0.01  # nonzero noise to generate error mechanisms

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=3,
        rounds=2,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
        after_clifford_depolarization=p,
    )

    dem = circuit.detector_error_model(decompose_errors=True)
    dem_mats = beliefmatching.detector_error_model_to_check_matrices(dem)

    H = dem_mats.check_matrix          # detectors × error mechanisms
    L = dem_mats.observables_matrix    # logicals × error mechanisms

    print("H shape:", H.shape)
    print("L shape:", L.shape)

    # Export to AList for Syndrilla
    write_alist(H, "Hx.alist")
    # For a circuit-level view, you can reuse H for Hz as well:
    write_alist(H, "Hz.alist")
    # Use L as (say) Lz (logical Z checks):
    write_alist(L, "Lz.alist")
    # Often you can omit Lx in the config, or define it separately if needed.

if __name__ == "__main__":
    main()