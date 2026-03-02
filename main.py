import stim
import beliefmatching
import numpy as np
from scipy import sparse

'''
https://github.com/oscarhiggott/PyMatching/issues/73
https://textbook.riverlane.com/en/latest/notebooks/ch5-decoding-surfcodes/simulating-surface-codes-stim.html

Only works on version 3.12
'''

'''

This is a plain MacKay AList file describing a binary parity-check matrix 
𝐻
𝑥
H
x
	​

.
I’ll decode it line by line and then show you the actual matrix.

1. First two lines: size and max degrees
6 13
4 2

6 13

6 = number of columns = number of variables (bits).

13 = number of rows = number of checks (parity constraints).

So 
𝐻
𝑥
H
x
	​

 is a 
13
×
6
13×6 matrix.

4 2

4 = maximum column weight (the most 1’s in any single column).

2 = maximum row weight (the most 1’s in any single row).

These are just upper bounds used to fix the layout of the next sections.

2. Column weights and row weights
3 4 3 3 4 3 
1 1 1 2 2 2 1 1 1 2 2 2 2 

Column weights (for each of 6 columns):

Column 1 has weight 3 (it participates in 3 checks).

Column 2 has weight 4.

Column 3 has weight 3.

Column 4 has weight 3.

Column 5 has weight 4.

Column 6 has weight 3.

Row weights (for each of 13 rows):

Row 1: weight 1

Row 2: weight 1

Row 3: weight 1

Row 4: weight 2

Row 5: weight 2

Row 6: weight 2

Row 7: weight 1

Row 8: weight 1

Row 9: weight 1

Row 10: weight 2

Row 11: weight 2

Row 12: weight 2

Row 13: weight 2

3. Column adjacency (which checks each variable appears in)

Next 6 lines: for each column (1 through 6), list the row indices where that column has a 1.
Indices are 1-based. If a column has fewer than max_col_weight entries, it’s padded with zeros (here they just stopped at the nonzero ones).

1 4 10 
2 5 10 11 
3 6 11 
4 7 12 
5 8 12 13 
6 9 13 

Interpretation:

Column 1 has 1’s in rows 1, 4, and 10.

Column 2 has 1’s in rows 2, 5, 10, 11.

Column 3 has 1’s in rows 3, 6, 11.

Column 4 has 1’s in rows 4, 7, 12.

Column 5 has 1’s in rows 5, 8, 12, 13.

Column 6 has 1’s in rows 6, 9, 13.

This completely defines the matrix from the variable-side view.

4. Row adjacency (which variables each check touches)

Final 13 lines: for each row (1 through 13), list the column indices where that row has a 1.
Again 1-based, padded up to max_row_weight = 2.

1 
2 
3 
1 4 
2 5 
3 6 
4 
5 
6 
1 2 
2 3 
4 5 
5 6 

Interpretation:

Row 1 has a 1 in column 1.

Row 2 has a 1 in column 2.

Row 3 has a 1 in column 3.

Row 4 has 1’s in columns 1 and 4.

Row 5 has 1’s in columns 2 and 5.

Row 6 has 1’s in columns 3 and 6.

Row 7 has a 1 in column 4.

Row 8 has a 1 in column 5.

Row 9 has a 1 in column 6.

Row 10 has 1’s in columns 1 and 2.

Row 11 has 1’s in columns 2 and 3.

Row 12 has 1’s in columns 4 and 5.

Row 13 has 1’s in columns 5 and 6.

This is redundant with the column adjacency lines, but it’s part of the AList format.

5. The actual 
13
×
6
13×6 matrix

If we translate all that into a plain 0/1 matrix 
𝐻
𝑥
H
x
	​

 with rows 1–13 and columns 1–6:

𝐻
𝑥
=
(
1
	
0
	
0
	
0
	
0
	
0


0
	
1
	
0
	
0
	
0
	
0


0
	
0
	
1
	
0
	
0
	
0


1
	
0
	
0
	
1
	
0
	
0


0
	
1
	
0
	
0
	
1
	
0


0
	
0
	
1
	
0
	
0
	
1


0
	
0
	
0
	
1
	
0
	
0


0
	
0
	
0
	
0
	
1
	
0


0
	
0
	
0
	
0
	
0
	
1


1
	
1
	
0
	
0
	
0
	
0


0
	
1
	
1
	
0
	
0
	
0


0
	
0
	
0
	
1
	
1
	
0


0
	
0
	
0
	
0
	
1
	
1
)
H
x
	​

=
	​

1
0
0
1
0
0
0
0
0
1
0
0
0
	​

0
1
0
0
1
0
0
0
0
1
1
0
0
	​

0
0
1
0
0
1
0
0
0
0
1
0
0
	​

0
0
0
1
0
0
1
0
0
0
0
1
0
	​

0
0
0
0
1
0
0
1
0
0
0
1
1
	​

0
0
0
0
0
1
0
0
1
0
0
0
1
	​

	​


Each column corresponds to one “X-error bit” (one variable in the decoder).

Each row is one parity check that must sum to 0 mod 2.

For a distance-3 surface code, this is the X-stabilizer parity-check matrix in some chosen ordering of qubits and stabilizers. The exact geometric meaning (which qubit is where) depends on the mapping used by whoever generated the file, but structurally this is all that’s in the .alist.

If you want, I can show how to parse this file in Python back into a SciPy sparse matrix (and vice versa, from a matrix into this exact format).


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
    print("H")
    print(H.toarray())
    print("L")
    print(L.toarray())
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