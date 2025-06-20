import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix


def coo2arr(coo: coo_matrix | tuple[coo_matrix, coo_matrix]):
    if isinstance(coo, coo_matrix):
        arr = np.vstack((
            coo.row,
            coo.col
        )).T
    else:
        df_A = pd.DataFrame({
            "idx_out": coo[0].row,
            "idx_left": coo[0].col,
        })
        df_B = pd.DataFrame({
            "idx_out": coo[1].row,
            "idx_right": coo[1].col,
        })
        df = pd.merge(df_A, df_B, on="idx_out", how="outer")
        arr = df.fillna(-1).to_numpy(dtype=int)

    return arr


def csr2arr(csr: csr_matrix | tuple[csr_matrix, csr_matrix]):
    if isinstance(csr, csr_matrix):
        row, col = csr.nonzero()
        arr = np.vstack((row, col)).T
    else:
        A_row, A_col = csr[0].nonzero()
        B_row, B_col = csr[1].nonzero()

        df_A = pd.DataFrame({
            "idx_out": A_row,
            "idx_left": A_col,
        })
        df_B = pd.DataFrame({
            "idx_out": B_row,
            "idx_right": B_col,
        })
        df = pd.merge(df_A, df_B, on="idx_out", how="outer")
        arr = df.fillna(-1).to_numpy(dtype=int)

    return arr
