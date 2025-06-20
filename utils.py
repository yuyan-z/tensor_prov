import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


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







