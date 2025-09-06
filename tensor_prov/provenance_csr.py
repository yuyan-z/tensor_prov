import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from provenance_base import ProvenanceBase
from utils_type import csr2arr


class Provenance(ProvenanceBase):
    def create_sparse_tensor(
            self,
            indices_out: np.ndarray,
            indices_in: np.ndarray,
            shape: tuple[int, int]
    ) -> csr_matrix:
        data = np.ones(indices_out.shape[0], dtype=np.int8)
        sparse_tensor = csr_matrix((data, (indices_out, indices_in)), shape=shape)
        return sparse_tensor

    def print_prov_result(
            self,
            df_in: pd.DataFrame,
            df_out: pd.DataFrame,
            id_in: int,
            id_out: int,
            result: tuple[csr_matrix, csr_matrix]
    ):
        print("-- id_in -> id_out--\n", id_in, '->', id_out)
        print(f"-- df_in: {id_in} --\n", df_in)
        print(f"-- df_out: {id_out} --\n", df_out)
        (tensor_record, tensor_attr) = result
        print("-- tensor_record --\n", csr2arr(tensor_record))
        # print(tensor_record.todense())
        print("-- tensor_attr --\n", csr2arr(tensor_attr))
        # print(tensor_attr.todense())


def trace(
        tensors: list[csr_matrix],
        direction: str = "backward",
        indices: list[int] | None = None,
        keep_path: bool = False
):
    if not keep_path:
        return trace_csr(tensors, direction=direction, indices=indices)
    else:
        return trace_pandas(tensors, direction=direction, indices=indices, keep_path=keep_path)


def trace_csr(
        tensors: list[csr_matrix],
        direction: str = "backward",
        indices: list[int] | None = None
):
    # csr matrix multiplication （einsum）
    if direction == "forward":
        tensors_ordered = [t.T.tocsr() for t in tensors]
    elif direction == "backward":
        tensors_ordered = tensors[::-1]
    else:
        raise ValueError("direction must be 'forward' or 'backward'")
    t1 = tensors_ordered[0] if indices is None else slice_csr(tensors_ordered[0], indices)
    for t2 in tensors_ordered[1:]:
        t1 = t1 @ t2
    result = csr2arr(t1)
    return result


def trace_pandas(
        tensors: list[csr_matrix],
        direction: str = "backward",
        indices: list[int] | None = None,
        keep_path: bool = False
):
    # pandas join
    if direction == "forward":
        t1_arr = csr2arr(tensors[0])
        df_path = pd.DataFrame(t1_arr, columns=["out", "in"])
        df_path = df_path[["in", "out"]]
        tensors_ordered = tensors[1:]
        i_right_on = 1
    elif direction == "backward":
        t1_arr = csr2arr(tensors[-1])
        df_path = pd.DataFrame(t1_arr, columns=["out", "t_0_in"])
        tensors_ordered = tensors[-2::-1]
        i_right_on = 0
    else:
        raise ValueError("direction must be 'forward' or 'backward'")

    if indices is not None:
        df_path = df_path[df_path.iloc[:, 0].isin(indices)]
    for i, t in enumerate(tensors_ordered):
        t_arr = csr2arr(t)
        df_t = pd.DataFrame(t_arr, columns=["out", "in"]).add_prefix(f't_{i + 1}_')
        df_path = df_path.merge(df_t, left_on=df_path.columns[-1], right_on=df_t.columns[i_right_on], how="inner")
        df_path = df_path.drop(columns=[df_t.columns[i_right_on]])
        if not keep_path:
            df_path = df_path.iloc[:, [0, -1]]
    df_path = df_path.sort_values(by=df_path.columns[0])
    result = df_path.to_numpy(dtype=int)

    return result


def slice_csr(csr: csr_matrix, indices: list | np.ndarray) -> csr_matrix:
    indices = np.array(indices)
    coo = csr.tocoo()
    mask = np.isin(coo.row, indices)
    new_row = coo.row[mask]
    new_col = coo.col[mask]
    new_data = coo.data[mask]
    return csr_matrix((new_data, (new_row, new_col)), shape=csr.shape)
