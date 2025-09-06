import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from provenance_base import ProvenanceBase
from utils_type import csr2arr, coo2arr
from watched_pandas import WatchedDataFrame


class Provenance(ProvenanceBase):
    def create_sparse_tensor(
            self,
            indices_out: np.ndarray,
            indices_in: np.ndarray,
            shape: tuple[int, int]
    ) -> coo_matrix:
        data = np.ones(indices_out.shape[0], dtype=np.int8)
        sparse_tensor = coo_matrix((data, (indices_out, indices_in)), shape=shape)
        return sparse_tensor

    def print_prov_result(
            self,
            df_in: pd.DataFrame,
            df_out: pd.DataFrame,
            id_in: int,
            id_out: int,
            result: tuple[coo_matrix, coo_matrix]
    ):
        print("-- id_in -> id_out--\n", id_in, '->', id_out)
        print(f"-- df_in: {id_in} --\n", df_in)
        print(f"-- df_out: {id_out} --\n", df_out)
        (tensor_record, tensor_attr) = result
        print("-- tensor_record --\n", coo2arr(tensor_record))
        # print(tensor_record.todense())
        print("-- tensor_attr --\n", coo2arr(tensor_attr))
        # print(tensor_attr.todense())


def trace(
        tensors: list[coo_matrix],
        direction: str = "backward",
        indices: list[int] | None = None,
        keep_path: bool = False
):
    if not keep_path:
        return trace_coo(tensors, direction=direction, indices=indices)
    else:
        return trace_pandas(tensors, direction=direction, indices=indices, keep_path=keep_path)


def trace_coo(
        tensors: list[coo_matrix],
        direction: str = "backward",
        indices: list[int] | None = None
):
    # csr matrix multiplication （einsum）
    if direction == "forward":
        tensors_ordered = [t.T for t in tensors]
    elif direction == "backward":
        tensors_ordered = tensors[::-1]
    else:
        raise ValueError("direction must be 'forward' or 'backward'")
    t1 = tensors_ordered[0] if indices is None else slice_coo(tensors_ordered[0], indices)
    t1 = t1.tocsr()
    for t2 in tensors_ordered[1:]:
        t1 = t1 @ t2
    result = csr2arr(t1)
    return result


def trace_pandas(
        tensors: list[coo_matrix],
        direction: str = "backward",
        indices: list[int] | None = None,
        keep_path: bool = False
):
    # pandas join
    if direction == "forward":
        t1_arr = coo2arr(tensors[0])
        df_path = pd.DataFrame(t1_arr, columns=["out", "in"])
        df_path = df_path[["in", "out"]]
        tensors_ordered = tensors[1:]
        i_right_on = 1
    elif direction == "backward":
        t1_arr = coo2arr(tensors[-1])
        df_path = pd.DataFrame(t1_arr, columns=["out", "t_0_in"])
        tensors_ordered = tensors[-2::-1]
        i_right_on = 0
    else:
        raise ValueError("direction must be 'forward' or 'backward'")

    if indices is not None:
        df_path = df_path[df_path.iloc[:, 0].isin(indices)]
    for i, t in enumerate(tensors_ordered):
        t_arr = coo2arr(t)
        df_t = pd.DataFrame(t_arr, columns=["out", "in"]).add_prefix(f't_{i+1}_')
        df_path = df_path.merge(df_t, left_on=df_path.columns[-1], right_on=df_t.columns[i_right_on], how="inner")
        df_path = df_path.drop(columns=[df_t.columns[i_right_on]])
        if not keep_path:
            df_path = df_path.iloc[:, [0, -1]]
    df_path = df_path.sort_values(by=df_path.columns[0])
    result = df_path.to_numpy(dtype=int)

    return result


def trace_numpy(
        tensors: list[coo_matrix],
        direction: str = "backward",
        indices: list[int] | None = None,
        keep_path: bool = False
):
    # numpy
    if direction == "forward":
        t1_arr = coo2arr(tensors[0])
        path_arr = t1_arr[:, [1, 0]]
        tensors_ordered = tensors[1:]
        i_right_on = 1
    elif direction == "backward":
        path_arr = coo2arr(tensors[-1])
        tensors_ordered = tensors[-2::-1]
        i_right_on = 0
    else:
        raise ValueError("direction must be 'forward' or 'backward'")

    for t in tensors_ordered:
        t_arr = coo2arr(t)
        i, j = np.where(path_arr[:, -1, None] == t_arr[:, i_right_on])
        if keep_path:
            path_arr = np.column_stack([path_arr[i, :], t_arr[j, 1 - i_right_on]])
        else:
            path_arr = np.column_stack([path_arr[i, 0], t_arr[j, 1 - i_right_on]])

    return path_arr


def slice_coo(coo: coo_matrix, indices: list | np.ndarray) -> coo_matrix:
    mask = np.isin(coo.row, indices)
    new_row = coo.row[mask]
    new_col = coo.col[mask]
    new_data = coo.data[mask]
    return coo_matrix((new_data, (new_row, new_col)), shape=coo.shape)
