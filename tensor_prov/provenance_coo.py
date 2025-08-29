import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from prov_graph import ProvGraph
from prov_tree import ProvTree
from utils import csr2arr, coo2arr
from watched_pandas import WatchedDataFrame


class Provenance:
    def __init__(self, verbose: int = 0):
        self.graph = ProvTree()
        self.verbose = verbose

    def subscribe(self, df):
        wdf = WatchedDataFrame(df, self)
        return wdf

    def capture_time(func):
        def wrapper(self, *args, **kwargs):
            start = time.time()
            result = func(self, *args, **kwargs)
            end = time.time()
            print(f"Runtime: {end - start:.4f} seconds")
            return result, end - start

        return wrapper

    @capture_time
    def capture(
            self,
            df_in: pd.DataFrame | tuple[pd.DataFrame, ...],
            df_out: pd.DataFrame,
            **kwargs
    ):
        primary_key = kwargs.get("primary_key", None)
        column_mapping = kwargs.get("column_mapping", None)

        result = []

        if isinstance(df_in, pd.DataFrame):
            tensor_record = capture_row_operation(df_in, df_out, primary_key)
            tensor_attr = capture_column_operation(df_in, df_out, column_mapping)
            result.append((tensor_record, tensor_attr))
            self.graph.add_child(df_in, df_out, (tensor_record, tensor_attr))

        elif isinstance(df_in, tuple):
            if primary_key is None:
                raise ValueError("primary_key is required")
            elif isinstance(primary_key, str):
                primary_key = tuple(primary_key for _ in df_in)

            for d, p in zip(df_in, primary_key):
                tensor_record = capture_row_operation(d, df_out, p)
                tensor_attr = capture_column_operation(d, df_out, column_mapping)
                result.append((tensor_record, tensor_attr))
                self.graph.add_child(d, df_out, (tensor_record, tensor_attr))
        else:
            raise ValueError("df_in should be DataFrame or tuple[pd.DataFrame, pd.DataFrame] !")

        return result


def create_sparse_tensor(
        indices_out: np.ndarray,
        indices_in: np.ndarray,
        shape: tuple[int, int]
):
    """
    Create a sparse tensor from indices.
    """
    data = np.ones(indices_out.shape[0], dtype=np.int8)
    sparse_tensor = coo_matrix((data, (indices_out, indices_in)), shape=shape)
    return sparse_tensor


def capture_row_operation(
        df_in: pd.DataFrame,
        df_out: pd.DataFrame,
        primary_key: str | None
):
    n_out, n_in = len(df_out), len(df_in)  # provenance shape (n_out, n_in)
    ids_out = df_out.index.to_numpy() if primary_key is None else df_out[primary_key].to_numpy()
    ids_in = df_in.index.to_numpy() if primary_key is None else df_in[primary_key].to_numpy()
    indices_out, indices_in = find_indices(ids_out, ids_in)

    # Create sparse tensor
    sparse_tensor = create_sparse_tensor(indices_out, indices_in, (n_out, n_in))

    return sparse_tensor


def capture_column_operation(
        df_in: pd.DataFrame,
        df_out: pd.DataFrame,
        column_mapping: dict | None
):
    cols_in = df_in.columns.values
    cols_out = df_out.columns.values
    n_out, n_in = len(cols_out), len(cols_in)

    if column_mapping is not None:
        out_to_in = {col_out: col_in for col_in, cols_out in column_mapping.items() for col_out in cols_out}
        cols_out = [out_to_in.get(col_out, col_out) for col_out in cols_out]

    indices_out, indices_in = find_indices(cols_out, cols_in)

    # Create sparse tensor
    sparse_tensor = create_sparse_tensor(indices_out, indices_in, (n_out, n_in))

    return sparse_tensor


def find_indices(ids_out: np.ndarray, ids_in: np.ndarray):
    # # I. np.where
    # indices_out, indices_in = np.where(ids_out[:, None] == ids_in[None, :])

    # # II. for loop
    # indices_out, indices_in = [], []
    # for out_idx, val_out in enumerate(ids_out):
    #     for in_idx, val_in in enumerate(ids_in):
    #         if val_out == val_in:
    #             indices_out.append(out_idx)
    #             indices_in.append(in_idx)
    #
    # indices_out = np.array(indices_out, dtype=np.int64)
    # indices_in = np.array(indices_in, dtype=np.int64)

    # III. dict and for loop
    id_to_indices_in = defaultdict(list)
    for idx_in, val in enumerate(ids_in):
        id_to_indices_in[val].append(idx_in)

    indices_out, indices_in = [], []
    for out_idx, val in enumerate(ids_out):
        if val in id_to_indices_in:
            in_indices = id_to_indices_in[val]
            indices_out.extend([out_idx] * len(in_indices))
            indices_in.extend(in_indices)

    indices_out = np.array(indices_out, dtype=np.int64)
    indices_in = np.array(indices_in, dtype=np.int64)

    return indices_out, indices_in


def print_prov_result(
        df_in: pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame],
        df_out: pd.DataFrame,
        result: list[tuple[coo_matrix, coo_matrix], ...],
        direction: str | None = None,
        num_examples: int = 5
):
    """
    Print the provenance result
    """
    for r in result:
        print(r)


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
        df_path = pd.DataFrame(t1_arr).iloc[:, [1, 0]]
        tensors_ordered = tensors[1:]
        i_right_on = 1
    elif direction == "backward":
        t1_arr = coo2arr(tensors[-1])
        df_path = pd.DataFrame(t1_arr)
        tensors_ordered = tensors[-2::-1]
        i_right_on = 0
    else:
        raise ValueError("direction must be 'forward' or 'backward'")

    if indices is not None:
        df_path = df_path[df_path.iloc[:, 0].isin(indices)]
    for t in tensors_ordered:
        t_arr = coo2arr(t)
        df_t = pd.DataFrame(t_arr).add_prefix('t_')
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
