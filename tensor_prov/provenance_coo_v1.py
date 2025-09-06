import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from tensor_prov.utils_type import coo2arr, csr2arr
from tensor_prov.utils_print import print_result_2d, print_result_3d


class Provenance:
    def __init__(self):
        self.step_count = 0

    def capture_time(func):
        def wrapper(self, *args, **kwargs):
            start = time.time()
            result = func(self, *args, **kwargs)
            end = time.time()
            print(f"Runtime: {end - start:.4f} seconds")
            return result, end - start

        return wrapper

    def create_sparse_tensor(self, indices_out: np.ndarray, indices_in: np.ndarray, shape: tuple[int, int]):
        """
        Create a sparse tensor from indices.
        """
        data = np.ones(indices_out.shape[0], dtype=np.int8)
        sparse_tensor = coo_matrix((data, (indices_out, indices_in)), shape=shape)
        return sparse_tensor

    def save_result(self, file):

        os.makedirs(self.save_dir, exist_ok=True)
        filename = os.path.join(self.save_dir, f'step_{self.step_count}_{operation}.npz')
        np.savez(
            filename,
            data=sparse_tensor.data,
            row=sparse_tensor.row,
            col=sparse_tensor.col,
            shape=sparse_tensor.shape,
            bitset=bitset
        )

    @capture_time
    def capture(
            self,
            df_in: pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame],
            df_out: pd.DataFrame,
            key_column: str | None = None,
            column_mapping: dict | None = None
    ):
        if isinstance(df_in, pd.DataFrame):
            tensor_record = capture_row_operation(df_in, df_out, key_column)
            tensor_attr = capture_column_operation(df_in, df_out, column_mapping)
        elif isinstance(df_in, tuple):
            if key_column is None:
                raise ValueError("key_column cannot be None")
            elif isinstance(key_column, str):
                key_column = [key_column, key_column]
            df_in1, df_in2 = df_in
            tensor_record1 = capture_row_operation(df_in1, df_out, key_column[0])
            tensor_record2 = capture_row_operation(df_in2, df_out, key_column[1])
            tensor_attr1 = capture_column_operation(df_in1, df_out, column_mapping)
            tensor_attr2 = capture_column_operation(df_in2, df_out, column_mapping)
            tensor_record = (tensor_record1, tensor_record2)
            tensor_attr = (tensor_attr1, tensor_attr2)
        else:
            raise ValueError("df_in must be a DataFrame or a tuple of DataFrames")

        return tensor_record, tensor_attr


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
        primary_key: str | None | list[str, str]
):
    n_out, n_in = len(df_out), len(df_in)  # provenance shape (n_out, n_in)

    if primary_key is None:
        ids_out = df_out.index.to_numpy()
        ids_in = df_in.index.to_numpy()
    elif isinstance(primary_key, str):
        ids_out = df_out[primary_key].to_numpy()
        ids_in = df_in[primary_key].to_numpy()
    elif isinstance(primary_key, list):
        ids_out = df_out[primary_key[1]].to_numpy()
        ids_in = df_in[primary_key[0]].to_numpy()
    else:
        raise TypeError("primary_key error")

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
        result: tuple,
        num_examples: int = 5
):
    """
    Print the provenance result
    """
    tensor_record, tensor_attr = result
    karg = {
        "tensor_record": tensor_record,
        "tensor_attr": tensor_attr,
        "arr_record": tensor_record if isinstance(tensor_record, np.ndarray) else csr2arr(tensor_record),
        "arr_attr": tensor_attr if isinstance(tensor_attr, np.ndarray) else csr2arr(tensor_attr)
    }
    if isinstance(df_in, pd.DataFrame):
        print_result_2d(df_in, df_out, num_examples, **karg)
    elif isinstance(df_in, tuple):
        print_result_3d(df_in, df_out, num_examples, **karg)


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
