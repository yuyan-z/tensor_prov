import time

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from tensor_prov.utils import coo2arr, csr2arr
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

    # def save_result(self, operation: str, result):
    #     sparse_tensor, bitset = result
    #     self.step_count += 1
    #     os.makedirs(self.save_dir, exist_ok=True)
    #     filename = os.path.join(self.save_dir, f'step_{self.step_count}_{operation}.npz')
    #     np.savez(
    #         filename,
    #         data=sparse_tensor.data,
    #         row=sparse_tensor.row,
    #         col=sparse_tensor.col,
    #         shape=sparse_tensor.shape,
    #         bitset=bitset
    #     )

    @capture_time
    def capture(
            self,
            df_in: pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame],
            df_out: pd.DataFrame,
            key_column: str,
            column_mapping: dict | None = None
    ):
        if isinstance(df_in, pd.DataFrame):
            tensor_record = self.capture_row_operation(df_in, df_out, key_column)
            tensor_attr = self.capture_column_operation(df_in, df_out, column_mapping)
        elif isinstance(df_in, tuple):
            df_in1, df_in2 = df_in
            tensor_record1 = self.capture_row_operation(df_in1, df_out, key_column)
            tensor_record2 = self.capture_row_operation(df_in2, df_out, key_column)
            tensor_attr1 = self.capture_column_operation(df_in1, df_out, column_mapping)
            tensor_attr2 = self.capture_column_operation(df_in2, df_out, column_mapping)
            tensor_record = (tensor_record1, tensor_record2)
            tensor_attr = (tensor_attr1, tensor_attr2)
        else:
            raise ValueError("df_in must be a DataFrame or a tuple of DataFrames")

        return tensor_record, tensor_attr

    def capture_row_operation(self, df_in: pd.DataFrame, df_out: pd.DataFrame, key_column: str):
        n_out, n_in = len(df_out), len(df_in)  # provenance shape (n_out, n_in)
        ids_out = df_out[key_column].to_numpy()
        ids_in = df_in[key_column].to_numpy()
        indices_out, indices_in = np.where(ids_out[:, None] == ids_in[None, :])

        # Create sparse tensor
        sparse_tensor = self.create_sparse_tensor(indices_out, indices_in, (n_out, n_in))

        return sparse_tensor

    def capture_column_operation(self, df_in: pd.DataFrame, df_out: pd.DataFrame, column_mapping: dict | None):
        cols_in = df_in.columns.values
        cols_out = df_out.columns.values
        n_out, n_in = len(cols_out), len(cols_in)

        if column_mapping is None:
            indices_out, indices_in = np.where(cols_out[:, None] == cols_in[None, :])
        else:
            out_to_in_map = {out: inp for inp, outs in column_mapping.items() for out in outs}
            col_in_index = {col: idx for idx, col in enumerate(cols_in)}
            mapped_in_cols = np.array([
                col if col in col_in_index else out_to_in_map.get(col, None)
                for col in cols_out
            ])
            indices_in = np.array([col_in_index[col] for col in mapped_in_cols])
            indices_out = np.arange(len(cols_out))

        # Create sparse tensor
        sparse_tensor = self.create_sparse_tensor(indices_out, indices_in, (n_out, n_in))

        return sparse_tensor


def print_prov_result(
        df_in: pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame],
        df_out: pd.DataFrame,
        result: tuple[coo_matrix, coo_matrix] | tuple[tuple[coo_matrix, coo_matrix], coo_matrix],
        num_examples: int = 5
):
    """
    Print the provenance result
    """
    tensor_record, tensor_attr = result
    karg = {
        "tensor_record": tensor_record,
        "tensor_attr": tensor_attr,
        "arr_record": coo2arr(tensor_record),
        "arr_attr": coo2arr(tensor_attr)
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
    if direction == "forward":
        tensors_ordered = [t.T for t in tensors]
    elif direction == "backward":
        tensors_ordered = tensors[::-1]
    else:
        raise ValueError("direction must be 'forward' or 'backward'")

    # Slicing
    t1 = tensors_ordered[0] if indices is None else slice_coo(tensors_ordered[0], indices)

    if not keep_path:
        # csr matrix multiplication
        t1 = t1.tocsr()
        for t2 in tensors_ordered[1:]:
            t1 = t1 @ t2
        result = csr2arr(t1)
    else:
        # pandas join
        t1_arr = coo2arr(t1)
        df_path = pd.DataFrame(t1_arr).add_prefix('path_')
        for t2 in tensors_ordered[1:]:
            t2_arr = coo2arr(t2)
            df_t2 = pd.DataFrame(t2_arr).add_prefix('t_')
            df_path = df_path.merge(df_t2, left_on=df_path.columns[-1], right_on='t_0', how='left')
            df_path = df_path.drop(columns=['t_0'])
        df_path = df_path.sort_values(by=df_path.columns[0])
        result = df_path.fillna(-1).to_numpy(dtype=int)

    return result


def slice_coo(coo_mat: coo_matrix, indices: list | np.ndarray) -> coo_matrix:
    mask = np.isin(coo_mat.row, indices)
    new_row = coo_mat.row[mask]
    new_col = coo_mat.col[mask]
    new_data = coo_mat.data[mask]
    return coo_matrix((new_data, (new_row, new_col)), shape=coo_mat.shape)

