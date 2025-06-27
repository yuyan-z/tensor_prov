import time

import numpy as np
import pandas as pd

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
        sparse_tensor = pd.DataFrame({
            "idx_out": indices_out,
            "idx_in": indices_in,
        })
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
            tensor_record = pd.merge(tensor_record1, tensor_record2, on="idx_out", how="outer")
            tensor_record = tensor_record.fillna(-1).astype(int)
            tensor_attr = pd.merge(tensor_attr1, tensor_attr2, on="idx_out", how="outer")
            tensor_attr = tensor_attr.fillna(-1).astype(int)
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
        result,
        num_examples: int = 5
):
    """
    Print the provenance result
    """
    tensor_record, tensor_attr = result
    karg = {
        "tensor_record": tensor_record,
        "tensor_attr": tensor_attr,
        "arr_record": tensor_record.to_numpy(),
        "arr_attr": tensor_attr.to_numpy(),
    }
    if isinstance(df_in, pd.DataFrame):
        print_result_2d(df_in, df_out, num_examples, **karg)
    elif isinstance(df_in, tuple):
        print_result_3d(df_in, df_out, num_examples, **karg)


def trace(
        tensors: list[pd.DataFrame],
        direction: str = "backward",
        indices: list[int] | None = None,
        keep_path: bool = False
):
    if direction == "forward":
        df_path = tensors[0].iloc[:, [1, 0]]
        if indices is not None:
            df_path = df_path[df_path.iloc[:, 0].isin(indices)]
        for t in tensors[1:]:
            t = t.add_prefix('t_')
            df_path = df_path.merge(t, left_on=df_path.columns[-1], right_on=t.columns[1], how="left")
            df_path = df_path.drop(columns=[t.columns[1]])

    elif direction == "backward":
        df_path = tensors[-1].copy()
        if indices is not None:
            df_path = df_path[df_path.iloc[:, 0].isin(indices)]
        for t in tensors[-2::-1]:
            t = t.add_prefix('t_')
            df_path = df_path.merge(t, left_on=df_path.columns[-1], right_on=t.columns[0], how="left")
            df_path = df_path.drop(columns=[t.columns[0]])
            if not keep_path:
                df_path = df_path.iloc[:, [0, -1]]
    else:
        raise ValueError("direction must be 'forward' or 'backward'")

    result = df_path.fillna(-1).to_numpy(dtype=int)

    return result

