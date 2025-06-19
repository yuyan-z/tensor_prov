import time

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


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

    def print_result(
            self,
            df_in: pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame],
            df_out: pd.DataFrame,
            results: tuple[coo_matrix, coo_matrix] | tuple[tuple[coo_matrix, coo_matrix], coo_matrix],
            num_examples: int = 5
    ):
        """
        Explain the provenance results
        """
        tensor_record, tensor_attr = results
        if isinstance(df_in, pd.DataFrame):
            self.print_result_2d(df_in, df_out, tensor_record, tensor_attr, num_examples)
        elif isinstance(df_in, tuple):
            self.print_result_3d(df_in, df_out, tensor_record, tensor_attr, num_examples)

    def print_result_2d(
            self,
            df_in: pd.DataFrame,
            df_out: pd.DataFrame,
            tensor_record: coo_matrix | np.ndarray,
            tensor_attr: coo_matrix | np.ndarray | None = None,
            num_examples: int = 5
    ):
        print("-- D_in --\n")
        print(df_in)
        print("\n-- D_out --\n")
        print(df_out)

        print("\n-- Record tensor --\n")
        print(tensor_record)
        print("\n-- Record examples --\n")
        if not isinstance(tensor_record, np.ndarray):
            tensor_record_arr = np.vstack((
                tensor_record.row,
                tensor_record.col
            )).T
        else:
            tensor_record_arr = tensor_record
        record_examples = list(tensor_record_arr[:num_examples])
        for idx_out, idx_in in record_examples:
            print(f"D_out[{idx_out}] is from D_in[{idx_in}]: ")
            print(f"\t{df_out.iloc[idx_out].tolist()}")
            print("\tis from")
            print(f"\t{df_in.iloc[idx_in].tolist()}\n")
        if len(tensor_record_arr) > num_examples:
            print(f"... {len(tensor_record_arr)} items in total.")

        if tensor_attr is not None:
            print("\n-- Attr tensor --\n")
            print(tensor_attr)
            print("\n-- Attr examples --\n")
            cols_in = df_in.columns.tolist()
            cols_out = df_out.columns.tolist()
            if not isinstance(tensor_attr, np.ndarray):
                tensor_attr_arr = np.vstack((
                    tensor_attr.row,
                    tensor_attr.col
                )).T
            else:
                tensor_attr_arr = tensor_attr
            attr_examples = list(tensor_attr_arr)
            for idx_out, idx_in in attr_examples:
                print(f"D_out[{cols_out[idx_out]}] is from D_in[{cols_in[idx_in]}]: ")

    def print_result_3d(
            self,
            dfs_in: tuple[pd.DataFrame, pd.DataFrame],
            df_out: pd.DataFrame,
            tensor_record: tuple[coo_matrix, coo_matrix],
            tensor_attr: tuple[coo_matrix, coo_matrix],
            num_examples: int = 5
    ):
        df_left, df_right = dfs_in

        print("-- D_left --\n")
        print(df_left)
        print("\n-- D_right --\n")
        print(df_right)
        print("\n-- D_out --\n")
        print(df_out)

        tensor_record1, tensor_record2 = tensor_record
        tensor_record_arr = np.vstack((
            tensor_record1.row,
            tensor_record1.col,
            tensor_record2.col
        )).T

        print("\n-- Record tensor --\n")
        print(tensor_record1)
        print(tensor_record2)
        print("\n-- Record examples --\n")
        record_examples = list(tensor_record_arr[:num_examples])
        for idx_out, idx_left, idx_right in record_examples:
            print(f"D_out[{idx_out}] is from D_left[{idx_left}] and D_right[{idx_right}]: ")
            print(f"\t{df_out.iloc[idx_out].tolist()}")
            print("\tis from")
            print(f"\t{df_left.iloc[idx_left].tolist()}")
            print("\tand")
            print(f"\t{df_right.iloc[idx_right].tolist()}\n")
        if len(tensor_record_arr) > num_examples:
            print(f"... {len(tensor_record_arr)} items in total.")

        tensor_attr1, tensor_attr2 = tensor_attr
        print("\n-- Attr tensor --\n")
        print(tensor_attr1)
        print(tensor_attr2)
