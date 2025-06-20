import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from utils import coo2arr


def print_result_2d(
        df_in: pd.DataFrame,
        df_out: pd.DataFrame,
        tensor_record: any,
        tensor_attr: any,
        num_examples: int
):
    print("-- D_in --\n")
    print(df_in)
    print("\n-- D_out --\n")
    print(df_out)
    arr_record = coo2arr(tensor_record)
    print("\n-- Record tensor --\n")
    print(tensor_record)
    print("\n-- Record examples --\n")
    record_examples = list(arr_record[:num_examples])
    for idx_out, idx_in in record_examples:
        print(f"D_out[{idx_out}] is from D_in[{idx_in}]: ")
        print(f"\t{df_out.iloc[idx_out].tolist()}")
        print("\tis from")
        print(f"\t{df_in.iloc[idx_in].tolist()}\n")
    if len(arr_record) > num_examples:
        print(f"... {len(arr_record)} items in total.")

    if tensor_attr is not None:
        print("\n-- Attr tensor --\n")
        print(tensor_attr)
        print("\n-- Attr examples --\n")
        cols_in = df_in.columns.tolist()
        cols_out = df_out.columns.tolist()
        arr_attr = coo2arr(tensor_attr)
        attr_examples = list(arr_attr)
        for idx_out, idx_in in attr_examples:
            print(f"D_out[{cols_out[idx_out]}] is from D_in[{cols_in[idx_in]}]: ")


def print_result_3d(
        dfs_in: tuple[pd.DataFrame, pd.DataFrame],
        df_out: pd.DataFrame,
        tensor_record: any,
        tensor_attr: any,
        num_examples: int
):
    df_left, df_right = dfs_in

    print("-- D_left --\n")
    print(df_left)
    print("\n-- D_right --\n")
    print(df_right)
    print("\n-- D_out --\n")
    print(df_out)

    print("\n-- Record tensor --\n")
    print(tensor_record[0])
    print()
    print(tensor_record[1])
    print("\n-- Record examples --\n")
    arr_record = coo2arr(tensor_record)
    record_examples = list(arr_record[:num_examples])
    for idx_out, idx_left, idx_right in record_examples:
        print(f"D_out[{idx_out}] is from D_left[{idx_left}] and D_right[{idx_right}]: ")
        print(f"\t{df_out.iloc[idx_out].tolist()}")
        print("\tis from")
        print(f"\t{df_left.iloc[idx_left].tolist()}")
        print("\tand")
        print(f"\t{df_right.iloc[idx_right].tolist()}\n")
    if len(arr_record) > num_examples:
        print(f"... {len(arr_record)} items in total.")

    print("\n-- Attr tensor --\n")
    print(tensor_attr[0])
    print()
    print(tensor_attr[1])
    print("\n-- Attr examples --\n")
    cols_left = df_left.columns.tolist()
    cols_right = df_right.columns.tolist()
    cols_out = df_out.columns.tolist()
    arr_attr = coo2arr(tensor_attr)
    attr_examples = list(arr_attr)
    for idx_out, idx_left, idx_right in attr_examples:
        s = f"D_out[{cols_out[idx_out]}] is from "
        s += f"D_left[{cols_left[idx_left]}] " if idx_left != -1 else ""
        s += f"D_right[{cols_right[idx_right]}] " if idx_right != -1 else ""
        print(s)

