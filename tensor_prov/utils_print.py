import pandas as pd


def print_result_2d(
        df_in: pd.DataFrame,
        df_out: pd.DataFrame,
        num_examples: int,
        tensor_record: any,
        tensor_attr: any,
        arr_record: list,
        arr_attr: list
):
    print("\n-- Record tensor --\n")
    print(tensor_record)
    print("\n-- Record examples --\n")
    record_examples = list(arr_record[:num_examples])
    for idx_out, idx_in in record_examples:
        print(f"D_out[{idx_out}] is from D_in[{idx_in}]: ")
        print(f"\t{df_out.iloc[[idx_out]]}")
        print("\tis from")
        print(f"\t{df_in.iloc[[idx_in]]}\n")
    if len(arr_record) > num_examples:
        print(f"... {len(arr_record)} items in total.")

    if tensor_attr is not None:
        print("\n-- Attr tensor --\n")
        print(tensor_attr)
        print("\n-- Attr examples --\n")
        cols_in = df_in.columns.tolist()
        cols_out = df_out.columns.tolist()

        attr_examples = list(arr_attr)
        for idx_out, idx_in in attr_examples:
            print(f"D_out[{cols_out[idx_out]}] is from D_in[{cols_in[idx_in]}]")


def print_result_3d(
        dfs_in: tuple[pd.DataFrame, pd.DataFrame],
        df_out: pd.DataFrame,
        num_examples: int,
        tensor_record: any,
        tensor_attr: any,
        arr_record: list,
        arr_attr: list
):
    df_left, df_right = dfs_in
    print("\n-- Record tensor --\n")
    if isinstance(tensor_record, tuple):
        print(tensor_record[0])
        print()
        print(tensor_record[1])
    else:
        print(tensor_record)
    print("\n-- Record examples --\n")
    record_examples = list(arr_record[:num_examples])
    for idx_out, idx_left, idx_right in record_examples:
        s = f"D_out[{idx_out}] is from "
        s_detail = f"\t{df_out.iloc[idx_out].tolist()} is from "
        if idx_left != -1:
            s += f"D_left[{idx_left}] "
            s_detail += f"\n\t{df_left.iloc[idx_left].tolist()}"
        if idx_right != -1:
            s += f"D_right[{idx_right}] "
            s_detail += f"\n\t{df_right.iloc[idx_right].tolist()}"
        print(s)
        print(s_detail)

    if len(arr_record) > num_examples:
        print(f"... {len(arr_record)} items in total.")

    print("\n-- Attr tensor --\n")
    if isinstance(tensor_attr, tuple):
        print(tensor_attr[0])
        print()
        print(tensor_attr[1])
    else:
        print(tensor_attr)
    print("\n-- Attr examples --\n")
    cols_left = df_left.columns.tolist()
    cols_right = df_right.columns.tolist()
    cols_out = df_out.columns.tolist()
    attr_examples = list(arr_attr)
    for idx_out, idx_left, idx_right in attr_examples:
        s = f"D_out[{cols_out[idx_out]}] is from "
        s += f"D_left[{cols_left[idx_left]}] " if idx_left != -1 else ""
        s += f"D_right[{cols_right[idx_right]}] " if idx_right != -1 else ""
        print(s)

