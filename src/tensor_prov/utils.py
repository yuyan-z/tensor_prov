def get_merge_column_mapping(cols_left, cols_right, cols_out, suffixes=("_x", "_y")):
    column_mapping = [{}, {}]
    cols_overlap = set(cols_left).intersection(cols_right)
    for col in cols_overlap:
        if f"{col}{suffixes[0]}" in cols_out:
            column_mapping[0][col] = [f"{col}{suffixes[0]}"]
        if f"{col}{suffixes[1]}" in cols_out:
            column_mapping[1][col] = [f"{col}{suffixes[1]}"]
    return column_mapping


def get_rename_column_mapping(cols_in, cols_out, **kwargs):
    column_mapping = None
    if kwargs.get("columns") or kwargs.get("axis") == "columns" or kwargs.get("axis") == 1:
        column_mapping = {col_in: [col_out] for col_in, col_out in zip(cols_in, cols_out) if col_out != col_in}
    return column_mapping

def get_column_mapping(attr, cols_in, cols_out, **kwargs):
    if attr == "rename":
        return get_rename_column_mapping(cols_in, cols_out, **kwargs)
    else:
        return None
