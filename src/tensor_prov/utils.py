def get_merge_column_mapping(cols_left, cols_right, cols_out, suffixes=("_x", "_y")):
    column_mapping = [{}, {}]
    cols_overlap = set(cols_left).intersection(cols_right)
    for col in cols_overlap:
        if f"{col}{suffixes[0]}" in cols_out:
            column_mapping[0][col] = [f"{col}{suffixes[0]}"]
        if f"{col}{suffixes[1]}" in cols_out:
            column_mapping[1][col] = [f"{col}{suffixes[1]}"]
    return column_mapping


