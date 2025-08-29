import numpy as np
import pandas as pd

from tensor_prov.provenance_coo import Provenance, print_prov_result, trace, trace_pandas, trace_numpy

# from tensor_prov.provenance_csr import Provenance, print_prov_result, trace
# from tensor_prov.provenance_pandas import Provenance, print_prov_result, trace

# Example datasets
df = pd.DataFrame({
    "ID": [10, 20, 30, 40],
    "Birthdate": ["1996-07-12", "1994-03-08", np.nan, "1987-11-23"],
    "Gender": ["F", "M", "F", "M"],
    "Postcode": ["90210", np.nan, "12345", "67890"]
})

name_df = pd.DataFrame({
    "ID": [20, 40],
    "Name": ["Alex", "Bob"]
})

prov = Provenance()


def example_horizontal_reduction():
    # Drop rows where containing NaN values
    df_horizontal_reduced = df.dropna()
    result, runtime = prov.capture(df, df_horizontal_reduced)
    print_prov_result(df, df_horizontal_reduced, result)


def example_horizontal_augmentation():
    # Add records and reorder
    df_horizontal_augmented = pd.DataFrame({
        "ID": [10, 20, 30, 40, 15],
        "Birthdate": ["1996-07-12", "1994-03-08", np.nan, "1987-11-23", "2000-10-20"],
        "Gender": ["F", "M", "F", "M", "F"],
        "Postcode": ["90210", np.nan, "12345", "67890", "75014"]
    })
    df_horizontal_augmented = df_horizontal_augmented.sort_values(by="ID")
    result, runtime = prov.capture(df, df_horizontal_augmented)
    print_prov_result(df, df_horizontal_augmented, result)


def example_vertical_reduction():
    # Select columns and reorder
    df_vertical_reduced = df[["ID", "Gender", "Birthdate"]]
    result, runtime = prov.capture(df, df_vertical_reduced)
    print_prov_result(df, df_vertical_reduced, result)


def example_vertical_augmentation():
    # Split BirthDate to Year, Month, Day.
    df_vertical_augmented = df.copy()
    birth_split = df_vertical_augmented["Birthdate"].str.split("-", expand=True)
    df_vertical_augmented["Year"] = birth_split[0]
    df_vertical_augmented["Month"] = birth_split[1]
    df_vertical_augmented["Day"] = birth_split[2]
    column_mapping = {
        "Birthdate": ["Year", "Month", "Day"]
    }
    result, runtime = prov.capture(df, df_vertical_augmented, column_mapping=column_mapping)
    print_prov_result(df, df_vertical_augmented, result)


def example_data_transformation():
    # Label Encoding
    df_transformed = df.copy()
    df_transformed["Gender"] = df_transformed["Gender"].map({"F": 0, "M": 1})
    result, runtime = prov.capture(df, df_transformed)
    print_prov_result(df, df_transformed, result)


def example_data_fusion():
    # Inner join
    df_inner_joined = pd.merge(df, name_df, on="ID", how="inner")
    result, runtime = prov.capture((df, name_df), df_inner_joined, key_column="ID")
    print_prov_result((df, name_df), df_inner_joined, result)


def example_trace(direction: str = "backward"):
    # 1. Drop rows with NaN in Birthdate
    df1 = df.copy()
    df1["Birthdate"] = pd.to_datetime(df1["Birthdate"], errors='coerce')
    df1 = df1.dropna(subset=["Birthdate"])
    result1, runtime = prov.capture(df, df1)

    # 2. Split BirthDate to Year, Month, Day. Drop BirthDate. Reorder columns
    df2 = df1.copy()
    df2["Year"] = df2["Birthdate"].dt.year
    df2["Month"] = df2["Birthdate"].dt.month
    df2["Day"] = df2["Birthdate"].dt.day
    df2 = df2.drop(columns=["Birthdate"])
    df2 = df2[["ID", "Gender", "Postcode", "Year", "Month", "Day"]]
    column_mapping = {
        "Birthdate": ["Year", "Month", "Day"]
    }
    result2, runtime = prov.capture(df1, df2, column_mapping=column_mapping)

    # 3. Reorder rows by Year
    df3 = df2.dropna(subset=["Postcode"])
    df3 = df3.sort_values(by="Year")
    result3, runtime = prov.capture(df2, df3)

    dfs = [df, df1, df2, df3]
    results = [result1, result2, result3]
    tensors_record = [result[0] for result in results]
    tensors_attr = [result[1] for result in results]

    for d in dfs:
        print(d)
        print()

    for t in tensors_record:
        print(t)
        print()

    # for t in tensors_attr:
    #     print(t)
    #     print()

    print("\n-- Trace records --\n")
    trace_record = trace_numpy(tensors_record, direction=direction)
    print(trace_record)
    # trace_attr = trace(tensors_attr, direction=direction)
    # print()
    # print(trace_attr)
    # print_prov_result(dfs[0], dfs[-1], (trace_record, trace_attr), direction=direction)

    print("\n-- Trace records sliced --\n")
    trace_record_sliced = trace_numpy(tensors_record, direction=direction, indices=[1, 3], keep_path=False)
    print(trace_record_sliced)

    print("\n-- Trace records with path --\n")
    trace_record_path = trace_numpy(tensors_record, direction=direction, indices=None, keep_path=True)
    print(trace_record_path)

    print("\n-- Trace records sliced with path --\n")
    trace_record_sliced_path = trace_numpy(tensors_record, direction=direction, indices=[1, 3], keep_path=True)
    print(trace_record_sliced_path)


if __name__ == '__main__':
    example_horizontal_reduction()
    # example_horizontal_augmentation()
    # example_vertical_reduction()
    # example_vertical_augmentation()
    # example_data_transformation()
    # example_data_fusion()
    # example_trace("backward")
    # example_trace("forward")
