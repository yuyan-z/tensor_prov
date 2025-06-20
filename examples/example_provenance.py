import numpy as np
import pandas as pd

# from provenance_coo import Provenance, print_prov_result
from provenance_csr import Provenance, print_prov_result

# Example datasets
df1 = pd.DataFrame({
    "ID": [10, 20, 30, 40],
    "Birthdate": ["1996-07-12", "1994-03-08", np.nan, "1987-11-23"],
    "Gender": ["F", "M", "F", "M"],
    "Postcode": ["90210", np.nan, "12345", "67890"]
})

df2 = pd.DataFrame({
    "ID": [20, 40],
    "Name": ["Alex", "Bob"]
})

prov = Provenance()


def example_horizontal_reduction():
    # Drop rows where containing NaN values
    df1_horizontal_reduced = df1.dropna().reset_index(drop=True)
    result, runtime = prov.capture(df1, df1_horizontal_reduced, key_column="ID")
    print_prov_result(df1, df1_horizontal_reduced, result)


def example_horizontal_augmentation():
    # Generate instances and resort
    df_horizontal_augmented = pd.DataFrame({
        "ID": [10, 20, 30, 40, 15, 35],
        "Birthdate": ["1996-07-12", "1994-03-08", np.nan, "1987-11-23", "2000-10-20", "1999-01-06"],
        "Gender": ["F", "M", "F", "M", "F", "M"],
        "Postcode": ["90210", np.nan, "12345", "67890", "75014", "29280"]
    })
    df_horizontal_augmented = df_horizontal_augmented.sort_values(by="ID").reset_index(drop=True)
    result, runtime = prov.capture(df1, df_horizontal_augmented, key_column="ID")
    print_prov_result(df1, df_horizontal_augmented, result)


def example_vertical_reduction():
    # Select columns and reorder
    df1_vertical_reduced = df1[["ID", "Gender", "Birthdate"]]
    result, runtime = prov.capture(df1, df1_vertical_reduced, "ID")
    print_prov_result(df1, df1_vertical_reduced, result)


def example_vertical_augmentation():
    # Split BirthDate to Year, Month, Day. Drop BirthDate. Reorder
    df1_vertical_augmented = df1.copy()
    birth_split = df1_vertical_augmented["Birthdate"].str.split("-", expand=True)
    df1_vertical_augmented["Year"] = birth_split[0]
    df1_vertical_augmented["Month"] = birth_split[1]
    df1_vertical_augmented["Day"] = birth_split[2]
    df1_vertical_augmented = df1_vertical_augmented[["ID", "Birthdate", "Gender", "Postcode", "Year", "Month", "Day"]]
    column_mapping = {
        "Birthdate": ["Year", "Month", "Day"]
    }

    result, runtime = prov.capture(df1, df1_vertical_augmented, "ID", column_mapping)
    print_prov_result(df1, df1_vertical_augmented, result)


def example_data_transformation():
    # Value transformation
    df1_transformed = df1.copy()
    df1_transformed["Gender"] = df1_transformed["Gender"].map({"F": 0, "M": 1})
    result, runtime = prov.capture(df1, df1_transformed, "ID")
    print_prov_result(df1, df1_transformed, result)


def example_data_fusion():
    # Inner join
    df_inner_joined = pd.merge(df1, df2, on="ID", how="outer")
    result, runtime = prov.capture((df1, df2), df_inner_joined, key_column="ID")
    print_prov_result((df1, df2), df_inner_joined, result)


if __name__ == '__main__':
    example_horizontal_reduction()
    # example_horizontal_augmentation()
    # example_vertical_reduction()
    # example_vertical_augmentation()
    # example_data_transformation()
    # example_data_fusion()


