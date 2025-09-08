import numpy as np
import pandas as pd

# from provenance_coo import Provenance, trace
from provenance_pandas import Provenance, trace

user_df = pd.DataFrame({
    "ID": [10, 20, 30, 40],
    "Birthdate": ["1996-07-12", "1994-03-08", np.nan, "1987-11-23"],
    "Gender": ["F", "M", "F", "M"],
    "Postcode": ["90210", np.nan, "12345", "67890"]
})

post_df = pd.DataFrame({
    "ID": [1, 2, 3, 4],
    "UID": [10, 40, 30, 10],
    "Topic": ["Art", "Football", "Travel", "Travel"]
})

prov = Provenance(verbose=0)
user_wdf = prov.subscribe(user_df)
post_wdf = prov.subscribe(post_df)


def example_capture():
    prov.pause()
    user_wdf_HA = user_wdf.copy()
    user_wdf_HA.loc[len(user_wdf_HA)] = [15, "2000-10-20", "F", "75014"]
    user_wdf_HA = user_wdf_HA.sort_values(by="ID")
    prov.start()
    user_wdf.set(user_wdf_HA)

    user_wdf["Gender"] = user_wdf["Gender"].map({"F": 0, "M": 1})

    prov.pause()
    user_wdf_VA = user_wdf.copy()
    user_wdf_VA["Birthdate"] = pd.to_datetime(user_wdf_VA["Birthdate"])
    user_wdf_VA["Year"] = user_wdf_VA["Birthdate"].dt.year
    user_wdf_VA["Month"] = user_wdf_VA["Birthdate"].dt.month
    user_wdf_VA["Day"] = user_wdf_VA["Birthdate"].dt.day
    column_mapping = {
        "Birthdate": ["Year", "Month", "Day"]
    }
    prov.start()
    prov.capture(user_wdf, user_wdf_VA, column_mapping=column_mapping)
    # user_wdf.set(user_wdf_VA, column_mapping=column_mapping)

    user_wdf_VR = user_wdf_VA[["ID", "Year", "Gender"]]

    join_wdf = user_wdf_VR.merge(post_wdf, left_on="ID", right_on="UID", how="inner")

    join_wdf = join_wdf.dropna()


def example_trace():
    path = prov.graph.get_edges(1, 14)[0]
    tensor_records = [p[0] for p in path]

    # backward trace
    trace_record = trace(tensor_records, "backward")
    print("--trace_record--\n", trace_record)
    trace_record_path = trace(tensor_records, "backward", keep_path=True)
    print("--trace_record keep path--\n", trace_record_path)
    trace_record_slice = trace(tensor_records, "backward", indices=[0, 2])
    print("--trace_record sliced--\n", trace_record_slice)
    trace_record_slice = trace(tensor_records, "backward", indices=[0, 2], keep_path=True)
    print("--trace_record sliced with path--\n", trace_record_slice)

    # forward trace
    trace_record = trace(tensor_records, "forward")
    print("--trace_record--\n", trace_record)
    trace_record_path = trace(tensor_records, "forward", keep_path=True)
    print("--trace_record keep path--\n", trace_record_path)
    trace_record_slice = trace(tensor_records, "forward", indices=[0, 2])
    print("--trace_record sliced--\n", trace_record_slice)
    trace_record_slice = trace(tensor_records, "forward", indices=[0, 2], keep_path=True)
    print("--trace_record sliced with path--\n", trace_record_slice)


if __name__ == "__main__":
    example_capture()
    example_trace()
