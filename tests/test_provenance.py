import pandas as pd

from tensor_prov.provenance_coo import Provenance, trace
# from tensor_prov.provenance_csr import Provenance, trace
# from tensor_prov.provenance_pandas import Provenance, trace
import tensor_prov.watched_pandas as wpd

save_dir = "results"

def test_capture(capsys):
    user_df = pd.DataFrame({
        "ID": [10, 20, 30, 40],
        "Birthdate": ["1996-07-12", "1994-03-08", None, "1987-11-23"],
        "Gender": ["F", "M", "F", "M"],
        "Postcode": ["90210", None, "12345", "67890"]
    })

    post_df = pd.DataFrame({
        "ID": [1, 2, 3, 4],
        "UID": [10, 40, 30, 10],
        "Topic": ["Art", "Football", "Travel", "Travel"]
    })

    prov = Provenance(save_dir=save_dir, verbose=2)
    user_wdf = prov.subscribe(user_df)
    post_wdf = prov.subscribe(post_df)

    prov.pause()
    user_wdf_HA = user_wdf.copy()
    user_wdf_HA.loc[len(user_wdf_HA)] = [15, "2000-10-20", "F", "75014"]
    user_wdf_HA = user_wdf_HA.sort_values(by="ID")
    prov.start()
    user_wdf.set(user_wdf_HA)
    assert capsys.readouterr().out == """-- id_in -> id_out--
 1 -> 5
-- tensor_record --
 [[0 0]
 [2 1]
 [3 2]
 [4 3]]
-- tensor_attr --
 [[0 0]
 [1 1]
 [2 2]
 [3 3]]
"""

    user_wdf["Gender"] = user_wdf["Gender"].map({"F": 0, "M": 1})
    assert capsys.readouterr().out == """-- id_in -> id_out--
 5 -> 6
-- tensor_record --
 [[0 0]
 [1 1]
 [2 2]
 [3 3]
 [4 4]]
-- tensor_attr --
 [[0 0]
 [1 1]
 [2 2]
 [3 3]]
"""

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
    assert capsys.readouterr().out == """-- id_in -> id_out--
 6 -> 11
-- tensor_record --
 [[0 0]
 [1 1]
 [2 2]
 [3 3]
 [4 4]]
-- tensor_attr --
 [[0 0]
 [1 1]
 [2 2]
 [3 3]
 [4 1]
 [5 1]
 [6 1]]
"""

    user_wdf_VR = user_wdf_VA[["ID", "Year", "Gender"]]
    assert capsys.readouterr().out == """-- id_in -> id_out--
 11 -> 12
-- tensor_record --
 [[0 0]
 [1 1]
 [2 2]
 [3 3]
 [4 4]]
-- tensor_attr --
 [[0 0]
 [1 4]
 [2 2]]
"""

    # join_wdf = user_wdf_VR.merge(post_wdf, left_on="ID", right_on="UID", how="inner")
    join_wdf = wpd.merge(user_wdf_VR, post_wdf, left_on="ID", right_on="UID", how="inner")
    assert capsys.readouterr().out == """-- id_in -> id_out--
 12 -> 13
-- tensor_record --
 [[0 0]
 [1 0]
 [2 3]
 [3 4]]
-- tensor_attr --
 [[0 0]
 [1 1]
 [2 2]]
-- id_in -> id_out--
 2 -> 13
-- tensor_record --
 [[0 0]
 [1 3]
 [2 2]
 [3 1]]
-- tensor_attr --
 [[3 0]
 [4 1]
 [5 2]]
"""

    join_wdf = join_wdf.dropna()
    assert capsys.readouterr().out == """-- id_in -> id_out--
 13 -> 14
-- tensor_record --
 [[0 0]
 [1 1]
 [2 3]]
-- tensor_attr --
 [[0 0]
 [1 1]
 [2 2]
 [3 3]
 [4 4]
 [5 5]]
"""


def test_trace(capsys):
    prov = Provenance(save_dir=save_dir, verbose=1)
    prov.load()
    history = prov.get_history(1, 14)[0]
    tensor_records = [h[0] for h in history]

    # backward trace
    trace_record = trace(tensor_records, "backward")
    print("--trace_record--\n", trace_record)
    trace_record_path = trace(tensor_records, "backward", keep_path=True)
    print("--trace_record keep path--\n", trace_record_path)
    trace_record_slice = trace(tensor_records, "backward", indices=[0, 2])
    print("--trace_record sliced--\n", trace_record_slice)
    trace_record_slice = trace(tensor_records, "backward", indices=[0, 2], keep_path=True)
    print("--trace_record sliced with path--\n", trace_record_slice)

    assert capsys.readouterr().out == """--trace_record--
 [[0 0]
 [1 0]
 [2 3]]
--trace_record keep path--
 [[0 0 0 0 0 0 0]
 [1 1 0 0 0 0 0]
 [2 3 4 4 4 4 3]]
--trace_record sliced--
 [[0 0]
 [2 3]]
--trace_record sliced with path--
 [[0 0 0 0 0 0 0]
 [2 3 4 4 4 4 3]]
"""

    # forward trace
    trace_record = trace(tensor_records, "forward")
    print("--trace_record--\n", trace_record)
    trace_record_path = trace(tensor_records, "forward", keep_path=True)
    print("--trace_record keep path--\n", trace_record_path)
    trace_record_slice = trace(tensor_records, "forward", indices=[0, 2])
    print("--trace_record sliced--\n", trace_record_slice)
    trace_record_slice = trace(tensor_records, "forward", indices=[0, 2], keep_path=True)
    print("--trace_record sliced with path--\n", trace_record_slice)

    assert capsys.readouterr().out == """--trace_record--
 [[0 0]
 [0 1]
 [3 2]]
--trace_record keep path--
 [[0 0 0 0 0 0 0]
 [0 0 0 0 0 1 1]
 [3 4 4 4 4 3 2]]
--trace_record sliced--
 [[0 0]
 [0 1]]
--trace_record sliced with path--
 [[0 0 0 0 0 0 0]
 [0 0 0 0 0 1 1]]
"""
