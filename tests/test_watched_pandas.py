import numpy as np
import pandas as pd
import pytest

from tensor_prov.watched_pandas import WatchedDataFrame
import tensor_prov.watched_pandas as wpd


class Provenance:
    def capture(
            self,
            df_in: pd.DataFrame | tuple[pd.DataFrame, ...],
            df_out: pd.DataFrame,
            primary_key: str | None | tuple[str, str] = None,
            column_mapping: dict | None = None
    ):
        print("-- df_in --")
        if isinstance(df_in, pd.DataFrame):
            print(df_in)
        elif isinstance(df_in, tuple) and all(isinstance(x, pd.DataFrame) for x in df_in):
            if primary_key is None:
                raise ValueError("primary_key is required !")
            elif isinstance(primary_key, str):
                primary_key = tuple(primary_key for _ in df_in)
            elif isinstance(primary_key, tuple) and len(primary_key) == len(df_in):
                pass
            else:
                raise ValueError("primary_key error !")

            for d in df_in:
                print(d)

            print("-- primary_key --")
            print(primary_key)
        else:
            raise ValueError("df_in should be DataFrame or tuple[pd.DataFrame, pd.DataFrame] !")

        print("-- df_out --")
        print(df_out)

        if column_mapping:
            print("-- column_mapping --")
            print(column_mapping)


def make_df():
    return pd.DataFrame({
        "ID": [10, 20, 30, 40],
        "Birthdate": ["1996-07-12", "1994-03-08", np.nan, "1987-11-23"],
        "Gender": ["F", "M", "F", "M"],
        "Postcode": ["90210", np.nan, "12345", "67890"]
    })


def make_df2():
    return pd.DataFrame({
        "ID": [10, 30, 40],
        "Name": ["Alice", "Chloe", "Bob"]
    })


prov = Provenance()


def test_set_df(capsys):
    df = make_df()
    df2 = make_df2()
    wdf = WatchedDataFrame(df, prov)

    wdf.set_df(df2)

    captured = capsys.readouterr()
    expected_output = """-- df_in --
   ID   Birthdate Gender Postcode
0  10  1996-07-12      F    90210
1  20  1994-03-08      M      NaN
2  30         NaN      F    12345
3  40  1987-11-23      M    67890
-- df_out --
   ID   Name
0  10  Alice
1  30  Chloe
2  40    Bob
"""
    assert captured.out == expected_output


def test_setitem(capsys):
    df = make_df()
    wdf = WatchedDataFrame(df, prov)

    wdf["Age"] = [16, 17, 18, 19]

    captured = capsys.readouterr()
    expected_output = """-- df_in --
   ID   Birthdate Gender Postcode
0  10  1996-07-12      F    90210
1  20  1994-03-08      M      NaN
2  30         NaN      F    12345
3  40  1987-11-23      M    67890
-- df_out --
   ID   Birthdate Gender Postcode  Age
0  10  1996-07-12      F    90210   16
1  20  1994-03-08      M      NaN   17
2  30         NaN      F    12345   18
3  40  1987-11-23      M    67890   19
"""
    assert captured.out == expected_output


def test_setitem_2(capsys):
    df = make_df()
    wdf = WatchedDataFrame(df, prov)

    wdf["ID"] = [1, 2, 3, 4]

    captured = capsys.readouterr()
    expected_output = """-- df_in --
   ID   Birthdate Gender Postcode
0  10  1996-07-12      F    90210
1  20  1994-03-08      M      NaN
2  30         NaN      F    12345
3  40  1987-11-23      M    67890
-- df_out --
   ID   Birthdate Gender Postcode
0   1  1996-07-12      F    90210
1   2  1994-03-08      M      NaN
2   3         NaN      F    12345
3   4  1987-11-23      M    67890
"""
    assert captured.out == expected_output


def test_getitem(capsys):
    df = make_df()
    wdf = WatchedDataFrame(df, prov)

    wdf = wdf[0:3]

    captured = capsys.readouterr()
    expected_output = """-- df_in --
   ID   Birthdate Gender Postcode
0  10  1996-07-12      F    90210
1  20  1994-03-08      M      NaN
2  30         NaN      F    12345
3  40  1987-11-23      M    67890
-- df_out --
   ID   Birthdate Gender Postcode
0  10  1996-07-12      F    90210
1  20  1994-03-08      M      NaN
2  30         NaN      F    12345
"""
    assert captured.out == expected_output


def test_getitem_2(capsys):
    df = make_df()
    wdf = WatchedDataFrame(df, prov)

    wdf = wdf[["ID", "Gender"]]

    captured = capsys.readouterr()
    expected_output = """-- df_in --
   ID   Birthdate Gender Postcode
0  10  1996-07-12      F    90210
1  20  1994-03-08      M      NaN
2  30         NaN      F    12345
3  40  1987-11-23      M    67890
-- df_out --
   ID Gender
0  10      F
1  20      M
2  30      F
3  40      M
"""
    assert captured.out == expected_output


def test_getitem_3(capsys):
    df = make_df()
    wdf = WatchedDataFrame(df, prov)

    wdf = wdf[wdf["Gender"] == "F"]

    captured = capsys.readouterr()
    expected_output = """-- df_in --
   ID   Birthdate Gender Postcode
0  10  1996-07-12      F    90210
1  20  1994-03-08      M      NaN
2  30         NaN      F    12345
3  40  1987-11-23      M    67890
-- df_out --
   ID   Birthdate Gender Postcode
0  10  1996-07-12      F    90210
2  30         NaN      F    12345
"""
    assert captured.out == expected_output


def test_setitem_getitem(capsys):
    df = make_df()
    wdf = WatchedDataFrame(df, prov)

    wdf["Female"] = wdf["Gender"] == "F"

    captured = capsys.readouterr()
    expected_output = """-- df_in --
   ID   Birthdate Gender Postcode
0  10  1996-07-12      F    90210
1  20  1994-03-08      M      NaN
2  30         NaN      F    12345
3  40  1987-11-23      M    67890
-- df_out --
   ID   Birthdate Gender Postcode  Female
0  10  1996-07-12      F    90210    True
1  20  1994-03-08      M      NaN   False
2  30         NaN      F    12345    True
3  40  1987-11-23      M    67890   False
"""
    assert captured.out == expected_output

def test_setitem_getitem_2(capsys):
    df = make_df()
    wdf = WatchedDataFrame(df, prov)

    wdf["Birthdate"] = pd.to_datetime(wdf["Birthdate"])  # Type conversion
    wdf['year'] = wdf['Birthdate'].dt.year  # Date time operations

    captured = capsys.readouterr()
    expected_output = """-- df_in --
   ID   Birthdate Gender Postcode
0  10  1996-07-12      F    90210
1  20  1994-03-08      M      NaN
2  30         NaN      F    12345
3  40  1987-11-23      M    67890
-- df_out --
   ID  Birthdate Gender Postcode
0  10 1996-07-12      F    90210
1  20 1994-03-08      M      NaN
2  30        NaT      F    12345
3  40 1987-11-23      M    67890
-- df_in --
   ID  Birthdate Gender Postcode
0  10 1996-07-12      F    90210
1  20 1994-03-08      M      NaN
2  30        NaT      F    12345
3  40 1987-11-23      M    67890
-- df_out --
   ID  Birthdate Gender Postcode    year
0  10 1996-07-12      F    90210  1996.0
1  20 1994-03-08      M      NaN  1994.0
2  30        NaT      F    12345     NaN
3  40 1987-11-23      M    67890  1987.0
"""
    assert captured.out == expected_output


def test_getattr_inplace(capsys):
    df = make_df()
    wdf = WatchedDataFrame(df, prov)

    wdf.dropna(inplace=True)

    captured = capsys.readouterr()
    expected_output = """-- df_in --
   ID   Birthdate Gender Postcode
0  10  1996-07-12      F    90210
1  20  1994-03-08      M      NaN
2  30         NaN      F    12345
3  40  1987-11-23      M    67890
-- df_out --
   ID   Birthdate Gender Postcode
0  10  1996-07-12      F    90210
3  40  1987-11-23      M    67890
"""
    assert captured.out == expected_output


def test_getattr_inplace_2(capsys):
    df = make_df()
    wdf = WatchedDataFrame(df, prov)

    wdf.sort_values("ID", ascending=False, inplace=True)
    wdf.reset_index(drop=True, inplace=True)

    captured = capsys.readouterr()
    expected_output = """-- df_in --
   ID   Birthdate Gender Postcode
0  10  1996-07-12      F    90210
1  20  1994-03-08      M      NaN
2  30         NaN      F    12345
3  40  1987-11-23      M    67890
-- df_out --
   ID   Birthdate Gender Postcode
3  40  1987-11-23      M    67890
2  30         NaN      F    12345
1  20  1994-03-08      M      NaN
0  10  1996-07-12      F    90210
-- df_in --
   ID   Birthdate Gender Postcode
3  40  1987-11-23      M    67890
2  30         NaN      F    12345
1  20  1994-03-08      M      NaN
0  10  1996-07-12      F    90210
-- df_out --
   ID   Birthdate Gender Postcode
0  40  1987-11-23      M    67890
1  30         NaN      F    12345
2  20  1994-03-08      M      NaN
3  10  1996-07-12      F    90210
"""
    assert captured.out == expected_output


def test_getattr_new(capsys):
    df = make_df()
    wdf = WatchedDataFrame(df, prov)

    wdf = wdf.head(3)

    captured = capsys.readouterr()
    expected_output = """-- df_in --
   ID   Birthdate Gender Postcode
0  10  1996-07-12      F    90210
1  20  1994-03-08      M      NaN
2  30         NaN      F    12345
3  40  1987-11-23      M    67890
-- df_out --
   ID   Birthdate Gender Postcode
0  10  1996-07-12      F    90210
1  20  1994-03-08      M      NaN
2  30         NaN      F    12345
"""
    assert captured.out == expected_output


def test_getattr_new_2(capsys):
    df = make_df()
    wdf = WatchedDataFrame(df, prov)

    wdf = wdf.dropna()

    captured = capsys.readouterr()
    expected_output = """-- df_in --
   ID   Birthdate Gender Postcode
0  10  1996-07-12      F    90210
1  20  1994-03-08      M      NaN
2  30         NaN      F    12345
3  40  1987-11-23      M    67890
-- df_out --
   ID   Birthdate Gender Postcode
0  10  1996-07-12      F    90210
3  40  1987-11-23      M    67890
"""
    assert captured.out == expected_output


def test_getattr_new_3(capsys):
    df = make_df()
    wdf = WatchedDataFrame(df, prov)

    wdf = wdf.sort_values("ID", ascending=False).reset_index(drop=True)

    captured = capsys.readouterr()
    expected_output = """-- df_in --
   ID   Birthdate Gender Postcode
0  10  1996-07-12      F    90210
1  20  1994-03-08      M      NaN
2  30         NaN      F    12345
3  40  1987-11-23      M    67890
-- df_out --
   ID   Birthdate Gender Postcode
3  40  1987-11-23      M    67890
2  30         NaN      F    12345
1  20  1994-03-08      M      NaN
0  10  1996-07-12      F    90210
-- df_in --
   ID   Birthdate Gender Postcode
3  40  1987-11-23      M    67890
2  30         NaN      F    12345
1  20  1994-03-08      M      NaN
0  10  1996-07-12      F    90210
-- df_out --
   ID   Birthdate Gender Postcode
0  40  1987-11-23      M    67890
1  30         NaN      F    12345
2  20  1994-03-08      M      NaN
3  10  1996-07-12      F    90210
"""
    assert captured.out == expected_output


def test_getattr_new_4(capsys):
    df = make_df()
    wdf = WatchedDataFrame(df, prov)
    df2 = make_df2()
    wdf2 = WatchedDataFrame(df2, prov)

    wdf3 = wdf.merge(wdf2, how='inner', on='ID')

    captured = capsys.readouterr()
    expected_output = """-- df_in --
   ID   Birthdate Gender Postcode
0  10  1996-07-12      F    90210
1  20  1994-03-08      M      NaN
2  30         NaN      F    12345
3  40  1987-11-23      M    67890
   ID   Name
0  10  Alice
1  30  Chloe
2  40    Bob
-- primary_key --
('ID', 'ID')
-- df_out --
   ID   Birthdate Gender Postcode   Name
0  10  1996-07-12      F    90210  Alice
1  30         NaN      F    12345  Chloe
2  40  1987-11-23      M    67890    Bob
"""
    assert captured.out == expected_output


def test_indexer_setitem(capsys):
    df = make_df()
    wdf = WatchedDataFrame(df, prov)

    wdf.iloc[0, 0] = 99

    captured = capsys.readouterr()
    expected_output = """-- df_in --
   ID   Birthdate Gender Postcode
0  10  1996-07-12      F    90210
1  20  1994-03-08      M      NaN
2  30         NaN      F    12345
3  40  1987-11-23      M    67890
-- df_out --
   ID   Birthdate Gender Postcode
0  99  1996-07-12      F    90210
1  20  1994-03-08      M      NaN
2  30         NaN      F    12345
3  40  1987-11-23      M    67890
"""
    assert captured.out == expected_output


def test_indexer_setitem_2(capsys):
    df = make_df()
    wdf = WatchedDataFrame(df, prov)

    wdf.loc[0, 'ID'] = 99

    captured = capsys.readouterr()
    expected_output = """-- df_in --
   ID   Birthdate Gender Postcode
0  10  1996-07-12      F    90210
1  20  1994-03-08      M      NaN
2  30         NaN      F    12345
3  40  1987-11-23      M    67890
-- df_out --
   ID   Birthdate Gender Postcode
0  99  1996-07-12      F    90210
1  20  1994-03-08      M      NaN
2  30         NaN      F    12345
3  40  1987-11-23      M    67890
"""
    assert captured.out == expected_output


def test_indexer_getitem(capsys):
    df = make_df()
    wdf = WatchedDataFrame(df, prov)

    wdf = wdf.iloc[[0,3], 0:3]

    captured = capsys.readouterr()
    expected_output = """-- df_in --
   ID   Birthdate Gender Postcode
0  10  1996-07-12      F    90210
1  20  1994-03-08      M      NaN
2  30         NaN      F    12345
3  40  1987-11-23      M    67890
-- df_out --
   ID   Birthdate Gender
0  10  1996-07-12      F
3  40  1987-11-23      M
"""
    assert captured.out == expected_output


def test_indexer_getitem_2(capsys):
    df = make_df()
    wdf = WatchedDataFrame(df, prov)

    wdf = wdf.iloc[[0,3], 0:3]

    captured = capsys.readouterr()
    expected_output = """-- df_in --
   ID   Birthdate Gender Postcode
0  10  1996-07-12      F    90210
1  20  1994-03-08      M      NaN
2  30         NaN      F    12345
3  40  1987-11-23      M    67890
-- df_out --
   ID   Birthdate Gender
0  10  1996-07-12      F
3  40  1987-11-23      M
"""
    assert captured.out == expected_output


def test_merge(capsys):
    df = make_df()
    wdf = WatchedDataFrame(df, prov)
    df2 = make_df2()
    wdf2 = WatchedDataFrame(df2, prov)

    wdf3 = wpd.merge(wdf, wdf2, on='ID', how='inner')

    captured = capsys.readouterr()
    expected_output = """-- df_in --
   ID   Birthdate Gender Postcode
0  10  1996-07-12      F    90210
1  20  1994-03-08      M      NaN
2  30         NaN      F    12345
3  40  1987-11-23      M    67890
   ID   Name
0  10  Alice
1  30  Chloe
2  40    Bob
-- primary_key --
('ID', 'ID')
-- df_out --
   ID   Birthdate Gender Postcode   Name
0  10  1996-07-12      F    90210  Alice
1  30         NaN      F    12345  Chloe
2  40  1987-11-23      M    67890    Bob
"""
    assert captured.out == expected_output

