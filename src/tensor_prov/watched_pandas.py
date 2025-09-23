from __future__ import annotations

import pandas as pd

from .utils import get_merge_column_mapping


class WatchedDataFrame:
    def __init__(self, df, prov, i=None):
        self.df = df.copy()
        self.prov = prov
        self.id = prov.graph.generate_id() if i is None else i

    def set(self, new: pd.DataFrame | WatchedDataFrame, **kwargs):
        if isinstance(new, pd.DataFrame):
            new_id = self.prov.graph.generate_id()
            if self.prov.is_tracking:
                self.prov.capture(self.df, new, id_in=self.id, id_out=new_id, **kwargs)
            self.df = new
            self.id = new_id
        elif isinstance(new, WatchedDataFrame):
            if self.prov.is_tracking:
                self.prov.capture(self, new, **kwargs)
            self.df = new.df
            self.id = new.id

    def __setitem__(self, key, value):
        # print("__setitem__")
        if self.prov.is_tracking:
            wdf_old = WatchedDataFrame(self.df, self.prov, self.id)
            self.df[key] = value
            self.id = self.prov.graph.generate_id()
            self.prov.capture(wdf_old, self)
        else:
            self.df[key] = value
            self.id = self.prov.graph.generate_id()

    def __getitem__(self, key):
        # print("__getitem__")
        result = self.df[key]
        if isinstance(result, pd.DataFrame):
            result = WatchedDataFrame(result, self.prov)
            if self.prov.is_tracking:
                self.prov.capture(self, result)
        return result

    def __getattr__(self, attr):
        # print("__getattr__", attr)
        orig_attr = getattr(self.df, attr)
        if callable(orig_attr) and not attr.startswith("_"):

            def hook(*args, **kwargs):
                # inplace methods
                if kwargs.get('inplace', False):
                    if self.prov.is_tracking:
                        wdf_old = WatchedDataFrame(self.df, self.prov, self.id)
                        result = orig_attr(*args, **kwargs)
                        self.id = self.prov.graph.generate_id()
                        self.prov.capture(wdf_old, self)
                    else:
                        result = orig_attr(*args, **kwargs)
                        self.id = self.prov.graph.generate_id()
                # return new object methods
                else:
                    result = orig_attr(*args, **kwargs)
                    if isinstance(result, pd.DataFrame):
                        result = WatchedDataFrame(result, self.prov)
                        if self.prov.is_tracking:
                            self.prov.capture(self, result)
                return result

            return hook
        else:
            return orig_attr

    def merge(self, *args, **kwargs):
        if kwargs.get("right", None):
            right_wdf = kwargs.pop("right")
        else:
            right_wdf, *args = args
        return _hook_merge(self, right_wdf, *args, **kwargs)

    @property
    def loc(self):
        return _WatchedIndexer(self, self.df.loc)

    @property
    def iloc(self):
        return _WatchedIndexer(self, self.df.iloc)

    def __len__(self):
        return len(self.df)


class _WatchedIndexer:
    def __init__(self, wdf, indexer):
        self._wdf = wdf
        self._indexer = indexer

    def __getitem__(self, key):
        # print("_WatchedIndexer __getitem__")
        result = self._indexer[key]
        if isinstance(result, pd.DataFrame):
            prov = self._wdf.prov
            result = WatchedDataFrame(result, prov)
            if self._wdf.prov.is_tracking:
                prov.capture(self._wdf, result)
        return result

    def __setitem__(self, key, value):
        # print("_WatchedIndexer __setitem__")
        if self._wdf.prov.is_tracking:
            wdf_old = WatchedDataFrame(self._wdf.df, self._wdf.prov, self._wdf.id)
            self._indexer[key] = value
            self._wdf.id = self._wdf.prov.graph.generate_id()
            self._wdf.prov.capture(wdf_old, self._wdf)
        else:
            self._indexer[key] = value
            self._wdf.id = self._wdf.prov.graph.generate_id()


def _hook_merge(left_wdf: WatchedDataFrame, right_wdf: WatchedDataFrame, *args, **kwargs):
    left_df = left_wdf.df.copy()
    right_df = right_wdf.df.copy()
    left_df["primary_key_x"] = range(len(left_df))
    right_df["primary_key_y"] = range(len(right_df))
    result = left_df.merge(right_df, *args, **kwargs)
    column_mapping = get_merge_column_mapping(
        left_wdf.columns,
        right_wdf.columns,
        result.columns
    )
    new_id = left_wdf.prov.graph.generate_id()
    left_wdf.prov.capture(
        [left_df, right_df],
        result,
        id_in=[left_wdf.id, right_wdf.id],
        id_out=new_id,
        primary_key=["primary_key_x", "primary_key_y"],
        column_mapping=column_mapping,
        column_ignore=["primary_key_x", "primary_key_y"]
    )
    result = result.drop(columns=["primary_key_x", "primary_key_y"])
    result = WatchedDataFrame(result, left_wdf.prov, new_id)
    return result


def merge(*args, **kwargs):
    if "left" in kwargs and "right" in kwargs:
        left_wdf = kwargs.pop("left")
        right_wdf = kwargs.pop("right")
    else:
        left_wdf, right_wdf, *args = args
    return _hook_merge(left_wdf, right_wdf, *args, **kwargs)


def get_dummies(*args, **kwargs):
    if "data" in kwargs:
        wdf = kwargs.pop("data")
    else:
        wdf, *args = args

    result = pd.get_dummies(wdf.df, *args, **kwargs)

    columns = kwargs.get("columns", None)
    if columns is None:
        columns = wdf.df.select_dtypes(include=["object", "category"]).columns.tolist()

    prefix = kwargs.get("prefix", None)
    if prefix is None:
        prefix_dict = {col: col for col in columns}
    elif isinstance(prefix, str):
        prefix_dict = {col: prefix for col in columns}
    elif isinstance(prefix, (list, tuple)):
        prefix_dict = dict(zip(columns, prefix))
    elif isinstance(prefix, dict):
        prefix_dict = {col: prefix.get(col, col) for col in columns}
    else:
        raise TypeError("prefix type")

    dummy_na = bool(kwargs.get("dummy_na", False))
    prefix_sep = kwargs.get("prefix_sep", "_")

    mapping = {}
    result_cols = set(result.columns.tolist())
    for col in columns:
        values = wdf.df[col]
        unique_vals = values.dropna().astype(str).unique().tolist()

        if dummy_na and values.isna().any():
            unique_vals.append("nan")

        encoded_cols = [f"{prefix_dict[col]}{prefix_sep}{v}" for v in unique_vals]
        mapping[col] = [c for c in encoded_cols if c in result_cols]

    result = WatchedDataFrame(result, wdf.prov)
    if wdf.prov.is_tracking:
        wdf.prov.capture(wdf, result, column_mapping=mapping)

    return result
