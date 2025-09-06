from __future__ import annotations

from collections import defaultdict

import pandas as pd

from utils import get_merge_column_mapping


class WatchedDataFrame:
    is_tracking = True

    def __init__(self, df, prov, i=None):
        self.df = df.copy()
        self.prov = prov
        self.id = prov.graph.new_id() if i is None else i

    def set(self, new: pd.DataFrame | WatchedDataFrame, **kwargs):
        if isinstance(new, pd.DataFrame):
            new_id = self.prov.graph.new_id()
            if WatchedDataFrame.is_tracking:
                self.prov.capture(self.df, new, id_in=self.id, id_out=new_id, **kwargs)
            self.df = new
            self.id = new_id
        elif isinstance(new, WatchedDataFrame):
            if WatchedDataFrame.is_tracking:
                self.prov.capture(self, new, **kwargs)
            self.df = new.df
            self.id = new.id

    def __setitem__(self, key, value):
        # print("__setitem__")
        if WatchedDataFrame.is_tracking:
            wdf_old = WatchedDataFrame(self.df, self.prov, self.id)
            self.df[key] = value
            self.id = self.prov.graph.new_id()
            self.prov.capture(wdf_old, self)
        else:
            self.df[key] = value
            self.id = self.prov.graph.new_id()

    def __getitem__(self, key):
        # print("__getitem__")
        result = self.df[key]
        if isinstance(result, pd.DataFrame):
            result = WatchedDataFrame(result, self.prov)
            if WatchedDataFrame.is_tracking:
                self.prov.capture(self, result)
        return result

    def __getattr__(self, attr):
        # print("__getattr__", attr)
        orig_attr = getattr(self.df, attr)
        if callable(orig_attr) and not attr.startswith("_"):

            def hooked(*args, **kwargs):
                # inplace methods
                if kwargs.get('inplace', False):
                    if WatchedDataFrame.is_tracking:
                        wdf_old = WatchedDataFrame(self.df, self.prov, self.id)
                        result = orig_attr(*args, **kwargs)
                        self.id = self.prov.graph.new_id()
                        self.prov.capture(wdf_old, self)
                    else:
                        result = orig_attr(*args, **kwargs)
                        self.id = self.prov.graph.new_id()
                # return new object methods
                else:
                    if attr == "merge":
                        left_df = self.df.copy()
                        left_df["primary_key_x"] = range(len(left_df))
                        if kwargs.get("right", None):
                            right_wdf = kwargs.pop("right")
                        else:
                            right_wdf, *args = args
                        right_df = right_wdf.df.copy()
                        right_df["primary_key_y"] = range(len(right_df))
                        result = left_df.merge(right_df, *args, **kwargs)
                        column_mapping = get_merge_column_mapping(self.df.columns, right_wdf.df.columns,
                                                                  result.columns)
                        new_id = self.prov.graph.new_id()
                        self.prov.capture(
                            [left_df, right_df],
                            result,
                            id_in=[self.id, right_wdf.id],
                            id_out=new_id,
                            primary_key=["primary_key_x", "primary_key_y"],
                            column_mapping=column_mapping,
                            column_ignore=["primary_key_x", "primary_key_y"]
                        )
                        result = result.drop(columns=["primary_key_x", "primary_key_y"])
                        result = WatchedDataFrame(result, self.prov, new_id)
                    else:
                        result = orig_attr(*args, **kwargs)
                        if isinstance(result, pd.DataFrame):
                            result = WatchedDataFrame(result, self.prov)
                            if WatchedDataFrame.is_tracking:
                                self.prov.capture(self, result)
                return result

            return hooked
        else:
            return orig_attr

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
            if WatchedDataFrame.is_tracking:
                prov.capture(self._wdf, result)
        return result

    def __setitem__(self, key, value):
        # print("_WatchedIndexer __setitem__")
        if WatchedDataFrame.is_tracking:
            wdf_old = WatchedDataFrame(self._wdf.df, self._wdf.prov, self._wdf.id)
            self._indexer[key] = value
            self._wdf.id = self._wdf.prov.graph.new_id()
            self._wdf.prov.capture(wdf_old, self._wdf)
        else:
            self._indexer[key] = value
            self._wdf.id = self._wdf.prov.graph.new_id()

# def merge(left_wdf: WatchedDataFrame, right_wdf: WatchedDataFrame, *args, **kwargs):
#     result = pd.merge(left_wdf.df, right_wdf.df, *args, **kwargs)
#     prov = left_wdf.prov
#     result = WatchedDataFrame(result, prov)
#     if WatchedDataFrame.is_tracking:
#         merge_key, column_mapping = _get_merge_key(kwargs, result.columns)
#         prov.capture([left_wdf, right_wdf], result, primary_key=merge_key, column_mapping=column_mapping)
#     return result
