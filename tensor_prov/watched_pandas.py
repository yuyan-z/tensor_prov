from __future__ import annotations

import pandas as pd


class WatchedDataFrame:
    is_watching = True
    is_tracking = True

    def __init__(self, df, prov):
        self._df = df
        self._prov = prov
        self.id = prov.graph.new_id()

    def get_df(self) -> pd.DataFrame:
        return self._df.copy()

    def set_df(self, df_new: pd.DataFrame):
        if WatchedDataFrame.is_tracking:
            df_old = self._df.copy()
            id_old = self.id
            self._df = df_new
            self.id = self._prov.graph.new_id()
            self._prov.capture(df_old, self._df, id_old, self.id)
        else:
            self._df = df_new
            self.id = self._prov.graph.new_id()

    def __setitem__(self, key, value):
        # print("__setitem__")
        if WatchedDataFrame.is_tracking:
            df_old = self._df.copy()
            id_old = self.id
            self._df[key] = value
            self.id = self._prov.graph.new_id()
            self._prov.capture(df_old, self._df, id_old, self.id)
        else:
            self._df[key] = value
            self.id = self._prov.graph.new_id()

    def __getitem__(self, key):
        # print("__getitem__")
        result = self._df[key]
        if isinstance(result, pd.DataFrame) and WatchedDataFrame.is_watching:
            result = WatchedDataFrame(result, self._prov)
            if WatchedDataFrame.is_tracking:
                self._prov.capture(self._df, result._df, self.id, result.id)
        return result

    def __getattr__(self, attr):
        # print("__getattr__", attr)
        orig_attr = getattr(self._df, attr)
        if callable(orig_attr) and not attr.startswith("_"):

            def hooked(*args, **kwargs):
                # inplace methods
                if kwargs.get('inplace', False):
                    if WatchedDataFrame.is_tracking:
                        df_old = self._df.copy()
                        id_old = self.id
                        result = orig_attr(*args, **kwargs)
                        self.id = self._prov.graph.new_id()
                        self._prov.capture(df_old, self._df, id_old, self.id)
                    else:
                        result = orig_attr(*args, **kwargs)
                        self.id = self._prov.graph.new_id()
                # return new object methods
                else:
                    if attr == "merge":
                        WatchedDataFrame.is_watching = False
                    result = orig_attr(*args, **kwargs)
                    if attr == "merge":
                        WatchedDataFrame.is_watching = True
                        result = WatchedDataFrame(result, self._prov)
                        if WatchedDataFrame.is_tracking:
                            right_wdf = args[0]
                            merge_key = get_merge_key(kwargs)
                            self._prov.capture((self._df, right_wdf._df), result._df, (self.id, right_wdf.id), result.id,
                                               primary_key=merge_key)
                    elif isinstance(result, pd.DataFrame) and WatchedDataFrame.is_watching:
                        result = WatchedDataFrame(result, self._prov)
                        if WatchedDataFrame.is_tracking:
                            self._prov.capture(self._df, result._df, self.id, result.id)
                return result

            return hooked
        else:
            return orig_attr

    @property
    def loc(self):
        return _WatchedIndexer(self, self._df.loc)

    @property
    def iloc(self):
        return _WatchedIndexer(self, self._df.iloc)

    def __len__(self):
        return len(self._df)


class _WatchedIndexer:
    def __init__(self, wdf, indexer):
        self._wdf = wdf
        self._indexer = indexer

    def __getitem__(self, key):
        # print("_WatchedIndexer __getitem__")
        result = self._indexer[key]
        if isinstance(result, pd.DataFrame) and WatchedDataFrame.is_watching:
            prov = self._wdf._prov
            result = WatchedDataFrame(result, prov)
            if WatchedDataFrame.is_tracking:
                prov.capture(self._wdf._df, result._df, self._wdf.id, result.id)
        return result

    def __setitem__(self, key, value):
        # print("_WatchedIndexer __setitem__")
        if WatchedDataFrame.is_tracking:
            df_old = self._wdf._df.copy()
            id_old = self._wdf.id
            self._indexer[key] = value
            self._wdf.id = self._wdf._prov.graph.new_id()
            self._wdf._prov.capture(df_old, self._wdf._df, id_old, self._wdf.id)
        else:
            self._indexer[key] = value
            self._wdf.id = self._wdf._prov.graph.new_id()


def get_merge_key(kwargs):
    if "on" in kwargs and kwargs["on"] is not None:
        merge_key = kwargs["on"]
    else:
        merge_key = (kwargs["left_on"], kwargs["right_on"])
    return merge_key


def merge(left_wdf: WatchedDataFrame, right_wdf: WatchedDataFrame, *args, **kwargs):
    left_df = left_wdf._df
    right_df = right_wdf._df
    result = pd.merge(left_df, right_df, *args, **kwargs)
    prov = left_wdf._prov
    result = WatchedDataFrame(result, prov)
    merge_key = get_merge_key(kwargs)
    prov.capture((left_df, right_df), result._df, (left_wdf.id, right_wdf.id), result.id, primary_key=merge_key)
    return result
