import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import wraps
from typing import Any

import numpy as np
import pandas as pd

from prov_graph import ProvGraph
from watched_pandas import WatchedDataFrame


def capture_time(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start = time.time()
        result = func(self, *args, **kwargs)
        runtime = time.time() - start
        self.last_runtime = runtime
        # if self.verbose > 0:
        #     print(f"\n[{func.__name__}] Runtime: {runtime:.4f} s")
        return result

    return wrapper


class ProvenanceBase(ABC):
    def __init__(self, save_dir: str, verbose: int = 0):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.graph = ProvGraph()
        self.verbose = verbose

    def subscribe(self, df) -> WatchedDataFrame:
        wdf = WatchedDataFrame(df, self)
        return wdf

    def pause(self):
        WatchedDataFrame.is_tracking = False

    def start(self):
        WatchedDataFrame.is_tracking = True

    @capture_time
    def capture(
            self,
            d_in: pd.DataFrame | list[pd.DataFrame] | WatchedDataFrame | list[WatchedDataFrame],
            d_out: pd.DataFrame | WatchedDataFrame,
            **kwargs: Any
    ) -> tuple[Any, Any] | list[tuple[Any, Any]]:
        id_in = kwargs.get("id_in", None)
        id_out = kwargs.get("id_out", None)
        primary_key = kwargs.get("primary_key", None)
        column_mapping = kwargs.get("column_mapping", None)

        if isinstance(d_in, pd.DataFrame):
            if not (isinstance(id_in, int) and isinstance(id_out, int)):
                raise TypeError("id_in, id_out error")
            result = self.capture_df(d_in, d_out, id_in, id_out, primary_key, column_mapping)
        elif isinstance(d_in, WatchedDataFrame):
            result = self.capture_df(d_in.df, d_out.df, d_in.id, d_out.id, primary_key, column_mapping)
        elif isinstance(d_in, list):
            if not isinstance(primary_key, list):
                primary_key = [primary_key] * len(d_in)
            if not isinstance(column_mapping, list):
                column_mapping = [column_mapping] * len(d_in)
            column_ignore = kwargs.get("column_ignore", [])

            result = []
            if isinstance(d_out, WatchedDataFrame):
                for i, d in enumerate(d_in):
                    r = self.capture_df(d.df, d_out.df, d.id, d_out.id, primary_key[i], column_mapping[i],
                                        column_ignore)
                    result.append(r)
            elif isinstance(d_out, pd.DataFrame) and isinstance(id_in, list) and isinstance(id_out, int):
                for i, d in enumerate(d_in):
                    r = self.capture_df(d, d_out, id_in[i], id_out, primary_key[i], column_mapping[i], column_ignore)
                    result.append(r)
            else:
                raise TypeError("d_in or d_out error")
        else:
            raise TypeError("d_in or d_out error")

        return result

    def capture_df(
            self,
            df_in: pd.DataFrame,
            df_out: pd.DataFrame,
            id_in: int,
            id_out: int,
            primary_key: str | None = None,
            column_mapping: dict | None = None,
            column_ignore: list[str] | None = None
    ) -> tuple[Any, Any]:
        tensor_record = self.capture_row_operation(df_in, df_out, primary_key)
        tensor_attr = self.capture_column_operation(df_in, df_out, column_mapping, column_ignore)
        result = (tensor_record, tensor_attr)
        self.graph.add_edge(id_in, id_out, (tensor_record, tensor_attr))

        if self.verbose > 0:
            if column_ignore:
                self.print_prov_result(
                    df_in.drop(columns=column_ignore, errors="ignore"),
                    df_out.drop(columns=column_ignore, errors="ignore"),
                    id_in,
                    id_out,
                    result)
            else:
                self.print_prov_result(df_in, df_out, id_in, id_out, result)

        self.save_prov_result(id_in, id_out, result)
        self.graph.save_graph(self.save_dir)

        return result

    def capture_row_operation(
            self,
            df_in: pd.DataFrame,
            df_out: pd.DataFrame,
            primary_key: str | None | list[str]
    ) -> Any:
        n_out, n_in = len(df_out), len(df_in)  # provenance shape (n_out, n_in)

        if primary_key is None:
            ids_out = df_out.index.to_numpy()
            ids_in = df_in.index.to_numpy()
        elif isinstance(primary_key, str):
            ids_out = df_out[primary_key].to_numpy()
            ids_in = df_in[primary_key].to_numpy()
        elif isinstance(primary_key, list):
            ids_out = df_out[primary_key[1]].to_numpy()
            ids_in = df_in[primary_key[0]].to_numpy()
        else:
            raise TypeError("primary_key error")

        indices_out, indices_in = find_indices(ids_out, ids_in)

        # Create sparse tensor
        sparse_tensor = self.create_sparse_tensor(indices_out, indices_in, (n_out, n_in))

        return sparse_tensor

    def capture_column_operation(
            self,
            df_in: pd.DataFrame,
            df_out: pd.DataFrame,
            column_mapping: dict | None,
            column_ignore: list[str] | None
    ) -> Any:
        cols_in = df_in.columns.drop(column_ignore, errors="ignore").values if column_ignore else df_in.columns.values
        cols_out = df_out.columns.drop(column_ignore,
                                       errors="ignore").values if column_ignore else df_out.columns.values
        n_out, n_in = len(cols_out), len(cols_in)

        if column_mapping is not None:
            out_to_in = {col_out: col_in for col_in, cols_out in column_mapping.items() for col_out in cols_out}
            cols_out = [out_to_in.get(col_out, col_out) for col_out in cols_out]

        indices_out, indices_in = find_indices(cols_out, cols_in)

        # Create sparse tensor
        sparse_tensor = self.create_sparse_tensor(indices_out, indices_in, (n_out, n_in))

        return sparse_tensor

    def load(self):
        self.graph.load_graph(self.save_dir, self)

    @abstractmethod
    def create_sparse_tensor(
            self,
            indices_out: np.ndarray,
            indices_in: np.ndarray,
            shape: tuple[int, int]
    ) -> Any:
        """
        Create a sparse tensor from indices.
        """
        raise NotImplementedError

    @abstractmethod
    def print_prov_result(
            self,
            df_in: pd.DataFrame,
            df_out: pd.DataFrame,
            id_in: int,
            id_out: int,
            result: tuple[Any, Any]
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_prov_result(
            self,
            id_in: int,
            id_out: int,
            result: tuple[Any, Any]
    ) -> None:
        """
          <save_dir>/<id_in>_<id_out>_record.npz
          <save_dir>/<id_in>_<id_out>_attr.npz
        """
        raise NotImplementedError

    @abstractmethod
    def load_prov_result(
            self,
            id_in: int,
            id_out: int
    ) -> Any:
        raise NotImplementedError


def find_indices(ids_out: np.ndarray, ids_in: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # # I. np.where
    # indices_out, indices_in = np.where(ids_out[:, None] == ids_in[None, :])

    # # II. for loop
    # indices_out, indices_in = [], []
    # for out_idx, val_out in enumerate(ids_out):
    #     for in_idx, val_in in enumerate(ids_in):
    #         if val_out == val_in:
    #             indices_out.append(out_idx)
    #             indices_in.append(in_idx)
    #
    # indices_out = np.array(indices_out, dtype=np.int64)
    # indices_in = np.array(indices_in, dtype=np.int64)

    # III. dict and for loop
    id_to_indices_in = defaultdict(list)
    for idx_in, val in enumerate(ids_in):
        id_to_indices_in[val].append(idx_in)

    indices_out, indices_in = [], []
    for out_idx, val in enumerate(ids_out):
        if val in id_to_indices_in:
            in_indices = id_to_indices_in[val]
            indices_out.extend([out_idx] * len(in_indices))
            indices_in.extend(in_indices)

    indices_out = np.array(indices_out, dtype=np.int64)
    indices_in = np.array(indices_in, dtype=np.int64)

    return indices_out, indices_in
