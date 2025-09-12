import os

import numpy as np
import pandas as pd

from .provenance_base import ProvenanceBase


class Provenance(ProvenanceBase):
    def create_sparse_tensor(
            self,
            indices_out: np.ndarray,
            indices_in: np.ndarray,
            shape: tuple[int, int]
    ) -> pd.DataFrame:
        sparse_tensor = pd.DataFrame({
            "out": indices_out,
            "in": indices_in,
        })
        return sparse_tensor

    def print_prov_result(
            self,
            df_in: pd.DataFrame,
            df_out: pd.DataFrame,
            id_in: int,
            id_out: int,
            result: tuple[pd.DataFrame, pd.DataFrame]
    ):
        print("-- id_in -> id_out--\n", id_in, '->', id_out)
        print(f"-- df_in: {id_in} --\n", df_in)
        print(f"-- df_out: {id_out} --\n", df_out)
        (tensor_record, tensor_attr) = result
        print("-- tensor_record --\n", tensor_record.to_numpy())
        print("-- tensor_attr --\n", tensor_attr.to_numpy())

    def save_prov_result(
            self,
            id_in: int,
            id_out: int,
            result: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        tensor_record, tensor_attr = result
        rec_path = os.path.join(self.save_dir, f"{id_in}_{id_out}_record.npz")
        attr_path = os.path.join(self.save_dir, f"{id_in}_{id_out}_attr.npz")
        np.savez(rec_path, **{col: tensor_record[col].to_numpy()
                              for col in tensor_record.columns})
        np.savez(attr_path, **{col: tensor_attr[col].to_numpy()
                               for col in tensor_attr.columns})

    def load_prov_result(
            self,
            id_in: int,
            id_out: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        rec_path = os.path.join(self.save_dir, f"{id_in}_{id_out}_record.npz")
        attr_path = os.path.join(self.save_dir, f"{id_in}_{id_out}_attr.npz")
        with np.load(rec_path, allow_pickle=True) as data:
            tensor_record = pd.DataFrame({k: data[k] for k in data.files})
        with np.load(attr_path, allow_pickle=True) as data:
            tensor_attr = pd.DataFrame({k: data[k] for k in data.files})
        return tensor_record, tensor_attr


def trace(
        tensors: list[pd.DataFrame],
        direction: str = "backward",
        indices: list[int] | None = None,
        keep_path: bool = False
) -> np.ndarray:
    # pandas join
    if direction == "forward":
        df_path = tensors[0].rename(columns={"out": "t_0_out"})
        df_path = df_path[["in", "t_0_out"]]
        tensors_ordered = tensors[1:]
        i_right_on = 1
    elif direction == "backward":
        df_path = tensors[-1].rename(columns={"in": "t_0_in"})
        tensors_ordered = tensors[-2::-1]
        i_right_on = 0
    else:
        raise ValueError("direction must be 'forward' or 'backward'")

    if indices is not None:
        df_path = df_path[df_path.iloc[:, 0].isin(indices)]
    for i, t in enumerate(tensors_ordered):
        df_t = t.add_prefix(f't_{i + 1}_')
        df_path = df_path.merge(df_t, left_on=df_path.columns[-1], right_on=df_t.columns[i_right_on], how="inner")
        df_path = df_path.drop(columns=[df_t.columns[i_right_on]])
        if not keep_path:
            df_path = df_path.iloc[:, [0, -1]]
    df_path = df_path.sort_values(by=df_path.columns[0])
    result = df_path.to_numpy(dtype=int)

    return result
