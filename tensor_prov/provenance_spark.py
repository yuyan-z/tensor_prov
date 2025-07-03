from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, monotonically_increasing_id
import time


class Provenance:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.step_count = 0

    @staticmethod
    def capture_time(func):
        def wrapper(self, *args, **kwargs):
            start = time.time()
            result = func(self, *args, **kwargs)
            end = time.time()
            print(f"[Step {self.step_count}] Runtime: {end - start:.4f} seconds")
            return result

        return wrapper

    @capture_time
    def capture(self, df_in: DataFrame, df_out: DataFrame, key_column: str, column_mapping: dict = None):
        self.step_count += 1

        if isinstance(df_in, DataFrame):
            record = self.capture_row_operation(df_in, df_out, key_column)
            attr = self.capture_column_operation(df_in, df_out, column_mapping)
        elif isinstance(df_in, tuple):
            df_in1, df_in2 = df_in
            record1 = self.capture_row_operation(df_in1, df_out, key_column)
            record2 = self.capture_row_operation(df_in2, df_out, key_column)
            attr1 = self.capture_column_operation(df_in1, df_out, column_mapping)
            attr2 = self.capture_column_operation(df_in2, df_out, column_mapping)

            record = record1.union(record2).dropDuplicates()
            attr = attr1.union(attr2).dropDuplicates()
        else:
            raise ValueError("df_in must be a DataFrame or a tuple of DataFrames")

        return record, attr

    def capture_row_operation(self, df_in, df_out, key_column: str):
        df_in_with_id = df_in.withColumn("idx_in", monotonically_increasing_id())
        df_out_with_id = df_out.withColumn("idx_out", monotonically_increasing_id())

        df_joined = df_in_with_id.join(df_out_with_id, on=key_column, how="inner")
        sparse_tensor = df_joined.select("idx_out", "idx_in")

        df_joined.show()
        sparse_tensor.show()

        return sparse_tensor

    def capture_column_operation(self, df_in, df_out, column_mapping: dict = None):
        cols_in = df_in.columns
        cols_out = df_out.columns

        if column_mapping is None:
            common_cols = set(cols_in) & set(cols_out)
            indices = [(cols_out.index(c), cols_in.index(c)) for c in common_cols]
        else:
            indices = []
            for out_col, in_cols in column_mapping.items():
                for in_col in in_cols:
                    if out_col in cols_out and in_col in cols_in:
                        indices.append((cols_out.index(out_col), cols_in.index(in_col)))

        sparse_tensor = self.spark.createDataFrame(indices, ["idx_out", "idx_in"])
        return sparse_tensor

    def trace(self, tensors: list, direction="backward", indices=None, keep_path=False):
        if direction == "forward":
            df_path = tensors[0].selectExpr("idx_in as src", "idx_out as dst")
            if indices:
                df_path = df_path.filter(col("src").isin(indices))

            for i, t in enumerate(tensors[1:], 1):
                t = t.selectExpr("idx_in as src", "idx_out as dst")
                df_path = df_path.join(t, df_path["dst"] == t["src"], "left").drop(t["src"])
                if not keep_path:
                    df_path = df_path.select("src", "dst")
        elif direction == "backward":
            df_path = tensors[-1].selectExpr("idx_out as src", "idx_in as dst")
            if indices:
                df_path = df_path.filter(col("src").isin(indices))
            for t in reversed(tensors[:-1]):
                t = t.selectExpr("idx_out as src", "idx_in as dst")
                df_path = df_path.join(t, df_path["dst"] == t["src"], "left").drop(t["src"])
                if not keep_path:
                    df_path = df_path.select("src", "dst")
        else:
            raise ValueError("Invalid direction")

        return df_path.fillna(-1)
