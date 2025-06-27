from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit, expr
from typing import Union, Tuple, Dict
import time


class Provenance:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.step_count = 0

    def capture_time(func):
        def wrapper(self, *args, **kwargs):
            start = time.time()
            result = func(self, *args, **kwargs)
            end = time.time()
            print(f"Runtime: {end - start:.4f} seconds")
            return result, end - start
        return wrapper

    def create_sparse_tensor(self, indices_out: DataFrame, indices_in: DataFrame) -> DataFrame:
        return indices_out.join(indices_in, on=["idx_out", "idx_in"], how="inner").select("idx_out", "idx_in")

    @capture_time
    def capture(
        self,
        df_in: Union[DataFrame, Tuple[DataFrame, DataFrame]],
        df_out: DataFrame,
        key_column: str,
        column_mapping: Dict[str, list] = None,
    ) -> Tuple[DataFrame, DataFrame]:
        if isinstance(df_in, DataFrame):
            tensor_record = self.capture_row_operation(df_in, df_out, key_column)
            tensor_attr = self.capture_column_operation(df_in, df_out, column_mapping)
        elif isinstance(df_in, tuple):
            df1, df2 = df_in
            tensor_record1 = self.capture_row_operation(df1, df_out, key_column)
            tensor_record2 = self.capture_row_operation(df2, df_out, key_column)
            tensor_attr1 = self.capture_column_operation(df1, df_out, column_mapping)
            tensor_attr2 = self.capture_column_operation(df2, df_out, column_mapping)
            tensor_record = tensor_record1.unionByName(tensor_record2).dropDuplicates()
            tensor_attr = tensor_attr1.unionByName(tensor_attr2).dropDuplicates()
        else:
            raise ValueError("df_in must be a DataFrame or a tuple of DataFrames")

        return tensor_record, tensor_attr

    def capture_row_operation(self, df_in: DataFrame, df_out: DataFrame, key_column: str) -> DataFrame:
        df_in_indexed = df_in.withColumnRenamed(key_column, "key_in").withColumn("idx_in", expr("monotonically_increasing_id()"))
        df_out_indexed = df_out.withColumnRenamed(key_column, "key_out").withColumn("idx_out", expr("monotonically_increasing_id()"))
        joined = df_out_indexed.join(df_in_indexed, df_out_indexed["key_out"] == df_in_indexed["key_in"], "inner")
        return joined.select("idx_out", "idx_in")

    def capture_column_operation(self, df_in: DataFrame, df_out: DataFrame, column_mapping: Dict[str, list] = None) -> DataFrame:
        cols_in = df_in.columns
        cols_out = df_out.columns

        if column_mapping is None:
            matched_pairs = [(i, j) for i, out_col in enumerate(cols_out)
                             for j, in_col in enumerate(cols_in) if out_col == in_col]
        else:
            reverse_map = {out: inp for inp, outs in column_mapping.items() for out in outs}
            matched_pairs = [
                (i, cols_in.index(reverse_map.get(out_col))) for i, out_col in enumerate(cols_out)
                if reverse_map.get(out_col) in cols_in
            ]

        return self.spark.createDataFrame(matched_pairs, ["idx_out", "idx_in"])

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Provenance").getOrCreate()
    # print(spark.sparkContext.defaultParallelism)  # 16
    prov = Provenance(spark)