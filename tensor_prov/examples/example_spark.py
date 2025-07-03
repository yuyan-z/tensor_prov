from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, year, month, dayofmonth, when

from tensor_prov.provenance_spark import Provenance

spark = SparkSession.builder.appName("Provenance").getOrCreate()



df = spark.createDataFrame([
    (10, "1996-07-12", "F", "90210"),
    (20, "1994-03-08", "M", None),
    (30, None, "F", "12345"),
    (40, "1987-11-23", "M", "67890")
], ["ID", "Birthdate", "Gender", "Postcode"])

name_df = spark.createDataFrame([
    (20, "Alex"),
    (40, "Bob")
], ["ID", "Name"])

prov = Provenance(spark)


def example_horizontal_reduction():
    df_filtered = df.dropna()
    # result = prov.capture(df, df_filtered, "ID")
    # result[0].show()
    # result[1].show()
    df_filtered.show(10, truncate=False)


def example_horizontal_augmentation():
    new_rows = spark.createDataFrame([
        (15, "2000-10-20", "F", "75014"),
        (35, "1999-01-06", "M", "29280")
    ], df.columns)
    df_aug = df.union(new_rows).orderBy("ID")
    result = prov.capture(df, df_aug, "ID")
    result[0].show()
    result[1].show()


def example_vertical_reduction():
    df_reduced = df.select("ID", "Gender", "Birthdate")
    result = prov.capture(df, df_reduced, "ID")
    result[0].show()
    result[1].show()


def example_vertical_augmentation():
    df_aug = df.withColumn("Year", year(to_date(col("Birthdate")))) \
        .withColumn("Month", month(to_date(col("Birthdate")))) \
        .withColumn("Day", dayofmonth(to_date(col("Birthdate")))) \
        .select("ID", "Gender", "Postcode", "Year", "Month", "Day")
    column_mapping = {
        "Birthdate": ["Year", "Month", "Day"]
    }
    result = prov.capture(df, df_aug, "ID", column_mapping)
    result[0].show()
    result[1].show()


def example_data_transformation():
    df_trans = df.withColumn("Gender", when(col("Gender") == "F", 0).otherwise(1))
    result = prov.capture(df, df_trans, "ID")
    result[0].show()
    result[1].show()


def example_data_fusion():
    df_joined = df.join(name_df, on="ID", how="outer")
    result = prov.capture((df, name_df), df_joined, "ID")
    result[0].show()
    result[1].show()


def example_trace(direction="backward"):
    # 1. Drop NaNs
    df1 = df.withColumn("Birthdate", to_date("Birthdate")).dropna(subset=["Birthdate"])
    result1 = prov.capture(df, df1, "ID")

    # 2. Split Birthdate
    df2 = df1.withColumn("Year", year("Birthdate")) \
        .withColumn("Month", month("Birthdate")) \
        .withColumn("Day", dayofmonth("Birthdate")) \
        .drop("Birthdate") \
        .select("ID", "Gender", "Postcode", "Year", "Month", "Day")
    result2 = prov.capture(df1, df2, "ID", {"Birthdate": ["Year", "Month", "Day"]})

    # 3. Sort by Year
    df3 = df2.orderBy("Year")
    result3 = prov.capture(df2, df3, "ID")

    # Trace across steps
    tensors = [r[0] for r in [result1, result2, result3]]

    print("\n-- Trace (no path) --")
    prov.trace(tensors, direction=direction, keep_path=False).show()

    print("\n-- Trace (path) --")
    prov.trace(tensors, direction=direction, keep_path=True).show()

    print("\n-- Trace (sliced) --")
    prov.trace(tensors, direction=direction, indices=[1], keep_path=True).show()


if __name__ == "__main__":
    df.show()
    # example_horizontal_reduction()
    # example_horizontal_augmentation(spark, prov)
    # example_vertical_reduction(spark, prov)
    # example_vertical_augmentation(spark, prov)
    # example_data_transformation(spark, prov)
    # example_data_fusion(spark, prov)
    # example_trace(direction="forward")


from pyspark.sql import SparkSession, DataFrame

spark = SparkSession.builder.appName("Provenance").getOrCreate()

df = spark.createDataFrame([
    (10, "1996-07-12", "F", "90210"),
    (20, "1994-03-08", "M", None),
    (30, None, "F", "12345"),
    (40, "1987-11-23", "M", "67890")
], ["ID", "Birthdate", "Gender", "Postcode"])

df.show()