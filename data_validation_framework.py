import json
import os
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr, col, abs as abs_, lit, coalesce
from pyspark.sql.types import DoubleType

# -----------------------------
# Utility: Load Config
# -----------------------------
def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

# -----------------------------
# Utility: Write JSON/CSV Reports
# -----------------------------
def save_report(spark, df, output_path, fmt="csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.coalesce(1).write.mode("overwrite").format(fmt).option("header", True).save(output_path)
    print(f"📁 Saved mismatch report to: {output_path}")

# -----------------------------
# Main Validation Framework
# -----------------------------
def main():
    start_time = time.time()
    spark = SparkSession.builder.appName("Data Validation Framework V2").getOrCreate()

    # Load Configuration
    CONFIG_PATH = "validation_config.json"
    config = load_config(CONFIG_PATH)

    source_path = config["source_path"]
    target_path = config["target_path"]
    primary_keys = config["primary_keys"]
    transformations = config["transformations"]
    tolerance = config.get("tolerance", 0.0)
    output_dir = config.get("output_dir", "validation_output")

    print("\n🚀 Starting Enhanced Data Validation Framework")

    # Load Data
    source_df = spark.read.format(config.get("source_format", "parquet")).load(source_path)
    target_df = spark.read.format(config.get("target_format", "parquet")).load(target_path)

    print(f"✅ Source rows: {source_df.count()} | Target rows: {target_df.count()}")

    # Row Count Comparison
    src_count, tgt_count = source_df.count(), target_df.count()
    row_count_match = src_count == tgt_count

    # Apply Transformations
    transformed_df = source_df.select(
        *primary_keys,
        *[expr(sql).alias(col_name) for col_name, sql in transformations.items()]
    )

    # Join on Primary Keys
    join_cond = [transformed_df[k] == target_df[k] for k in primary_keys]
    joined_df = transformed_df.alias("src").join(target_df.alias("tgt"), join_cond, "outer")

    # Comparison and Report Collection
    mismatch_summaries = []
    total_mismatch_records = 0

    for col_name in transformations.keys():
        src_col = col(f"src.{col_name}")
        tgt_col = col(f"tgt.{col_name}")

        # Handle numeric tolerance
        diff_expr = abs_(coalesce(src_col.cast(DoubleType()), lit(0)) - coalesce(tgt_col.cast(DoubleType()), lit(0)))

        mismatch_df = joined_df.filter(
            (
                (src_col.isNull() & tgt_col.isNotNull()) |
                (src_col.isNotNull() & tgt_col.isNull()) |
                (
                    (src_col != tgt_col)
                    if tolerance == 0 else diff_expr > tolerance
                )
            )
        ).select(*[col(f"src.{k}") for k in primary_keys],
                 src_col.alias("source_value"), tgt_col.alias("target_value"))

        mismatch_count = mismatch_df.count()
        total_mismatch_records += mismatch_count

        if mismatch_count > 0:
            print(f"❌ Column {col_name}: {mismatch_count} mismatches")
            save_report(spark, mismatch_df, f"{output_dir}/{col_name}_mismatches", fmt="csv")
        else:
            print(f"✅ Column {col_name}: values match")

        mismatch_summaries.append({
            "column": col_name,
            "mismatch_count": mismatch_count
        })

    # Summary DataFrame
    summary_df = spark.createDataFrame(mismatch_summaries)
    save_report(spark, summary_df, f"{output_dir}/summary_report", fmt="csv")

    end_time = time.time()
    duration = round(end_time - start_time, 2)

    print("\n📊 Validation Summary")
    print(f"Row Count Match: {'✅' if row_count_match else '❌'}")
    print(f"Column-Level Mismatches: {total_mismatch_records}")
    print(f"Execution Time: {duration}s")

    if row_count_match and total_mismatch_records == 0:
        print("\n✅ All Data Validation Checks Passed!")
    else:
        print("\n❌ Data Validation Issues Found — see reports for details.")

    spark.stop()


if __name__ == "__main__":
    main()
