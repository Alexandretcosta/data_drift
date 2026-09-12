
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

@dataclass
class DriftConfig:
	numeric_features: Optional[List[str]] = None
	categorical_features: Optional[List[str]] = None
	n_bins: int = 10
	relative_error: float = 0.01
	categorical_top_k: int = 50
	eps: float = 1e-6
	psi_thresholds: Tuple[float, float] = (0.10, 0.25)
	bounded_thresholds: Tuple[float, float, float] = (0.05, 0.10, 0.20)
	dataset_drift_min_share: float = 0.30

## Calculo de Bins Numericos
def compute_numeric_edges(df, col_name, n_bins=10, relative_error=0.01):

	probs = [i / n_bins for i in range(n_bins + 1)]
	edges = df.approxQuantile(col_name, probs, relative_error)

	# remove duplicatas para evitar bins inválidos
	edges = sorted(set([float(x) for x in edges if x is not None]))
	if len(edges) < 2:
		return None

	edges[0] = float("-inf")
	edges[-1] = float("inf")

	return edges


## Bucketizacao Simples em Spark Sql
from pyspark.sql import functions as F

def add_numeric_bucket(df, col_name, edges, bucket_col):

	expr = F.when(F.col(col_name).isNull(), F.lit(-1))
	for i in range(len(edges) - 1):
		cond = (F.col(col_name) >= F.lit(edges[i])) & (F.col(col_name) < F.lit(edges[i + 1]))
		expr = expr.when(cond, F.lit(i))

	return df.withColumn(bucket_col, expr.otherwise(F.lit(len(edges) - 2)))

## Distribuicao por Bucket

from pyspark.sql.window import Window
def distribution_by_col(df, bucket_col):
	counts = df.groupBy(bucket_col).agg(F.count("*").alias("n"))
	total = counts.agg(F.sum("n").alias("total")).first()["total"]

	return counts.withColumn("p", F.col("n") / F.lit(total))

## Computacao das Metricas
def compute_distribution_metrics(ref_dist, cur_dist, key_col, eps=1e-6):
	joined = (
	ref_dist.select(F.col(key_col), F.col("p").alias("p_ref"))
	.join(cur_dist.select(F.col(key_col), F.col("p").alias("p_cur")), key_col, "outer")
	.fillna(0.0, subset=["p_ref", "p_cur"])
	.withColumn("p_ref_s", F.greatest(F.col("p_ref"), F.lit(eps)))
	.withColumn("p_cur_s", F.greatest(F.col("p_cur"), F.lit(eps)))
	.withColumn("m", (F.col("p_ref_s") + F.col("p_cur_s")) / 2)
	)

	agg = joined.agg(
	F.sum((F.col("p_cur_s") - F.col("p_ref_s")) * F.log(F.col("p_cur_s") /
	F.col("p_ref_s"))).alias("psi"),
	(
		F.lit(0.5) * F.sum(F.col("p_ref_s") * F.log(F.col("p_ref_s") / F.col("m"))) +
		F.lit(0.5) * F.sum(F.col("p_cur_s") * F.log(F.col("p_cur_s") / F.col("m")))
	).alias("js"),
	F.sqrt(F.lit(0.5) * F.sum((F.sqrt(F.col("p_ref_s")) - F.sqrt(F.col("p_cur_s"))) **
	2)).alias("hellinger")
	)

	return agg.first().asDict()

## KS aproximado por Histograma
def compute_ks_approx(joined_dist, bucket_col):

	w = Window.orderBy(bucket_col).rowsBetween(Window.unboundedPreceding, Window.currentRow)
	return (
	joined_dist
	.withColumn("cdf_ref", F.sum("p_ref").over(w))
	.withColumn("cdf_cur", F.sum("p_cur").over(w))
	.withColumn("abs_diff", F.abs(F.col("cdf_ref") - F.col("cdf_cur")))
	.agg(F.max("abs_diff").alias("ks_approx"))
	.first()["ks_approx"]
	)

detector = SparkDataDriftDetector(
	config=DriftConfig(
	numeric_features=["idade", "renda", "score"],
	categorical_features=["uf", "segmento", "canal"],
	n_bins=10,
	categorical_top_k=50,
	metrics_numeric=["psi", "js", "hellinger", "ks_approx"],
	metrics_categorical=["psi", "js", "hellinger", "linf", "chi_square"],
	psi_thresholds=(0.10, 0.25),
	bounded_thresholds=(0.05, 0.10, 0.20),
	dataset_drift_min_share=0.30,
	eps=1e-6
	)
)

detector.fit(reference_df)


report = detector.compare(current_df)
report.feature_metrics.show()
report.dataset_summary.show()