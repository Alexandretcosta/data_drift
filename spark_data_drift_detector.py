
"""
SparkDataDriftDetector
======================

Classe PySpark para detecção univariada e distribuída de Data Drift.

Componentes:
    - DriftConfig
    - SchemaValidator
    - FeatureProfiler
    - NumericProfiler
    - CategoricalProfiler
    - MetricCalculator
    - SeverityClassifier
    - DriftReport
    - PersistenceAdapter
    - SparkDataDriftDetector

Dependências:
    PySpark >= 3.3

Uso básico:

    config = DriftConfig(
        numeric_features=["idade", "renda", "score"],
        categorical_features=["uf", "segmento", "canal"]
    )

    detector = SparkDataDriftDetector(config)

    detector.fit(reference_df)

    report = detector.compare(current_df)

    report.feature_metrics.show(truncate=False)
    report.dataset_summary.show(truncate=False)

    detector.save_report(report, "/tmp/drift_report")
    detector.save_reference_profile("/tmp/reference_profile")

Observação metodológica:
    O detector compara perfis estatísticos da referência e do conjunto atual.
    Ele não compara linhas individualmente e não afirma, sozinho, que o modelo
    precisa ser retreinado. Drift estatístico é tratado como sinal para
    investigação de impacto no modelo.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    NumericType,
    StringType,
    BooleanType,
    DateType,
    TimestampType,
)
from pyspark.ml.feature import Bucketizer


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

@dataclass
class DriftConfig:
    """Parâmetros gerais do detector."""

    numeric_features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None

    n_bins: int = 10
    relative_error: float = 0.01
    categorical_top_k: int = 50

    metrics_numeric: List[str] = field(
        default_factory=lambda: [
            "psi",
            "js",
            "hellinger",
            "ks_approx",
        ]
    )

    metrics_categorical: List[str] = field(
        default_factory=lambda: [
            "psi",
            "js",
            "hellinger",
            "chi_square",
        ]
    )

    psi_thresholds: Tuple[float, float] = (0.10, 0.25)

    # Para métricas limitadas / normalizadas em [0, 1]
    bounded_thresholds: Tuple[float, float, float] = (
        0.05,
        0.10,
        0.20,
    )

    dataset_drift_min_share: float = 0.30

    eps: float = 1e-6

    # Número máximo de categorias explicitamente armazenadas no perfil.
    # As demais são agrupadas em __OTHER__.
    max_profile_categories: Optional[int] = None

    # Se True, tenta inferir features automaticamente.
    infer_features: bool = False

    # Se True, calcula importância de cada métrica no ranking.
    use_metric_consensus: bool = True

    def __post_init__(self):
        if self.numeric_features is None:
            self.numeric_features = []

        if self.categorical_features is None:
            self.categorical_features = []

        if self.n_bins < 2:
            raise ValueError("n_bins deve ser >= 2.")

        if not 0 < self.relative_error <= 1:
            raise ValueError("relative_error deve estar entre 0 e 1.")

        if self.categorical_top_k < 1:
            raise ValueError("categorical_top_k deve ser >= 1.")

        if self.eps <= 0:
            raise ValueError("eps deve ser > 0.")

        if not 0 <= self.dataset_drift_min_share <= 1:
            raise ValueError(
                "dataset_drift_min_share deve estar entre 0 e 1."
            )


# =============================================================================
# UTILITÁRIOS
# =============================================================================

class _Utils:

    @staticmethod
    def validate_features_exist(df: DataFrame, features: List[str]) -> List[str]:
        available = set(df.columns)
        return [f for f in features if f in available]

    @staticmethod
    def safe_divide(numerator, denominator, eps=1e-6):
        return numerator / F.greatest(denominator, F.lit(eps))

    @staticmethod
    def normalized_distribution(
        df: DataFrame,
        key_col: str,
        count_col: str = "n",
        probability_col: str = "p",
    ) -> DataFrame:

        total = df.agg(F.sum(F.col(count_col)).alias("_total")).first()["_total"]

        if total is None or total == 0:
            return df.withColumn(probability_col, F.lit(0.0))

        return df.withColumn(
            probability_col,
            F.col(count_col) / F.lit(float(total))
        )

    @staticmethod
    def collect_single_row(df: DataFrame) -> Dict[str, Any]:
        row = df.first()
        return row.asDict() if row is not None else {}

    @staticmethod
    def is_numeric_type(dtype) -> bool:
        return isinstance(dtype, NumericType)

    @staticmethod
    def is_categorical_type(dtype) -> bool:
        return (
            isinstance(dtype, (StringType, BooleanType))
        )

    @staticmethod
    def canonical_category(value: Any) -> str:
        if value is None:
            return "__MISSING__"
        return str(value)


# =============================================================================
# SCHEMA VALIDATOR
# =============================================================================

class SchemaValidator:
    """
    Valida diferenças estruturais entre reference_df e current_df.

    Verifica:
        - colunas ausentes no current;
        - colunas inesperadas no current;
        - tipos diferentes;
        - percentual de nulos;
        - categorias novas para features categóricas.
    """

    def __init__(self, categorical_top_k: int = 50):
        self.categorical_top_k = categorical_top_k

    def validate(
        self,
        reference_df: DataFrame,
        current_df: DataFrame,
        categorical_features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:

        categorical_features = categorical_features or []

        ref_schema = {
            field.name: field.dataType.simpleString()
            for field in reference_df.schema.fields
        }

        cur_schema = {
            field.name: field.dataType.simpleString()
            for field in current_df.schema.fields
        }

        ref_cols = set(reference_df.columns)
        cur_cols = set(current_df.columns)

        missing_columns = sorted(ref_cols - cur_cols)
        unexpected_columns = sorted(cur_cols - ref_cols)

        common_columns = sorted(ref_cols & cur_cols)

        type_changes = []

        for col_name in common_columns:
            ref_type = ref_schema[col_name]
            cur_type = cur_schema[col_name]

            if ref_type != cur_type:
                type_changes.append({
                    "feature": col_name,
                    "reference_type": ref_type,
                    "current_type": cur_type,
                })

        # Nulos: calculados em uma única agregação por dataset.
        null_exprs_ref = [
            F.avg(F.col(c).isNull().cast("double")).alias(c)
            for c in common_columns
        ]

        null_exprs_cur = [
            F.avg(F.col(c).isNull().cast("double")).alias(c)
            for c in common_columns
        ]

        ref_null_row = (
            reference_df.agg(*null_exprs_ref).first()
            if common_columns
            else None
        )

        cur_null_row = (
            current_df.agg(*null_exprs_cur).first()
            if common_columns
            else None
        )

        feature_rows = []

        type_change_map = {
            x["feature"]: x for x in type_changes
        }

        for feature in common_columns:
            missing_ref = (
                float(ref_null_row[feature])
                if ref_null_row is not None
                else 0.0
            )

            missing_cur = (
                float(cur_null_row[feature])
                if cur_null_row is not None
                else 0.0
            )

            feature_rows.append({
                "feature": feature,
                "status": (
                    "TYPE_CHANGED"
                    if feature in type_change_map
                    else "OK"
                ),
                "reference_type": ref_schema[feature],
                "current_type": cur_schema[feature],
                "missing_ref": missing_ref,
                "missing_cur": missing_cur,
                "missing_delta": missing_cur - missing_ref,
            })

        # Categorias novas são calculadas apenas para as features
        # categóricas explicitamente informadas.
        new_categories = {}

        for feature in categorical_features:
            if feature not in ref_cols or feature not in cur_cols:
                continue

            ref_categories = (
                reference_df
                .select(
                    F.coalesce(
                        F.col(feature).cast("string"),
                        F.lit("__MISSING__")
                    ).alias("_category")
                )
                .distinct()
            )

            cur_categories = (
                current_df
                .select(
                    F.coalesce(
                        F.col(feature).cast("string"),
                        F.lit("__MISSING__")
                    ).alias("_category")
                )
                .distinct()
            )

            new_rows = (
                cur_categories
                .join(ref_categories, "_category", "left_anti")
                .limit(1000)
                .collect()
            )

            new_categories[feature] = [
                row["_category"] for row in new_rows
            ]

        return {
            "missing_columns": missing_columns,
            "unexpected_columns": unexpected_columns,
            "type_changes": type_changes,
            "feature_quality": feature_rows,
            "new_categories": new_categories,
        }

    def validate_as_dataframe(
        self,
        spark: SparkSession,
        validation_result: Dict[str, Any],
    ) -> DataFrame:

        rows = validation_result["feature_quality"]

        if not rows:
            return spark.createDataFrame(
                [],
                """
                feature string,
                status string,
                reference_type string,
                current_type string,
                missing_ref double,
                missing_cur double,
                missing_delta double
                """
            )

        return spark.createDataFrame(rows)


# =============================================================================
# NUMERIC PROFILER
# =============================================================================

class NumericProfiler:
    """
    Gera perfil distribuído para features numéricas.

    O perfil contém:
        - count;
        - null_count;
        - null_proportion;
        - min;
        - max;
        - approximate quantiles;
        - bin_edges;
        - histogram por bins.
    """

    def __init__(
        self,
        n_bins: int = 10,
        relative_error: float = 0.01,
        eps: float = 1e-6,
    ):
        self.n_bins = n_bins
        self.relative_error = relative_error
        self.eps = eps

    def compute_numeric_edges(
        self,
        df: DataFrame,
        feature: str,
    ) -> Optional[List[float]]:

        probabilities = [
            i / self.n_bins
            for i in range(self.n_bins + 1)
        ]

        values = df.select(feature).where(
            F.col(feature).isNotNull()
        )

        if values.limit(1).count() == 0:
            return None

        edges = values.approxQuantile(
            feature,
            probabilities,
            self.relative_error,
        )

        # Remove duplicatas causadas por variáveis com muitos valores iguais.
        edges = sorted(
            set(
                float(x)
                for x in edges
                if x is not None
            )
        )

        if len(edges) < 2:
            return None

        edges[0] = float("-inf")
        edges[-1] = float("inf")

        return edges

    def add_numeric_bucket(
        self,
        df: DataFrame,
        feature: str,
        edges: List[float],
        bucket_col: str = "__drift_bucket",
    ) -> DataFrame:

        expr = F.when(
            F.col(feature).isNull(),
            F.lit(-1)
        )

        for i in range(len(edges) - 1):
            lower = edges[i]
            upper = edges[i + 1]

            if i == len(edges) - 2:
                condition = (
                    (F.col(feature) >= F.lit(lower))
                    & (F.col(feature) <= F.lit(upper))
                )
            else:
                condition = (
                    (F.col(feature) >= F.lit(lower))
                    & (F.col(feature) < F.lit(upper))
                )

            expr = expr.when(
                condition,
                F.lit(i)
            )

        return df.withColumn(
            bucket_col,
            expr.otherwise(F.lit(-1))
        )

    def distribution_by_bucket(
        self,
        df: DataFrame,
        bucket_col: str,
    ) -> DataFrame:

        counts = (
            df.groupBy(bucket_col)
            .agg(F.count("*").alias("n"))
        )

        return _Utils.normalized_distribution(
            counts,
            bucket_col,
            "n",
            "p",
        )

    def profile(
        self,
        df: DataFrame,
        feature: str,
        bin_edges: Optional[List[float]] = None,
    ) -> Dict[str, Any]:

        if bin_edges is None:
            bin_edges = self.compute_numeric_edges(
                df,
                feature,
            )

        row = (
            df.agg(
                F.count(F.col(feature)).alias("count"),
                F.count("*").alias("total_count"),
                F.min(F.col(feature)).alias("min"),
                F.max(F.col(feature)).alias("max"),
            )
            .first()
        )

        total_count = int(row["total_count"] or 0)
        count = int(row["count"] or 0)
        null_count = total_count - count

        probabilities = [
            i / self.n_bins
            for i in range(self.n_bins + 1)
        ]

        quantiles = []

        if count > 0:
            quantiles = df.select(feature).where(
                F.col(feature).isNotNull()
            ).approxQuantile(
                feature,
                probabilities,
                self.relative_error,
            )

        histogram = []

        if bin_edges is not None and count > 0:
            bucketed = self.add_numeric_bucket(
                df,
                feature,
                bin_edges,
            )

            dist = self.distribution_by_bucket(
                bucketed,
                "__drift_bucket",
            )

            histogram = [
                row.asDict()
                for row in dist.orderBy("__drift_bucket").collect()
            ]

        return {
            "feature": feature,
            "feature_type": "numeric",
            "count": count,
            "total_count": total_count,
            "null_count": null_count,
            "null_proportion": (
                null_count / total_count
                if total_count > 0
                else 0.0
            ),
            "min": float(row["min"]) if row["min"] is not None else None,
            "max": float(row["max"]) if row["max"] is not None else None,
            "quantiles": [
                float(x) for x in quantiles
            ],
            "bin_edges": bin_edges,
            "histogram": histogram,
        }


# =============================================================================
# CATEGORICAL PROFILER
# =============================================================================

class CategoricalProfiler:
    """
    Gera perfil distribuído para features categóricas.

    O perfil contém:
        - contagem total;
        - nulos;
        - cardinalidade;
        - top-K;
        - categorias raras;
        - proporções;
        - categoria agregada __OTHER__.

    As categorias são tratadas como strings para facilitar a persistência
    e a comparação entre datasets.
    """

    def __init__(
        self,
        top_k: int = 50,
        rare_threshold: float = 0.01,
    ):
        self.top_k = top_k
        self.rare_threshold = rare_threshold

    def profile(
        self,
        df: DataFrame,
        feature: str,
        reference_categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:

        category_col = "__drift_category"

        prepared = df.withColumn(
            category_col,
            F.coalesce(
                F.col(feature).cast("string"),
                F.lit("__MISSING__")
            )
        )

        counts = (
            prepared
            .groupBy(category_col)
            .agg(F.count("*").alias("n"))
        )

        total_row = counts.agg(
            F.sum("n").alias("total")
        ).first()

        total = int(total_row["total"] or 0)

        if total == 0:
            return {
                "feature": feature,
                "feature_type": "categorical",
                "count": 0,
                "null_count": 0,
                "null_proportion": 0.0,
                "cardinality": 0,
                "top_k": [],
                "rare_categories": [],
                "categories": [],
            }

        dist = counts.withColumn(
            "p",
            F.col("n") / F.lit(float(total))
        )

        top_rows = (
            dist
            .orderBy(F.desc("n"))
            .limit(self.top_k)
            .collect()
        )

        top_categories = [
            row[category_col]
            for row in top_rows
        ]

        # Raras = categorias cuja proporção é menor que o threshold.
        rare_rows = (
            dist
            .filter(F.col("p") < F.lit(self.rare_threshold))
            .orderBy(F.desc("p"))
            .limit(1000)
            .collect()
        )

        rare_categories = [
            row[category_col]
            for row in rare_rows
        ]

        profile_categories = (
            reference_categories
            if reference_categories is not None
            else top_categories
        )

        profile_categories = list(
            dict.fromkeys(profile_categories)
        )

        category_rows = [
            {
                "category": row[category_col],
                "n": int(row["n"]),
                "p": float(row["p"]),
            }
            for row in (
                dist
                .filter(
                    F.col(category_col).isin(profile_categories)
                )
                .collect()
            )
        ]

        known_category_set = set(profile_categories)

        # Soma tudo que não estiver explicitamente no perfil.
        other_row = (
            dist
            .filter(
                ~F.col(category_col).isin(profile_categories)
            )
            .agg(
                F.sum("n").alias("n"),
                F.sum("p").alias("p"),
            )
            .first()
        )

        if other_row["n"] is not None and int(other_row["n"]) > 0:
            category_rows.append({
                "category": "__OTHER__",
                "n": int(other_row["n"]),
                "p": float(other_row["p"]),
            })

        null_row = (
            dist
            .filter(F.col(category_col) == "__MISSING__")
            .first()
        )

        null_count = (
            int(null_row["n"])
            if null_row is not None
            else 0
        )

        return {
            "feature": feature,
            "feature_type": "categorical",
            "count": total - null_count,
            "total_count": total,
            "null_count": null_count,
            "null_proportion": null_count / total,
            "cardinality": dist.count(),
            "top_k": category_rows,
            "rare_categories": rare_categories,
            "categories": [
                x["category"] for x in category_rows
            ],
        }


# =============================================================================
# FEATURE PROFILER
# =============================================================================

class FeatureProfiler:
    """
    Orquestra NumericProfiler e CategoricalProfiler.

    O perfil retornado é um dicionário Python compacto, composto por
    estatísticas agregadas. Os dados brutos não são coletados no driver.
    """

    def __init__(self, config: DriftConfig):
        self.config = config

        self.numeric_profiler = NumericProfiler(
            n_bins=config.n_bins,
            relative_error=config.relative_error,
            eps=config.eps,
        )

        self.categorical_profiler = CategoricalProfiler(
            top_k=config.categorical_top_k
        )

    def infer_features(
        self,
        df: DataFrame,
    ) -> Tuple[List[str], List[str]]:

        numeric = []
        categorical = []

        for field in df.schema.fields:
            if _Utils.is_numeric_type(field.dataType):
                numeric.append(field.name)
            elif _Utils.is_categorical_type(field.dataType):
                categorical.append(field.name)

        return numeric, categorical

    def profile(
        self,
        df: DataFrame,
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        reference_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        if self.config.infer_features:
            inferred_numeric, inferred_categorical = (
                self.infer_features(df)
            )

            numeric_features = (
                numeric_features or inferred_numeric
            )

            categorical_features = (
                categorical_features or inferred_categorical
            )

        numeric_features = (
            numeric_features
            if numeric_features is not None
            else self.config.numeric_features
        )

        categorical_features = (
            categorical_features
            if categorical_features is not None
            else self.config.categorical_features
        )

        numeric_features = _Utils.validate_features_exist(
            df,
            numeric_features,
        )

        categorical_features = _Utils.validate_features_exist(
            df,
            categorical_features,
        )

        profile = {
            "numeric": {},
            "categorical": {},
        }

        for feature in numeric_features:

            reference_edges = None

            if reference_profile:
                reference_edges = (
                    reference_profile
                    .get("numeric", {})
                    .get(feature, {})
                    .get("bin_edges")
                )

            profile["numeric"][feature] = (
                self.numeric_profiler.profile(
                    df,
                    feature,
                    bin_edges=reference_edges,
                )
            )

        for feature in categorical_features:

            reference_categories = None

            if reference_profile:
                reference_categories = (
                    reference_profile
                    .get("categorical", {})
                    .get(feature, {})
                    .get("categories")
                )

            profile["categorical"][feature] = (
                self.categorical_profiler.profile(
                    df,
                    feature,
                    reference_categories=reference_categories,
                )
            )

        return profile


# =============================================================================
# METRIC CALCULATOR
# =============================================================================

class MetricCalculator:
    """
    Calcula métricas de distância entre duas distribuições.

    Métricas:
        - PSI
        - Jensen-Shannon
        - Hellinger
        - KS aproximado
        - Chi-Square

    Todas operam sobre distribuições agregadas.
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def _prepare(
        self,
        reference_distribution: List[Dict[str, Any]],
        current_distribution: List[Dict[str, Any]],
        key: str = "category",
    ) -> List[Dict[str, float]]:

        ref = {
            str(x[key]): float(x["p"])
            for x in reference_distribution
        }

        cur = {
            str(x[key]): float(x["p"])
            for x in current_distribution
        }

        keys = sorted(set(ref) | set(cur))

        result = []

        for k in keys:
            p_ref = max(ref.get(k, 0.0), self.eps)
            p_cur = max(cur.get(k, 0.0), self.eps)

            result.append({
                "key": k,
                "p_ref": p_ref,
                "p_cur": p_cur,
            })

        # Renormalização após smoothing.
        total_ref = sum(x["p_ref"] for x in result)
        total_cur = sum(x["p_cur"] for x in result)

        for x in result:
            x["p_ref"] /= total_ref
            x["p_cur"] /= total_cur

        return result

    def psi(
        self,
        distribution: List[Dict[str, Any]],
    ) -> float:

        value = 0.0

        for x in distribution:
            p = max(x["p_ref"], self.eps)
            q = max(x["p_cur"], self.eps)

            value += (q - p) * __import__("math").log(q / p)

        return float(value)

    def js(
        self,
        distribution: List[Dict[str, Any]],
    ) -> float:

        import math

        value = 0.0

        for x in distribution:
            p = max(x["p_ref"], self.eps)
            q = max(x["p_cur"], self.eps)

            m = (p + q) / 2.0

            value += (
                0.5 * p * math.log(p / m)
                + 0.5 * q * math.log(q / m)
            )

        return float(value)

    def hellinger(
        self,
        distribution: List[Dict[str, Any]],
    ) -> float:

        import math

        total = sum(
            (
                math.sqrt(x["p_ref"])
                - math.sqrt(x["p_cur"])
            ) ** 2
            for x in distribution
        )

        return float(math.sqrt(0.5 * total))

    def ks_approx(
        self,
        distribution: List[Dict[str, Any]],
    ) -> float:

        cdf_ref = 0.0
        cdf_cur = 0.0
        max_difference = 0.0

        for x in distribution:
            cdf_ref += x["p_ref"]
            cdf_cur += x["p_cur"]

            max_difference = max(
                max_difference,
                abs(cdf_ref - cdf_cur),
            )

        return float(max_difference)

    def chi_square(
        self,
        distribution: List[Dict[str, Any]],
    ) -> float:

        # Estatística baseada nas proporções.
        # O resultado é uma medida auxiliar e não substitui um teste
        # estatístico completo com graus de liberdade/p-value.
        value = 0.0

        for x in distribution:
            expected = max(
                x["p_ref"],
                self.eps,
            )

            observed = x["p_cur"]

            value += (
                (observed - expected) ** 2
                / expected
            )

        return float(value)

    def calculate(
        self,
        reference_distribution: List[Dict[str, Any]],
        current_distribution: List[Dict[str, Any]],
        key: str = "category",
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:

        metrics = metrics or [
            "psi",
            "js",
            "hellinger",
            "ks_approx",
            "chi_square",
        ]

        distribution = self._prepare(
            reference_distribution,
            current_distribution,
            key=key,
        )

        result = {}

        for metric in metrics:
            if metric == "psi":
                result["psi"] = self.psi(distribution)

            elif metric == "js":
                result["js"] = self.js(distribution)

            elif metric == "hellinger":
                result["hellinger"] = self.hellinger(distribution)

            elif metric == "ks_approx":
                result["ks_approx"] = self.ks_approx(distribution)

            elif metric == "chi_square":
                result["chi_square"] = self.chi_square(distribution)

            else:
                raise ValueError(
                    f"Métrica desconhecida: {metric}"
                )

        return result


# =============================================================================
# SEVERITY CLASSIFIER
# =============================================================================

class SeverityClassifier:
    """
    Classifica a severidade do drift.

    PSI:
        < low_threshold  -> none
        low..high        -> medium
        >= high          -> high

    Para métricas normalizadas em [0, 1]:
        < 0.05           -> none
        0.05..0.10       -> low
        0.10..0.20       -> medium
        >= 0.20          -> high

    'critical' é reservado para evidências estruturais muito fortes,
    como categorias novas em proporção elevada, schema incompatível ou
    consenso forte entre métricas.
    """

    def __init__(
        self,
        psi_thresholds: Tuple[float, float] = (0.10, 0.25),
        bounded_thresholds: Tuple[float, float, float] = (
            0.05,
            0.10,
            0.20,
        ),
    ):
        self.psi_thresholds = psi_thresholds
        self.bounded_thresholds = bounded_thresholds

    @staticmethod
    def severity_score(severity: str) -> int:
        return {
            "none": 0,
            "low": 1,
            "medium": 2,
            "high": 3,
            "critical": 4,
        }.get(severity, 0)

    def classify_metric(
        self,
        metric: str,
        value: Optional[float],
    ) -> str:

        if value is None:
            return "none"

        if metric == "psi":
            low, high = self.psi_thresholds

            if value < low:
                return "none"

            if value < high:
                return "medium"

            return "high"

        if metric in {
            "js",
            "hellinger",
            "ks_approx",
        }:
            low, medium, high = self.bounded_thresholds

            if value < low:
                return "none"

            if value < medium:
                return "low"

            if value < high:
                return "medium"

            return "high"

        # Chi-square não é diretamente comparável à escala acima.
        return "none"

    def classify(
        self,
        metrics: Dict[str, float],
        primary_metric: str = "psi",
        new_category_share: float = 0.0,
        schema_issue: bool = False,
    ) -> Dict[str, Any]:

        primary_value = metrics.get(primary_metric)

        severity = self.classify_metric(
            primary_metric,
            primary_value,
        )

        metric_severities = {
            metric: self.classify_metric(metric, value)
            for metric, value in metrics.items()
        }

        # Consenso entre métricas.
        high_count = sum(
            self.severity_score(x) >= 3
            for x in metric_severities.values()
        )

        medium_or_higher_count = sum(
            self.severity_score(x) >= 2
            for x in metric_severities.values()
        )

        if high_count >= 2:
            severity = "high"

        elif medium_or_higher_count >= 2:
            severity = max(
                severity,
                "medium",
                key=self.severity_score,
            )

        # Evidência estrutural forte.
        if schema_issue:
            severity = "critical"

        elif new_category_share >= 0.20:
            severity = "critical"

        elif new_category_share >= 0.10:
            severity = max(
                severity,
                "high",
                key=self.severity_score,
            )

        return {
            "severity": severity,
            "severity_score": self.severity_score(severity),
            "metric_severities": metric_severities,
        }


# =============================================================================
# DRIFT REPORT
# =============================================================================

class DriftReport:
    """
    Objeto de saída do detector.

    Atributos:
        feature_metrics:
            DataFrame com uma linha por feature.

        dataset_summary:
            Resumo agregado do dataset.

        recommendations:
            Recomendações textuais por feature.
    """

    def __init__(
        self,
        feature_metrics: DataFrame,
        dataset_summary: DataFrame,
        recommendations: Optional[DataFrame] = None,
    ):
        self.feature_metrics = feature_metrics
        self.dataset_summary = dataset_summary
        self.recommendations = recommendations

    def show(
        self,
        n: int = 20,
        truncate: bool = False,
    ):
        print("=== FEATURE METRICS ===")
        self.feature_metrics.show(
            n,
            truncate=truncate,
        )

        print("=== DATASET SUMMARY ===")
        self.dataset_summary.show(
            n,
            truncate=truncate,
        )

        if self.recommendations is not None:
            print("=== RECOMMENDATIONS ===")
            self.recommendations.show(
                n,
                truncate=truncate,
            )

    def order_by_severity(self) -> DataFrame:
        return self.feature_metrics.orderBy(
            F.desc("severity_score"),
            F.desc("rank_score"),
        )

    def drifted_features(self) -> DataFrame:
        return (
            self.feature_metrics
            .filter(F.col("drift_detected") == True)
            .orderBy(
                F.desc("severity_score"),
                F.desc("rank_score"),
            )
        )


# =============================================================================
# PERSISTENCE ADAPTER
# =============================================================================

class PersistenceAdapter:
    """
    Persistência simples em Parquet ou Delta.

    Não persiste o dataset bruto. Persiste somente perfis e resultados.
    """

    def __init__(self, format: str = "parquet"):
        format = format.lower()

        if format not in {"parquet", "delta"}:
            raise ValueError(
                "format deve ser 'parquet' ou 'delta'."
            )

        self.format = format

    def save_dataframe(
        self,
        df: DataFrame,
        path: str,
        mode: str = "overwrite",
    ):
        (
            df.write
            .format(self.format)
            .mode(mode)
            .save(path)
        )

    def load_dataframe(
        self,
        spark: SparkSession,
        path: str,
    ) -> DataFrame:
        return (
            spark.read
            .format(self.format)
            .load(path)
        )


# =============================================================================
# DETECTOR PRINCIPAL
# =============================================================================

class SparkDataDriftDetector:
    """
    Classe principal para detecção de Data Drift.

    Fluxo:

        detector.fit(reference_df)

        report = detector.compare(current_df)

    O método fit() constrói o perfil de referência.

    O método compare() utiliza os mesmos bins e categorias da referência
    para construir o perfil atual e calcular as métricas.
    """

    def __init__(
        self,
        config: DriftConfig,
        spark: Optional[SparkSession] = None,
        persistence: Optional[PersistenceAdapter] = None,
    ):
        self.config = config

        self.spark = (
            spark
            if spark is not None
            else SparkSession.getActiveSession()
        )

        if self.spark is None:
            raise ValueError(
                "Informe uma SparkSession ativa."
            )

        self.schema_validator = SchemaValidator(
            categorical_top_k=config.categorical_top_k
        )

        self.profiler = FeatureProfiler(config)

        self.metric_calculator = MetricCalculator(
            eps=config.eps
        )

        self.severity_classifier = SeverityClassifier(
            psi_thresholds=config.psi_thresholds,
            bounded_thresholds=config.bounded_thresholds,
        )

        self.persistence = (
            persistence
            if persistence is not None
            else PersistenceAdapter("parquet")
        )

        self.reference_profile = None
        self.reference_df_schema = None

    # -------------------------------------------------------------------------
    # FEATURE DISCOVERY
    # -------------------------------------------------------------------------

    def _resolve_features(
        self,
        reference_df: DataFrame,
    ):

        numeric_features = list(
            self.config.numeric_features
        )

        categorical_features = list(
            self.config.categorical_features
        )

        if self.config.infer_features:
            inferred_numeric, inferred_categorical = (
                self.profiler.infer_features(reference_df)
            )

            if not numeric_features:
                numeric_features = inferred_numeric

            if not categorical_features:
                categorical_features = inferred_categorical

        return (
            numeric_features,
            categorical_features,
        )

    # -------------------------------------------------------------------------
    # FIT
    # -------------------------------------------------------------------------

    def fit(
        self,
        reference_df: DataFrame,
    ):
        """
        Constrói o perfil de referência.

        Returns:
            self
        """

        (
            numeric_features,
            categorical_features,
        ) = self._resolve_features(reference_df)

        missing_numeric = [
            x for x in numeric_features
            if x not in reference_df.columns
        ]

        missing_categorical = [
            x for x in categorical_features
            if x not in reference_df.columns
        ]

        if missing_numeric or missing_categorical:
            raise ValueError(
                "Features ausentes no reference_df: "
                f"{missing_numeric + missing_categorical}"
            )

        self.reference_profile = self.profiler.profile(
            reference_df,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )

        self.reference_df_schema = {
            field.name: field.dataType.simpleString()
            for field in reference_df.schema.fields
        }

        return self

    # -------------------------------------------------------------------------
    # DISTRIBUTION HELPERS
    # -------------------------------------------------------------------------

    @staticmethod
    def _numeric_histogram_to_distribution(
        histogram: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        return [
            {
                "category": str(row["__drift_bucket"]),
                "p": float(row["p"]),
            }
            for row in histogram
        ]

    @staticmethod
    def _categorical_profile_to_distribution(
        categories: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        return [
            {
                "category": str(row["category"]),
                "p": float(row["p"]),
            }
            for row in categories
        ]

    @staticmethod
    def _new_category_share(
        reference_profile: Dict[str, Any],
        current_profile: Dict[str, Any],
    ) -> float:

        ref_categories = set(
            reference_profile.get("categories", [])
        )

        cur_categories = current_profile.get(
            "categories",
            []
        )

        if not cur_categories:
            return 0.0

        new_count = sum(
            1
            for category in cur_categories
            if category not in ref_categories
        )

        return new_count / len(cur_categories)

    # -------------------------------------------------------------------------
    # RECOMMENDATION
    # -------------------------------------------------------------------------

    @staticmethod
    def _recommendation(
        severity: str,
        drift_detected: bool,
        new_category_share: float = 0.0,
        missing_delta: float = 0.0,
    ) -> str:

        if severity == "critical":
            return (
                "Investigar imediatamente: evidência estrutural "
                "ou mudança de distribuição muito relevante."
            )

        if severity == "high":
            return (
                "Avaliar impacto no modelo e investigar a causa "
                "do drift antes de considerar recalibração ou retreinamento."
            )

        if severity == "medium":
            return (
                "Monitorar a evolução do drift e verificar impacto "
                "sobre as métricas do modelo."
            )

        if severity == "low":
            return (
                "Acompanhar historicamente; não indica, isoladamente, "
                "necessidade de intervenção."
            )

        if abs(missing_delta) > 0.05:
            return (
                "Investigar alteração relevante na qualidade dos dados "
                "e na proporção de valores nulos."
            )

        if new_category_share > 0:
            return (
                "Monitorar categorias novas e verificar alteração "
                "do processo gerador dos dados."
            )

        return "Nenhuma ação necessária."

    # -------------------------------------------------------------------------
    # COMPARE
    # -------------------------------------------------------------------------

    def compare(
        self,
        current_df: DataFrame,
    ) -> DriftReport:

        if self.reference_profile is None:
            raise RuntimeError(
                "Execute fit(reference_df) antes de compare(current_df)."
            )

        validation = self.schema_validator.validate(
            reference_df=self._reference_schema_df_placeholder(
                current_df
            ),
            current_df=current_df,
            categorical_features=self.config.categorical_features,
        )

        # A validação acima precisa do reference DataFrame real para verificar
        # categorias. Como o detector não guarda o dataset bruto, fazemos a
        # validação estrutural diretamente com o schema armazenado.
        validation = self._validate_current_schema(current_df)

        current_profile = self.profiler.profile(
            current_df,
            numeric_features=self.config.numeric_features,
            categorical_features=self.config.categorical_features,
            reference_profile=self.reference_profile,
        )

        rows = []

        # -------------------------
        # NUMÉRICAS
        # -------------------------

        for feature, ref_info in (
            self.reference_profile["numeric"].items()
        ):

            cur_info = (
                current_profile["numeric"]
                .get(feature)
            )

            if cur_info is None:
                continue

            ref_dist = (
                self._numeric_histogram_to_distribution(
                    ref_info.get("histogram", [])
                )
            )

            cur_dist = (
                self._numeric_histogram_to_distribution(
                    cur_info.get("histogram", [])
                )
            )

            metrics = self.metric_calculator.calculate(
                ref_dist,
                cur_dist,
                key="category",
                metrics=self.config.metrics_numeric,
            )

            primary_metric = (
                "psi"
                if "psi" in metrics
                else next(iter(metrics))
            )

            severity_result = (
                self.severity_classifier.classify(
                    metrics,
                    primary_metric=primary_metric,
                )
            )

            severity = severity_result["severity"]

            drift_detected = severity != "none"

            rank_score = (
                severity_result["severity_score"]
                * (
                    1
                    + (
                        sum(
                            1
                            for s in severity_result[
                                "metric_severities"
                            ].values()
                            if self.severity_classifier.severity_score(s) >= 2
                        )
                        / max(len(metrics), 1)
                    )
                    if self.config.use_metric_consensus
                    else 0
                )
            )

            missing_delta = (
                cur_info["null_proportion"]
                - ref_info["null_proportion"]
            )

            recommendation = self._recommendation(
                severity,
                drift_detected,
                missing_delta=missing_delta,
            )

            rows.append({
                "feature_name": feature,
                "feature_type": "numeric",
                "primary_metric": primary_metric,
                "primary_value": float(metrics[primary_metric]),
                "psi": metrics.get("psi"),
                "js": metrics.get("js"),
                "hellinger": metrics.get("hellinger"),
                "ks_approx": metrics.get("ks_approx"),
                "chi_square": metrics.get("chi_square"),
                "severity": severity,
                "severity_score": int(
                    severity_result["severity_score"]
                ),
                "drift_detected": bool(drift_detected),
                "rank_score": float(rank_score),
                "missing_ref": float(
                    ref_info["null_proportion"]
                ),
                "missing_cur": float(
                    cur_info["null_proportion"]
                ),
                "missing_delta": float(missing_delta),
                "new_category_share": 0.0,
                "recommendation": recommendation,
            })

        # -------------------------
        # CATEGÓRICAS
        # -------------------------

        for feature, ref_info in (
            self.reference_profile["categorical"].items()
        ):

            cur_info = (
                current_profile["categorical"]
                .get(feature)
            )

            if cur_info is None:
                continue

            ref_dist = (
                self._categorical_profile_to_distribution(
                    ref_info.get("top_k", [])
                )
            )

            cur_dist = (
                self._categorical_profile_to_distribution(
                    cur_info.get("top_k", [])
                )
            )

            metrics = self.metric_calculator.calculate(
                ref_dist,
                cur_dist,
                key="category",
                metrics=self.config.metrics_categorical,
            )

            primary_metric = (
                "psi"
                if "psi" in metrics
                else next(iter(metrics))
            )

            new_share = self._new_category_share(
                ref_info,
                cur_info,
            )

            severity_result = (
                self.severity_classifier.classify(
                    metrics,
                    primary_metric=primary_metric,
                    new_category_share=new_share,
                )
            )

            severity = severity_result["severity"]
            drift_detected = severity != "none"

            rank_score = (
                severity_result["severity_score"]
                * (
                    1
                    + (
                        sum(
                            1
                            for s in severity_result[
                                "metric_severities"
                            ].values()
                            if self.severity_classifier.severity_score(s) >= 2
                        )
                        / max(len(metrics), 1)
                    )
                    if self.config.use_metric_consensus
                    else 0
                )
            )

            missing_delta = (
                cur_info["null_proportion"]
                - ref_info["null_proportion"]
            )

            recommendation = self._recommendation(
                severity,
                drift_detected,
                new_category_share=new_share,
                missing_delta=missing_delta,
            )

            rows.append({
                "feature_name": feature,
                "feature_type": "categorical",
                "primary_metric": primary_metric,
                "primary_value": float(metrics[primary_metric]),
                "psi": metrics.get("psi"),
                "js": metrics.get("js"),
                "hellinger": metrics.get("hellinger"),
                "ks_approx": metrics.get("ks_approx"),
                "chi_square": metrics.get("chi_square"),
                "severity": severity,
                "severity_score": int(
                    severity_result["severity_score"]
                ),
                "drift_detected": bool(drift_detected),
                "rank_score": float(rank_score),
                "missing_ref": float(
                    ref_info["null_proportion"]
                ),
                "missing_cur": float(
                    cur_info["null_proportion"]
                ),
                "missing_delta": float(missing_delta),
                "new_category_share": float(new_share),
                "recommendation": recommendation,
            })

        # Criar DataFrame final.
        schema = """
            feature_name string,
            feature_type string,
            primary_metric string,
            primary_value double,
            psi double,
            js double,
            hellinger double,
            ks_approx double,
            chi_square double,
            severity string,
            severity_score int,
            drift_detected boolean,
            rank_score double,
            missing_ref double,
            missing_cur double,
            missing_delta double,
            new_category_share double,
            recommendation string
        """

        feature_metrics = self.spark.createDataFrame(
            rows,
            schema=schema,
        )

        feature_metrics = (
            feature_metrics
            .orderBy(
                F.desc("severity_score"),
                F.desc("rank_score"),
            )
            .withColumn(
                "rank",
                F.row_number().over(
                    Window.orderBy(
                        F.desc("severity_score"),
                        F.desc("rank_score"),
                    )
                )
            )
        )

        # Dataset drift:
        # percentual de features que apresentaram drift.
        summary = (
            feature_metrics
            .agg(
                F.count("*").alias("total_features"),
                F.sum(
                    F.col("drift_detected").cast("int")
                ).alias("drifted_features"),
                F.sum(
                    F.when(
                        F.col("severity").isin(
                            "high",
                            "critical",
                        ),
                        1,
                    ).otherwise(0)
                ).alias("high_or_critical_features"),
            )
            .withColumn(
                "drift_share",
                F.when(
                    F.col("total_features") > 0,
                    F.col("drifted_features")
                    / F.col("total_features"),
                ).otherwise(F.lit(0.0)),
            )
            .withColumn(
                "dataset_drift",
                F.col("drift_share")
                >= F.lit(
                    self.config.dataset_drift_min_share
                ),
            )
        )

        # Nível agregado do dataset.
        summary = summary.withColumn(
            "dataset_severity",
            F.when(
                F.col("high_or_critical_features") > 0,
                F.lit("high"),
            )
            .when(
                F.col("dataset_drift"),
                F.lit("medium"),
            )
            .when(
                F.col("drift_share") > 0,
                F.lit("low"),
            )
            .otherwise(
                F.lit("none")
            )
        )

        summary = summary.select(
            "total_features",
            "drifted_features",
            "high_or_critical_features",
            "drift_share",
            "dataset_drift",
            "dataset_severity",
        )

        recommendations = feature_metrics.select(
            "rank",
            "feature_name",
            "severity",
            "rank_score",
            "recommendation",
        )

        return DriftReport(
            feature_metrics=feature_metrics,
            dataset_summary=summary,
            recommendations=recommendations,
        )

    # -------------------------------------------------------------------------
    # SCHEMA VALIDATION SEM DATAFRAME DE REFERÊNCIA
    # -------------------------------------------------------------------------

    def _validate_current_schema(
        self,
        current_df: DataFrame,
    ) -> Dict[str, Any]:

        current_schema = {
            field.name: field.dataType.simpleString()
            for field in current_df.schema.fields
        }

        reference_columns = set(
            self.reference_df_schema.keys()
        )

        current_columns = set(
            current_df.columns
        )

        missing_columns = sorted(
            reference_columns - current_columns
        )

        unexpected_columns = sorted(
            current_columns - reference_columns
        )

        type_changes = []

        for feature in sorted(
            reference_columns & current_columns
        ):
            ref_type = self.reference_df_schema[feature]
            cur_type = current_schema[feature]

            if ref_type != cur_type:
                type_changes.append({
                    "feature": feature,
                    "reference_type": ref_type,
                    "current_type": cur_type,
                })

        return {
            "missing_columns": missing_columns,
            "unexpected_columns": unexpected_columns,
            "type_changes": type_changes,
        }

    @staticmethod
    def _reference_schema_df_placeholder(
        current_df: DataFrame,
    ) -> DataFrame:
        """
        Método auxiliar mantido apenas para compatibilidade estrutural.
        A implementação final de compare() utiliza o schema persistido,
        pois o dataset de referência bruto não é mantido pelo detector.
        """
        return current_df

    # -------------------------------------------------------------------------
    # PERSISTÊNCIA
    # -------------------------------------------------------------------------

    def _profile_to_dataframe(
        self,
        profile: Dict[str, Any],
    ) -> DataFrame:

        rows = []

        for feature, info in profile["numeric"].items():
            rows.append({
                "feature_name": feature,
                "feature_type": "numeric",
                "count": info["count"],
                "total_count": info["total_count"],
                "null_count": info["null_count"],
                "null_proportion": info["null_proportion"],
                "min": info["min"],
                "max": info["max"],
                "quantiles": [
                    float(x) for x in info["quantiles"]
                ],
                "bin_edges": [
                    float(x)
                    for x in info["bin_edges"]
                ]
                if info["bin_edges"] is not None
                else None,
                "histogram_json": str(
                    info["histogram"]
                ),
                "categories_json": None,
            })

        for feature, info in profile["categorical"].items():
            rows.append({
                "feature_name": feature,
                "feature_type": "categorical",
                "count": info["count"],
                "total_count": info["total_count"],
                "null_count": info["null_count"],
                "null_proportion": info["null_proportion"],
                "min": None,
                "max": None,
                "quantiles": None,
                "bin_edges": None,
                "histogram_json": None,
                "categories_json": str(
                    info["top_k"]
                ),
            })

        schema = """
            feature_name string,
            feature_type string,
            count long,
            total_count long,
            null_count long,
            null_proportion double,
            min double,
            max double,
            quantiles array<double>,
            bin_edges array<double>,
            histogram_json string,
            categories_json string
        """

        return self.spark.createDataFrame(
            rows,
            schema=schema,
        )

    def save_reference_profile(
        self,
        path: str,
        mode: str = "overwrite",
    ):
        if self.reference_profile is None:
            raise RuntimeError(
                "Nenhum reference_profile disponível. Execute fit()."
            )

        profile_df = self._profile_to_dataframe(
            self.reference_profile
        )

        self.persistence.save_dataframe(
            profile_df,
            path,
            mode=mode,
        )

    def save_report(
        self,
        report: DriftReport,
        path: str,
        mode: str = "overwrite",
    ):
        self.persistence.save_dataframe(
            report.feature_metrics,
            path,
            mode=mode,
        )

    def save_dataset_summary(
        self,
        report: DriftReport,
        path: str,
        mode: str = "overwrite",
    ):
        self.persistence.save_dataframe(
            report.dataset_summary,
            path,
            mode=mode,
        )


# =============================================================================
# EXEMPLO DE UTILIZAÇÃO
# =============================================================================

if __name__ == "__main__":

    spark = (
        SparkSession.builder
        .appName("SparkDataDriftDetectorExample")
        .getOrCreate()
    )

    # -------------------------------------------------------------------------
    # Exemplo de configuração
    # -------------------------------------------------------------------------

    config = DriftConfig(
        numeric_features=[
            "idade",
            "renda",
            "score",
        ],
        categorical_features=[
            "uf",
            "segmento",
            "canal",
        ],
        n_bins=10,
        relative_error=0.01,
        categorical_top_k=50,
        metrics_numeric=[
            "psi",
            "js",
            "hellinger",
            "ks_approx",
        ],
        metrics_categorical=[
            "psi",
            "js",
            "hellinger",
            "chi_square",
        ],
        psi_thresholds=(0.10, 0.25),
        bounded_thresholds=(0.05, 0.10, 0.20),
        dataset_drift_min_share=0.30,
        eps=1e-6,
    )

    # -------------------------------------------------------------------------
    # Criar detector
    # -------------------------------------------------------------------------

    detector = SparkDataDriftDetector(
        config=config,
        spark=spark,
        persistence=PersistenceAdapter(
            format="parquet"
        ),
    )

    # -------------------------------------------------------------------------
    # Dados
    #
    # Substitua pelas suas fontes reais:
    #
    # reference_df = spark.read.parquet(...)
    # current_df = spark.read.parquet(...)
    # -------------------------------------------------------------------------

    reference_df = spark.createDataFrame(
        [
            (25, 3000.0, 700.0, "DF", "PF", "APP"),
            (30, 5000.0, 720.0, "DF", "PF", "APP"),
            (42, 9000.0, 760.0, "SP", "PJ", "WEB"),
            (35, 7000.0, 740.0, "DF", "PF", "APP"),
            (50, 12000.0, 800.0, "SP", "PJ", "AGENCIA"),
        ],
        [
            "idade",
            "renda",
            "score",
            "uf",
            "segmento",
            "canal",
        ],
    )

    current_df = spark.createDataFrame(
        [
            (26, 10000.0, 680.0, "DF", "PF", "APP"),
            (31, 15000.0, 690.0, "SP", "PJ", "WEB"),
            (45, 20000.0, 700.0, "SP", "PJ", "WEB"),
            (39, 18000.0, 710.0, "RJ", "MPE", "APP"),
            (52, 25000.0, 720.0, "RJ", "PJ", "AGENCIA"),
        ],
        [
            "idade",
            "renda",
            "score",
            "uf",
            "segmento",
            "canal",
        ],
    )

    # -------------------------------------------------------------------------
    # FIT
    # -------------------------------------------------------------------------

    detector.fit(reference_df)

    # -------------------------------------------------------------------------
    # COMPARE
    # -------------------------------------------------------------------------

    report = detector.compare(current_df)

    report.show(
        n=100,
        truncate=False,
    )

    # -------------------------------------------------------------------------
    # Persistência
    # -------------------------------------------------------------------------

    # detector.save_reference_profile(
    #     "/tmp/drift/reference_profile"
    # )

    # detector.save_report(
    #     report,
    #     "/tmp/drift/report"
    # )

    # detector.save_dataset_summary(
    #     report,
    #     "/tmp/drift/dataset_summary"
    # )

    spark.stop()
