# Data Drift Detector for Big Data

Algoritmo para **detecção e diagnóstico de Data Drift em ambientes de Big Data**, desenvolvido em **Python e Apache PySpark**.

O projeto foi desenvolvido como parte do trabalho de conclusão de curso **"Arquitetura para detecção automática de Data Drift para BigData"**, com o objetivo de identificar alterações nas distribuições das variáveis de entrada de modelos de Machine Learning e fornecer informações para apoiar o monitoramento dos dados.

## Objetivo

O algoritmo compara dois conjuntos de dados:

- **Referência:** normalmente os dados utilizados no treinamento do modelo;
- **Produção:** dados atuais utilizados pelo modelo.

A partir dessa comparação, são identificadas alterações nas distribuições das variáveis numéricas e categóricas, calculadas métricas de drift e atribuídos níveis de severidade por variável e para o conjunto de dados.

## Arquitetura

O detector foi organizado em componentes independentes:

```text
SparkDataDriftDetector
│
├── DriftConfig
├── SchemaValidator
├── FeatureProfiler
│   ├── NumericProfiler
│   └── CategoricalProfiler
├── MetricCalculator
│   ├── PSI
│   ├── Jensen-Shannon
│   ├── Hellinger
│   ├── KS Aproximado
│   └── Chi-Square
├── SeverityClassifier
├── DriftReport
└── PersistenceAdapter
```

Cada componente possui uma responsabilidade específica, permitindo a evolução e manutenção do algoritmo de forma modular.

## Fluxo

O processamento segue as seguintes etapas:

```text
Dados de Referência
        │
        ▼
Validação do Schema
        │
        ▼
Perfil Estatístico
        │
        ├──────────────┐
        ▼              ▼
   Numéricas      Categóricas
        │              │
        └──────┬───────┘
               ▼
       Cálculo das Métricas
               │
               ▼
      Classificação de Severidade
               │
               ▼
         Drift Report
```

### 1. Validação

Verifica possíveis diferenças estruturais entre os conjuntos de dados, incluindo:

- colunas ausentes;
- alterações de tipos;
- valores ausentes;
- categorias inesperadas.

### 2. Perfilamento

São construídos perfis estatísticos resumidos dos dados.

Para variáveis numéricas são utilizados histogramas e quantis. Para variáveis categóricas são calculadas frequências e proporções.

### 3. Métricas

O algoritmo utiliza cinco métricas principais:

| Métrica | Aplicação |
|---|---|
| PSI | Numéricas e categóricas |
| Kolmogorov-Smirnov (KS) | Variáveis numéricas |
| Jensen-Shannon (JS) | Numéricas e categóricas |
| Hellinger | Numéricas e categóricas |
| Qui-quadrado (χ²) | Variáveis categóricas |

As métricas são calculadas comparando a distribuição dos dados de referência com a distribuição dos dados monitorados.

### 4. Severidade

Os resultados são classificados em níveis de severidade para facilitar a priorização das variáveis que apresentam maiores indícios de alteração.

### 5. Relatório

Ao final, o algoritmo consolida:

- métricas por variável;
- indicação de drift;
- severidade;
- informações sobre o conjunto de dados;
- recomendações para análise.

## Instalação

Clone o repositório:

```bash
git clone https://github.com/Alexandretcosta/data_drift.git
cd data_drift
```

Instale as dependências necessárias para execução com Python e PySpark.

> Recomenda-se utilizar uma instalação do Apache Spark compatível com a versão do PySpark utilizada no projeto.

## Utilização

Com uma sessão Spark criada:

```python
from spark_data_drift_detector import (
    DriftConfig,
    SparkDataDriftDetector
)

categorical_columns = [
    "ProductCD",
    "P_emaildomain",
    "card6"
]

numerical_columns = [
    "V96",
    "V127",
    "V133",
    "V160"
]

config = DriftConfig(
    categorical_columns=categorical_columns,
    numerical_columns=numerical_columns
)

detector = SparkDataDriftDetector(
    spark=spark,
    config=config
)

result = detector.detect(
    reference_df=df_train,
    production_df=df_test
)

result.show(truncate=False)
```

Nesse exemplo:

- `df_train` representa os dados de referência;
- `df_test` representa os dados monitorados;
- as variáveis são separadas entre numéricas e categóricas;
- o Spark realiza o processamento distribuído.

## Exemplo utilizado no trabalho

Para validar o algoritmo, foi utilizada a base **IEEE-CIS Fraud Detection**, disponibilizada pela Vesta Corporation na plataforma Kaggle.

O experimento utilizou:

- **590.540 registros** de treinamento;
- **506.691 registros** de teste;
- **3 variáveis categóricas:** `ProductCD`, `P_emaildomain` e `card6`;
- **4 variáveis numéricas:** `V96`, `V127`, `V133` e `V160`.

O conjunto de treino foi utilizado como referência e o conjunto de teste como dados de monitoramento.

### Resultado do experimento

Nenhuma das sete variáveis analisadas apresentou drift no experimento realizado.

```text
Variáveis analisadas:       7
Variáveis com drift:        0
Variáveis com alta crítica: 0
Proporção com drift:        0%
Drift no conjunto:          Não
Severidade geral:           Nenhuma
```

O perfilamento da base de referência levou aproximadamente **3 minutos**, enquanto a comparação com o conjunto de teste e a geração do relatório levaram aproximadamente **3 minutos** na infraestrutura utilizada.

## Estrutura do projeto

```text
data_drift/
│
├── spark_data_drift_detector.py
├── rotina-teste.ipynb
├── README.md
└── ...
```

### `spark_data_drift_detector.py`

Implementação principal do detector de Data Drift.

### `rotina-teste.ipynb`

Notebook utilizado para carregar os dados, configurar o algoritmo e executar os testes.

## Escalabilidade

A solução utiliza PySpark para explorar o processamento distribuído dos dados.

As métricas baseadas em histogramas e contagens permitem trabalhar com perfis estatísticos compactos, evitando a necessidade de comparar individualmente todos os registros entre as bases. Essa característica é especialmente relevante em ambientes de grandes volumes de dados.

## Limitações

O experimento realizado teve como objetivo principal validar a implementação e o fluxo do algoritmo. Como as variáveis analisadas não apresentaram drift entre treino e teste, o experimento não permite avaliar completamente a sensibilidade do detector diante de mudanças reais.

Como próximos passos, podem ser realizados:

- testes com drift sintético;
- avaliação com outras bases de dados;
- inclusão de novas métricas;
- avaliação da sensibilidade das métricas;
- testes em ambientes de Big Data de maior escala.

Essas possibilidades são indicadas como trabalhos futuros no estudo.

## Tecnologias

- Python
- Apache PySpark
- Apache Spark
- Google Colab
- Jupyter Notebook

## Autor

**Alexandre Teixeira Costa**

Projeto disponível no GitHub:

`github.com/Alexandretcosta/data_drift`