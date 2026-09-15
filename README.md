# Arquitetura para detecção automática de Data Drift para BigData

# Resumo
Modelos de Machine Learning e processos de negócio dependem da estabilidade e da qualidade dos dados utilizados como entrada. Ao longo do tempo, entretanto, as características estatísticas e as distribuições desses dados podem sofrer alterações, fenômeno conhecido como Data Drift. Embora a ocorrência de drift não implique necessariamente degradação do desempenho de um modelo, sua identificação é importante para permitir o monitoramento e a investigação de possíveis impactos nos resultados produzidos. Neste trabalho, é apresentada uma arquitetura para diagnóstico de Data Drift voltada a ambientes de Big Data, contemplando o monitoramento de variáveis numéricas e categóricas. A proposta aborda a construção de perfis de referência e de produção, a aplicação de diferentes métricas estatísticas para identificação de alterações nas distribuições e a classificação da severidade do drift por variável e para o conjunto de dados. Como parte da solução, são desenvolvidos componentes em PySpark para realizar o perfilamento, o cálculo das métricas e a geração de relatórios de diagnóstico. Dessa forma, busca-se apresentar uma abordagem distribuída e aplicável a grandes volumes de dados, contribuindo para a identificação de mudanças nos padrões dos dados e para o monitoramento contínuo de modelos de Machine Learning.

# Abstract
Title: Architecture for automatic data drift detection in Big Data.

Machine Learning models and business processes depend on the stability and quality of the input data used. Over time, however, the statistical characteristics and distributions of these data can undergo changes, a phenomenon known as Data Drift. Although the occurrence of drift does not necessarily imply performance degradation of a model, its identification is important to enable monitoring and investigation of potential impacts on the generated results. In this work, an architecture for Data Drift diagnosis targeted at Big Data environments is presented, covering the monitoring of numerical and categorical variables. The proposal addresses the construction of reference and production profiles, the application of different statistical metrics to identify distribution changes, and the classification of drift severity per variable and for the dataset as a whole. As part of the solution, components are developed in PySpark to perform profiling, metric calculation, and the generation of diagnostic reports. Thus, the objective is to present a distributed approach applicable to large data volumes, contributing to the identification of changes in data patterns and the continuous monitoring of Machine Learning models.

Keywords: Data Drift, Machine Learning, Big Data, PySpark, Data Monitoring

# Introdução

Dentro do mundo das empresas as tomadas de decisões sempre vem com dados. Geralmente, esses dados eles são a entrada de algum processo negocial ou um modelo de machine learning. Vamos pegar um exemplo, no ano de 2020, ano da Pandemia do COVID-19 os atendimentos da agência bancárias foram diminuidos aos montes, enquanto, o número de canais digitais aumentou. Isso é uma mudança significativa na distribuição dos dados. Se eu tiver um modelo de Machine Learning de previsão de Numerário das Agências, com certeza, eu teria que rever os dados de entrada nesse cenário. 

O conceito dessa mudança dos dados de entrada de um modelo de Machine Learning é conhecido como Data Drift (Referência: A Survey on Concept Drift Adaptation (2014)). No estudo de Gama (Referência: A Survey on Concept Drift Adaptation (2014)), ele define que o drift pode acontecer de diferentes formas. Pode acontecer de forma abrupta, quando de uma hora para outra a distribuição dos dados muda de repente, o exemplo da pandemia pode ser interessante para mostrar essa mudança. Outra forma de Data Drift é ser incremental/gradual, ou seja, vai mudando de pouco a pouco. Esse tipo de drift é um tipo de drift mais suave e mais difícil de identificar. Um dos cuidados que deve ter é não confundir Outlier com Data Drift. Isso é um dos casos mais cuidadosos que devem ser identificados em modelos de Machine Learning. Na figura abaixo, ilustra bem os significados de cada um de drift. 

Os modelos de Machine Learning funciona da seguinte forma, recebe os dados de entrada e retorna os dados de saída. Os dados de entrada variam de diferentes formatos, podem ser textos (que são convertidos em vetores), podem ser imagens, mas na maioria dos trabalhos os modelos de machine learning recebe dados númericos e categóricos. Como falado, se acontece uma mudança significativa nos dados de entrada, pode ocorrer que o resultado do modelo de Machine Learning se degrada e dessa forma é interessante identificar qual feature do modelo fez com que tivesse essa mudança de resultado. 

Tem que deixar claro também que tem dois tipos de drifts importantes. No que iremos tratar nesse trabalho é apenas o Data Drift, ou seja, quando a distribuição de entrada P(x) em produção difere da distribuição P(x) de treino. E além disso, quando isso impacta no resultado do modelo. No estudo de Gama e Ackerman et al. (2021), trazem a definição de Concept Drift, ou seja,  quando a relação estatística entre X e Y muda, mesmo a distribuição de entrada P(X) continua a mesma. Na forma matemática, a definição seria da seguinte forma, P(X_train) = P(X_prod), entretanto, P(Y/X_train) "sinal de diferente" P(Y/X_prod). Dentro do estudo de Ackerman et al. (2021), apresenta técnicas e métricas para verificar esses tipo de casos. 

Dentro dos estudos apresentados, oferecem diversas formas de identificar o drift. No caso de Ackerman et al. (2022), ele cria método com a diminuição de componentes para analisar o drift e impacto no modelo. No nosso estudo, iremos focar apenas no primeiro conceito apresentado antes, mudança de P(X) que impacta o modelo de machine learning, porém, o foco será construir uma arquitetura para o ambiente BigData visto que no ano de 2026 o número de dados movidos no mundo deve chegar a (Fonte). Devido a isso, iremos construir essa infraestrutura dentro do ambiente do Pyspark. 

# Data Drift

Neste tópico vamos explicar de forma resumida um pouco dos tipos de data drift e se esses tipos de drift impactam um modelo de ML. Como visto no tópico anterior, temos diferentes tipos de drifts e vamos explorar cada um deles. 

O primeiro Drift e o qual iremos trabalhar dentro do artigo é o Data/Covariate, quando olhamos apenas para a distribuição de uma unica variavel, ou seja, em forma matemática é quando seu P(X_treino) difere estatistcamente do P(X_prod). Para mostrar isso de forma mais lúcida, vamos trazer o exemplo de Gama et al. (2014), ele traz um exemplo de um modelo de credit scoring que é treinado com dados históricos em que a variável "renda mensal" segue uma distribuição relativamente estável, centrada em uma faixa específica. Após um evento macroeconômico (por exemplo, um período de inflação alta ou uma crise que reduz o poder de compra), a distribuição dessa única variável se desloca: a média cai, a variância aumenta, ou a forma da distribuição muda (passa a ter mais assimetria). Isso caracteriza um drift univariado de covariável: a distribuição de entrada P(renda) mudou, mesmo que a relação entre renda e risco de inadimplência (P(Y|X)) permaneça a mesma. Como o modelo foi treinado sob a distribuição antiga, seu desempenho degrada porque passa a operar numa região do espaço de entrada pouco representada nos dados de treino. Esse tipo de exemplo (mudança em P(X) sem necessariamente mudar a relação com o alvo) está alinhado à distinção clássica entre drift real (mudança em P(Y|X)) e drift virtual/covariate shift (mudança apenas em P(X)) discutida na literatura de concept drift. 

A figura abaixo ilustra um pouco sobre o assunto que estamos falando. Para esses casos, a melhor forma para verificar se uma distribuição muda de uma para outra é a função de probabilidade acumulada, diferentemente, do Histograma.

Outro ponto é o Concept Drift que foi colocado já no tópico anterior.  


(Gama et al., 2014)

GAMA, J. et al. A survey on concept drift adaptation. ACM Computing Surveys, v. 46, n. 4, p. 1-37, 2014.

Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). A change in user's interests when following an online news stream is described as concept drift, and the survey distinguishes cases where the conditional distribution of the target given the input changes from cases where the input distribution itself may shift while that relationship stays the same. ACM Computing Surveys, 46(4), Artigo 44, 1–37.




# Info sobre a Dissertação

Nome Orientador:
Carlos Henrique Rodrigues Sarro

Email Orientador: 
chsarro@gmail.com