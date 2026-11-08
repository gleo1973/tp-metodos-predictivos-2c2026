# Trabajo Práctico – Métodos Predictivos para la Gestión

## Popularidad de artículos digitales: análisis estadístico y modelos predictivos

**Tecnicatura Universitaria en Gestión y Análisis de Datos – FCE, UBA**  
**Segundo cuatrimestre de 2026**

---

## Resumen

Este trabajo analiza la relación entre las características de artículos publicados por Mashable y su popularidad, medida mediante la cantidad de veces que fueron compartidos en redes sociales (`shares`). Se utiliza el conjunto de datos *Online News Popularity*, disponible en el repositorio UCI Machine Learning Repository.

El análisis combina exploración y limpieza de datos, muestreo, inferencia estadística y modelos predictivos. Debido a la marcada asimetría de `shares`, se incorpora la transformación `shares_log = log(1 + shares)` como variable objetivo para los modelos. Se estima una regresión lineal múltiple mediante mínimos cuadrados ordinarios (OLS) y se desarrolla un modelo Random Forest Regressor, cuya configuración se optimiza mediante búsqueda aleatoria y validación cruzada.

En la comparación sobre el mismo conjunto de prueba, Random Forest obtiene un desempeño ligeramente superior al de OLS: MAE de 0,640 frente a 0,657; RMSE de 0,858 frente a 0,879; y R² de 0,141 frente a 0,100. Sin embargo, la diferencia es moderada y el modelo presenta una brecha entre entrenamiento y prueba, por lo que su capacidad de generalización es limitada. Los resultados permiten identificar asociaciones y patrones predictivos, pero no establecer relaciones causales.

## 1. Descripción y fuente del dataset

Se utiliza el dataset *Online News Popularity*, publicado en UCI Machine Learning Repository y elaborado a partir de artículos de Mashable. Cada observación representa un artículo e incluye variables sobre su contenido, estructura, temática, publicación, características lingüísticas y difusión.

La descarga utilizada inicialmente contiene 39.644 observaciones y 59 columnas, incluida la variable objetivo `shares`. Luego del proceso de revisión y limpieza se trabaja con 39.643 artículos.

La variable `shares` representa la cantidad de veces que un artículo fue compartido en redes sociales. Como es cuantitativa, el problema se aborda como una tarea de regresión supervisada.

**Fuente:** UCI Machine Learning Repository, *Online News Popularity*. Autores del dataset: Kelwin Fernandes, Pedro Vinagre, Paulo Cortez y Pedro Sernadela.

## 2. Carga, exploración inicial y limpieza

Se cargan las librerías y el dataset, se inspeccionan sus dimensiones, tipos de datos, nombres de columnas y estructura general. También se identifican las familias de variables y sus escalas.

La revisión de calidad contempla valores faltantes, registros duplicados, consistencia de variables binarias, indicadores de días de publicación y variables LDA. Se identifica una observación con valores incompatibles con la definición de ciertas variables de proporciones y con ausencia simultánea de información en variables LDA y de sentimiento. Tras revisar el registro, se excluye y la base de trabajo queda conformada por 39.643 observaciones.

Se identifican además campos no utilizados como predictores, como `url` y `timedelta`, por su función de identificación o por no representar características disponibles y pertinentes para el objetivo de modelado.

### 2.1 Detección de valores atípicos

El análisis de valores atípicos se aborda desde una perspectiva multivariada mediante tres métodos:

- **Distancia de Mahalanobis:** mide la distancia de una observación respecto del centro multivariado, teniendo en cuenta las escalas y covarianzas.
- **Local Outlier Factor (LOF):** identifica observaciones cuya densidad local difiere de la de sus vecinos. Requiere estandarizar las variables para evitar que las escalas dominen las distancias.
- **Isolation Forest:** utiliza divisiones aleatorias en árboles para identificar observaciones que pueden aislarse con mayor facilidad.

Estos procedimientos permiten detectar registros potencialmente atípicos. La identificación de un outlier no implica automáticamente que el dato sea erróneo ni que deba eliminarse: su tratamiento requiere considerar la definición de las variables y el contexto de cada observación.

## 3. Análisis exploratorio

### 3.1 Variable objetivo

La variable `shares` presenta una distribución fuertemente asimétrica hacia la derecha. En la base de 39.643 artículos, la media es aproximadamente 3.395 compartidos y la mediana es 1.400. La desviación estándar es cercana a 11.627 y el máximo alcanza 843.300 compartidos. La diferencia entre media y mediana y la presencia de valores extremos muestran que unos pocos artículos alcanzan una difusión excepcional.

Para reducir la asimetría y facilitar el modelado se utiliza `shares_log = log(1 + shares)`. La transformación no elimina la heterogeneidad, pero comprime los valores altos y permite trabajar con una escala menos dominada por los casos extremos.

### 3.2 Variables predictoras y relaciones

Se exploran características de estructura y extensión de los artículos, enlaces, imágenes, videos, palabras clave, canales temáticos, días de publicación, referencias a artículos propios y medidas de sentimiento y subjetividad.

Las correlaciones individuales entre los predictores y la variable objetivo son en general débiles. Esto sugiere que la popularidad no queda explicada por una única característica y que puede depender de la combinación de distintos factores.

El análisis de correlación y del factor de inflación de la varianza (VIF) identifica grupos de predictores correlacionados entre sí. Esto es especialmente relevante para la regresión lineal, porque la multicolinealidad puede aumentar la incertidumbre de los coeficientes y dificultar su interpretación. A partir del análisis y de criterios conceptuales se define un conjunto de 28 predictores para los modelos posteriores.

## 4. Definición del problema predictivo

La pregunta de investigación es: **¿en qué medida las características de un artículo —contenido, estructura, temática, sentimiento y popularidad previa de sus palabras clave— permiten predecir la cantidad de veces que será compartido en redes sociales?**

El objetivo general es estimar la popularidad de los artículos de Mashable y comparar dos enfoques complementarios:

1. Un modelo estadístico OLS, orientado a estudiar asociaciones lineales e interpretar coeficientes.
2. Un modelo de Machine Learning, orientado a capturar patrones predictivos, incluidas posibles relaciones no lineales e interacciones.

La unidad de análisis es el artículo. La variable objetivo modelada es `shares_log`, y se utilizan los mismos 28 predictores en los modelos comparados.

Las hipótesis iniciales se organizan en cinco grupos:

- **H1 – Multimedia:** más imágenes y videos se asocian con más compartidos.
- **H2 – Extensión:** la longitud del contenido se relaciona con la popularidad.
- **H3 – Temática:** la popularidad difiere según el canal temático y los temas del artículo.
- **H4 – Sentimiento:** el tono emocional se asocia con la cantidad de compartidos.
- **H5 – Popularidad previa de keywords:** las palabras clave con mayor popularidad histórica se asocian con más compartidos.

Estas hipótesis plantean asociaciones esperadas y no relaciones causales.

## 5. Muestreo e inferencia

Se extrae una muestra aleatoria simple sin reposición de 750 artículos de la población de 39.643 observaciones. La muestra se utiliza para estudiar su representatividad y desarrollar una primera estimación OLS.

Se comparan características de la muestra y de la población mediante medidas descriptivas y análisis de variables. También se estudian la media y la varianza de `shares`, así como el comportamiento de los promedios al considerar distintos tamaños muestrales.

La comparación permite observar que una muestra puede reproducir razonablemente algunos aspectos de la población sin representar con la misma precisión todas sus características, especialmente cuando se trata de una variable muy asimétrica y dispersa como `shares`.

## 6. Regresión lineal múltiple (OLS)

### 6.1 Preparación y estimación

Se prepara un conjunto común de 28 predictores, combinando variables cuantitativas y variables dummy para representar categorías. Se omiten categorías de referencia para evitar la trampa de las variables dummy. La variable dependiente es `shares_log`.

Se estima un primer modelo OLS con la muestra de 750 artículos y un segundo modelo con la base completa de 39.643 observaciones. En ambos casos se utilizan los mismos predictores.

### 6.2 Coeficientes y significatividad

Los coeficientes representan asociaciones entre cada predictor y `shares_log`, manteniendo constantes los demás predictores del modelo. Los signos indican la dirección de la asociación estimada. Los errores estándar, estadísticos t y p-valores permiten evaluar la incertidumbre y la significatividad estadística.

Se utiliza un nivel de significación del 5 %. Un p-valor inferior a 0,05 aporta evidencia de que el coeficiente es distinto de cero, condicionado al modelo especificado. La significatividad estadística no demuestra causalidad ni equivale necesariamente a importancia práctica.

### 6.3 Comparación entre muestra y población

En la muestra de 750 observaciones, el modelo presenta R² aproximado de 0,146 y R² ajustado de 0,112. En la base completa, el R² es aproximadamente 0,100 y el R² ajustado, 0,099.

La prueba F global resulta estadísticamente significativa en ambos modelos. El error estándar residual es cercano a 0,874 en la muestra y 0,882 en la base completa.

Estas métricas describen el ajuste dentro de los datos utilizados para estimar cada modelo. No deben confundirse con una evaluación de predicción sobre datos nuevos. Las diferencias entre muestra y población también muestran que el aumento del tamaño muestral no implica necesariamente un R² mayor.

## 7. Modelos de Machine Learning

### 7.1 Preparación y partición

Se utilizan los mismos 28 predictores y `shares_log` como variable objetivo. La base de 39.643 observaciones se divide en 80 % para entrenamiento y 20 % para prueba, con `random_state=42`. La partición produce 31.714 observaciones de entrenamiento y 7.929 de prueba.

El conjunto de prueba se reserva para evaluar el desempeño sobre observaciones que no se utilizaron durante el ajuste.

### 7.2 Primer modelo: Random Forest inicial

Se entrena un Random Forest Regressor con parámetros iniciales y se evalúa mediante validación cruzada de cinco particiones sobre el conjunto de entrenamiento. El RMSE promedio obtenido es 0,8702, con un desvío estándar de 0,0139 entre los folds. Este resultado funciona como referencia para la etapa de optimización.

### 7.3 Segundo modelo: optimización de hiperparámetros

Se aplica `RandomizedSearchCV` con diez configuraciones aleatorias y cinco particiones de validación cruzada. La búsqueda utiliza MSE negativo como criterio de selección. La mejor configuración guardada utiliza 300 árboles (`n_estimators=300`), profundidad máxima de 15 (`max_depth=15`), mínimo de 5 observaciones para dividir un nodo (`min_samples_split=5`) y mínimo de 4 observaciones por hoja (`min_samples_leaf=4`).

### 7.4 Evaluación del modelo optimizado

Las métricas del Random Forest optimizado son:

| Métrica | Train | Test |
|---|---:|---:|
| MAE | 0,479 | 0,640 |
| MSE | 0,420 | 0,737 |
| RMSE | 0,648 | 0,858 |
| R² | 0,515 | 0,141 |

El desempeño es mejor en entrenamiento que en prueba. La diferencia entre ambos conjuntos sugiere cierto grado de sobreajuste y una capacidad de generalización limitada.

### 7.5 Comparación OLS vs. Random Forest

Para que la comparación sea justa, OLS también se ajusta únicamente con el conjunto de entrenamiento y ambos modelos se evalúan sobre el mismo conjunto de prueba.

| Métrica | OLS | Random Forest |
|---|---:|---:|
| MAE | 0,657 | **0,640** |
| RMSE | 0,879 | **0,858** |
| R² | 0,100 | **0,141** |

Random Forest presenta errores ligeramente menores y un R² algo mayor. La mejora es moderada y no implica que el modelo explique una proporción alta de la variabilidad de la popularidad.

### 7.6 Importancia de variables

En el Random Forest, las variables con mayor importancia relativa guardada son `kw_avg_avg` (0,195), `self_reference_avg_sharess` (0,107), `global_subjectivity` (0,070), `global_sentiment_polarity` (0,069), `n_unique_tokens` (0,068), `average_token_length` (0,065) y `n_tokens_content` (0,061).

La importancia basada en reducción de impureza describe la participación de las variables en las divisiones de los árboles. No indica por sí sola la dirección de la relación, no demuestra causalidad y puede distribuirse entre predictores correlacionados.

## 8. Comparación contextual: estadística y Machine Learning

OLS y Random Forest cumplen funciones complementarias. OLS facilita la interpretación de asociaciones lineales mediante coeficientes, errores estándar y pruebas de significatividad. Random Forest permite representar relaciones no lineales e interacciones, aunque su interpretación es menos directa.

En la partición evaluada, Random Forest supera ligeramente a OLS en MAE, RMSE y R². Sin embargo, su R² de prueba es 0,141 y existe una diferencia considerable entre Train y Test. Por lo tanto, no corresponde afirmar que el modelo tenga alta precisión ni que esté listo para producción.

Antes de una implementación real sería necesario validar el modelo con datos nuevos, considerar una partición temporal, comparar con una referencia simple y monitorear el desempeño a lo largo del tiempo.

## 9. Conclusiones

### 9.1 Principales hallazgos

La popularidad de los artículos es heterogénea y `shares` presenta una fuerte asimetría. Las variables consideradas muestran asociaciones individuales generalmente débiles, lo que indica que la difusión depende de múltiples características y de su combinación.

OLS permitió analizar asociaciones y significatividad estadística. Random Forest permitió explorar patrones predictivos más complejos e identificar la importancia relativa de los predictores. Entre estos, `kw_avg_avg` fue el de mayor importancia en el modelo optimizado.

En la evaluación sobre el mismo conjunto de prueba, Random Forest obtuvo mejores métricas que OLS, aunque la mejora fue moderada y la capacidad predictiva general continuó siendo limitada.

### 9.2 Evaluación de las hipótesis

- **H1 – Multimedia:** evidencia favorable en OLS, ya que `num_imgs` y `num_videos` presentan coeficientes positivos y estadísticamente significativos en la base completa. Se trata de asociaciones, no de efectos causales.
- **H2 – Extensión:** evidencia parcial. `n_tokens_content` no resulta significativo al 5 % en OLS sobre la base completa, mientras que `average_token_length` sí presenta asociación significativa. Algunas variables de texto también tienen importancia predictiva en Random Forest.
- **H3 – Temática:** evidencia favorable. Varias categorías temáticas presentan diferencias estadísticamente significativas respecto de la categoría de referencia utilizada en el modelo.
- **H4 – Sentimiento:** evidencia parcial. `global_subjectivity` es significativa en OLS, mientras que `global_sentiment_polarity` no lo es al 5 %; ambas aparecen entre las variables importantes de Random Forest.
- **H5 – Popularidad previa de keywords:** evidencia favorable. `kw_avg_avg` presenta asociación positiva y significativa en OLS y es el predictor de mayor importancia relativa en Random Forest.

### 9.3 Limitaciones

El dataset corresponde a un único medio digital y a un período histórico específico, por lo que los resultados no se generalizan automáticamente a otros medios o períodos. `shares` mide compartidos, pero no representa todas las dimensiones de la popularidad, como visualizaciones, alcance o lectores únicos.

Los datos son observacionales y no permiten establecer causalidad. OLS presenta capacidad explicativa limitada y Random Forest muestra una brecha entre entrenamiento y prueba. La evaluación se realizó mediante una partición aleatoria y validación cruzada sobre entrenamiento; no se efectuó validación temporal ni externa. Tampoco se incorporó un modelo de referencia simple, por lo que no se cuantificó la mejora respecto de una predicción básica.

### 9.4 Variables adicionales

El modelo podría enriquecerse con información agregada y anonimizada sobre audiencia, impresiones, visualizaciones, clics, alcance, características del autor, horario y contexto temporal de publicación, plataformas de difusión, características semánticas del contenido y tendencias externas.

Para evitar fuga de información, las variables adicionales deberían estar disponibles en el momento en que se realiza la predicción y no depender de resultados posteriores a la publicación.

### 9.5 Conclusión final

El trabajo permitió integrar análisis exploratorio, inferencia estadística y Machine Learning para estudiar la popularidad de artículos digitales. Los resultados muestran que ciertas características del contenido, la temática, los elementos multimedia y la popularidad histórica de las palabras clave se asocian con la cantidad de compartidos.

OLS ofrece mayor facilidad de interpretación estadística, mientras que Random Forest permite explorar patrones predictivos más complejos. Aunque Random Forest obtuvo mejores resultados en la comparación realizada, la diferencia fue moderada y su desempeño en prueba indica que todavía existen limitaciones importantes.

Por lo tanto, los modelos constituyen herramientas de apoyo para comprender y estimar patrones de difusión, pero no permiten garantizar el éxito de una publicación ni reemplazan el criterio editorial. La capacidad predictiva futura dependerá de la calidad y amplitud de los datos, de la validación con nuevas observaciones y del monitoreo continuo del modelo.

---

## Nota de consistencia antes de entregar

Este informe sintetiza el contenido y los resultados guardados en el notebook revisado. Antes de entregar, ejecuten el notebook completo desde el inicio y actualicen las tablas, gráficos y cifras si cambian. Verifiquen también que la sección 9 del notebook tenga incorporado el desarrollo de conclusiones y que los títulos del informe coincidan con la versión final acordada por el grupo.
