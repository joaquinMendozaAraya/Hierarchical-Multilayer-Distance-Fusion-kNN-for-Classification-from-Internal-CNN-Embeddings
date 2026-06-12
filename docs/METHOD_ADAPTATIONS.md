# Adaptaciones de competidores que deben declararse en el articulo

Fecha: 2026-06-05.

## Proposito

Este documento registra las diferencias entre los algoritmos descritos en sus
fuentes originales y las implementaciones utilizadas en la comparacion contra
WinMax.

Estas diferencias deben explicitarse en metodologia, protocolo experimental o
limitaciones. No se deben presentar las implementaciones como reproducciones
identicas cuando los artefactos disponibles obligan a adaptar la entrada.

## Condicion comun del experimento

Los perfiles disponibles contienen vectores obtenidos mediante global average
pooling (GAP) en endpoints semanticos de stages CNN. No contienen todos los
mapas espaciales ni las activaciones de cada bloque interno.

Todos los metodos reciben:

- el mismo backbone congelado;
- los mismos embeddings multicapa;
- los mismos splits train/validation/test;
- las mismas etiquetas;
- ninguna fuente externa de datos;
- ningun backbone alternativo.

Las transformaciones y clasificadores se ajustan con train. Los
hiperparametros se seleccionan con validation. Test se evalua una sola vez
despues de congelar la configuracion elegida.

## Adaptaciones por metodo

### MLCFF

Fuente: Fradi, Fradi y Dugelay, 2021.

El metodo original extrae activaciones de multiples capas convolucionales y
realiza pooling por canal. La implementacion actual utiliza los endpoints de
stages disponibles, cuyo GAP ya fue calculado durante la extraccion.

Se conservan:

1. normalizacion L2 independiente por capa;
2. concatenacion de todas las representaciones;
3. PCA;
4. LDA;
5. SVM lineal multiclase one-vs-one.

Denominacion recomendada:

`faithful stage-level adaptation of MLCFF`

No denominarla `exact reproduction`.

### Head2Toe

Fuente: Evci et al., 2022.

El metodo original utiliza representaciones de todos los bloques y controla la
dimension espacial mediante pooling. La implementacion actual recibe vectores
GAP de endpoints de stages.

Se conservan:

1. normalizacion unitaria por representacion;
2. concatenacion de features;
3. probe lineal con penalizacion group-lasso L2,1;
4. ranking por norma de las conexiones de cada feature;
5. seleccion de una fraccion top-K;
6. reentrenamiento de una cabeza lineal sobre las features seleccionadas.

La cabeza final sin regularizacion se aproxima mediante regresion logistica con
`C=1e6`.

Denominacion recomendada:

`stage-level adaptation preserving the Head2Toe selection mechanism`

### EasyMKL

Fuente: Aiolli y Donini, 2015.

Cada capa genera un kernel lineal despues de normalizar sus embeddings con
norma L2. EasyMKL aprende los pesos de los kernels exclusivamente con train.

Adaptaciones:

1. la comparacion principal utiliza kernels lineales, no una busqueda abierta
   sobre todas las posibles familias de kernels;
2. para controlar memoria puede utilizarse un subconjunto estratificado de
   train;
3. el limite predeterminado es 1000 muestras, pero puede desactivarse usando
   el conjunto completo;
4. el numero efectivo de muestras debe reportarse en cada resultado.

El limite de muestras es un adaptador computacional y no debe ocultarse.

### Regularized MAXVAR-GCCA

Las capas se interpretan como vistas pareadas de una misma imagen.

Adaptaciones:

1. se ajusta PCA por vista utilizando solo train para controlar rango y
   estabilidad numerica;
2. se emplea regularizacion en las transformaciones fuera de muestra;
3. las representaciones proyectadas de las capas de una imagen se promedian
   antes de aplicar kNN.

El objetivo MAXVAR se conserva. El promedio de las vistas proyectadas y el
mapeo fuera de muestra son decisiones de esta implementacion.

### GMLDA

Fuente: Sharma et al., 2012.

El escenario original considera vistas o modalidades diferentes. En este
experimento, los stages CNN se tratan como vistas pareadas.

Se conservan:

1. el problema generalizado de autovalores de GMA;
2. los objetivos de dispersion LDA;
3. las medias de clase como ejemplares entre vistas.

Adaptaciones:

1. PCA por vista ajustado solo con train;
2. regularizacion para matrices singulares;
3. promedio de las proyecciones de las capas de una misma imagen.

La interpretacion de capas CNN como vistas es una hipotesis experimental del
presente trabajo, no el escenario original del paper.

### MvDA

Fuente: Kan et al., 2016.

Se preservan las matrices de dispersion multivista y el problema generalizado
de autovalores descrito en las ecuaciones 7-12 del paper.

Adaptaciones:

1. los stages CNN se consideran vistas pareadas;
2. se aplica PCA por vista antes de construir las matrices de dispersion;
3. se agrega regularizacion para estabilidad;
4. las vistas proyectadas de cada imagen se promedian antes de kNN.

Debe describirse como una adaptacion del algoritmo MvDA a vistas
representacionales internas.

### Concatenation + NCA + kNN

Fuente del algoritmo NCA: Goldberger et al., 2004.

No corresponde a un metodo multicapa original completo. Es un baseline
compuesto:

1. normalizacion L2 por capa;
2. concatenacion fija de todas las capas;
3. PCA ajustado con train;
4. NCA supervisado;
5. kNN.

Por costo, NCA puede ajustarse con un subconjunto estratificado de train. El
tamano efectivo debe reportarse. La transformacion seleccionada se aplica
despues al conjunto train completo para entrenar kNN.

Denominacion recomendada:

`all-layer concatenation followed by NCA and kNN`

No presentarlo como una reproduccion de un paper de fusion multicapa.

### Uniform layer soft vote

Es un control de fusion a nivel de decision:

1. se ajusta un clasificador independiente por cada capa fija;
2. se promedian uniformemente sus probabilidades;
3. no se seleccionan capas;
4. no se utilizan pesos, ranking ni mecanismos de WinMax.

No corresponde a la variante soft-vote interna desarrollada anteriormente
dentro del framework propuesto.

### Uniform multi-view linear kernel SVM

Es un control de kernels multiples uniformes.

Se implementa mediante concatenacion de representaciones L2 escaladas por
`1/sqrt(numero de vistas)`. Esta representacion es matematicamente equivalente
al promedio uniforme de los kernels lineales normalizados de las capas.

Debe explicitarse esta equivalencia para evitar que la implementacion parezca
un algoritmo de pesos de kernel aprendidos.

### Proposed method - WinMax

WinMax se recalcula independientemente en cada combinacion dataset-backbone.

Se seleccionan con validation:

1. ranking de capas;
2. subconjunto de capas;
3. numero de vecinos;
4. pesos de fusion de distancias.

Los competidores externos no reciben el ranking, los pesos ni las decisiones
de WinMax. Test se calcula solo para la configuracion WinMax seleccionada.

## Texto sugerido para metodologia

> Since the available layer bank contains global-average-pooled stage
> endpoints rather than every raw convolutional activation, methods originally
> defined over all intermediate feature maps were implemented as stage-level
> adaptations. Their central optimization, selection, and fusion mechanisms
> were preserved, while numerical regularization and hyperparameters were
> determined exclusively from the training and validation partitions.

Texto adicional para GMLDA y MvDA:

> For multi-view projection methods, intermediate CNN stages were treated as
> paired representational views of the same image. This extends the original
> cross-view formulation to internal network representations; projected views
> were aggregated uniformly before classification.

Texto adicional para limites computacionales:

> When the full quadratic or kernel optimization exceeded the predefined
> computational budget, a deterministic stratified subset of the training
> partition was used to fit the projection or kernel weights. The effective
> fitting sample size is reported for every affected experiment.

## Informacion que debe reportarse

Para cada metodo:

- capas y dimensiones utilizadas;
- dimension final;
- hiperparametros seleccionados;
- numero de candidatos de validacion;
- tamano total de train;
- tamano efectivo de ajuste si hubo limite computacional;
- tipo y valor de regularizacion;
- accuracy, macro-F1 y balanced accuracy;
- delta respecto de HMDF-kNN en el mismo contexto;
- tiempo de ajuste;
- etiqueta de fidelidad de la implementacion.

## Limitacion pendiente

Los perfiles importados no siempre contienen IDs persistentes de las imagenes.
Actualmente se verifican dimensiones, etiquetas, clases y valores finitos, pero
la deteccion directa de imagenes duplicadas entre splits queda marcada como
`unavailable_no_saved_sample_ids` cuando no existen `paths.npy` o `paths.csv`.

## Rutas relacionadas en este repositorio

- Notebook introductorio:
  `notebooks/HMDF_kNN_friendly_walkthrough.ipynb`
- Implementaciones:
  `src/raw_multiview_competitors.py`
- Runner:
  `pipelines/run_raw_literature_multidim_competitors.py`
- Protocolo general:
  `docs/METHOD_FIDELITY_AND_PROTOCOL.md`
