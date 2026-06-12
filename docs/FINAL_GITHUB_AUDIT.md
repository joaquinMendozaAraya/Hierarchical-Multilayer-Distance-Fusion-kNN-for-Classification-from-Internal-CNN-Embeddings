# Auditoría final verificable de HMDF-kNN antes de GitHub

Fecha de auditoría: 12 de junio de 2026  
Proyecto: `C:\Users\jesqu\Desktop\Tesis\Tesis_2026`  
Manuscrito recibido:
`C:\Users\jesqu\Downloads\Hierarchical_Multilayer_Distance_Fusion_kNN_for_Brain_Tumor_MRI_Classification_from_Internal_CNN_Embeddings.zip`

## A. Resumen ejecutivo

### Estado general

El proyecto está cerca de una versión publicable, pero el ZIP recibido no debe
subirse directamente a GitHub. La auditoría encontró una discrepancia real en
el agregador de métricas: cuando una clase verdadera no era predicha, la
implementación manual dejaba su precisión y F1 como `NaN` y luego la excluía
del promedio macro. Esto inflaba el macro-F1 de algunos métodos, principalmente
en Brain MRI 44C.

El error fue corregido en:

`C:\Users\jesqu\Desktop\Tesis\Tesis_2026\scripts\build_result_matrices_for_paper.py`

Después de la corrección:

- las 630 filas método-contexto con predicciones guardadas reproducen
  exactamente accuracy, macro-F1 y balanced accuracy;
- los 45 resultados HMDF-kNN reproducen sus métricas por muestra;
- los 45 contextos HMDF-kNN seleccionan el máximo lexicográfico de validación;
- ninguno de los registros auditados declara uso de test para selección;
- los valores principales de HMDF-kNN no cambiaron;
- MAXVAR-GCCA cambió de `0.9519` a `0.9515` en macro-F1 medio;
- las selecciones de la referencia más fuerte por contexto y los conteos
  numéricos, prácticos y estadísticos no cambiaron.

La copia corregida y compilable está en:

`C:\Users\jesqu\Desktop\Tesis\Tesis_2026\audits\github_preparation_2026_06_12\manuscript_corrected_audit`

PDF principal corregido:

`C:\Users\jesqu\Desktop\Tesis\Tesis_2026\audits\github_preparation_2026_06_12\manuscript_corrected_audit\audit_build\main.pdf`

Suplemento corregido:

`C:\Users\jesqu\Desktop\Tesis\Tesis_2026\audits\github_preparation_2026_06_12\manuscript_corrected_audit\supplementary\build\context_robustness.pdf`

El ZIP original no fue modificado.

### Dictamen

**Resultados principales: verificados después de la reparación.**

**Repositorio GitHub: todavía no listo para publicación.**

Bloqueos principales:

1. falta construir una carpeta de release limpia;
2. falta reemplazar el marcador `[GitHub repository placeholder]`;
3. falta guardar predicciones por muestra del baseline
   `last_layer_selected_classifier`;
4. falta un README específico de HMDF-kNN;
5. falta un entorno reproducible fijado por versiones;
6. falta excluir aproximadamente 40 GB de datos y resultados intermedios;
7. falta eliminar de la copia de release fuentes LaTeX y tablas obsoletas;
8. la separación por paciente no está verificada;
9. el estudio principal usa una sola semilla.

## B. Claims cuantitativos verificados

La tabla exhaustiva de 40 claims se encuentra en:

`C:\Users\jesqu\Desktop\Tesis\Tesis_2026\audits\github_preparation_2026_06_12\claim_verification.csv`

| Claim | Paper/copia corregida | Recalculado | Estado |
|---|---:|---:|---|
| Datasets brain-MRI | 5 | 5 | Verified |
| Backbones | 9 | 9 | Verified |
| Contextos brain-MRI | 45 | 45 | Verified |
| Contextos HAM10000 separados | 9 | 9 | Verified |
| HMDF-kNN macro-F1 | 0.9616 | 0.9616074461 | Verified |
| HMDF-kNN accuracy | 0.9683 | 0.9682912755 | Verified |
| HMDF-kNN balanced accuracy | 0.9597 | 0.9596812622 | Verified |
| Softmax macro-F1 | 0.9326 | 0.9325708162 | Verified |
| Final-embedding block macro-F1 | 0.9470 | 0.9470028432 | Verified |
| Multilayer reference block macro-F1 | 0.9550 | 0.9550267592 | Verified |
| MAXVAR-GCCA macro-F1 | 0.9515 | 0.9514750181 | Verified after correction |
| MvDA macro-F1 | 0.9504 | 0.9504274741 | Verified |
| Final-embedding kNN macro-F1 | 0.9469 | 0.9468933514 | Verified |
| Numerical W/T/L | 28/0/17 | 28/0/17 | Verified |
| Practical W/T/L, margin 0.005 | 19/24/2 | 19/24/2 | Verified |
| Paired bootstrap W/NS/L | 10/28/0 over 38 | 10/28/0 | Verified |
| Holm-corrected W/NS/L | 2/36/0 over 38 | 2/36/0 | Verified |

### Discrepancia corregida

El ZIP original y una tabla obsoleta todavía contenían MAXVAR-GCCA = `0.9519`.
El valor respaldado por las predicciones es `0.951475...`, que se reporta como
`0.9515`.

La reparación del agregador detectó:

- 24 discrepancias de `test_f1_macro`;
- 21 discrepancias de `test_precision_macro`;
- 1 discrepancia de `val_f1_macro`;
- 1 discrepancia de `val_precision_macro`.

Registro:

`C:\Users\jesqu\Desktop\Tesis\Tesis_2026\audits\github_preparation_2026_06_12\master_vs_selected_json_mismatches.csv`

Backup anterior a la reparación:

`C:\Users\jesqu\Desktop\Tesis\Tesis_2026\audits\github_preparation_2026_06_12\pre_metric_fix_backup`

## C. Auditoría de datasets y contextos

### Contextos principales

Los 45 contextos principales existen y corresponden exactamente a:

`5 datasets brain-MRI x 9 backbones = 45 contextos`

Datasets:

| ID interno | Nombre usado en el paper | Clases | Train/val/test | Total |
|---|---|---:|---:|---:|
| `brain_tumor_mri_14c` | Brain tumor MRI 15C | 15 | 3120/667/669 | 4456 |
| `brain_tumor_mri_17c` | Brain tumor MRI 17C | 17 | 3111/668/669 | 4448 |
| `brain_tumor_mri_44c` | Brain tumor MRI 44C | 44 | 3137/671/671 | 4479 |
| `brain_tumor_mri_4c` | Brain tumor MRI 4C | 4 | 4760/840/1600 | 7200 |
| `sciencedb_brain_tumor_3c` | ScienceDB brain tumor 3C | 3 | 2145/459/460 | 3064 |

Backbones:

- ResNet-18;
- ResNet-34;
- ResNet-50;
- DenseNet-121;
- EfficientNet-B0;
- EfficientNet-B2;
- EfficientNet-B3;
- MobileNetV3-Large;
- ConvNeXt-Tiny.

### HAM10000

HAM10000 contiene 9 contextos adicionales, uno por backbone. Está correctamente
excluido de:

- los promedios principales brain-MRI;
- el conteo de 45 contextos;
- los tests bootstrap y Holm principales;
- las figuras agregadas del manuscrito.

Se mantiene como control externo:

- 7 clases;
- 7010/1502/1503 muestras train/val/test;
- total 10015.

### Integridad de splits

Los 54 perfiles evaluados, incluyendo HAM10000, registran:

- `sample_identity_audit=passed`;
- cero IDs duplicados entre splits;
- etiquetas alineadas;
- embeddings finitos.

Limitación no resuelta: los metadatos disponibles no permiten verificar
separación por paciente. El paper lo declara correctamente.

### Checkpoints

Existen 55 archivos `best.pt` y 55 `last.pt`, pero uno corresponde a una copia
de respaldo incompleta:

`brain_tumor_mri_14c\efficientnet_b0\brain_tumor_mri_14c__efficientnet_b0__full__seed42__img224.incomplete_backup_20260601_174816`

Por tanto, el conteo lógico sigue siendo 54 ejecuciones finales:

`6 datasets x 9 backbones = 54`

La carpeta `.incomplete_backup_*` debe excluirse del release.

Auditoría tabular:

`C:\Users\jesqu\Desktop\Tesis\Tesis_2026\audits\github_preparation_2026_06_12\dataset_context_audit.csv`

## D. Auditoría de HMDF-kNN y métodos

### Pipeline de entrenamiento y embeddings

Entrenamiento/fine-tuning:

`C:\Users\jesqu\Desktop\Tesis\Tesis_2026\google_drive_full_finetuning_pack\notebooks\RUN_FULL_FINETUNING_ALL_MODELS_COLAB.ipynb`

Reparación y extracción:

`C:\Users\jesqu\Desktop\Tesis\Tesis_2026\google_drive_full_finetuning_pack\notebooks\REPAIR_MISSING_AND_EXTRACT_EMBEDDINGS_COLAB.ipynb`

Extractor local:

`C:\Users\jesqu\Desktop\Tesis\Tesis_2026\scripts\extract_full_finetuned_colab_embeddings.py`

### Implementación exacta de HMDF-kNN

Implementación:

`C:\Users\jesqu\Desktop\Tesis\Tesis_2026\scripts\run_raw_literature_multidim_competitors.py`

Función actual:

`run_winmax_reference` (línea aproximada 364)

La función conserva un nombre histórico. Antes de GitHub conviene renombrarla
a `run_hmdf_knn` y mantener un alias temporal para compatibilidad.

La auditoría confirmó:

- entrada mediante embeddings internos guardados de backbones fine-tuned;
- normalización L2 por capa;
- evaluación independiente de cada capa en validación;
- ranking por validation macro-F1, balanced accuracy y accuracy;
- preferencia por menor `k` en empate de la evaluación individual;
- construcción de prefijos rankeados de hasta cuatro capas;
- pesos uniformes;
- pesos basados en score con potencias 0.5, 1 y 2;
- ocho candidatos Dirichlet reproducibles por prefijo;
- seed Dirichlet = 42;
- selección de `k` en `{1,3,5,7,11}`;
- selección lexicográfica usando validación;
- desempate final por menor dimensionalidad sumada;
- evaluación test solo después de congelar la configuración.

Cada contexto HMDF-kNN contiene exactamente 185 candidatos de validación.
Los archivos de candidatos no contienen columnas `test_*`.

Resultados de auditoría por contexto:

`C:\Users\jesqu\Desktop\Tesis\Tesis_2026\audits\github_preparation_2026_06_12\hmdf_context_recalculation.csv`

### Configuraciones seleccionadas

- 1 capa: 6 contextos;
- 2 capas: 11;
- 3 capas: 15;
- 4 capas: 13;
- `k=1`: 41 contextos;
- pesos uniformes: 7;
- score-based: 2;
- Dirichlet: 36.

### Métodos y trazabilidad

| Método | Contextos | Predicciones por muestra | Candidatos de validación | Estado |
|---|---:|---:|---:|---|
| Softmax head | 45 | 45 | N/A | Verified |
| Final-embedding selected classifier | 45 | 0 | matriz agregada | Metrics verified; predictions missing |
| kNN/SVM/RF/logreg/GMM/XGBoost/GNB finales | 45 cada uno | reconstrucción desde candidatos | candidatos guardados | Verified |
| Raw concat + linear | 45 | 45 | 45 | Verified |
| Concat + PCA + linear | 45 | 45 | 45 | Verified |
| Uniform soft vote | 45 | 45 | 45 | Verified |
| Uniform kernel SVM | 45 | 45 | 45 | Verified |
| MLCFF-style | 45 | 45 | 45 | Verified stage-level adaptation |
| Head2Toe-style | 45 | 45 | 45 | Verified stage-level adaptation |
| EasyMKL | 45 | 45 | 45 | Verified stage-level adaptation |
| MAXVAR-GCCA | 45 | 45 | 45 | Verified |
| GMLDA | 45 | 45 | 45 | Verified |
| MvDA | 45 | 45 | 45 | Verified |
| NCA + kNN | 45 | 45 | 45 | Verified |
| KMEx | 45 | 45 | 45 | Verified adaptation |
| HMDF-kNN | 45 | 45 | 45 | Verified |

Detalle:

`C:\Users\jesqu\Desktop\Tesis\Tesis_2026\audits\github_preparation_2026_06_12\method_artifact_audit.csv`

Recomputación desde predicciones:

`C:\Users\jesqu\Desktop\Tesis\Tesis_2026\audits\github_preparation_2026_06_12\method_prediction_recalculation.csv`

Recomputación de clasificadores finales:

`C:\Users\jesqu\Desktop\Tesis\Tesis_2026\audits\github_preparation_2026_06_12\final_classifier_recalculation.csv`

### Adaptaciones de literatura

Documentación principal:

- `C:\Users\jesqu\Desktop\Tesis\Tesis_2026\experiments\78_raw_literature_multidim_competitors\METHOD_FIDELITY_AND_PROTOCOL.md`
- `C:\Users\jesqu\Desktop\Tesis\Tesis_2026\experiments\78_raw_literature_multidim_competitors\PAPER_DISCLOSURE_METHOD_ADAPTATIONS.md`

La corrección aplicada cambió el límite EasyMKL documentado de 2500 a 1000,
coincidiendo con el ejecutor y los resultados.

Los métodos MLCFF, Head2Toe, EasyMKL y KMEx deben seguir describiéndose como
adaptaciones stage-level sobre las vistas de embeddings guardadas. No deben
presentarse como reproducciones completas end-to-end de sus papers originales.

## E. Auditoría de ablaciones

Los valores se reconstruyeron desde resultados reales:

| Variante | Macro-F1 | Capas medias | Estado |
|---|---:|---:|---|
| Best single layer | 0.9560 | 1.00 | Verified |
| All-layer uniform distance fusion | 0.9582 | 5.67 | Verified |
| Ranked-prefix uniform fusion | 0.9599 | 2.49 | Verified |
| Ranked-prefix score-weighted fusion | 0.9599 | 2.49 | Verified |
| Greedy score-weighted fusion | 0.9604 | 2.56 | Verified |
| Greedy uniform fusion | 0.9606 | 2.60 | Verified |
| HMDF-kNN | 0.9616 | 2.78 | Verified |

La redacción actual es metodológicamente correcta al indicar que estas variantes
pueden cambiar conjuntamente selección de capas, agregación y `k`. No deben
describirse como una prueba causal aislada de la familia de pesos.

## F. Auditoría de figuras, tablas y compilación

### Activos principales

| Activo | Fuente | Estado |
|---|---|---|
| Fig. 1, pipeline | imagen metodológica | Existe; coincide con el método |
| Fig. 2, capas internas | `prepare_hmdf_results_section.py` | Verified |
| Fig. 3, configuración | `prepare_hmdf_results_section.py` | Verified |
| Fig. 4, familias | `prepare_hmdf_results_section.py` | Verified; HAM excluido |
| Table I, datasets | auditoría de datasets | Verified |
| Table II, familias | inventario de métodos | Verified |
| Table III, ablación | resultados reales | Verified |
| Table IV, métodos completos | matriz corregida | Verified |
| Fig. S1, heatmap | resultados pareados | Verified |
| Fig. S2, confusión | predicciones por muestra | Verified |
| Fig. S3, geometría | distancias fusionadas | Verified; cualitativa |
| Table S1, dataset summary | resultados corregidos | Verified |
| Table S2, outcomes | bootstrap/holm | Verified |

Inventario:

`C:\Users\jesqu\Desktop\Tesis\Tesis_2026\audits\github_preparation_2026_06_12\figure_table_audit.csv`

### Compilación

El manuscrito principal compila en 13 páginas y el suplemento en 2 páginas
usando Tectonic.

No se detectaron:

- referencias indefinidas;
- citas indefinidas;
- cajas `overfull`;
- texto o figuras fuera de la página;
- páginas completamente vacías.

Se detectaron:

- advertencias `underfull` en tablas, bibliografía y algunos párrafos;
- una advertencia de codificación en `algorithm.sty`, proveniente del paquete
  LaTeX cargado por Tectonic;
- espacio vacío visible en la página 8 debido a dos tablas `table*` que deben
  pasar a la página siguiente.

Se probó mover ambas tablas al pie de la página. El cambio redujo parte del
vacío, pero hizo que Table I y Table II aparecieran después del inicio de
Results. El cambio fue revertido porque alteraba negativamente la narrativa.

La inspección visual completa está en:

- `C:\Users\jesqu\Desktop\Tesis\Tesis_2026\audits\github_preparation_2026_06_12\main_contact_sheet.png`
- `C:\Users\jesqu\Desktop\Tesis\Tesis_2026\audits\github_preparation_2026_06_12\supplement_contact_sheet.png`

## G. Inconsistencias y riesgos

### Críticos, corregidos

1. **Macro-F1 inflado para clases verdaderas no predichas.**
   Se corrigió el cálculo para asignar precisión/F1 cero en esas clases.
2. **MAXVAR-GCCA 0.9519 en el ZIP original.**
   Se corrigió a 0.9515 en la copia auditada.
3. **Tablas y figuras derivadas de la matriz anterior.**
   Se regeneraron desde la matriz corregida.

### Importantes, pendientes

1. **Predicciones del baseline final seleccionado.**
   Hay métricas y candidatos, pero no un artefacto por muestra para las 45
   selecciones `last_layer_selected_classifier`.
2. **Una sola semilla.**
   Los 45 contextos principales corresponden a `seed42`. El paper debe mantener
   esta limitación explícita.
3. **Separación por paciente no verificada.**
   Debe conservarse como limitación central.
4. **Placeholder GitHub.**
   Aparece en `04-experimental-protocol.tex` y `05-results.tex`.
5. **Nombres históricos en código.**
   HMDF-kNN todavía aparece como `run_winmax_reference`,
   `proposed_method_reference` o `Proposed method` en artefactos internos.
6. **README desactualizado.**
   El README raíz describe principalmente triplet learning y Method Lab, no el
   pipeline final HMDF-kNN.
7. **Entorno no fijado.**
   `requirements.txt` no tiene versiones y no incluye explícitamente `scipy` o
   `seaborn`, usados por el pipeline de resultados.
8. **Rutas absolutas en artefactos.**
   Varios CSV/JSON guardan rutas locales. Los scripts principales resuelven el
   root de forma relativa, pero el release necesita un manifest portable.
9. **Git no disponible en el entorno actual.**
   No fue posible ejecutar `git status`, `git check-ignore` ni verificar el
   historial.

### Menores

1. La carpeta del ZIP incluye `main (2).tex`, con resultados y terminología
   antiguos.
2. `tables/table_top_methods.tex` conserva MAXVAR-GCCA = 0.9519.
3. El ZIP contiene una copia anidada del mismo ZIP.
4. Incluye `main.aux`, `main.bbl`, `main.blg`, `main.log` y `main.out`.
5. La página 8 tiene espacio vacío por restricciones de floats IEEE.
6. Hay advertencias `underfull` y sustitución de fuentes al compilar con
   Tectonic, sin desbordamientos visibles.

Lista de artefactos obsoletos:

`C:\Users\jesqu\Desktop\Tesis\Tesis_2026\audits\github_preparation_2026_06_12\stale_zip_artifacts.csv`

## H. Preparación recomendada para GitHub

### Prioridad alta

1. Crear una carpeta nueva de release; no inicializar Git en todo `Tesis_2026`.
2. Copiar únicamente scripts finales, módulos requeridos, configuración,
   manuscrito limpio y resultados compactos.
3. Guardar las 45 predicciones del baseline final seleccionado.
4. Reemplazar el placeholder por la URL definitiva.
5. Crear README de HMDF-kNN con:
   - definición del método;
   - protocolo train/validation/test;
   - datasets y licencias;
   - preparación de embeddings;
   - reproducción de tablas y figuras;
   - estructura de resultados.
6. Agregar `.gitignore` antes del primer `git add`.
7. Agregar un environment fijado y probarlo en un entorno limpio.
8. Agregar un manifest de resultados con rutas relativas y checksums.
9. Excluir la copia `.incomplete_backup_*`.
10. Mantener los datos, checkpoints y embeddings fuera del Git normal.

### Prioridad media

1. Renombrar identificadores históricos WinMax/VCHMF a HMDF-kNN con aliases.
2. Incluir los documentos de fidelidad y adaptación en `docs/`.
3. Agregar una prueba rápida que:
   - cargue un contexto pequeño;
   - ejecute selección solo con validación;
   - confirme que candidatos no contienen métricas test;
   - regenere una fila de la tabla final.
4. Incluir `LICENSE`, `CITATION.cff` y la licencia de cada dataset.
5. Publicar resultados grandes mediante Zenodo, OSF o un release externo, con
   checksums y enlaces desde el README.

### Prioridad baja

1. Optimizar la página 8 al adaptar el paper al formato editorial final.
2. Reducir advertencias `underfull`.
3. Regenerar Fig. 1 desde una fuente editable o script documentado.

Propuesta de `.gitignore`:

`C:\Users\jesqu\Desktop\Tesis\Tesis_2026\audits\github_preparation_2026_06_12\GITHUB_RELEASE_GITIGNORE_PROPOSED.txt`

Inventario de release:

`C:\Users\jesqu\Desktop\Tesis\Tesis_2026\audits\github_preparation_2026_06_12\github_release_inventory.csv`

Entorno propuesto:

`C:\Users\jesqu\Desktop\Tesis\Tesis_2026\audits\github_preparation_2026_06_12\requirements_publication_proposed.txt`

## I. Cambios aplicados durante la auditoría

1. Creación del auditor reproducible:
   `scripts\audit_hmdf_github_readiness.py`.
2. Reparación del cálculo manual de macro precision y macro-F1.
3. Backup de matrices anteriores.
4. Regeneración de la matriz maestra.
5. Regeneración de resúmenes consolidados.
6. Regeneración del paquete de publicación.
7. Regeneración de tablas y figuras del manuscrito.
8. Corrección textual de MAXVAR-GCCA de 0.9519 a 0.9515.
9. Corrección de la documentación EasyMKL de 2500 a 1000 muestras.
10. Compilación del manuscrito principal y suplemento.
11. Inspección visual página por página.
12. Creación de propuestas de `.gitignore`, inventario y requirements.

No se reentrenaron backbones ni se alteró el ZIP original.

## J. Conclusión

El resultado central del paper es respaldado por los artefactos actuales:
HMDF-kNN alcanza macro-F1 medio 0.9616 sobre 45 contextos brain-MRI, con HAM10000
separado del análisis principal. Las selecciones HMDF-kNN son reproducibles,
dependen de validación y se evalúan en test después de congelarse.

El hallazgo crítico fue un error de agregación macro en métodos comparativos,
no en HMDF-kNN. Tras corregirlo, las conclusiones principales se mantienen y
la diferencia frente al bloque multilayer aumenta ligeramente. La conclusión
debe seguir siendo acotada: ventaja agregada favorable, no superioridad
universal, no validación clínica y no generalización por paciente demostrada.

El siguiente paso correcto es construir un repositorio de release limpio a
partir de la copia corregida y los artefactos auditados, no publicar el
workspace ni el ZIP tal como están.
