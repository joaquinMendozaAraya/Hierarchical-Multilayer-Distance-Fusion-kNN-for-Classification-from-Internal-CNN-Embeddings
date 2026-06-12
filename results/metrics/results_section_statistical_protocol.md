# HMDF-kNN Results comparison protocol

Generated from frozen artifacts. No model was retrained.

## Main comparison

- Contexts: 45 brain-MRI dataset-backbone pairs.
- Proposed method: HMDF-kNN.
- Reference selection: highest validation macro-F1 among complete non-proposed
  methods in the same context; validation balanced accuracy and validation
  accuracy are tie-breakers.
- Test use: one frozen comparison after reference selection.
- Practical equivalence margin: 0.005 macro-F1.
- Statistical analysis: paired class-stratified bootstrap over identical test
  samples, 2000 replicates, base seed 42. Paired predictions were
  available for 38 of 45 contexts.
- Multiple comparisons: Holm correction over the 38 available
  context-level p-values.

## Outcome definitions

1. Numerical: sign of the test macro-F1 delta.
2. Practical: win/loss only when the absolute delta reaches 0.005.
3. Statistical, unadjusted: paired bootstrap 95% CI excludes zero.
4. Statistical, corrected: Holm-adjusted p < 0.05 among contexts with paired
   prediction artifacts; the delta sign determines which method is ahead.

| Criterion | HMDF ahead | Tie/NS | Reference ahead |
|---|---:|---:|---:|
| Numerical sign | 28 | 0 | 17 |
| Practical margin (0.005) | 19 | 24 | 2 |
| Paired bootstrap 95% CI | 10 | 28 | 0 |
| Paired bootstrap + Holm | 2 | 36 | 0 |

## Retrospective test-envelope analysis

The legacy context envelope chooses the highest observed non-proposed test
macro-F1 after evaluation. It gives 24 numerical wins and
21 numerical losses for HMDF-kNN; with the 0.005 practical margin,
the counts are 15/26/4.

This envelope is retained as a conservative descriptive analysis only. It is
not used for inferential claims because the reference method is chosen using
the same test outcomes being compared.

