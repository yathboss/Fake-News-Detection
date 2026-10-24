# Evaluation Results (TEST split: dataset/MMFakeBench_test.json)

n = 10000 test examples, fixed seed = 42

| Method | Accuracy | Precision (macro) | Recall (macro) | Macro-F1 |
|---|---|---|---|---|
| Majority-class baseline | 0.7000 | 0.3500 | 0.5000 | 0.4118 |
| Text-only (frozen CLIP text + MLP) | 0.6773 | 0.6431 | 0.6635 | 0.6454 |
| CLIP multimodal, no RAG | 0.7853 | 0.7446 | 0.7488 | 0.7466 |
| CLIP multimodal + RAG | 0.7719 | 0.7306 | 0.7430 | 0.7358 |

## Majority-class baseline - Confusion matrix

| | Pred Real | Pred Fake |
|---|---|---|
| True Real | 0 | 3000 |
| True Fake | 0 | 7000 |

## Text-only (frozen CLIP text + MLP) - Confusion matrix

| | Pred Real | Pred Fake |
|---|---|---|
| True Real | 1887 | 1113 |
| True Fake | 2114 | 4886 |

## CLIP multimodal, no RAG - Confusion matrix

| | Pred Real | Pred Fake |
|---|---|---|
| True Real | 1973 | 1027 |
| True Fake | 1120 | 5880 |

## CLIP multimodal + RAG - Confusion matrix

| | Pred Real | Pred Fake |
|---|---|---|
| True Real | 2012 | 988 |
| True Fake | 1293 | 5707 |
