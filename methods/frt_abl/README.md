# FRT-ABL implementation

Core implementation for fuzzy-rough three-way abductive learning on handwritten arithmetic expression recognition. This package contains model training and evaluation, experiment runners and analysis, ablation variants, configurations, and unit tests.

Dataset files and generated experiment outputs are kept outside the published source. Set the dataset paths in the selected configuration before running training or evaluation.

Run tests from the repository root:

```powershell
py -m unittest discover -s methods/frt_abl/tests
```
