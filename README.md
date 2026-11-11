# FRT-ABL-FSS

This project implements a fuzzy-rough three-way abductive learning method for handwritten arithmetic expression recognition. It includes training and evaluation code, ablation variants, experiment analysis tools, and tests.

本项目实现了面向手写算术表达式识别的模糊粗糙三支溯因学习方法，并整理了训练评估、消融实验、实验分析及测试代码。

The implementation is in [`methods/frt_abl`](methods/frt_abl). Dataset files and experiment outputs are not included. Install the dependencies in `methods/frt_abl/requirements.txt`, then run the unit tests from the repository root:

```powershell
py -m unittest discover -s methods/frt_abl/tests
```
