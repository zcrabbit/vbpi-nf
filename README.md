# vbpi-nf
Code for [Improved Variational Bayesian Phylogenetic Inference with Normalizing Flows](http://arxiv.org/abs/2012.00459)

Please consider citing the paper when any of the material is used for your research.
```
@incollection{VBPI-NF,
title = {Improved Variational Bayesian Phylogenetic Inference with Normalizing Flows},
author = {Zhang, Cheng},
booktitle = {Arxiv arXiv:2012.00459},
year = {2020},
}
```

## Mini demo

Use command line
```python
python main.py --dataset DS1 --flow_type identity --empFreq
python main.py --dataset DS1 --flow_type planar --Lnf 16 --stepszBranch 0.0003 --empFreq
python main.py --dataset DS1 --flow_type realnvp --Lnf 5 --stepszBranch 0.0001 --empFreq

```
You can also load the data, set up and train the model on your own. See more details in [main.py](https://github.com/zcrabbit/vbpi-nf/blob/main/code/main.py).




