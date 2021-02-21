# vbpi-nf
Code for [Improved Variational Bayesian Phylogenetic Inference with Normalizing Flows](http://arxiv.org/abs/2012.00459)

Please consider citing the paper when any of the material is used for your research.
```
@inproceedings{VBPI-NF,
 author = {Zhang, Cheng},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {18760--18771},
 publisher = {Curran Associates, Inc.},
 title = {Improved Variational Bayesian Phylogenetic Inference with Normalizing Flows},
 url = {https://proceedings.neurips.cc/paper/2020/file/d96409bf894217686ba124d7356686c9-Paper.pdf},
 volume = {33},
 year = {2020}
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




