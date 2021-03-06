MSC
============================================

 
The PyTorch implementation and dataset of "Community Value Prediction in Social E-commerce" (WebConf 2021).


<p align="center">
  <img width="1000" src="msc.png">
</p>

### Abstract

<p align="justify">
The phenomenal success of the newly-emerging social e-commerce has demonstrated that utilizing social relations is becoming a promising approach to promote e-commerce platforms. In this new scenario, one of the most important problems is to predict the value of a community formed by closely connected users in social networks due to its tremendous business value. However, few works have addressed this problem because of 1) its novel setting and 2) its challenging nature that the structure of a community has complex effects on its value. To bridge this gap, we develop a Multi-scale Structure-aware Community value prediction network (MSC) that jointly models the structural information of different scales, including peer relations, community structure, and inter-community connections, to predict the value of given communities. Specifically, we first proposed a Masked Edge Learning Graph Convolutional Network (MEL-GCN) based on a novel masked propagation mechanism to model peer influence. Then, we design a Pair-wise Community Pooling (PCPool) module to capture critical community structures. Finally, we model the inter-community connections by distinguishing intra-community edges from inter-community edges and employing a Multi-aggregator Framework (MAF). Extensive experiments on a large-scale real-world social e-commerce dataset demonstrate our method's superior performance over state-of-the-art baselines, with a relative performance gain of 11.40\%, 10.01\%, and 10.97\% in MAE, RMSE, and NRMSE, respectively. Further ablation study shows the effectiveness of our designed components.</p>

This repository provides a PyTorch implementation of MSC as described in the paper:

> Community Value Prediction in Social E-commerce.
> Guozhen Zhang, Yong Li, Yuan Yuan, Fengli Xu, Hancheng Cao, Yujian Xu, Depeng Jin.
> WebConf, 2021.


### Citing

If you find MSC useful in your research, please consider citing the following paper:

> Community Value Prediction in Social E-commerce.
> Guozhen Zhang, Yong Li, Yuan Yuan, Fengli Xu, Hancheng Cao, Yujian Xu, Depeng Jin.
> WebConf, 2021.


### Usage

Simply run the following command to reproduce the experiments on corresponding dataset and model:

`
python run_experiments.py -p 'exp_config/msc_bd.cfg' -d [device]
`

Here, the `device` can be `cpu` or `cuda`.
