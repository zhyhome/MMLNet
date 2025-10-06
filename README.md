# Towards Robust and Realible Multimodal Fake News Detection with Incomplete Modality

#### If you have any questions, don't hesitate to get in touch with us: hyzhou03@gmail.com
---

## Installation
Follow these steps to set up the environment:
```
conda create -n news python=3.8.10 -y

conda activate news

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch 
```
pip install list：
```
pytorch_lightning==2.4.0
transformers==4.23.1
numpy==1.21.6
tqdm
wandb==0.13.3
scikit-learn==1.0.2
```

---

## Data Preparation

Download and prepare the datasets from the following sources:
- **Weibo**: [Paper](https://doi.org/10.1145/3123266.3123454).
- **Weibo21**: [GitHub](https://github.com/kennqiang/MDFEND-Weibo21).
- **Pheme**: [Paper](https://link.springer.com/chapter/10.1007/978-3-319-67217-5_8).


## Incomplete modality preprocessing
```
python gen_mask.py
```

## Training

```Shell
# To ensure fairness and reproducibility, MMLNet adopts a consistent set of hyperparameters across all datasets
python3 main.py --model MMLNet --weight_decay 0.005 --train_batch_size 16 --dev_batch_size 16 --learning_rate 1e-4 --clip_learning_rate 3e-6 --num_train_epochs 20 --layers 5 --max_grad_norm 6 --dropout_rate 0.3 --optimizer_name adam --text_size 768 --image_size 1024 --warmup_proportion 0.2 --device 0
```



## Reference

If this project helps your research, please consider citing the following papers:

```

```

If you use the weibo dataset, please cite the paper below:
```
@inproceedings{weibo,
author = {Jin, Zhiwei and Cao, Juan and Guo, Han and Zhang, Yongdong and Luo, Jiebo},
title = {Multimodal Fusion with Recurrent Neural Networks for Rumor Detection on Microblogs},
year = {2017},
isbn = {9781450349062},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3123266.3123454},
doi = {10.1145/3123266.3123454},
pages = {795–816},
numpages = {22},
keywords = {rumor detection, multimodal fusion, microblog, lstm, attention mechanism},
location = {Mountain View, California, USA},
series = {MM '17}
}
```
If you use the weibo21 dataset, please cite the paper below:
```
@inproceedings{weibo21,
  title={MDFEND: Multi-domain Fake News Detection},
  author={Nan, Qiong and Cao, Juan and Zhu, Yongchun and Wang, Yanyan and Li, Jintao},
  booktitle={Proceedings of the 30th ACM International Conference on Information \& Knowledge Management},
  pages={3343--3347},
  year={2021}
}
```
If you use the pheme dataset, please cite the paper below:
```
@inproceedings{pheme,
  title={Exploiting Context for Rumour Detection in Social Media},
  author={Arkaitz Zubiaga and Maria Liakata and Rob Procter},
  booktitle={Social Informatics},
  year={2017},
}
```
