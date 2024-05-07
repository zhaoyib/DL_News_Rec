<!--
 * @Description: 
 * @Author: Yibo Zhao
 * @Date: 2024-05-07 16:51:24
-->
<h1 align="center">多模态深度学习新闻推荐系统</h1>
<div align="center">
    <a href="./LICENSE">
      <img src="https://img.shields.io/badge/license-Apache--2.0-yellow">
    </a>
</div>
<br>

<p align="center">
  <strong style="background-color: green;">English</strong>
  |
  <a href="./READM_zh.md" target="_Self">简体中文</a>
</p>

<details open="open">
<summary>点击目录跳转</summary>

- <a href="#-Experimental Background" target="_Self">💡 Experimental Background</a>
- <a href="#-环境配置" target="_Self">🌐 环境配置</a>
- <a href="#-快速开始" target="_Self">🚀 快速开始</a>
- <a href="#-实验结果" target="_Self">🍎 实验结果</a>
- <a href="#-实验分析" target="_Self">📖 实验分析</a>
  - <a href="#微调的影响" target="_Self">微调的影响</a>
  - <a href="#多模态的影响" target="_Self">多模态的影响</a>
- <a href="#-模型权重下载" target="_Self">⚙️ 模型权重下载</a>
- <a href="#-后续工作" target="_Self">📈 后续工作</a>
- <a href="#-联系作者" target="_Self">🧲 联系作者</a>
- <a href="#-引用本实验" target="_Self">✏️ 引用本实验</a>
- <a href="#-授权许可" target="_Self">🔐 授权许可</a>
- <a href="#-参考文献" target="_Self">🔗 参考文献</a>

</details>
<br>

## 💡 Experimental Background

In today's information explosion era, people can increasingly easily access a vast array of information resources. However, due to the overwhelming amount of new information, it becomes challenging to find useful and engaging content. News recommendation systems can infer user preferences based on their historical clicks, browsing, likes, and other behaviors, thus providing personalized recommendations to enhance user experience.<br>

News, being a collection of text and image information, benefits from the current maturity of text and image pre-trained models. These models have shown to be adaptable to various downstream tasks and deliver excellent performance. If a transferable general feature extractor could be identified to represent news, it could significantly alter the paradigm of news recommendation. Therefore, this project aims to explore the following questions:<br>

1. Can a frozen multimodal pre-trained model be used as a transferable feature extractor for a news recommendation system?
2. Would fine-tuning a multimodal pre-trained model on a news recommendation dataset improve model performance? And by how much?
3. Does the introduction of multimodality bring performance improvements to the model?<br>

To answer these questions, the project is designed with the following four experiments:<br>

1. Use the frozen weights of the BLIP model for feature extraction of news, train a ranker based on this, and then evaluate the impact of multimodal pre-trained models on news recommendation systems.
2. Use the frozen weights of Bert-base-uncased and Vit-base-patch-16-224 for feature extraction of news, train a ranker based on this, compare with Experiment 1 to assess the effect of multimodal joint pre-training, and compare with Experiment 3 to evaluate the impact of fine-tuning.
3. Use Bert-base-uncased and Vit-base-patch-16-224 for feature extraction of news and fine-tune on the MIND-Small dataset to assess the impact of fine-tuning.
4. Use only Bert-base-uncased for news feature extraction, fine-tune with the same hyperparameters as Experiment 3 on the MIND-Small dataset, and evaluate the impact of introducing the image modality.<br>

## 🌐 Environment Setup

This experiment rented a server GPU, using a single RTX 4090 (24GB), and configured 90GB of memory.<br>

The experiment used `python = 3.8.10`. You can create the virtual environment needed for this experiment by executing the following command in conda:
```
conda create --name NRMS python=3.8 && activate NRMS
```

The Python library dependencies for this experiment have been summarized in the `requirements.txt file`. Please execute the following command to install the dependencies:
```
pip install -r requirements.txt
```

*Tips:* It is recommended to manually install `torch` and the relevant `CUDA` version to avoid automatically installing the CPU version of `torch`.

File structure:：
```
```

## 🚀 Quick Start

If you want to conduct Experiments 1 and 2, first execute the following code:
```
$ export PYTHONPATH=$(pwd)/experiment_fixed:$PYTHONPATH
```
Afterward, please configure your relevant configuration file, and then proceed in `Total_Pipeline.ipynb`.


For experiment 3, 4, run the following code first:
```
$ export PYTHONPATH=$(pwd)/experiment_finetuning/src:$PYTHONPATH
```
If you only want to verify the results of Experiment 3, download and configure the model path, then execute `evaluate.py`. It will validate on the MIND Small validation set and output the AUC, MRR, nDCG@5, nDCG@10 four evaluation metrics.<br>
If you wish to fine-tune on the dataset, execute `train.py`. Please organize your file structure in advance to avoid unexpected errors.

## 🍎 Experimental Results

For Experiments 1 and 2, disappointing results were obtained. After training for 20 epochs on the training set, the model loss remained oscillating around a low value without further decline. The final verified metric data was almost indistinguishable from random prediction.<br>

For Experiment 3, the model loss converged after training for 2 epochs. The verified metrics on the validation set showed about a 20% improvement over completely random prediction.<br>

For Experiment 4, after training for 3 epochs, the model loss converged. The verified metrics on the validation set showed about a 10% improvement over the fine-tuned multimodal model.<br>

The specific experimental results are shown in the table below:

| experiment | AUC | MRR | nDCG@5 | nDCG@10 |
|:----:|:---:|:---:|:------:|:------:|
| Random Prediction | 0.500 | 0.201 | 0.203 | 0.267 |
| Experiment 1 | 0.494 | 0.217 | 0.222 | 0.285 |
| Experiment 2 | 0.505 | 0.222 | 0.226 | 0.289 |
| Experiment 3 | 0.638 | 0.254 | 0.281 | 0.343 |
| Experiment 4 | 0.689 | 0.306 | 0.336 | 0.400 |

## 📖 Experimental Analysis

Firstly, regarding the fairness of the experiments, the following analysis is made:<br>

A total of four models were trained in the experiments. To ensure a fair comparison, each model was trained for approximately 29 hours. Specifically, for the models in Experiments 1 and 2, they were trained for 20 epochs, totaling 30 hours; for the model in Experiment 3, it was trained for 2 epochs, totaling 29 hours; for the model in Experiment 4, it was trained for 3 epochs, totaling 28.5 hours. Additionally, the model structures were designed to ensure that the trainable parameter count for the four models was essentially consistent, maximizing fairness.<br>

For Experiments 1 and 2, the results were essentially the same. After attempting to modify model hyperparameters, loss function weights, learning rate schedules, batch sizes, and other training-related settings, the best model achieved an optimal state after training for 1500 batches, with 64,000 samples, with the loss value oscillating around 0.6 and not decreasing further. Compared to the 0.96 loss value of random binary classification prediction, it can be inferred that the model has learned some knowledge and has converged. However, its performance on the validation set, with an AUC of only 0.506, slightly better than random prediction, indicates that the other three evaluation metrics were not significantly better than random prediction.<br>

For Experiment 3, after training for 2 epochs, the loss converged, and the model achieved an AUC of 0.638, MRR of 0.255, nDCG@5 of 0.280, and nDCG@10 of 0.343 on the validation set, which is significantly better than the performance of the non-fine-tuned model.<br>

For Experiment 4, although it only used text information, after training for 3 epochs, it achieved an AUC of 0.689, MRR of 0.306, nDCG@5 of 0.336, and nDCG@10 of 0.400 on the validation set, which is better than the fine-tuned multimodal model.<br>

The experiment also considered whether the large number of parameters in the multimodal model prevented full convergence within 2 epochs, leading to the observed issue. However, after training for 3 epochs, the AUC, MRR, and other metrics on the validation set did not improve, and the training set loss did not decrease. The training set grad_norm indicator also consistently fluctuated around a relatively large value, indicating that the model had indeed converged after 2 epochs, and the experimental comparison is fair.<br>

### Impact of Fine-Tuning

Firstly, comparing the aforementioned Experiments 1 and 2, it can be observed that there is no practical difference between using a pre-trained model with multimodality trained simultaneously and using models trained separately in a frozen weight scenario. Both are incapable of performing the task of a recaller in a recommendation system.<br>

However, by comparing Experiment 2 with Experiment 3, it is evident that fine-tuning the pre-trained model on a news recommendation dataset can enhance its recommendation capabilities. Due to limited GPU resources, fine-tuning larger multimodal pre-trained models such as BLIP on the news recommendation dataset was not explored in this experiment, which could be a potential direction for further research.<br>

### Impact of Multimodality

Comparing Experiment 3 with Experiment 4, it is found that the introduction of the image modality has negatively impacted model performance. Compared to the recommendation system that only uses text information, the multimodal recommendation system not only has a higher training cost and longer duration but also performs worse than the single-modality recommendation system. The reasons for this disadvantage may include two main aspects: firstly, the feature fusion effect may not be sufficient, and the multimodal representation calculated directly using the Hadamard product may not have sufficient expressive power; secondly, there is a high rate of missing image data in the dataset. The MIND Small dataset contains a total of 65,238 news items, with 36,546 missing image information, accounting for 56% of the total number of news items. Even if the image is arranged on the right side of a blank image as described in the paper "Why do we click," it is challenging for ViT to extract effective information from it.<br>

Therefore, constructing a high-quality news recommendation dataset is also one of the important foundations for promoting subsequent research.<br>

## ⚙️ 模型权重下载

Since the models trained in Experiments 1 and 2 were not effective, the models from these experiments will not be provided here. Only the models from Experiments 3 and 4 are available for download through the following Baidu Pan link:<br>

[模型下载链接]()

## 📈 Future Work

1. Research on fine-tuning methods for multimodal pre-trained models in recommendation systems. This experiment only used the most basic full-parameter fine-tuning. Perhaps future experiments could explore the fine-tuning effects of LoRA on pre-trained models or find more advanced fine-tuning algorithms.
2. Construction of a multimodal news recommendation dataset. The biggest limitation of this experiment was the lack of a multimodal news dataset. Most papers related to news recommendation use proprietary or commercial datasets that are not publicly available. An open-source, high-quality multimodal news recommendation dataset would significantly advance the research of news recommendation system algorithms.
3. Research on algorithms for updating user preferences in real-time based on user click information. This experiment only trained on static datasets, whereas in practical applications, users are constantly generating new data. How to quickly update user features and preferences is also a major challenge in the design of news recommendation systems.
4. Research on algorithm model efficiency optimization. The training costs of the models in Experiments 1 and 2 were too high. Are there any methods to reduce the training costs of the models?<br>

## 🧲 Contact the Author

If you have any questions, please raise an Issue on Github or contact the author Yibo Zhao @ East China Normal University<br>
10203330408@stu.ecnu.edu.cn

## ✏️ Citing This Project

If this project has been helpful or inspirational to you, you can cite it in the following format:<br>
```
@misc{YiboZhao2024Thesis,
  author       = {Yibo Zhao},
  title        = {Multimodal News Recommendation System},
  howpublished = {Bachelor's Thesis},
  year         = 2024,
  school       = {East China Normal University},
  address      = {Shanghai China},
  month        = May
}
```

## 🔐 License

This project is licensed under the [Apache 2.0 License](./LICENSE)

## 🔗 参考文献

[1]  Wang, Z. "Deep Learning for Recommender Systems" [M]. Beijing: Publishing House of Electronics Industry, 2020.3<br>

[2] Elkan C. The foundations of cost-sensitive learning[J]. International joint conference on artificial intelligence: volume 17. Lawrence Erlbaum Associates Ltd, 2001, 973-978. <br>

[3] Li, J., Li, D., Xiong, C., & Hoi, S. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation[C]. International conference on machine learning, 2022: 12888-12900.<br>

[4] Liu X Y, Zhou Z H. The influence of class imbalance on cost-sensitive learning: An empirical study[C]. Sixth International Conference on Data Mining (ICDM’06), IEEE, 2006: 970-974.<br>

[5] Lv, Q., Josephson, W., Wang, Z., Charikar, M., & Li, K. Multi-probe LSH: efficient indexing for high-dimensional similarity search[C]. Proceedings of the 33rd international conference on Very large data bases, 2007: 950-961.<br>

[6] Neve, J., & McConville, R. ImRec: Learning reciprocal preferences using images[C]. Proceedings of the 14th ACM Conference on Recommender Systems, 2020: 170-179.<br>

[7] Qiu, R., Huang, Z., Yin, H., & Wang, Z. Contrastive learning for representation degeneration problem in sequential recommendation[C]. Proceedings of the fifteenth ACM international conference on web search and data mining, 2022: 813-823.<br>

[8] Wang, X., He, X., Nie, L., & Chua, T.-S. Item Silk Road: Recommending Items from Information Domains to Social Users[C]. Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2017. https://doi.org/10.1145/3077136.3080771<br>

[9] Wu, F., Qiao, Y., Chen, J. H., Wu, C., Qi, T., Lian, J., ... & Zhou, M. Mind: A large-scale dataset for news recommendation[C]. Proceedings of the 58th annual meeting of the association for computational linguistics, 2020: 3597-3606.<br>

[10] Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.  [E] https://www.deeplearningbook.org/<br>

[11] Li, R., Deng, W., Cheng, Y., Yuan, Z., Zhang, J., & Yuan, F. Exploring the Upper Limits of Text-Based Collaborative Filtering Using Large Language Models: Discoveries and Insights [E] https://arxiv.org/abs/2305.11700<br>

[12] Paszke, A., Gross, S., Chintala, S., Chanan, G., Yang, E., DeVito, Z., Lin, Z., Desmaison, A., Antiga, L., & Lerer, A. Automatic differentiation in PyTorch [E] https://pytorch.org/docs/stable/index.html.<br>

[13] Wu, C., Wu, F., Qi, T., & Huang, Y. Empowering News Recommendation with Pre-trained Language Models [E] https://arxiv.org/abs/2104.07413<br>

[14] Wu, C., et al. NewsBERT: Distilling Pre-trained Language Model for Intelligent News Application. [E] https://arxiv.org/abs/2102.04887 2021<br>
