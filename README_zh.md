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
  <strong style="background-color: green;">简体中文</strong>
  |
  <a href="./READM.md" target="_Self">English</a>
</p>

<details open="open">
<summary>点击目录跳转</summary>

- <a href="#-实验背景" target="_Self">💡 实验背景</a>
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

## 💡 实验背景

在信息爆炸的今天，人们能够越来越轻松地获取大量的信息资源，但是由于新增信息过于庞大，人们难以找到有用的感兴趣的内容。而新闻推荐系统可以通过用户的历史点击、浏览、点赞等行为对用户的偏好进行推断，进而为用户提供个性化的推荐服务，提高用户的体验。<br>

而新闻属于文本和图像信息的集合，而当下的文本、图像预训练模型发展均较为成熟，能够适应不同下游任务，并且有优秀的表现。而如果能够找到一个可迁移的通用特征提取器来提取新闻的表征，那么将极大地改变新闻推荐的范式。因此，本项目旨在探索以下几个问题：<br>

1. 是否可以利用冻结权重的多模态预训练模型作为新闻推荐系统的可迁移特征提取器？
2. 如果在新闻推荐数据集上对多模态预训练模型进行微调，是否会提升模型性能？会多大程度提升模型性能？
3. 多模态的引入是否为模型带来了性能上的提升？<br>

为了回答上述的三个问题，本项目设计了以下四个实验：<br>

1. 使用冻结权重的BLIP模型对新闻进行特征提取，在此基础上训练排序器，进而评估多模态预训练模型对新闻推荐系统的影响。
2. 使用冻结权重的Bert-base-uncased以及Vit-base-patch-16-224对新闻进行特征提取，在此基础上训练排序器，与实验1对比评估多模态共同预训练的效果，与实验3对比评估微调的影响。
3. 使用Bert-base-uncased以及Vit-base-patch-16-224对新闻进行特征提取，并且在MIND-Small数据集上进行微调，评估微调带来的影响。
4. 仅使用Bert-base-uncased对新闻特征进行提取，在MIND-Small数据集上使用和实验3相同的超参数进行微调，评估图像模态引入带来的影响。<br>

## 🌐 环境配置

本实验租用了服务器GPU，使用了单卡RTX 4090 (24GB)，并且配置了90GB的内存。<br>

本实验使用`python = 3.8.10`，可以在conda中执行如下命令创建本实验所需的虚拟环境：
```
conda create --name NRMS python=3.8 && activate NRMS
```

本实验的python库依赖已经汇总在`requirements.txt`文件中，请执行下面的命令行命令来安装依赖：
```
pip install -r requirements.txt
```

*Tips:* 推荐手动安装`torch`以及相关CUDA版本，避免自动安装CPU版本的`torch`

文件结构：
```
```

## 🚀 快速开始

如果想要进行实验1、2，那么首先请执行下列代码：
```
$ export PYTHONPATH=$(pwd)/experiment_fixed:$PYTHONPATH
```
此后，请配置你的相关配置文件，然后在`Total_Pipeline.ipynb`中进行


如果想要进行实验3、4，那么首先请执行下列代码：
```
$ export PYTHONPATH=$(pwd)/experiment_finetuning/src:$PYTHONPATH
```
如果仅希望验证实验3的结果，那么请在下载并配置好模型路径之后执行`evaluate.py`，其会在`MIND Small`的验证集上进行验证，并且输出 `AUC`, `MRR`, `nDCG@5`, `nDCG@10`四个评价指标。
如果希望在数据集上进行微调，那么请执行`train.py`，请预先组织好你的文件结构，以免出现意外错误。

## 🍎 实验结果

对于实验1、2，得到了令人沮丧的实验结果，在训练集上训练了20个epoch之后，模型loss始终在一个较低的值附近震荡，不会进一步下降，最后验证得到的指标数据几乎无异于随机预测。<br>

对于实验3，我们在训练了2个epoch之后，模型loss收敛，验证得到的指标相较于完全随机预测有20%左右的提升。<br>

对于实验4，我们在训练了3个epoch之后，模型loss收敛，验证得到的指标相较于实验3有10%左右的提升。<br>

具体实验结果见下表：<br>

| 实验 | AUC | MRR | nDCG@5 | nDCG@10 |
|:----:|:---:|:---:|:------:|:------:|
| 随机预测 | 0.500 | 0.201 | 0.203 | 0.267 |
| 实验1 | 0.494 | 0.217 | 0.222 | 0.285 |
| 实验2 | 0.505 | 0.222 | 0.226 | 0.289 |
| 实验3 | 0.638 | 0.254 | 0.281 | 0.343 |
| 实验4 | 0.689 | 0.306 | 0.336 | 0.400 |

## 📖 实验分析

首先，关于实验的公平性，在这里先做出如下分析：<br>

实验共计训练了四个模型，为了保证对比的公平性，每个模型的训练时长都是29小时左右，具体而言，针对实验1、2的模型，其训练了20个epoch，共计30个小时；针对实验3的模型，训练了2个epoch，共计29小时，针对实验4的模型，训练了3个epcoh，共计28.5小时。同时，模型结构方面也尽可能地确保了四个模型的可训练参数量基本一致，能够最大程度确保其公平性。<br>

于实验1、2，其结果基本一致，尝试修改模型超参数、损失函数权重、学习率变化函数、批次大小等训练相关设置之后，得到的最优模型在训练1500个批次，64000个样本的学习之后，损失值保持在0.6左右震荡，并且在后续的几十万个批次中不再下降。与二分类随机预测的0.96的损失值相比，我们可以认为其从中学习到了一定的知识，并且已经收敛。而其在验证集上的指标显示其AUC仅有0.506，略优于随机预测的0.5，而其余三个评价指标也均没有明显优于随机预测。<br>

对于实验3，在训练了2个epoch之后，损失基本收敛，在验证集上验证得到其AUC为0.638，MRR为0.255，NDCG@5为0.280，NDCG@10为0.343，显著优于未经微调的模型效果。<br>

对于实验4，虽然其仅仅使用了文本信息，但是在训练3个epoch之后，在验证集上得到其AUC为0.689，MRR为0.306，NDCG@5为0.336，NDCG@10为0.400，可以看到其效果优于多模态的微调模型。<br>

实验同样考虑了是否是由于多模态模型参数量大，2个epoch没有完全收敛导致了这样的问题，但是在训练了3个epoch之后，其验证集上AUC、MRR等指标均没有提升，且训练集Loss也并没有下降，训练集grad_norm指标也始终在一个较大的值附近浮动[36]，因此可以判断2个epoch的时候模型实际上已经收敛，实验对比是公平的。<br>

### 微调的影响

首先对比上述实验1、2，可以发现在冻结权重的情况下，使用多模态同时预训练的预训练模型和使用分别预训练的模型并不会有实际上的区别，二者均不能够胜任推荐系统的召回器的工作。<br>

但是通过对比实验2、3，我们可以发现，如果能够在新闻推荐数据集上对我们的预训练模型进行微调，那么其可以提升其推荐性能。受限于显卡资源，我并没有在新闻推荐数据集上对BLIP这类相对较大的多模态预训练模型进行微调，这可能是后续进一步实验的可能探索方向<br>

### 多模态的影响

对比实验3、4，我们可以发现：图片模态的引入损害了模型的性能，相比于仅使用文本信息的推荐系统，多模态的推荐系统不仅训练开销大、时间长，而且效果还弱于单一模态的推荐系统。造成这一劣势的原因可能主要有两方面：其一是特征融合的效果可能并不够好，直接使用哈达玛积计算出来的多模态表征的表达能力并不足；其二是数据集的图片缺失过多，MIND Small数据集中涉及到的新闻共计65238条，其中有36546条图片信息缺失，占总新闻数目的56%。即使是依照论文 Why do we click的处理方式，将标题排版在空白图右侧，ViT也很难从中提取出有效的信息。<br>

因此，构建高质量的新闻推荐数据集也是推动后续研究的重要基础之一。<br>

## ⚙️ 模型权重下载

由于实验1、2所训练得到的模型并不有效，因此在这里就不提供实验1、2中的模型。仅提供实验3、4的模型，可以通过下面的百度网盘链接下载：<br>

[模型下载链接]()

## 📈 后续工作

1. 多模态预训练模型在推荐系统中微调的方法研究，本实验仅使用了最基础的全参数微调，或许后续实验可以探究LoRA在预训练模型上的微调效果，或者寻找更先进的微调算法。
2. 多模态新闻推荐数据集的构建，本实验的最大限制就在于多模态新闻数据集的缺乏，大部分新闻推荐相关的论文都使用了私有的或者商业数据集，并不对外公开，一个开源的高质量多模态新闻推荐数据集将大幅推动新闻推荐系统算法的研究。
3. 实时根据用户的点击信息更新其偏好算法研究，本实验仅仅在静态的数据集上进行训练，而实际应用中，用户每时每刻都在产生新的数据，如何迅速更新用户的特征、偏好同样是新闻推荐系统设计中的一大挑战。 
4. 算法模型效率优化研究，实验1、2中模型训练开销过大，是否有什么方法能够减少模型的训练开销。

## 🧲 联系作者

如果有任何问题，请在Github上提出Issue，也可以联系作者 赵艺博@华东师范大学<br>
10203330408@stu.ecnu.edu.cn

## ✏️ 引用本项目

如果本项目对你有帮助或者有启发，可以按照如下格式引用：<br>
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

## 🔐 授权许可

本项目授权许可为[Apache 2.0 License](./LICENSE)

## 🔗 参考文献

[1] 王喆 《深度学习推荐系统》[M] 北京：电子工业出版社，2020.3 <br>
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
