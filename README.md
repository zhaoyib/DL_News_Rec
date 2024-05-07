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

*Tips:*推荐手动安装`torch`以及相关CUDA版本，避免自动安装CPU版本的`torch`

文件结构：
```
```

## 🚀 快速开始

如果想要进行实验1、2，那么首先请执行下列代码：
```
```
此后，请配置你的相关配置文件，


如果想要进行实验3、4，那么首先请执行下列代码：
```
```
如果仅希望验证实验3的结果，那么请执行`evaluate.py`，其会在`MIND Small`的验证集上进行验证，并且输出 `AUC`, `MRR`, `nDCG@5`, `nDCG@10`四个评价指标。
如果希望在数据集上进行微调，那么请执行`train.py`，请预先组织好你的文件结构，以免出现意外错误。

## 🍎 实验结果

## 📖 实验分析

### 微调的影响

### 多模态的影响

## ⚙️ 模型权重下载

## 📈 后续工作

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

## 🔗 参考文献
