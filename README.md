# CLIP

[[Blog]](https://openai.com/blog/clip/) [[Paper]](https://arxiv.org/abs/2103.00020) [[Model Card]](model-card.md) [[Colab]](https://colab.research.google.com/github/openai/clip/blob/master/notebooks/Interacting_with_CLIP.ipynb)

CLIP（对比语言-图像预训练）是一种在各种（图像、文本）对上训练的神经网络。它可以用自然语言指导(instructed)下，在给定图像的情况下预测最相关的文本片段，无需直接优化任务，类似于 GPT-2 和 3 的零样本能力。我们发现 CLIP 与 ImageNet“零样本”上原始 ResNet50 的性能相匹配，无需使用任何原始 128 万个标记示例，克服了计算机视觉中的几个主要挑战。



## 方法 Approach

![CLIP](CLIP.png)



## 使用方法 Usage


首先， [安装 PyTorch 1.7.1](https://pytorch.org/get-started/locally/)（或更高版本）和 torchvision，以及一些额外的小依赖项，然后将此存储库安装为 Python 包。在 CUDA GPU 机器上，使用以下命令完成安装：

```bash
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
```

替换上面的 `cudatoolkit=11.0` 为您机器上适当的 CUDA 版本，或者在没有 GPU 的机器上使用 `cpuonly`。

```python
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
```


## 接口 API

CLIP 模块 `clip` 提供以下方法：

#### `clip.available_models()`

返回可用的 CLIP 模型名称。

#### `clip.load(name, device=..., jit=False)`

返回指定模型名称（由 `clip.available_models()` 返回）的模型和模型所需的 TorchVision 变换。它会在必要时下载模型。`name` 参数也可以是本地检查点的路径。
可以选择指定运行模型的设备，默认情况下，如果有 CUDA 设备，则使用第一个 CUDA 设备，否则使用 CPU。当 `jit` 为 `False` 时，将加载非 JIT 版本的模型。

#### `clip.tokenize(text: Union[str, List[str]], context_length=77)`

返回一个 LongTensor，包含给定文本输入的标记化序列。该张量可以用作模型的输入。

---

模型由 `clip.load()` 返回，支持以下方法：

#### `model.encode_image(image: Tensor)`

给定一批图像，返回由 CLIP 模型的视觉部分编码的图像特征。
#### `model.encode_text(text: Tensor)`

给定一批文本标记，返回由 CLIP 模型的语言部分编码的文本特征。

#### `model(image: Tensor, text: Tensor)`

给定一批图像和一批文本标记，返回两个张量，包含对应每个图像和文本输入的对数几率分数。数值是对应图像和文本特征之间的余弦相似度，乘以 100。


## 更多示例 More Examples

### 零样本预测 Zero-Shot Prediction

下面的代码演示了如何使用 CLIP 进行零样本预测，如论文附录 B 所示。该示例从 [CIFAR-100 数据集](https://www.cs.toronto.edu/~kriz/cifar.html) 中取出一张图像，并预测该图像在数据集的 100 个文本标签中最可能的标签。

```python
import os
import clip
import torch
from torchvision.datasets import CIFAR100

# 加载模型 Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# 下载数据集 Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

# 准备输入 Prepare the inputs
image, class_id = cifar100[3637]
image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

# 计算特征 Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# 选择图像最相似的前 5 个标签 Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)

# 打印结果 Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")
```

结果输出如下（具体数字可能因计算设备略有不同）：

```
Top predictions:

           snake: 65.31%
          turtle: 12.29%
    sweet_pepper: 3.83%
          lizard: 1.88%
       crocodile: 1.75%
```

注意，该示例使用了返回给定输入编码特征的 `encode_image()` 和 `encode_text()` 方法。


### 线性探测评估 Linear-probe evaluation

下面的代码示例使用 [scikit-learn](https://scikit-learn.org/) 对图像特征执行逻辑回归。

```python
import os
import clip
import torch

import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm

# 加载模型 Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# 加载数据集 Load the dataset
root = os.path.expanduser("~/.cache")
train = CIFAR100(root, download=True, train=True, transform=preprocess)
test = CIFAR100(root, download=True, train=False, transform=preprocess)


def get_features(dataset):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

# 计算图像特征 Calculate the image features
train_features, train_labels = get_features(train)
test_features, test_labels = get_features(test)

# 执行逻辑回归 Perform logistic regression
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
classifier.fit(train_features, train_labels)

# 使用逻辑回归分类器进行评估 Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
print(f"Accuracy = {accuracy:.3f}")
```

注意，`C` 值应通过使用验证集的超参数搜索确定。


## 参见 See Also

* [OpenCLIP](https://github.com/mlfoundations/open_clip): 包含更大且独立训练的 CLIP 模型，最高可达 ViT-G/14
* [Hugging Face implementation of CLIP](https://huggingface.co/docs/transformers/model_doc/clip): 便于与 HF 生态系统集成