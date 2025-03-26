# Deep Mutual Learning (DML) with Pytorch

## Usage

```bash
$ git clone git@github.com:HHorimoto/pytorch-dml.git
$ cd pytorch-dml
$ ~/python3.10/bin/python3 -m venv .venv
$ . .venv/bin/activate
$ pip install -r requirements.txt
$ source run_.sh
```

## Features

### Deep Mutual Learning (DML)
I trained two student models using Deep Mutual Learning for 50 epochs on CIFAR-10. 
For evaluation, I used one of the student models. 
The table below presents the experimental results.

**Comparison Table**

The table shows that **DML** achieves higher accuracy than **Independent**.

|             |  Accuracy  |
| ----------- | :--------: |
| Independent |   0.7942   |
| DML         | **0.8145** |

#### Reference
[1] [https://github.com/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/11_cnn_pytorch/11_deep_mutual_learning.ipynb](https://github.com/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/11_cnn_pytorch/11_deep_mutual_learning.ipynb)