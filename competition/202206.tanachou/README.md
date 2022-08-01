#  タナチョー 部材の画像認識 

Personal note about this : [https://signate.jp/competitions/720](https://signate.jp/competitions/720)

## Problem summary 


```bash
# dataset includes 122 classes, each class contains about 10 images
# need to recognize type of image
```

## What i have tried

```bash
# 1. image classification
# split into train/val : 80%/20%
# framework : mmclassification
# model : efficient-net b0, frozen last 4 layer 
# result
Inference : 0.361
Score : 0.8834071
Unfortunately, the sore on private test set is : 0.3034661

# 2. image recognition / feature learning
# 2.a training baseline with arcface
# https://signate.jp/competitions/720/discussions/training-baseline-with-arcface

# 2.b training recognition using PaddleCls
# this framework is very well written and mature
# it includes entire image recognition work flows (object detection, feature extraction, vector search, ....)
# train = 80% of classes, val = 20% of classes

Inference : 0.44
Score : 0.806
# the score on public set is lower, so I didn't choose it
# but it should have better result than image classification 
```

Result: 66位 / 143人投稿 、brozen medalも受けられない ..... 

## 個人ノート

1. 高いランクを取得できない原因は problem整理をちゃんとしませんでした。　
今回のタスクは顔認識問題と似ているので、顔認識に良く使われる ArcFace、Feature Learningを利用すべきになります。
しかしproblem整理せずに、個人のknow-howによって進めたので、時間が無駄になります。
後、parameter tunning段階で大幅時間が食われました。

2. good approach for competition：

```bash
1. データセットのクラスーや数分布、解析
2. 似ているデータセット検索
3. 最新アルゴリズムを調査
4. 評価
5. パラメータtunning

```

End