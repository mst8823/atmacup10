## atmacup10
結果：public:7th, private:9th

コンペURL: [https://www.guruguru.science/competitions/16](https://www.guruguru.science/competitions/16)

### directory structure
```
├─input
├─notebooks
├─output
│  └─exp044
│      ├─cols
│      ├─feature
│      ├─preds
│      ├─reports
│      └─trained
├─scr
│  └─mypipe
│    ├─experiment
│    ├─features
│    └─models
└─submission
```
- outputフォルダは, 実験スクリプトを走らせたときに動的に作成されます. input と scr だけあればokです.
- scr ディレクトリ内に実験スクリプトを作成する必要があります.
- 実験管理は, 実験1スクリプト・mlflow を使っています(今回は諸事情でmlflow未使用ですが). 特徴量作成～submit作成まで一つのスクリプトに収めるようにしてます.

### 1. feature enginnering
1. 色情報の集約特徴量
2. 作者, シリーズもの作品ごとの集約特徴量(min, max, mean, max-min, std, z-score, ununique, etc)
3. 他テーブルの one hot 特徴量, w2v 特徴量
4. main のテーブルと他テーブルの情報を混ぜた w2v 特徴量, 数値掛け合わせ特徴量
5. title, more_title, long_title, description, concat_title(title+more_title+long_title)のtfidf->svd(各256次元)の特徴量, doc2vec 特徴量

### 2. cv 
1. group k fold (art_series_id)
2. num fold: 5~10

### 3. model
**1. single model**
  - lightgbm * 11, xgboost * 1
  - それぞれ異なる特徴量で学習
  - 2000~3000個の特徴量を importance で上位700コ程度に絞ってから学習
  - 3 seed average
  - best public score: lightgbm cv:0.9392, public:0.9463, private:0.9725

**2. stacking (1st layer)**
  - ridge * 2, mlp * 2, lightgbm * 4, xgboost * 2, extra_trees * 1, cat_boost * 2
  - 5~10 seed average
  - best public score: cat_boost cv:0.9177, public:0.9401, private:0.9638 
 
**3. stacking (2nd layer)**
  - lightgbm * 1, cat_boost * 2, ridge * 1
  - singe model, stacking 1st layer の予測値すべてを学習に用いた
  - 5~10 seed average
  - best public score: cat_boost cv:0.9164, public:0.9400, private:0.9639 

**4. stacking (final layer)**
  - cat_boost
  - singe model, stacking 1st layer, 2nd layer の予測値をすべて学習に用いた
  - 5 seed average
  - best public score: cat_boost cv:0.9181, public:0.9396, private:0.9639 

### 4. summary
1. seed average の効果があった (cv 0.002　程度)
2. ensembe (stacking) の効果があった. 線形モデルやmlpよりもGBDTモデルでのstakingのほうが効果があった.

### おわりに
とても楽しく参加することができました！運営さん、参加者の皆さんありがとうございました！！

次の atmacup はもっといい順位でフィニッシュできるように頑張ります!
