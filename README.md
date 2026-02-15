# 🧬 Stanford Ribonanza RNA Folding 2
> **딥러닝을 활용한 RNA 3D 구조 및 반응성 예측**

이 저장소는 Kaggle에서 진행된 [Stanford Ribonanza RNA Folding 2](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2) 경진대회 솔루션을 담고 있습니다. RNA 서열 데이터를 활용해 화학적 반응성을 예측하고, 이를 통해 RNA의 복잡한 3D 구조를 파악하는 모델을 구축하는 것이 목표입니다.

---

## 📌 프로젝트 개요

RNA는 단백질만큼이나 생명 현상에서 중요한 역할을 수행하며, 특히 **mRNA 백신이나 유전자 치료제** 개발의 핵심 요소입니다. RNA의 기능은 그 서열이 스스로 접히며 형성하는 **3D 구조**에 의해 결정되는데, 이를 실험적으로 밝혀내는 데는 막대한 비용과 시간이 소요됩니다.

본 프로젝트에서는 스탠퍼드 대학교 **Das Lab**에서 제공하는 대규모 실험 데이터(Ribonanza)를 학습하여, 미지의 RNA 서열에 대한 **화학적 반응성(Reactivity)**을 정확하게 예측하는 AI 모델을 개발합니다.

## 💡 주요 과제 (The Challenge)

RNA는 A, C, G, U의 네 가지 염기로 구성된 단순한 사슬처럼 보이지만, 실제로는 복잡하게 꼬여 3차원 구조를 만듭니다. 본 경진대회에서는 다음과 같은 실험 데이터를 예측합니다:

1. **DMS (Dimethyl Sulfate) 반응성:** 특정 염기가 외부로 노출되어 있는 정도를 측정합니다.
2. **SHAPE (Selective 2'-hydroxyl acylation analyzed by primer extension) 반응성:** RNA 백본의 유연성을 측정하여 구조적 특징을 나타냅니다.

이 지표들을 정확히 예측할 수 있다면, 값비싼 실험 없이도 RNA가 어떻게 접힐지(Folding) 컴퓨터로 시뮬레이션할 수 있습니다.

## 📊 데이터셋 정보

* **Input:** RNA 염기 서열 (Sequence)
* **Output:** 각 염기 위치에서의 DMS 및 SHAPE 반응성 수치
* **데이터 규모:** 수천만 개의 RNA 서열을 포함하는 세계 최대 규모의 Ribonanza 데이터셋 활용

## 📏 평가 지표 (Evaluation)

모델의 성능은 실제 실험 결과와 예측값 사이의 **MAE (Mean Absolute Error)**를 통해 평가됩니다.

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

---

## 🛠️ 기술 스택 (Tech Stack)

* **Language:** Python
* **Framework:** PyTorch / TensorFlow
* **Models:** Transformer, Graph Neural Networks (GNN), CNN-based models
* **Tools:** Pandas, NumPy, Scikit-learn, WandB (실험 기록)

## 📂 저장소 구조

```text
├── data/               # 데이터셋 (Kaggle 제공)
├── notebooks/          # EDA 및 실험 코드
├── src/                # 모델 아키텍처 및 학습 스크립트
├── models/             # 학습된 모델 가중치
└── README.md           # 프로젝트 설명
```

## 🔗 관련 링크
- [Kaggle 경진대회 페이지: Stanford Ribonanza RNA Folding 2](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2)
- [주최 기관 (Stanford Das Lab): Das Lab Official Site](https://daslab.stanford.edu/)

## ✍️ Author
이름/닉네임 - [GitHub Profile](https://github.com/)

Email: your-email@example.com
