# 🧬 Stanford Ribonanza RNA Folding 2
> **AI를 활용한 RNA 3D 구조 예측 도전 (Predicting RNA 3D Structures)**

이 저장소는 Kaggle에서 진행된 [Stanford Ribonanza RNA Folding 2](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2) 경진대회 솔루션을 담고 있습니다. RNA 서열 데이터를 활용해 **3차원 구조(3D Coordinates)**를 정밀하게 예측하는 모델을 구축하는 것이 목표입니다.

---

## 📌 프로젝트 개요

RNA는 단백질만큼이나 생명 현상에서 중요한 역할을 수행하며, 특히 **mRNA 백신이나 유전자 치료제** 개발의 핵심 요소입니다. RNA의 기능은 그 서열이 스스로 접히며 형성하는 **3D 구조**에 의해 결정되는데, 이를 실험적으로 밝혀내는 데는 막대한 비용과 시간이 소요됩니다.

본 프로젝트에서는 스탠퍼드 대학교 **Das Lab**에서 제공하는 데이터를 학습하여, RNA 서열로부터 **3D 원자 좌표(Coordinates)**를 예측하는 AI 모델을 개발합니다.

## 💡 주요 과제 (The Challenge: 3D Structure Prediction)

본 경진대회(Ribonanza 2)의 목표는 **RNA 분자의 3차원 형상**을 예측하는 것입니다.

1.  **Input:** RNA 염기 서열 (Sequence)
2.  **Output:** 각 염기(Residue)의 C1' 원자에 대한 **(x, y, z) 3차원 좌표**.
3.  **Best-of-5:** RNA는 역동적이어서 여러 가지 구조를 가질 수 있습니다. 따라서 모델은 **5개의 서로 다른 구조 후보**를 제출해야 합니다.

## 📏 평가 지표 (Evaluation)

모델의 성능은 예측된 구조와 실제 구조 간의 **TM-Score (Template Modeling Score)**를 기반으로 평가됩니다.
*   단순한 거리 오차(RMSD)와 달리, 구조의 전체적인 위상(Topology) 유사도를 측정합니다.
*   제출한 5개의 구조 중, 실제 구조와 가장 유사한(TM-Score가 높은) 1개의 점수가 채택됩니다 (**Best-of-5 Rule**).

---

## 🛠️ 솔루션 접근법 (Solution Approach)

`final_solution_fixed.py` 파일에 개선된 솔루션이 구현되어 있습니다.

### 1. 모델 아키텍처
*   **Transformer Encoder:** RNA 서열의 긴 의존성(Long-range dependency)을 학습하기 위해 Transformer를 사용합니다.
*   **Best-of-5 Head:** 단일 구조가 아닌, 5개의 독립적인 구조 변형을 예측하여 채점 시 가장 잘 맞는 구조가 선택될 확률을 높입니다.

### 2. 학습 전략 (Loss Function)
*   **Kabsch Algorithm + RMSD Loss:** 분자의 **회전 및 이동 불변성(Rotation/Translation Invariance)**을 고려합니다. 예측된 구조를 정답 구조에 최적으로 정렬(Align)한 후 오차를 최소화합니다. 단순 좌표거리(L1/MSE)의 문제를 해결했습니다.

### 3. 데이터 처리
*   **ID Parsing:** `target_id_residue` 형식의 ID를 파싱하여 서열과 좌표를 매핑합니다.
*   **Sequence Alignment:** 누락된 잔기(Residue)나 불일치를 처리하는 로직이 포함되어 있습니다.

---

## 📂 저장소 구조

```text
├── data/                    # 데이터셋 (Kaggle 제공)
│   ├── train_sequences.csv
│   ├── train_labels.csv
│   └── ...
├── final_solution.ipynb     # (구버전) 초기 노트북
├── final_solution_fixed.py  # (개선됨) 3D 구조 예측 솔루션 스크립트
├── crate_code_by-GPT.ipynb  # 참고용 베이스라인 코드
└── README.md                # 프로젝트 설명
```

## 🔗 관련 링크
- [Kaggle 경진대회 페이지: Stanford Ribonanza RNA Folding 2](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2)
- [주최 기관 (Stanford Das Lab): Das Lab Official Site](https://daslab.stanford.edu/)
