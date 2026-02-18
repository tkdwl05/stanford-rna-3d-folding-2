# improve_data_label 버전 비교

| version | best_tm | best_val | nan_mentions |
|---|---:|---:|---:|
| improve_data_label_v2.ipynb | - | - | 0 |
| improve_data_label_v3.ipynb | - | 17.177842 | 1 |
| improve_data_label_v4.ipynb | - | 0.289603 | 1 |
| improve_data_label_v5.ipynb | - | 0.29828451772530873 | 2 |
| improve_data_label_v6.ipynb | - | 54.14275932312012 | 2 |
| improve_data_label_v6_1.ipynb | - | - | 2 |
| improve_data_label_v6_2.ipynb | - | 54.43951187133789 | 2 |
| improve_data_label_v7.ipynb | - | - | 2 |
| improve_data_label_v7_1.ipynb | - | - | 86 |
| improve_data_label_v8.ipynb | 0.05699617266654968 | - | 146 |
| improve_data_label_v9.ipynb | 0.03536276463419199 | - | 78 |

- 최고 TM-score 출력 로그는 **improve_data_label_v8.ipynb (0.0570)** 입니다.
- NaN 출력이 많은 버전은 Stage2에서 수치 불안정이 발생한 것으로 보입니다.
- `improve_data_label_v10.py`는 v8 기반 + NaN 방어 로직(비유한 loss/grad skip, local bond regularization)으로 개선했습니다.
