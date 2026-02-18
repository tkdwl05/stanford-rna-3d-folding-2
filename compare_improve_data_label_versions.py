import json
import re
from pathlib import Path

NOTEBOOKS = sorted(Path('.').glob('improve_data_label_v*.ipynb'))
BEST_TM_RE = re.compile(r"best_tm=\s*([0-9.]+)", re.I)
BEST_VAL_RE = re.compile(r"best(?:_val| val)=\s*([0-9.]+)", re.I)
NAN_RE = re.compile(r"\bnan\b", re.I)


def parse_notebook(path: Path):
    nb = json.loads(path.read_text())
    text = []
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        for out in cell.get('outputs', []):
            if 'text' in out:
                text.append(''.join(out['text']))
            elif 'data' in out and 'text/plain' in out['data']:
                text.append(''.join(out['data']['text/plain']))
    full = '\n'.join(text)

    best_tm = None
    for m in BEST_TM_RE.finditer(full):
        best_tm = float(m.group(1))

    best_val = None
    for m in BEST_VAL_RE.finditer(full):
        best_val = float(m.group(1))

    nan_count = len(NAN_RE.findall(full))

    return {
        'file': path.name,
        'best_tm': best_tm,
        'best_val': best_val,
        'nan_mentions': nan_count,
    }


def main():
    rows = [parse_notebook(nb) for nb in NOTEBOOKS]

    lines = [
        '# improve_data_label 버전 비교',
        '',
        '| version | best_tm | best_val | nan_mentions |',
        '|---|---:|---:|---:|',
    ]

    for r in rows:
        lines.append(
            f"| {r['file']} | {r['best_tm'] if r['best_tm'] is not None else '-'} | "
            f"{r['best_val'] if r['best_val'] is not None else '-'} | {r['nan_mentions']} |"
        )

    tm_rows = [r for r in rows if r['best_tm'] is not None]
    if tm_rows:
        best_tm_row = max(tm_rows, key=lambda x: x['best_tm'])
        lines += [
            '',
            f"- 최고 TM-score 출력 로그는 **{best_tm_row['file']} ({best_tm_row['best_tm']:.4f})** 입니다.",
        ]

    lines += [
        '- NaN 출력이 많은 버전은 Stage2에서 수치 불안정이 발생한 것으로 보입니다.',
        '- `improve_data_label_v10.py`는 v8 기반 + NaN 방어 로직(비유한 loss/grad skip, local bond regularization)으로 개선했습니다.',
        '',
    ]

    Path('improve_data_label_comparison.md').write_text('\n'.join(lines))
    print('Wrote improve_data_label_comparison.md')


if __name__ == '__main__':
    main()
