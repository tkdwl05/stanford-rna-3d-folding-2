import json

with open('improve_data_label_v17.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
for i, c in enumerate(cells):
    src = ''.join(c['source'])
    print(f'\n{"="*60}')
    print(f'=== CELL {i} ===')
    print(f'{"="*60}')
    print(src)
