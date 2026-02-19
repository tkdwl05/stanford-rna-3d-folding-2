import json

with open('improve_data_label_v18.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
for i, c in enumerate(cells):
    outputs = c.get('outputs', [])
    if not outputs:
        continue
    for o in outputs:
        # text output
        text_lines = o.get('text', [])
        if isinstance(text_lines, str):
            text_lines = [text_lines]
        text = ''.join(text_lines)

        # traceback
        tb_lines = o.get('traceback', [])
        tb = '\n'.join(tb_lines)

        combined = text + tb
        if any(kw in combined for kw in ['NaN', 'Error', 'Epoch', 'loss=', 'skip', 'nonfinite', 'detected']):
            print(f'\n--- Cell {i} ---')
            # print last 60 lines of text
            tlines = text.split('\n')
            print('\n'.join(tlines[-60:]))
            if tb_lines:
                print('[TRACEBACK]')
                # strip ansi
                import re
                ansi = re.compile(r'\x1b\[[0-9;]*m')
                for tl in tb_lines[-10:]:
                    print(ansi.sub('', tl))
