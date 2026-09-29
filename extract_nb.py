import json

with open('notebooks_01_rcaeval_dataset_inspection (1).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('notebook_content.txt', 'w', encoding='utf-8') as out:
    for cell in nb.get('cells', []):
        out.write(f"\n--- {cell['cell_type']} ---\n")
        out.write(''.join(cell.get('source', [])) + "\n")
