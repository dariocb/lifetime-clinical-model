"""Validate/export Mermaid source, then render its canonical topology to SVG."""
import argparse
import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import jsonschema

ROOT = Path(__file__).resolve().parent


def embed_diagrams(names):
    targets = {
        'tpp_a': ('01_tpp_models.ipynb', 'a8c59504'),
        'tpp_b': ('01_tpp_models.ipynb', '461b0499'),
        'tpp_c': ('01_tpp_models.ipynb', '70c26934'),
        'gp_a': ('02_gp_models_1_2.ipynb', 'c07fbd1b'),
        'gp_b': ('02_gp_models_1_2.ipynb', 'e88af4b2'),
        'gp_c': ('03_gp_model_3.ipynb', '8ed79a09'),
    }
    paths = sorted({targets[name][0] for name in names})
    for filename in paths:
        notebook_path = ROOT.parent / 'notebooks/models' / filename
        notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
        chosen = [name for name in names if targets[name][0] == filename]
        anchors = {targets[name][1]: name for name in chosen}
        diagram_ids = {name.replace('_', '-') + '-diagram' for name in chosen}
        cells, found = [], set()
        for cell in notebook['cells']:
            if cell.get('id') in diagram_ids:
                continue
            cells.append(cell)
            name = anchors.get(cell.get('id'))
            if name is None:
                continue
            found.add(name)
            cells.append({
                'cell_type': 'code', 'id': name.replace('_', '-') + '-diagram', 'metadata': {},
                'source': ['from IPython.display import display\n', 'import base64\n',
                           f'# {name.replace("_", "-").upper()}: inputs → outputs\n',
                           'display({\n',
                           f"    'image/svg+xml': (ROOT / 'diagrams/{name}.svg').read_text(encoding='utf-8'),\n",
                           f"    'image/png': base64.b64encode((ROOT / 'diagrams/{name}.png').read_bytes()).decode('ascii'),\n",
                           '}, raw=True)'],
                'execution_count': None,
                'outputs': [{'output_type': 'display_data', 'metadata': {}, 'data': {
                    'image/svg+xml': (ROOT / f'{name}.svg').read_text(encoding='utf-8'),
                    'image/png': base64.b64encode((ROOT / f'{name}.png').read_bytes()).decode('ascii'),
                }}],
            })
        if found != set(chosen):
            raise ValueError(f'Could not locate model explanations for {set(chosen) - found}')
        notebook['cells'] = cells
        notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill-dir", type=Path, help="Mermaid Diagrams skill directory")
    parser.add_argument("--node", help="Node.js executable")
    parser.add_argument("--node-modules", type=Path, help="Directory containing @viz-js/viz")
    parser.add_argument("--models", choices=("all", "tpp", "gp"), default="all",
                        help="Which model diagrams to regenerate and embed")
    args = parser.parse_args()
    skill = args.skill_dir
    if skill is None:
        matches = sorted((Path.home() / ".codex/plugins/cache/openai-curated-remote/mermaid-diagrams").glob("*/skills/mermaid-diagrams"))
        if not matches:
            parser.error("Provide --skill-dir for the installed Mermaid Diagrams skill")
        skill = matches[-1]
    tool = skill / "scripts/mermaid_tool.py"
    schema = json.loads((skill / "references/canonical-v2.schema.json").read_text(encoding="utf-8"))
    runtime = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node"
    node = args.node or shutil.which("node") or str(runtime / "bin/node.exe")
    modules = args.node_modules or Path(os.environ.get("CODEX_NODE_MODULES", str(runtime / "node_modules")))
    names = [f"{family}_{letter}" for family in ("tpp", "gp")
             if args.models in ("all", family) for letter in "abc"]
    for name in names:
        spec = ROOT / f"{name}.json"
        jsonschema.validate(json.loads(spec.read_text(encoding="utf-8")), schema)
        subprocess.run([sys.executable, str(tool), "create", str(spec), str(ROOT / f"{name}.mmd"),
                        "--format", "mmd", "--overwrite"], check=True)
        subprocess.run([sys.executable, str(tool), "lint", str(spec)], check=True)
        subprocess.run([node, str(ROOT / "render_svg.cjs"), str(spec),
                        str(ROOT / f"{name}.svg"), str(modules)], check=True)
    embed_diagrams(names)
    print("Generated diagrams and embedded SVG outputs with PNG fallbacks.")


if __name__ == "__main__":
    main()
