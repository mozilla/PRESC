"""Add tags to all code cells in the notebooks."""

import nbformat as nbf
from pathlib import Path

TAGS_ALL_CELLS = ["remove-input"]
REPORT_DIR = Path(__file__).parent / ".." / "presc" / "report" / "resources"

notebooks = REPORT_DIR.glob("*.ipynb")

for ipath in notebooks:
    print(f"Adding tags in {ipath}")
    ntbk = nbf.read(ipath, nbf.NO_CONVERT)
    for cell in ntbk.cells:
        if cell.cell_type != "code":
            continue

        cell_tags = cell.metadata.get("tags", [])
        for tag_to_add in TAGS_ALL_CELLS:
            if tag_to_add not in cell_tags:
                cell_tags.append(tag_to_add)
        if len(cell_tags) > 0:
            cell.metadata["tags"] = cell_tags

    nbf.write(ntbk, ipath)
