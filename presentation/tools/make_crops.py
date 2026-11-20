"""Crops two regions out of REAL screenshots so that their text is readable on a slide.

    python presentation/tools/make_crops.py        (needs Pillow; tooling only, not in requirements.txt)

The crops are plain rectangular cut-outs of the screenshots in docs/screenshots/ (nothing is redrawn or edited); the originals are untouched.
"""
from pathlib import Path
from PIL import Image

root = Path(__file__).resolve().parent.parent.parent
shots = root / "docs" / "screenshots"
out = root / "presentation" / "images"
out.mkdir(parents=True, exist_ok=True)

crops = [
    ("08_candidate_details_and_score_explanation.png", (380, 700, 1400, 1205), "crop_candidate_details.png"),
    ("04b_file_validation_edge_cases_scrolled.png", (380, 268, 1245, 670), "crop_status_edge_cases.png"),
]
for source, box, name in crops:
    image = Image.open(shots / source)
    image.crop(box).save(out / name)
    print("wrote", (out / name).relative_to(root), "from", source, box)
