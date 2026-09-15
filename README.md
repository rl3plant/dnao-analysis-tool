# DNAO Analysis Tool

Desktop tool (PySide6) for annotating DNA-origami instances in AFM images:
click or box-select to segment with SAM2, review/correct proposals,
fuse/split annotations, label them, and train a classifier from corrections.

## Setup

Check out [`neo-toolkit`](../neo-toolkit) as a sibling directory (needed for
segmentation), then:

```bash
./setup.sh
./run.sh
```

See `setup.sh`/`requirements.txt` for manual install steps.

## Tests

```bash
pytest tests/
```
