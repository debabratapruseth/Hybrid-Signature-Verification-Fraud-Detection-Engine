# Consolidated Google Colab runner

For normal case analysis, use only:

```text
complete_signature_case_runner_cells.py
```

It contains the complete fresh-case workflow: setup, uploads, YOLO preview,
interactive padding, human box selection, all three branches, deterministic
fusion, evidence images, OpenAI reports, and persistent output saving.

The file uses `# %%` markers for readability. Copy each marked section into a
separate Google Colab cell and run them sequentially.

Required reusable code remains under `../src/`, configuration under
`../config.yaml`, and persistent checkpoints under `../models/`.

For occasional model development or checkpoint recreation, use:

```text
train_models_cells.py
```

The training notebook is deliberately separate from case inference. It creates
versioned YOLO, Siamese, calibration, held-out-test, and optional CEDAR
artifacts, and promotes them to canonical model paths only when explicit
`PROMOTE_*` flags are enabled.
