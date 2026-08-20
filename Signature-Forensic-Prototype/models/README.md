# Model checkpoints

Place locally trained or downloaded checkpoints in this directory. Checkpoint
files are ignored by Git because they can be large and may have separate licence
or redistribution terms.

Expected prototype paths from `config.yaml`:

```text
models/
├── yolo_signature_detector.pt
├── siamese_resnet18.pt
├── verification_calibration.json
└── cedar_external_test_report.json
```

The SAM checkpoint is expected to be downloaded through its model library cache.
The precise implementation will be documented in the segmentation notebook.

For reproducible experiments, record:

- model architecture and base checkpoint;
- dataset name, version, and licence;
- writer-disjoint split definition where applicable;
- training configuration and random seed;
- preprocessing version;
- validation metrics;
- calibration method and threshold values; and
- checkpoint hash.

A pretrained YOLO model is not automatically a signature detector, and a
pretrained ResNet-18 is not automatically a signature verifier. Both task-
specific models require appropriate training and validation.

Never treat a model confidence or similarity threshold as proof of authenticity.
