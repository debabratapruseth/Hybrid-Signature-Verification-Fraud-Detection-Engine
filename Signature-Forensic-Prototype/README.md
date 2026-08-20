# AI Signature Detection, Verification and Forensic Analysis

A Google Colab–first research prototype that compares one questioned signature
with three reference signatures and screens the surrounding document for
possible digital-editing indicators.

For a file-by-file explanation, data contracts, conventions, and guidance for
future code-analysis systems, see [`CODE_GUIDE.md`](CODE_GUIDE.md).

> This system produces review-priority evidence. It does not determine
> authenticity, forgery, intent, identity, or legal validity.

## Run the project

The single annotated Colab runner is:

```text
colab_working/complete_signature_case_runner_cells.py
```

Copy each `# %%` section into a separate Colab cell and run them in order. The
runner:

1. mounts Google Drive and installs inference dependencies;
2. uploads one questioned PDF/image and exactly three references;
3. runs YOLO and pauses for human selection of the intended signature;
4. optionally expands or manually defines the signature box;
5. runs SAM segmentation and conservative OpenCV cleaning;
6. runs Branch 1 learned comparison, Branch 2 structural comparison, and
   Branch 3 document-forensics screening;
7. calculates deterministic three-branch fusion;
8. displays input and evidence images; and
9. uses OpenAI to generate senior and technical reports.

The OpenAI API key is requested securely at runtime and is not saved to Drive.

## Architecture

```text
Questioned document + 3 reference signatures
                         |
              Document quality processing
                         |
                 YOLO candidate boxes
                         |
                  Human box selection
                         |
             SAM segmentation + cleaning
                         |
       +-----------------+-----------------+
       |                 |                 |
    Branch 1          Branch 2          Branch 3
    Learned           Structural        Document
    comparison        AI geometry       forensics
       |                 |                 |
       +-----------------+-----------------+
                         |
          Reliability-adjusted deterministic fusion
                         |
          Evidence dashboard + two OpenAI reports
                         |
                 Qualified human review
```

SAM currently runs on every human-accepted box. It is not restricted to
low-quality images.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── config.yaml
├── colab_working/
│   ├── README.md
│   ├── complete_signature_case_runner_cells.py
│   └── train_models_cells.py
├── src/
│   ├── fresh_case_pipeline.py
│   ├── document.py
│   ├── detection.py
│   ├── segmentation.py
│   ├── cleaning.py
│   ├── verification.py
│   ├── forensics.py
│   ├── risk.py
│   ├── geometry/
│   ├── document_forensics/
│   └── branch_fusion/
├── models/
│   └── README.md
├── data/
│   └── README.md
└── outputs/
    └── .gitkeep
```

## Required persistent model artifacts

Place these files under `models/` in the Google Drive project:

```text
models/
├── yolo_signature_detector.pt
├── siamese_resnet18.pt
├── verification_calibration.json
└── cedar_external_test_report.json
```

SAM (`facebook/sam-vit-base` by default) is downloaded into the temporary Colab
cache when required.

## Branches and fusion

- Branch 1 compares learned signature representations and includes extraction
  quality, explainable comparison, duplicate evidence, and the saved CEDAR
  generalization result.
- Branch 2 compares skeletons, contours, curvature, critical points, Hu moments,
  Fourier descriptors, Shape Context, and skeleton graphs.
- Branch 3 screens for copy-paste placement, compression differences, noise
  differences, and possible reference-image reuse.
- Final fusion starts with weights of Branch 1 `45%`, Branch 2 `35%`, and
  Branch 3 `20%`. Reliability factors modify and renormalize these weights.

All thresholds and starting weights are visible in `config.yaml`. They are
prototype settings, not universally validated forensic thresholds.

## Data handling

- Process only documents and signatures you are authorized to use.
- Do not commit case documents, reference signatures, model weights, API keys,
  or generated output runs.
- Preserve original evidence files and store derived artifacts separately.
- Confirm that all three references represent the same claimed writer and
  normal signature form.
- Visually verify the complete selected signature box before running analysis.

## Output

Each case receives a timestamped directory:

```text
outputs/<run-id>/
├── case_manifest.json
├── risk_fusion_result.json
├── branch2_geometry/
├── branch3_document_forensics/
└── branch_fusion/
    ├── branch_fusion_result.json
    ├── report_evidence.json
    ├── senior_stakeholder_report.md
    ├── technical_fusion_report.md
    ├── report_metadata.json
    └── branch_fusion_dashboard.png
```
