# Code Guide and Data Contracts

This document explains the active codebase for students, reviewers, architects,
and code-analysis systems. It describes responsibilities and data movement; the
Python source remains authoritative for exact implementation details.

## 1. Safety and interpretation boundary

The software is a research prototype that prioritizes cases for human review.
No module is authorized to conclude that a signature is genuine, authentic,
forged, fraudulent, copied, or legally valid.

Three distinctions must remain visible:

1. **Detection** asks where signature-like content appears.
2. **Comparison** measures similarity or difference between supplied images.
3. **Document forensics** screens for configured editing indicators.

None of these questions independently answers who signed a document or whether
an observed difference is innocent, intentional, or fraudulent.

## 2. Main entry point

`colab_working/complete_signature_case_runner_cells.py` is the only user-facing
case-analysis runner. It is an annotated Python file divided by `# %%` markers
so each section can be pasted into one Google Colab cell.

The runner owns interactive concerns:

- mounting Google Drive;
- dependency installation;
- input upload;
- displaying reference signatures;
- displaying all YOLO candidates;
- interactive bounding-box padding;
- human selection of the relevant signature;
- displaying evidence;
- secure API-key entry; and
- displaying final reports.

The runner delegates analysis to reusable functions under `src/`. It should not
contain hidden scoring logic.

`colab_working/train_models_cells.py` is a separate model-development notebook.
It is not part of case inference. It stages datasets on the Colab VM, creates
versioned artifacts in Drive, and changes canonical inference checkpoints only
when explicit promotion flags are enabled.

## 3. End-to-end sequence

```text
Questioned PDF/image + exactly 3 reference images
                         |
                 src.document.load_document
                         |
             original_image + preprocessed_image
                         |
            src.detection.detect_signatures
                         |
          all YOLO candidates displayed to a human
                         |
        selected YOLO box OR recorded manual box
                         |
       src.segmentation.segment_accepted_detections
                         |
        src.cleaning.clean_segmentation_results
                         |
              cleaned questioned signature
                         |
       +-----------------+------------------+
       |                 |                  |
   Branch 1          Branch 2           Branch 3
   learned +         structural         document
   explainable       geometry           forensics
       |                 |                  |
       +-----------------+------------------+
                         |
       src.branch_fusion.calculate_branch_fusion
                         |
      deterministic result + evidence dashboard
                         |
      OpenAI senior report + technical report
```

## 4. Coordinate and image conventions

Understanding these conventions prevents subtle bugs:

- Color images are NumPy arrays in **RGB** order inside project code.
- OpenCV reads and writes **BGR**, so save/load helpers explicitly convert.
- Binary ink masks use Boolean `True` for foreground internally.
- Conventional saved binary images use black ink (`0`) on white (`255`).
- Bounding boxes use pixel coordinates `[x1, y1, x2, y2]`.
- `x1, y1` is the inclusive top-left corner.
- `x2, y2` is the exclusive bottom-right boundary used by NumPy slicing.
- Detection numbers shown to users are one-based.
- Python list indices are zero-based; orchestration converts between them.

## 5. Persistent case contract

Every run has a unique timestamped directory:

```text
outputs/<run-id>/
```

The three canonical inputs required by final fusion are:

```text
risk_fusion_result.json
branch2_geometry/branch2_geometry_result.json
branch3_document_forensics/branch3_document_forensics.json
```

`src/branch_fusion/evidence.py` fails early if one is absent or structurally
incompatible. This prevents evidence from different runs being silently mixed.

## 6. Top-level source modules

### `src/fresh_case_pipeline.py`

Inference-only orchestration for the three branches. It contains no model
training. Its public functions are:

- `validate_fresh_case`: validates one questioned document and exactly three
  readable reference paths.
- `preview_branch1_detections`: performs document preparation and YOLO only,
  saves preview artifacts, and returns candidates for human selection.
- `run_branch1_fresh`: accepts a selected candidate or manual box, then runs
  SAM, cleaning, saved Siamese inference, explainable comparisons, and Branch 1
  risk fusion.
- `run_branch2_fresh`: runs Structural AI and saves its JSON and images.
- `run_branch3_fresh`: runs Document Forensics and saves its JSON and images.

The optional `detection_preview` argument avoids running YOLO twice. A manual
box is marked `human_supplied_bounding_box` and receives no fabricated model
confidence.

### `src/document.py`

Loads the first or selected PDF page with PyMuPDF, or reads an image. It keeps an
independent copy of the original RGB evidence and produces a bounded-size
analysis copy. Quality observations include dimensions, brightness, blur, and
skew. Preprocessing may improve analysis but never replaces the preserved
original.

### `src/detection.py`

Loads a signature-specific Ultralytics YOLO checkpoint, predicts signature-like
boxes, creates unchanged crops, evaluates box quality, draws numbered previews,
and saves JSON-safe metadata. It does not decide which detected signature is
relevant to the case; that is a human selection in the Colab runner.

### `src/segmentation.py`

Uses a human-accepted box as a prompt for SAM. SAM may return several candidate
masks. The module scores candidates using predicted mask quality, plausible
area, boundary behavior, rectangularity, and agreement with an OpenCV ink
estimate. The selected mask keeps likely signature pixels and whites out the
background.

SAM currently runs for every accepted box, not only for poor-quality images.

### `src/cleaning.py`

Conservatively converts the SAM output into a normalized signature image. It
uses adaptive thresholding, straight-line screening, connected-component
filtering, light morphology, and a fixed output canvas.

The cleaning fallback is important: if proposed cleaning removes too much ink
or changes skeleton endpoints excessively, the module restores the original
SAM-derived ink and records a warning. A quality score of `1.0` can therefore
coexist with `passed=False` when fallback preserved the input.

### `src/verification.py`

Contains both reusable inference and retained model-definition/training
utilities. Production inference uses:

- `load_siamese_checkpoint`;
- `load_calibration`;
- `validate_reference_signatures`; and
- `compare_with_references`.

The shared ResNet-18 encoder maps each image to an L2-normalized embedding.
Cosine similarity is calculated between the questioned embedding and each
reference embedding. The median and spread are interpreted through saved
calibration boundaries.

Similarity is not an authenticity probability. Three references provide only
three reference-to-reference pairs, so observed writer variation remains
limited.

### `src/external_validation.py`

Recreates untouched CEDAR external validation after Siamese retraining. It
discovers CEDAR writers, creates balanced genuine/genuine and
genuine/skilled-forgery pairs, scores them with the already-trained model, and
reports operational metrics at the already-fixed BHSig260 validation threshold.
Its CEDAR-derived EER threshold is descriptive only and must not be promoted as
the operating threshold.

### `src/forensics.py`

Branch 1’s explainable comparison layer. It normalizes ink, extracts visible
features, aligns questioned and reference signatures, calculates XOR
differences, and screens for exact or geometrically supported digital reuse.
Its results contribute to Branch 1 risk and are distinct from Branch 2’s richer
Structural AI descriptors.

### `src/risk.py`

Branch 1 quality gates and risk fusion. It combines:

- verification risk;
- pipeline-quality risk;
- explainable forensic difference risk; and
- duplicate-reuse risk.

The configured starting weights are visible in `config.yaml`. The verification
weight is reduced when the saved untouched CEDAR report indicates weak transfer
to an external dataset. The result is a review-priority score, not a calibrated
probability.

## 7. Branch 2 — `src/geometry/`

Branch 2 contains deterministic shape and stroke measurements.

### `geometry/common.py`

Shared image loading, ink-mask normalization, JSON conversion, and visualization
helpers. All Branch 2 descriptors depend on its normalized mask contract.

### `geometry/skeleton.py`

Thins ink to a one-pixel-wide skeleton, estimates eight-connected length, and
finds endpoints, junctions, and raw branch pixels.

### `geometry/contours.py`

Finds external ink contours, chooses the dominant contour, resamples it to a
fixed number of points, and measures contour properties.

### `geometry/curvature.py`

Estimates local turning along a sampled contour and summarizes the curvature
distribution.

### `geometry/critical_points.py`

Combines curvature extrema with skeleton endpoints and junctions to describe
structurally important locations.

### `geometry/hu_moments.py`

Calculates seven scale/translation/rotation-oriented Hu shape moments and a
distance between two moment vectors.

### `geometry/fourier_descriptors.py`

Transforms a sampled contour into a compact frequency-domain representation and
calculates descriptor distance.

### `geometry/shape_context.py`

Describes the distribution of other contour points around each sampled point
using log-polar histograms. Matching cost is computed between two descriptor
sets.

### `geometry/graph.py`

Builds a graph-like summary from skeleton endpoints and junctions, then compares
node and connectivity statistics. It is a descriptive topology approximation,
not a recovered pen-stroke sequence.

### `geometry/similarity.py`

Maps individual descriptor distances to bounded similarities and combines them
using visible `geometry.metric_weights`. These weights are prototype settings,
not universal forensic constants.

### `geometry/visualization.py`

Creates overlays and mismatch heatmaps. Visuals are review aids and are not
independent model evidence.

### `geometry/pipeline.py`

Orchestrates every Branch 2 descriptor for one questioned signature and each
reference. It also compares reference signatures with each other so the
questioned scores can be described relative to the small observed reference
variation. It saves the canonical Branch 2 JSON and visualization files.

## 8. Branch 3 — `src/document_forensics/`

Branch 3 screens the original page and selected signature region for configured
digital-editing indicators.

### `document_forensics/common.py`

Loads RGB inputs, validates bounding boxes, and crops regions consistently.

### `document_forensics/orb_match.py`

Extracts ORB keypoints, applies ratio-tested descriptor matching, and optionally
uses RANSAC to count geometrically consistent matches. Thin or low-resolution
signatures often have too few keypoints; this is reported as unavailable
evidence rather than a negative finding.

### `document_forensics/perceptual_hash.py`

Calculates image hashes and Hamming distances. Perceptual similarity may indicate
reuse but does not prove copying or intent.

### `document_forensics/duplicate_detection.py`

Combines exact normalized equality, perceptual hashes, ORB matches, and RANSAC
support for questioned-to-reference reuse screening.

### `document_forensics/copy_paste.py`

Searches the document outside the source box for a possible second placement.
It combines ORB evidence with multiscale template matching and phase
correlation. PDF rendering, scale changes, compression, and sparse strokes can
limit every method.

### `document_forensics/compression.py`

Performs JPEG recompression/error-level screening and estimates blockiness.
Different rendering or scan history can produce the same patterns as editing,
so results remain nonspecific indicators.

### `document_forensics/noise_analysis.py`

Compares high-frequency residual statistics inside the signature region with a
surrounding region. Paper texture, ink density, shadows, scanners, and
compression can all change residuals.

### `document_forensics/risk.py`

Combines available copy-paste, compression, noise, and duplicate components
using visible weights. It separately records evidence availability so missing
ORB features do not masquerade as proof of no editing.

### `document_forensics/visualization.py`

Draws analysis regions and evidence panels.

### `document_forensics/pipeline.py`

Runs all Branch 3 screens against the preserved original page, removes large
working arrays from JSON, saves the canonical report, and writes visual evidence.

## 9. Final fusion — `src/branch_fusion/`

### `branch_fusion/evidence.py`

Defines canonical file locations, loads JSON, validates required fields, and
discovers available evidence images.

### `branch_fusion/fusion.py`

Calculates deterministic three-branch fusion. Starting weights are:

- Branch 1: `0.45`;
- Branch 2: `0.35`; and
- Branch 3: `0.20`.

Each weight is multiplied by a visible reliability factor and then normalized
again. Branch 2 reliability reflects limited reference-pair evidence. Branch 3
reliability reflects evidence coverage and missing feature methods. Quality
limitations can block a lower-priority conclusion.

### `branch_fusion/visualization.py`

Builds a compact dashboard containing representative input and output images,
branch risks, the fusion conclusion, and the required disclaimer.

### `branch_fusion/reports.py`

Creates a compact structured evidence bundle and sends it to OpenAI through the
Responses API. Images are not transmitted. Separate prompts produce:

- a plain-language senior stakeholder report; and
- a detailed technical report.

Returned Markdown is validated for mandatory headings, an explicit limitation,
and unsupported affirmative forensic verdicts. OpenAI explains the locked
deterministic result; it does not recalculate fusion.

### `branch_fusion/pipeline.py`

Provides the final orchestration boundary:

1. load configuration and all three canonical branch results;
2. calculate deterministic fusion;
3. generate both required OpenAI reports;
4. save JSON, Markdown, metadata, and the dashboard.

## 10. Configuration map

`config.yaml` is grouped by responsibility:

- `document`: rendering, resizing, and page quality;
- `detection`: YOLO checkpoint and box thresholds;
- `segmentation`: SAM model and mask-quality rules;
- `cleaning`: conservative ink-processing rules;
- `verification`: Siamese checkpoint, transforms, and saved-calibration paths;
- `forensics` and `duplicate_detection`: Branch 1 explainable checks;
- `geometry`: Branch 2 descriptor settings and metric weights;
- `document_forensics`: Branch 3 methods and component weights;
- `quality`, `risk_weights`, and `risk_decisions`: Branch 1 fusion;
- `reporting`: retained reporting preferences; and
- `branch_fusion`: final branch weights, reliability limits, decision
  boundaries, OpenAI model, and API-key environment-variable name.

Code should read configuration rather than duplicate thresholds in notebooks.
Fallback defaults exist only for compatibility and must remain visible.

## 11. Model and validation artifacts

The consolidated run expects:

```text
models/yolo_signature_detector.pt
models/siamese_resnet18.pt
models/verification_calibration.json
models/cedar_external_test_report.json
```

The YOLO and Siamese checkpoints are inference artifacts. The calibration JSON
maps similarities to prototype decision regions. The CEDAR report describes
external model behavior and adjusts reliability; CEDAR signatures are not used
as references for the current case.

## 12. Guidance for future code changes

When modifying the project:

1. Preserve original evidence separately from working images.
2. Never auto-select a signature merely because it has the highest confidence.
3. Record manual boxes explicitly.
4. Do not let SAM or cleaning silently remove substantial ink.
5. Keep missing evidence distinct from negative evidence.
6. Keep numerical fusion deterministic and outside the LLM.
7. Never describe a review-risk score as forgery probability.
8. Save enough intermediate evidence for a reviewer to reproduce the path.
9. Keep all three canonical branch JSON files tied to the same run ID.
10. Update this guide whenever a data contract or pipeline boundary changes.
