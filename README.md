# Anomalib Trainer

## Application purpose

Anomalib Trainer is a Windows desktop application for creating reproducible anomaly-detection training runs from inspection images. PatchCore, PaDiM, Dinomaly-DINOv2, and Dinomaly-DINOv3 have validated training paths; AnomalyDINO, SuperADD, EfficientAD, and SuperSimpleNet are explicitly experimental. No model is currently declared portable-export validated. Promotion requires a real train, evaluation, Torch export, reload, and score-parity pass for that exact configuration; unit tests alone are not production evidence.

## Screenshots

- `[placeholder]` Home / Projects
- `[placeholder]` Dataset import and validation
- `[placeholder]` Training progress and logs
- `[placeholder]` Results gallery and metrics
- `[placeholder]` Inference page

## Supported Windows versions

- Windows 10
- Windows 11

## Environment requirements

- Miniconda or Anaconda with Conda available on `PATH`
- PowerShell 5.1+ or PowerShell 7+

## Conda installation

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Backend cpu
```

For NVIDIA CUDA:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Backend cuda
```

The setup script detects RTX 50-series (Blackwell) GPUs and installs the matching
PyTorch 2.7.1 CUDA 12.8 build. Other NVIDIA GPUs use the CUDA 12.6 build. CUDA
setup verifies that the installed PyTorch supports the active GPU architecture
before it completes.

## GitHub installation

```powershell
git clone https://github.com/stefanoKS/aigaikan_training.git
cd aigaikan_training
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Backend cpu
```

## Running from source

```powershell
conda run --live-stream -n anomalib-trainer python -m app.main
```

or

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run.ps1
```

## Creating a project

1. Open **Home / Projects**.
2. Click **New Project**.
3. Enter a project name.
4. The project structure is created under `Documents\AnomalibProjects\<ProjectName>`.
5. Use **Save Project** to explicitly save its current state.

## Dataset folder meanings

- `ok_train`: required normal images used to train the model.
- `ok_validation` and `ng_validation`: optional explicit calibration images. When absent, the application derives a deterministic calibration partition from the available held-out images.
- `ok_test` and `ng_test`: final held-out test images. If `ok_test` is absent, the application reserves a deterministic normal holdout from `ok_train` before training. `ng_test` is optional for normal-only operation.
- `masks`: optional pixel-level masks for anomalous images. A missing or unselected mask folder never blocks training; pixel-level metrics are simply unavailable.

The application hashes every selected source image and rejects duplicates across training, calibration, and final-test partitions. Each run copies its exact deterministic split into `dataset_snapshot`; source folders are never moved, renamed, or modified.

## Fixed inspection region

The Inspection Region page can define one fixed four-point perspective quadrilateral on the original camera resolution. Its pixel corners are canonicalized as top-left, top-right, bottom-right, bottom-left; the natural rectified width and height come from the quadrilateral edge lengths. The application does not resize this region to an application-defined fixed size.

When enabled, every training, calibration, final-test, inference, reevaluation, and deployment-parity image follows the same contract:

```text
raw camera frame -> InspectionRegionProcessor -> rectified ROI -> Anomalib model
```

The project and every run persist canonical `inspection_region.json` metadata. Runs store its SHA-256 hash, source resolution, and rectified dimensions in `run_manifest.json`. Inputs at another resolution are rejected; the source files remain unchanged. A changed ROI requires retraining and prevents stale result/inference-run selection.

## Training

1. Select the OK folders and, when available, genuine NG folders from the Dataset page.
2. Optionally select the NG mask folder.
3. Validate the dataset from the Dataset page.
4. Choose a model with a lifecycle status appropriate for the work. Training-validated models are preferred; experimental models require local acceptance evidence.
5. Start training from the Training page.

The supported profiles are fixed and use Anomalib-native preprocessing: PatchCore uses `wide_resnet50_2`, `layer2`/`layer3`, coreset ratio `0.1`, nine neighbors, batch size 8, and one epoch; PaDiM uses `resnet18` with `layer1`/`layer2`/`layer3` and one epoch; Dinomaly-DINOv2 uses `vit_base_patch14_reg4_dinov2`; and Dinomaly-DINOv3 uses `vit_base_patch16_dinov3.lvd1689m`. Both Dinomaly profiles use decoder depth 8, bottleneck dropout 0.2, context recentering disabled, and an automatic budget of $\max(5000, \lceil \text{training images} / \text{batch size} \rceil)$ optimizer steps unless an explicit advanced override is saved. Dinomaly-DINOv3 uses Anomalib's `448x448` resize with a `384x384` center crop, because its $16$-pixel patch grid requires dimensions divisible by $16$. The application does not impose a global image-size preprocessor override.

Threshold calibration always uses held-out calibration predictions, never final-test predictions. Automatic calibration uses labeled F1 when genuine held-out OK and NG samples exist; otherwise it uses normal-only conformal calibration at the selected normal false-reject target (default $0.5\%$). Normal-only calibration establishes an operating point, not universal defect detection. The legacy maximum-score method is explicitly marked as conservative; synthetic anomaly calibration is unavailable until a dedicated generator can provide honest provenance.

Every prediction records its raw model score separately from its Anomalib-postprocessed score and map. Calibration and operational image thresholds apply only to the declared postprocessed score semantic and use `score >= threshold`. Raw scores are preserved for provenance and are not clamped into the postprocessed $[0, 1]$ domain.

For preprocessing contract v3, overlapping external tiles are feather-blended and the image decision score is calculated from the reconstructed valid-ROI map. Legacy v2 keeps its original maximum-overlap reconstruction behavior. SuperADD does not allow external tiling: its native patching and top-$0.1\%$ mean score aggregation are retained. Its native normalization and percentile thresholds are fitted from held-out OK validation images only; the application still performs its own post-fit calibration on all held-out evidence.

Every completed run records `dataset_manifest.json`, `calibration_manifest.json`, `final_test_manifest.json`, `config.json`, `environment.json`, `inspection_region.json`, `run_manifest.json`, `results.json`, final-test `predictions.csv`, and an immutable `evaluation_revisions` record. The run manifest identifies the Anomalib-selected checkpoint by path and SHA-256 hash and stores threshold method, value, revision, calibration manifest hash, sample counts, operating target, observed false-reject rate, score-distribution evidence, and fixed-ROI provenance. Inference and export reject runs whose canonical checkpoint, threshold, or inspection ROI metadata is missing, mismatched, or changed.

Before staging a run snapshot, deterministic model-ready image tiles are cached under the project `prepared_data_cache` directory. Each immutable entry is keyed by source SHA-256 and the resolved preprocessing-plan SHA-256, and records tile checksums and expected RGB dimensions. Invalid or corrupt entries are discarded and rebuilt before use; each run records cache hit, miss, and rebuild counts in its metrics and manifest. Run snapshots remain independent copied files for reproducibility. Delete `prepared_data_cache` while no training job is active to reclaim its storage; it will be rebuilt on the next run.

## Reading the results

The Results page shows:

- final-test predictions with source path, dataset role, ground truth, prediction, score, threshold, and correctness
- factory quality metrics including NG tested/detected/missed, NG detection rate, escape rate, false-reject count/rate, and the complete OK/NG confusion matrix when genuine NG test data exists
- threshold method, revision, calibration counts, normal false-reject target/observation, and observed score quantiles
- an active, immutable image/pixel threshold revision. Creating a revision regenerates labels, masks, contours, and visual artifacts from saved postprocessed continuous maps without running a model. Original `results.json` and final-test artifacts remain unchanged. `active_threshold_revision.json` selects a checksummed revision; the revision CSV and artifact directory are stored under `threshold_revisions`.
- anomaly maps rendered with a fixed unit-interval color transform. The continuous map is retained unchanged, valid-ROI masks are saved with it, and padded or invalid pixels remain transparent in heatmaps and unchanged in overlays.
- `NOT VERIFIED` and the prominent warning `NO GENUINE NG TEST DATA. DEFECT-DETECTION PERFORMANCE HAS NOT BEEN VERIFIED.` when no genuine NG final-test data exists. NG detection, escape, AUROC, precision, recall, and F1 are then `NOT MEASURED`.
- filterable result rows plus CSV/JSON export, result-folder access, and multi-run comparison. Direct quality comparisons require identical complete source split manifests and inspection ROI hashes.

AUROC and F1 are supplementary ranking metrics. An escaped NG is always treated as the primary quality failure.

## Model export

Export is blocked for every default model until it reaches `TORCH_EXPORT_VALIDATED`; the interface does not treat an unverified export as deployable. Once a model passes that evidence gate, export uses the manifest-verified canonical checkpoint rather than the newest file in a folder and packages its own `canonical_checkpoint.ckpt` with a relative manifest reference. OpenVINO exports must include both the `.xml` graph and `.bin` weights file. The package includes configuration, environment report, dataset/calibration/final-test/run manifests, `inspection_region.json`, result report, validation sidecars, and `deployment_manifest.json` hashes. Deployment validation requires exact decision parity and format-specific image-score parity (Torch: `0.0001`; ONNX/OpenVINO: `0.001`) against stored final-test predictions, using the same fixed ROI and reconstructed valid-map v3 scoring as the application. This is Anomalib deployment evidence only; it is not compatibility evidence for the separate AIGAIKAN runtime or defect-detection evidence when genuine NG test data is absent.

## Reevaluation without retraining

`EvaluationRevisionService` supports later evaluation using the original manifest-verified canonical checkpoint plus new calibration and final-test folders. It never calls model fitting. Each recalibration writes a new immutable revision with the unchanged checkpoint hash, new threshold metadata, new data-manifest hashes, quality evidence, and prediction CSV.

Torch `.pt` inference uses Anomalib's legacy `TorchInferencer`, which requires `TRUST_REMOTE_CODE=1` because PyTorch model loading can execute serialized Python code. Enable it only for artifacts from a trusted source. Exporting an artifact is not a substitute for acceptance testing it in the exact target runtime.

## Building the executable

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -EnvironmentName anomalib-trainer
```

## Creating the installer

```powershell
iscc installer\AnomalibTrainer.iss
```

## Offline installation

Use the weight download step before packaging:

```powershell
conda run --live-stream -n anomalib-trainer python scripts\download_weights.py
```

Then copy the built `release\` artifacts to the target PC.

## Common errors

### CUDA out of memory

- reduce batch size for PaDiM or Dinomaly
- switch device to CPU if necessary

### GPU architecture is unsupported

- rerun `powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Backend cuda`
- RTX 50-series GPUs require the CUDA 12.8 PyTorch build selected automatically by setup

### Missing model weights

- run `conda run --live-stream -n anomalib-trainer python scripts\download_weights.py`
- verify that `weights\wide_resnet50_2-default.pth` exists
- if internet access is blocked, download the weights on a connected PC and copy them into the `weights` folder

### PyTorch DLL issues

- confirm the `anomalib-trainer` Conda environment uses Python 3.11
- rerun `powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Backend cpu` or replace `cpu` with `cuda`
- verify VC++ runtime availability on the target machine

## Developer architecture

- `app/models`: typed dataclasses for project, dataset, config, predictions, and runs
- `app/core`: project management, validation, parsing, settings, environment inspection
- `app/services`: Anomalib, export, image, and visualization abstractions
- `app/workers`: separate worker entrypoints for training and inference
- `app/ui`: Qt Widgets pages inside a `QMainWindow` shell

## Model support policy

Lifecycle status is evidence-based. Training-validated models have automated training-path coverage; experimental models have not met that bar. `TORCH_EXPORT_VALIDATED` requires real train/export/reload numerical and decision-parity evidence, and is the only status permitted to export. No default definition currently has that status.
