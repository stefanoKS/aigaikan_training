# Anomalib Trainer

## Application purpose

Anomalib Trainer is a Windows desktop application for creating reproducible anomaly-detection training runs from inspection images. Its production workflow is validated for PatchCore and stock Anomalib Dinomaly with DINOv2 encoders. Dinomaly-DINOv3 and SuperADD-DINOv3 are separate experimental options.

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

## Training

1. Select the OK folders and, when available, genuine NG folders from the Dataset page.
2. Optionally select the NG mask folder.
3. Validate the dataset from the Dataset page.
4. Choose PatchCore, stock `dinomaly_dinov2`, or an explicitly marked experimental DINOv3 option on the Training Configuration page.
5. Start training from the Training page.

PatchCore and native SuperADD build a memory bank in one epoch. Dinomaly-DINOv2 defaults to a target of 3000 optimizer steps; the application derives its saved epoch count from the eligible training image count and batch size. Dinomaly-DINOv3 is an application-side reconstruction adapter, not stock Anomalib Dinomaly. It inspects the loaded encoder's patch size, embedding width, transformer depth, heads, preprocessing, and intermediate-token contract at runtime; incompatible encoders are rejected rather than guessed.

Threshold calibration always uses held-out calibration predictions, never final-test predictions. Automatic calibration uses labeled F1 when genuine held-out OK and NG samples exist; otherwise it uses normal-only conformal calibration at the selected normal false-reject target (default $0.5\%$). Normal-only calibration establishes an operating point, not universal defect detection. The legacy maximum-score method is explicitly marked as conservative; synthetic anomaly calibration is unavailable until a dedicated generator can provide honest provenance.

Every completed run records `dataset_manifest.json`, `calibration_manifest.json`, `final_test_manifest.json`, `config.json`, `environment.json`, `run_manifest.json`, `results.json`, final-test `predictions.csv`, and an immutable `evaluation_revisions` record. The run manifest identifies the Anomalib-selected checkpoint by path and SHA-256 hash and stores threshold method, value, revision, calibration manifest hash, sample counts, operating target, observed false-reject rate, and score-distribution evidence. Inference and export reject runs whose canonical checkpoint or threshold is missing or has changed.

## Reading the results

The Results page shows:

- final-test predictions with source path, dataset role, ground truth, prediction, score, threshold, and correctness
- factory quality metrics including NG tested/detected/missed, NG detection rate, escape rate, false-reject count/rate, and the complete OK/NG confusion matrix when genuine NG test data exists
- threshold method, revision, calibration counts, normal false-reject target/observation, and observed score quantiles
- `NOT VERIFIED` and the prominent warning `NO GENUINE NG TEST DATA. DEFECT-DETECTION PERFORMANCE HAS NOT BEEN VERIFIED.` when no genuine NG final-test data exists. NG detection, escape, AUROC, precision, recall, and F1 are then `NOT MEASURED`.
- filterable result rows plus CSV/JSON export, result-folder access, and multi-run comparison. Direct quality comparisons require identical complete source split manifests; preprocessing differences do not prevent comparison.

AUROC and F1 are supplementary ranking metrics. An escaped NG is always treated as the primary quality failure.

## Model export

Select OpenVINO, ONNX, and/or Torch in the Results page. Export uses the run-manifest checkpoint rather than the newest file in a folder, verifies its SHA-256 hash first, and checks that every produced artifact is nonempty. OpenVINO exports must include both the `.xml` graph and `.bin` weights file. Every export creates a named deployment folder containing the validated artifact, final-test predictions, configuration, environment report, dataset/calibration/final-test/run manifests, result report, validation sidecars, and `deployment_manifest.json` hashes. Deployment sidecars and manifests preserve the exact threshold metadata and require decision parity against stored final-test predictions; this verifies runtime consistency, not defect-detection performance when genuine NG test data is absent.

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

- reduce batch size
- reduce image width and height
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

## How to add another Anomalib model

1. Add a new `ModelDefinition` to `app/core/model_registry.py`.
2. Extend `AnomalibService` with the correct model constructor and defaults.
3. Expose the model on the Training Configuration page.
4. Normalize any new metrics in `app/core/result_parser.py`.
5. Add focused tests for the new configuration and result parsing behavior.
