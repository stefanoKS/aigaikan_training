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
decode RGB camera frame
	-> InspectionRegionProcessor four-point perspective rectification
	-> deterministic image preprocessing profile
	-> model-specific alignment/padding
	-> Anomalib preprocessing and normalization
```

## Preprocess Images

The **Preprocess Images** tab configures the deterministic profile that runs after the four-point ROI and before model-specific padding. It opens on **Project Good Images** by default; these images are preview-only and never estimate learned preprocessing parameters. A preview may instead use exactly one **Custom Image** or a deterministic non-recursive **Custom Folder**. Custom preview paths are saved only as project draft UI state and never enter run or deployment metadata. The tab provides Previous, Next, Random, and Reset-to-Project-Good navigation.

Custom raw camera frames must match the saved ROI source resolution. Mismatched inputs are displayed with a warning and are never stretched. A custom image can be explicitly declared already rectified; this is never inferred automatically and must match the saved rectified ROI size.

Profiles use RGB `uint8` input/output in the fixed $[0,255]$ range, preserve rectified dimensions, and serialize a strict ordered operation list:

```json
{
	"schema_version": 1,
	"profile_id": "custom_v1",
	"input_color_order": "RGB",
	"input_dtype": "uint8",
	"input_range": "0_255",
	"operation_order": "roi_then_image_operations_then_padding",
	"operations": [
		{"type": "grayscale", "luminance_standard": "itu_r_bt601_full_range", "output_channels": 3, "channel_replication": true},
		{"type": "gaussian_blur", "sigma": 1.0, "kernel_size": "automatic", "border_mode": "reflect"},
		{"type": "disk_morphological_opening", "radius": 2, "diameter": 5, "iterations": 1, "border_mode": "reflect"}
	]
}
```

Available color modes are Preserve RGB and BT.601 grayscale replicated to three identical channels. Smoothing supports None, Box Blur, Gaussian Blur, and Median Blur with curated Reflect or Replicate borders. Gaussian is the recommended interactive choice, but None remains the compatibility default; blur can suppress fine texture and weaken defects below its scale. Disk opening is grayscale erosion followed by grayscale dilation using an elliptical kernel with diameter $2r+1$. Select a disk larger than expected fiber thickness but smaller than the smallest important defect. Optional operator-supplied fiber, defect, and pixels-per-millimetre values produce only evidence-based warnings and physical disk metadata; the application never estimates them from Good images.

Convenience presets populate the same explicit controls: No Additional Preprocessing, Grayscale Only, Grayscale + Gaussian, Grayscale + Median, Grayscale + Disk Opening, and Grayscale + Gaussian + Disk Opening. The preview shows raw image plus ROI overlay, rectified ROI, final preprocessed ROI, and a fixed-range absolute difference. It uses the same `PreprocessingPipeline` as staging and inference; it does not use per-image display normalization.

Saving changed operations or parameters updates the preprocessing hash, marks prior results stale, and requires training plus recalibration. Changing only the preview source never changes a dataset split, model profile, hash, manifest, or result. Projects and historic runs without profile metadata resolve to `legacy_none_v1`, reproducing the prior RGB no-op behavior exactly.

New runs record the full profile in `config.json`, `preprocessing_plan.json` when non-legacy, `image_preprocessing.json`, `results.json`, `run_manifest.json`, and model provenance. The deployment bundle contains standalone `preprocessing.json`, its semantic SHA-256, runtime OpenCV/NumPy versions, `reference_runner/run_preprocessing_reference.py`, and `reference_runner/golden_vectors.json`. Run the bundle checker with:

```powershell
python reference_runner\run_preprocessing_reference.py --golden reference_runner\golden_vectors.json
```

For exact raw-frame model inputs, give the runner `--input`, `--output`, `--inspection-region inspection_region.json`, and `--resolved-plan preprocessing_plan.json`. The golden vectors verify both the profile and the full ROI-to-model-input route before a future runtime accepts a bundle.

The project and every run persist canonical `inspection_region.json` metadata. Runs store its SHA-256 hash, source resolution, and rectified dimensions in `run_manifest.json`. Inputs at another resolution are rejected; the source files remain unchanged. A changed ROI requires retraining and prevents stale result/inference-run selection.

## Training

1. Select the OK folders and, when available, genuine NG folders from the Dataset page.
2. Optionally select the NG mask folder.
3. Validate the dataset from the Dataset page.
4. Choose a model with a lifecycle status appropriate for the work. Training-validated models are preferred; experimental models require local acceptance evidence.
5. Start training from the Training page.

The supported profiles are fixed and use Anomalib-native preprocessing: PatchCore uses `wide_resnet50_2`, `layer2`/`layer3`, coreset ratio `0.1`, nine neighbors, batch size 8, and one epoch; PaDiM uses `resnet18` with `layer1`/`layer2`/`layer3` and one epoch; Dinomaly-DINOv2 uses `vit_base_patch14_reg4_dinov2`; and Dinomaly-DINOv3 uses `vit_base_patch16_dinov3.lvd1689m`. Both Dinomaly profiles use decoder depth 8, bottleneck dropout 0.2, context recentering disabled, and an automatic budget of $\max(5000, \lceil \text{training images} / \text{batch size} \rceil)$ optimizer steps unless an explicit advanced override is saved. Dinomaly-DINOv3 uses Anomalib's `448x448` resize with a `384x384` center crop, because its $16$-pixel patch grid requires dimensions divisible by $16$. The application does not impose a global image-size preprocessor override.

### SuperADD backbones and precision

SuperADD has a dedicated **SuperADD Settings** group. It exposes only the installed, curated DINOv3 `timm` backbones: Small, Small+, Base, Large, and Huge+. New selections persist their exact `timm` identifier and SuperADD precision (`float32` or CUDA-only `float16`) in the project and every completed run. Historical configurations that do not contain SuperADD fields continue to use the existing Huge+ `float32` profile, automatic feature layers, patch size `448`, overlap `16`, score quantile `0.001`, and the unchanged Anomalib memory-bank default.

Small is the fastest candidate, Small+ is the recommended real-time candidate, Base is a balanced quality/latency candidate, Large is slow, and Huge+ is the current/reference configuration and is expected to be very slow. Changing the SuperADD backbone or precision requires retraining and recalibration. Never copy a threshold between different backbone or precision runs: its score distribution is run-specific. SuperADD external tiling remains disabled and native top-$0.1\%$ anomaly-map aggregation remains unchanged.

Use this controlled experiment on the same inputs, ROI, preprocessing profile, padding, automatic feature layers, patch size, overlap, memory-bank default, and seeds:

1. Huge+ FP32 reference.
2. Base FP32.
3. Small+ FP32.
4. Base FP16 on CUDA.
5. Small+ FP16 on CUDA.

Retrain and recalibrate each run before comparison.

Threshold calibration always uses held-out calibration predictions, never final-test predictions. Automatic calibration uses labeled F1 when genuine held-out OK and NG samples exist; otherwise it uses normal-only conformal calibration at the selected normal false-reject target (default $0.5\%$). Normal-only calibration establishes an operating point, not universal defect detection. The legacy maximum-score method is explicitly marked as conservative; synthetic anomaly calibration is unavailable until a dedicated generator can provide honest provenance.

Every prediction records its raw model score separately from its Anomalib-postprocessed score and map. Calibration and operational image thresholds apply only to the declared postprocessed score semantic and use `score >= threshold`. Raw scores are preserved for provenance and are not clamped into the postprocessed $[0, 1]$ domain.
Every prediction records its raw model score separately from its Anomalib-postprocessed score and map. Calibration and operational image thresholds apply only to the declared decision-score semantic and use `score >= threshold`. Raw scores are preserved for provenance and are not clamped into the postprocessed $[0, 1]$ domain. SuperADD retains its native top-$0.1\%$ score aggregation and internal patching; its operational score semantic is recorded explicitly and its native normalized values remain provenance only.

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

Export is blocked for every default model until it reaches `TORCH_EXPORT_VALIDATED`; the interface does not treat an unverified export as deployable. **No default model is currently Torch-export validated.** PatchCore, PaDiM, and both Dinomaly profiles are training-validated only; AnomalyDINO, SuperADD, EfficientAD, and SuperSimpleNet remain experimental. The blocker is missing demonstrated real Anomalib export, reload, map, score, and decision-parity evidence for the exact model/configuration. No registry flag is changed by this repository work.

Once a model passes that separate evidence gate, a Torch package uses the manifest-verified canonical checkpoint rather than the newest file in a folder and packages `canonical_checkpoint.ckpt`, `deployment_manifest.json`, `decision_policy.json`, `preprocessing_plan.json`, `inspection_region.json`, `preprocessing.json`, `environment.json`, `config.json`, the Torch artifact, validation report, reference runners, and golden preprocessing vectors. Every referenced artifact has a SHA-256 recorded in the manifest. Contract version 3 adds the sidecar decision policy; version 2 manifests can be read for migration/audit but fail closed for production reference inference because they have no explicit decision policy.

`decision_policy.json` is authoritative for the image decision. It is independent from Anomalib's embedded thresholds and uses the unchanged rule `score >= threshold -> NG`. Thresholds are finite but are not restricted to $[0,1]$ because SuperADD distance scores may legitimately exceed one:

```json
{
	"decision_policy_version": 1,
	"threshold": 1.7,
	"comparator": ">=",
	"above_or_equal_label": "NG",
	"below_label": "OK",
	"score_semantic": "superadd_native_top_quantile_score_v1",
	"source": "operator_override",
	"base_calibrated_threshold": 0.7,
	"revision_id": "threshold-003",
	"operator_note": "line trial",
	"model_sha256": "...",
	"preprocessing_plan_sha256": "..."
}
```

The Results page keeps calibrated and active deployment thresholds separate from the proposed operator value. Preview uses persisted validation/final-test scores only, reports OK-to-NG and NG-to-OK changes plus measurable false-reject/recall values, and warns when an operator threshold lies beyond calibration observations. Saving creates and atomically activates an immutable `threshold-NNN` revision with the operator note; it preserves continuous maps and heatmaps. A pixel threshold remains independent and changes only binary masks and contour overlays. The Inference page calls its post-result copy filter **NG image copy filter**; it never changes the deployment policy or prediction labels.

The metadata-driven in-memory reference runner accepts raw RGB arrays without temporary PNG staging, verifies package checksums, applies saved ROI -> image profile -> padding/tiling -> model transform order, resolves the same semantic-safe decision score as training/inference/export validation, and keeps pixel masks independent:

```powershell
python scripts\deployment_reference_inference.py --package path\to\deployment --input frame.png --output result.json --device cpu
```

`result.json` contains the score, score semantic/source, decision, and versioned timing record. Run the batch-one benchmark separately; it excludes artifact saving from model latency and reports P50/P95/P99/maximum/throughput after 10 warm-ups and 100 measured frames by default:

```powershell
python scripts\benchmark_deployment_reference.py --package path\to\deployment --input frames --output benchmark.json
```

Example timing payload:

```json
{
	"timing_record_version": 2,
	"preprocess_total_ms": 1.42,
	"anomalib_transform_ms": 0.28,
	"host_to_device_ms": 0.14,
	"model_forward_ms": 4.83,
	"native_postprocess_ms": 0.31,
	"decision_postprocess_ms": 0.03,
	"model_pipeline_ms": 5.31,
	"end_to_end_compute_ms": 7.01,
	"artifact_io_ms": 0.0,
	"batch_size": 1,
	"tile_count": 1,
	"warmup_status": "not_warmed"
}
```

The existing temporary-PNG trainer inference path remains available for compatibility and records `staging_io_ms` separately. It reports batch wall time and amortized per-image time; true batch-one timing is recorded only when a real batch contains one image. CUDA reference timing uses synchronized CUDA events when a CUDA runner is selected. This is Anomalib deployment evidence only; it is not compatibility evidence for the separate AIGAIKAN runtime or defect-detection evidence when genuine NG test data is absent.

## Industrial checkpoint benchmark

The **Industrial Inference Benchmark** section on the Inference page launches a separate process, so the GUI remains responsive. It loads a completed SuperADD checkpoint once, keeps it resident, uses `model.eval()` and eager `torch.inference_mode()`, and measures batch size one. This is a checkpoint benchmark only, **not** a validated deployment export and it does not change any export-support status.

`camera-equivalent` preloads sorted source images as RGB `uint8` arrays before warmup. It measures from an available camera frame in RAM and explicitly excludes file decode, camera exposure/transport, PLC, and actuator latency. `file-end-to-end` decodes each source once per measured frame, reports decode separately, and includes it in file-source end-to-end latency. Neither mode saves PNGs, masks, heatmaps, overlays, NPZ files, or CSV rows in its timed production path.

The result reports P50/P95/P99, mean, standard deviation, min, and max for each measured phase. Industrial pass/fail uses P95 end-to-end compute latency with the selected reserve, not model-forward timing alone:

$$
	ext{allowed compute budget} = \frac{1000}{\text{target FPS}} \left(1 - \frac{\text{reserve percent}}{100}\right)
$$

Run the same benchmark from PowerShell for each completed training run:

```powershell
python scripts\benchmark_run_inference.py --run-dir "C:\runs\superadd-huge-fp32" --input "C:\benchmark-images" --device cuda --mode camera-equivalent --warmup 20 --iterations 200 --target-fps 10 --reserve-percent 20 --output "C:\benchmarks\huge-fp32.json" --csv-output "C:\benchmarks\huge-fp32.csv"
python scripts\benchmark_run_inference.py --run-dir "C:\runs\superadd-base-fp32" --input "C:\benchmark-images" --device cuda --mode camera-equivalent --warmup 20 --iterations 200 --target-fps 10 --reserve-percent 20 --output "C:\benchmarks\base-fp32.json"
python scripts\benchmark_run_inference.py --run-dir "C:\runs\superadd-small-plus-fp32" --input "C:\benchmark-images" --device cuda --mode camera-equivalent --warmup 20 --iterations 200 --target-fps 10 --reserve-percent 20 --output "C:\benchmarks\small-plus-fp32.json"
python scripts\benchmark_run_inference.py --run-dir "C:\runs\superadd-base-fp16" --input "C:\benchmark-images" --device cuda --mode camera-equivalent --warmup 20 --iterations 200 --target-fps 10 --reserve-percent 20 --output "C:\benchmarks\base-fp16.json"
python scripts\benchmark_run_inference.py --run-dir "C:\runs\superadd-small-plus-fp16" --input "C:\benchmark-images" --device cuda --mode camera-equivalent --warmup 20 --iterations 200 --target-fps 10 --reserve-percent 20 --output "C:\benchmarks\small-plus-fp16.json"
python scripts\compare_inference_benchmarks.py C:\benchmarks\base-fp32.json C:\benchmarks\base-fp16.json C:\benchmarks\small-plus-fp32.json C:\benchmarks\small-plus-fp16.json --output C:\benchmarks\comparison.csv
```

The comparison script warns when benchmark documents differ in input manifest, ROI, preprocessing profile, prepared canvas size, warmup count, measured count, or target FPS.

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
