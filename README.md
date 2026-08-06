# Anomalib Trainer

## Application purpose

Anomalib Trainer is a Windows desktop application for non-programmers who need to create anomaly-detection projects, import OK/NG images, validate datasets, launch PatchCore training through a GUI, review results, and package the application for deployment on other Windows PCs.

## Screenshots

- `[placeholder]` Home / Projects
- `[placeholder]` Dataset import and validation
- `[placeholder]` Training progress and logs
- `[placeholder]` Results gallery and metrics
- `[placeholder]` Inference page

## Supported Windows versions

- Windows 10
- Windows 11

## Python requirements

- Python 3.11
- PowerShell 5.1+ or PowerShell 7+

## CPU installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements\cpu.txt
python scripts\verify_installation.py
```

## CUDA installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements\cuda.txt
python scripts\verify_installation.py
```

## GitHub installation

```powershell
git clone https://github.com/stefanoKS/aigaikan_training.git
cd aigaikan_training
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

## Running from source

```powershell
.venv\Scripts\Activate.ps1
python -m app.main
```

or

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run.ps1
```

## Creating a project

1. Open **Home / Projects**.
2. Click **New Project**.
3. Choose a parent directory.
4. The project structure is created under `AnomalibProjects\<ProjectName>`.

## Dataset folder meanings

- `ok_train`: normal images used for PatchCore training
- `ok_test`: normal images used during evaluation
- `ng_test`: anomalous images used during evaluation
- `masks`: optional pixel-level masks for anomalous test images

## Training PatchCore

1. Import `ok_train`, `ok_test`, and `ng_test`.
2. Optionally import `masks`.
3. Validate the dataset from the Dataset page.
4. Configure device, image size, batch size, and PatchCore settings on the Training Configuration page.
5. Start training from the Training page.

## Reading the results

The Results page is designed to show:

- normalized metrics such as Image AUROC, Image F1, precision, recall, and threshold
- a prediction gallery with original image, anomaly map, overlay, predicted label, ground-truth label, and score
- CSV and JSON export options

## Building the executable

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build.ps1
```

## Creating the installer

```powershell
iscc installer\AnomalibTrainer.iss
```

## Offline installation

Use the weight download step before packaging:

```powershell
.venv\Scripts\Activate.ps1
python scripts\download_weights.py
```

Then copy the built `release\` artifacts to the target PC.

## Common errors

### CUDA out of memory

- reduce batch size
- reduce image width and height
- switch device to CPU if necessary

### Missing model weights

- run `python scripts\download_weights.py`
- verify that `weights\wide_resnet50_2-default.pth` exists
- if internet access is blocked, download the weights on a connected PC and copy them into the `weights` folder

### PyTorch DLL issues

- confirm Python 3.11 is used
- reinstall the pinned torch/torchvision package pair
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
