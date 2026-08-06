# Implementation Plan

## Phase 1: Foundation
- Create the application package structure and packaging files.
- Implement typed configuration models and project persistence.
- Implement logging, settings, and a modular `QMainWindow` shell.
- Add CPU-safe unit tests for the foundational services.

## Phase 2: Dataset
- Implement dataset import/copy-reference behavior and thumbnail/statistics support.
- Implement dataset validation with separate warnings and errors.
- Add tests for empty folders, unsupported files, corrupt images, Unicode paths, and duplicate detection.

## Phase 3: Training
- Implement the model registry and Anomalib service abstraction.
- Implement the training worker with JSON Lines progress events.
- Control training from the UI via `QProcess`, including cancellation and logging.

## Phase 4: Results
- Parse worker metrics and predictions into internal result models.
- Render metrics, confusion matrix, histogram, and result galleries in the UI.
- Support CSV/JSON export and persisted run metadata.

## Phase 5: Inference
- Implement model loading and image/folder inference in a separate worker process.
- Display inference results and export them to CSV.

## Phase 6: Deployment
- Pin dependencies and add setup/build/installer scripts.
- Add a PyInstaller spec and Inno Setup script.
- Expand README instructions for source usage, offline setup, and packaging.
