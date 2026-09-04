"""Runtime localization for visible desktop UI text only."""

from __future__ import annotations

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QAbstractButton, QComboBox, QGroupBox, QLabel, QLineEdit, QTableWidget, QWidget


class UiTranslator(QObject):
    """Translate fixed UI labels between English and Japanese without changing application data."""

    language_changed = Signal()
    _TEXT_SOURCE_PROPERTY = "ui_translation_source_text"
    _TOOLTIP_SOURCE_PROPERTY = "ui_translation_source_tooltip"
    _ITEM_SOURCE_ROLE = int(Qt.ItemDataRole.UserRole) + 1

    _JAPANESE = {
        "English": "英語",
        "Japanese": "日本語",
        "Language": "言語",
        "PROJECT WORKSPACE": "プロジェクト ワークスペース",
        "NO PROJECT OPEN": "プロジェクト未選択",
        "Home / Projects": "ホーム / プロジェクト",
        "HOME / PROJECTS": "ホーム / プロジェクト",
        "Dataset": "データセット",
        "Inspection Region": "検査領域",
        "Preprocess Images": "画像前処理",
        "Training Configuration": "学習設定",
        "Training": "学習",
        "Results": "結果",
        "Inference": "推論",
        "New Project": "新規プロジェクト",
        "Open Project": "プロジェクトを開く",
        "Save Project": "プロジェクトを保存",
        "Current Project": "現在のプロジェクト",
        "Project Name": "プロジェクト名",
        "Project Path": "プロジェクト パス",
        "Created": "作成日時",
        "Last Opened": "最終オープン",
        "Last Training Status": "最終学習ステータス",
        "Not trained": "未学習",
        "Recent Projects": "最近のプロジェクト",
        "Import Mode": "取り込み方法",
        "Copy images into project": "画像をプロジェクトへコピー",
        "Reference original folder": "元フォルダーを参照",
        "Validate Dataset": "データセットを検証",
        "Clear Dataset Selection": "データセット選択をクリア",
        "OK Train Folder": "OK 学習フォルダー",
        "OK Validation Folder (Optional)": "OK 検証フォルダー（任意）",
        "NG Validation Folder (Optional)": "NG 検証フォルダー（任意）",
        "OK Final Test Folder (Optional)": "OK 最終テストフォルダー（任意）",
        "NG Final Test Folder": "NG 最終テストフォルダー",
        "NG Mask Folder (Optional)": "NG マスクフォルダー（任意）",
        "Folder Path": "フォルダー パス",
        "Image Count": "画像数",
        "Invalid Images": "無効な画像",
        "Source Resolution": "元画像解像度",
        "Color Mode": "カラーモード",
        "Actions": "操作",
        "Preview": "プレビュー",
        "Select Folder": "フォルダーを選択",
        "Open Folder": "フォルダーを開く",
        "No preview\navailable": "プレビューなし",
        "Preview\nunavailable": "プレビューを表示できません",
        "Expected Mask": "想定マスク",
        "Use a grayscale binary PNG when possible: 0 = background, 255 = defect. Keep the same dimensions as its NG image and name it image.png or image_mask.png.": "可能な場合はグレースケールの二値 PNG を使用してください。0 = 背景、255 = 欠陥です。対応する NG 画像と同じ寸法にし、image.png または image_mask.png と命名してください。",
        "Dataset Validation": "データセット検証",
        "Summary": "概要",
        "Effective Split": "有効な分割",
        "Level": "レベル",
        "Role": "役割",
        "Message": "メッセージ",
        "Path": "パス",
        "Training Configuration": "学習設定",
        "Model": "モデル",
        "Compatibility": "互換性",
        "Device": "デバイス",
        "Random Seed": "乱数シード",
        "Split Seed": "分割シード",
        "Data-loader Workers": "データローダー ワーカー数",
        "Auto": "自動",
        "Trainer Settings": "トレーナー設定",
        "Batch Size": "バッチサイズ",
        "Max Epochs": "最大エポック数",
        "Estimated Training Steps": "推定学習ステップ数",
        "Validate Every (Epochs)": "検証間隔（エポック）",
        "Gradient Clip Norm": "勾配クリップ ノルム",
        "Accumulate Batches": "バッチ累積数",
        "Disabled": "無効",
        "Dinomaly Training": "Dinomaly 学習",
        "Encoder": "エンコーダー",
        "Availability": "利用可能性",
        "Training Steps Override": "学習ステップ上書き",
        "Automatic baseline": "自動ベースライン",
        "Preprocessing Policy": "前処理ポリシー",
        "Rectified ROI Size": "補正済み ROI サイズ",
        "Padding Policy": "パディング ポリシー",
        "Automatic Right Padding": "右自動パディング",
        "Automatic Bottom Padding": "下自動パディング",
        "Custom Right Padding": "右カスタムパディング",
        "Custom Bottom Padding": "下カスタムパディング",
        "Prepared Image Size": "準備済み画像サイズ",
        "Model Alignment Requirement": "モデル整列要件",
        "Use Nearest Valid Size": "最寄りの有効サイズを使用",
        "Reset to Automatic": "自動に戻す",
        "Tiling": "タイル処理",
        "Enable horizontal tile processing": "水平方向のタイル処理を有効化",
        "Score Aggregation": "スコア集約",
        "Top-k Fraction": "上位 k の割合",
        "Automatic": "自動",
        "Custom": "カスタム",
        "Maximum valid-pixel score": "有効ピクセルの最大スコア",
        "Top-k valid-pixel mean": "有効ピクセル上位 k の平均",
        "Decision Threshold Calibration": "判定しきい値の校正",
        "Calibration Method": "校正方法",
        "Normal False Reject Target": "正常品の誤排出目標",
        "Custom Normal False Reject Target": "カスタム正常品誤排出目標",
        "NG Recall Target": "NG 再現率目標",
        "Require minimum NG recall": "最小 NG 再現率を要求",
        "Required NG Recall": "必要な NG 再現率",
        "Pixel Mask": "ピクセルマスク",
        "Enable pixel mask threshold": "ピクセルマスクしきい値を有効化",
        "Pixel Map Threshold": "ピクセルマップしきい値",
        "Final-Test Acceptance Policy": "最終テスト受け入れポリシー",
        "Maximum False Reject Rate": "最大誤排出率",
        "Minimum OK Test Images": "最小 OK テスト画像数",
        "Minimum NG Test Images": "最小 NG テスト画像数",
        "Save Configuration": "設定を保存",
        "Load Configuration": "設定を読み込む",
        "Reset to Defaults": "既定値に戻す",
        "Start Training": "学習を開始",
        "Cancel Training": "学習をキャンセル",
        "Open Log File": "ログファイルを開く",
        "Run Evaluation": "評価を実行",
        "Training Status": "学習ステータス",
        "Current Stage": "現在のステージ",
        "Stage Progress": "ステージ進捗",
        "Overall Progress": "全体進捗",
        "Elapsed Time": "経過時間",
        "Active Model": "使用モデル",
        "Active Device": "使用デバイス",
        "Dataset Counts": "データセット数",
        "Filter": "フィルター",
        "All": "すべて",
        "Correct OK": "正解 OK",
        "Correct NG": "正解 NG",
        "False OK": "見逃し NG",
        "False NG": "誤検出 NG",
        "Highest anomaly score": "異常スコアが高い順",
        "Lowest anomaly score": "異常スコアが低い順",
        "Export Results CSV": "結果 CSV を出力",
        "Export Metrics JSON": "メトリクス JSON を出力",
        "Open Result Folder": "結果フォルダーを開く",
        "Compare Runs": "実行結果を比較",
        "Model Export": "モデル出力",
        "Project directory": "プロジェクト ディレクトリ",
        "Browse": "参照",
        "Destination": "出力先",
        "Export for AIGAIKAN": "AIGAIKAN 用に出力",
        "Advanced Formats": "追加形式",
        "Metrics": "メトリクス",
        "Metric": "メトリクス",
        "Value": "値",
        "Original": "元画像",
        "Anomaly Map": "異常マップ",
        "Overlay": "オーバーレイ",
        "Continuous Map": "連続マップ",
        "Pixel Mask": "ピクセルマスク",
        "Contour Overlay": "輪郭オーバーレイ",
        "Pixel Threshold": "ピクセルしきい値",
        "Predicted": "予測",
        "Ground Truth": "正解ラベル",
        "Score": "スコア",
        "Source Path": "ソースパス",
        "Not available": "利用不可",
        "Not produced": "未生成",
        "Inspection Region": "検査領域",
        "Enable fixed ROI": "固定 ROI を有効化",
        "Previous dataset image": "前のデータセット画像",
        "Next dataset image": "次のデータセット画像",
        "Random": "ランダム",
        "Reset ROI": "ROI をリセット",
        "Save ROI": "ROI を保存",
        "No dataset image available": "利用可能なデータセット画像がありません",
        "Rectified ROI": "補正済み ROI",
        "ROI disabled": "ROI は無効です",
        "Select four corners": "4 つの角を選択",
        "No source image": "元画像なし",
        "Load Training Run": "学習済み実行を読み込む",
        "Select Image": "画像を選択",
        "Select Folder": "フォルダーを選択",
        "Run Inference": "推論を実行",
        "Cancel": "キャンセル",
        "Export Inference CSV": "推論 CSV を出力",
        "Export NG Images": "NG 画像を出力",
        "Select Training Run": "学習実行を選択",
        "Select Benchmark Image": "ベンチマーク画像を選択",
        "Select Benchmark Folder": "ベンチマークフォルダーを選択",
        "Start Benchmark": "ベンチマークを開始",
        "Export Benchmark JSON": "ベンチマーク JSON を出力",
        "Export Benchmark CSV": "ベンチマーク CSV を出力",
        "Exports selected NG rows, or every NG detection when no rows are selected.": "選択した NG 行を出力します。行を選択しない場合は、すべての NG 検出を出力します。",
        "Inference Summary": "推論概要",
        "Training Run": "学習実行",
        "Input": "入力",
        "Anomaly Score": "異常スコア",
        "Predicted Result": "予測結果",
        "Training Threshold": "学習しきい値",
        "NG Export Threshold": "NG 出力しきい値",
        "Use custom export threshold": "カスタム出力しきい値を使用",
        "Use custom NG image copy filter": "カスタム NG 画像コピー条件を使用",
        "Status": "ステータス",
        "Inference Log": "推論ログ",
        "Selected Prediction": "選択した予測",
        "Source": "ソース",
        "Prediction": "予測",
        "Heat Map": "ヒートマップ",
        "No training run loaded": "学習済み実行が未選択です",
        "No image or folder selected": "画像またはフォルダーが未選択です",
        "No image": "画像なし",
        "Ready": "準備完了",
        "New Project": "新規プロジェクト",
        "Project name": "プロジェクト名",
        "Open Anomalib Project": "Anomalib プロジェクトを開く",
        "Could Not Create Project": "プロジェクトを作成できません",
        "Could Not Open Project": "プロジェクトを開けません",
        "Select Model Export Folder": "モデル出力フォルダーを選択",
        "No Project": "プロジェクト未選択",
        "Missing Dependencies": "依存関係が不足しています",
        "Open a project before exporting a model.": "モデルを出力する前にプロジェクトを開いてください。",
        "No Training Run": "学習実行未選択",
        "Complete or load a training run before exporting a model.": "モデルを出力する前に学習を完了するか、学習実行を読み込んでください。",
        "Model Export Failed": "モデル出力に失敗しました",
        "Model Export Partially Completed": "モデル出力が一部完了しました",
        "Model Export Completed": "モデル出力が完了しました",
        "No model formats were exported.": "モデル形式は出力されませんでした。",
        "Complete or load a training run before exporting results.": "結果を出力する前に学習を完了するか、学習実行を読み込んでください。",
        "Export Results CSV": "結果 CSV を出力",
        "Export Results JSON": "結果 JSON を出力",
        "Complete or load a training run before opening results.": "結果を開く前に学習を完了するか、学習実行を読み込んでください。",
        "Complete or load a training run before comparing results.": "比較する前に学習を完了するか、学習実行を読み込んでください。",
        "Select Two or More Results JSON Files to Compare": "比較する 2 つ以上の結果 JSON ファイルを選択",
        "Could Not Compare Runs": "実行結果を比較できません",
        "Run Comparison": "実行結果の比較",
        "Run Comparison Evidence Warning": "実行結果の比較に関する証拠の警告",
        "Create or open a project first.": "最初にプロジェクトを作成または開いてください。",
        "Could Not Save Project": "プロジェクトを保存できません",
        "Project Saved": "プロジェクトを保存しました",
        "Project settings were saved.": "プロジェクト設定を保存しました。",
        "Could Not Select Folder": "フォルダーを選択できません",
        "Folder Not Available": "フォルダーを利用できません",
        "No existing folder is selected for this dataset role.": "このデータセットの役割に既存フォルダーは選択されていません。",
        "Create or open a project before validating data.": "データを検証する前にプロジェクトを作成または開いてください。",
        "Dataset Ready": "データセットの準備ができました",
        "The selected dataset is ready for training.": "選択したデータセットは学習に使用できます。",
        "Dataset Needs Attention": "データセットを確認してください",
        "Resolve the listed errors before training.": "学習前に一覧のエラーを解決してください。",
        "Invalid Inspection ROI": "検査 ROI が無効です",
        "Inspection ROI Saved": "検査 ROI を保存しました",
        "The fixed inspection region was saved to the project.": "固定検査領域をプロジェクトに保存しました。",
        "Invalid Training Settings": "学習設定が無効です",
        "Configuration Saved": "設定を保存しました",
        "Training settings were saved to the project.": "学習設定をプロジェクトに保存しました。",
        "Create or open a project before starting training.": "学習開始前にプロジェクトを作成または開いてください。",
        "Could Not Start Training": "学習を開始できません",
        "Training Failed": "学習に失敗しました",
        "Select Completed Training Run": "完了した学習実行を選択",
        "Invalid Training Run": "学習実行が無効です",
        "Select Image for Inference": "推論する画像を選択",
        "Select Image Folder for Inference": "推論する画像フォルダーを選択",
        "No Inference Input": "推論入力未選択",
        "Load a completed training run first.": "最初に完了した学習実行を読み込んでください。",
        "Select an image or image folder first.": "最初に画像または画像フォルダーを選択してください。",
        "Could Not Start Inference": "推論を開始できません",
        "Inference Failed": "推論に失敗しました",
        "No Inference Results": "推論結果がありません",
        "Run inference before exporting predictions.": "予測を出力する前に推論を実行してください。",
        "Export Inference Results": "推論結果を出力",
        "No NG Detections": "NG 検出はありません",
        "No selected results meet the current NG export threshold.": "選択した結果に現在の NG 出力しきい値を満たすものはありません。",
        "Export Raw NG Images": "生の NG 画像を出力",
        "Could Not Export NG Images": "NG 画像を出力できません",
        "Threshold Revision Failed": "しきい値改訂に失敗しました",
        "Deployment Threshold Preview Failed": "デプロイしきい値プレビューに失敗しました",
        "Invalid Preprocessing Profile": "前処理プロファイルが無効です",
        "Preprocessing Profile Saved": "前処理プロファイルを保存しました",
        "Image preprocessing changed. Train and calibrate a new run before using inference.": "画像前処理が変更されました。推論を使用する前に新しい実行を学習して校正してください。",
        "Training Failed": "学習に失敗しました",
        "Invalid Benchmark Run": "ベンチマーク実行が無効です",
        "Benchmark Input Required": "ベンチマーク入力が必要です",
        "Select a completed SuperADD training run and benchmark image or folder.": "完了した SuperADD 学習実行とベンチマーク画像またはフォルダーを選択してください。",
        "Benchmark Could Not Start": "ベンチマークを開始できません",
        "No Benchmark Result": "ベンチマーク結果がありません",
        "Run an industrial benchmark before exporting its result.": "結果を出力する前に産業用ベンチマークを実行してください。",
        "Decision Revision Failed": "判定改訂に失敗しました",
        "Confirm Decision Revision": "判定改訂の確認",
        "Load a completed training run before changing decisions.": "判定を変更する前に完了した学習実行を読み込んでください。",
        "No Inference Input": "推論入力未選択",
        "Select an image or image folder first.": "最初に画像または画像フォルダーを選択してください。",
        "Could Not Start Inference": "推論を開始できません",
        "No Inference Results": "推論結果がありません",
        "Run inference before exporting predictions.": "予測を出力する前に推論を実行してください。",
        "Select Custom Image": "カスタム画像を選択",
        "Select Custom Folder": "カスタムフォルダーを選択",
        "Reset to Project Good Images": "プロジェクトの正常画像に戻す",
        "Save Preprocessing Profile": "前処理プロファイルを保存",
        "Project Good Images": "プロジェクトの正常画像",
        "Custom Image": "カスタム画像",
        "Custom Folder": "カスタムフォルダー",
        "Custom source is already rectified": "カスタム入力はすでに補正済み",
        "Automatic odd kernel": "奇数カーネルを自動選択",
        "Enable preview zoom": "プレビュー拡大を有効化",
        "Enable pixel-value inspection": "ピクセル値の確認を有効化",
        "Torch (.pt)": "PyTorch（.pt）",
        "OpenVINO IR (.xml + .bin)": "OpenVINO IR（.xml + .bin）",
        "ONNX (.onnx)": "ONNX（.onnx）",
        "Generate pixel mask": "ピクセルマスクを生成",
        "Preview Effect": "影響をプレビュー",
        "Save and Activate Decision Revision": "判定改訂を保存して有効化",
        "Preview a different NG score threshold": "別の NG スコアしきい値をプレビュー",
        "Reset to Active Threshold": "有効なしきい値に戻す",
        "Optional operator note": "任意の作業者メモ",
        "Image Decision Threshold Preview": "画像判定しきい値プレビュー",
        "Proposed NG Score Threshold": "提案 NG スコアしきい値",
        "Operator Note": "作業者メモ",
        "Preview Summary": "プレビュー概要",
        "Decision-only preview. The model is not rerun, and heatmaps and pixel masks do not change.": "判定のみのプレビューです。モデルは再実行されず、ヒートマップとピクセルマスクは変更されません。",
        "Inference-Time Prediction": "推論時の判定",
        "Inference-Time Threshold": "推論時のしきい値",
        "Active / Preview Decision": "有効 / プレビュー判定",
        "Decision Change": "判定変更",
        "Displayed Decision Threshold": "表示中の判定しきい値",
        "Active Deployment NG Score Threshold": "有効なデプロイ NG スコアしきい値",
        "NG image copy filter": "NG 画像コピー条件",
        "ANOMALIB TRAINER": "ANOMALIB TRAINER",
        "Deployment Decision Revision": "デプロイ判定改訂",
        "Image Preprocessing Profile": "画像前処理プロファイル",
        "Industrial Inference Benchmark": "産業用推論ベンチマーク",
        "Preview Source": "プレビュー元",
        "SuperADD Settings": "SuperADD 設定",
        "Absolute Difference": "絶対差分",
        "Active preview source: Project Good Images": "使用中のプレビュー元: プロジェクトの正常画像",
        "Allowed Compute Budget": "許容演算時間",
        "Amortized Batch Time per Image": "画像あたりの平均バッチ時間",
        "AIGAIKAN Compatibility": "AIGAIKAN 互換性",
        "Anomalib Export Parity": "Anomalib 出力一致性",
        "Assessment": "評価",
        "Backbone": "バックボーン",
        "Benchmark Image/Folder": "ベンチマーク画像 / フォルダー",
        "Box Kernel Height": "ボックスカーネル高さ",
        "Box Kernel Width": "ボックスカーネル幅",
        "Calibrated NG Threshold": "校正済み NG しきい値",
        "Calibration False Reject Observed": "校正時の実測誤排出率",
        "Calibration False Reject Target": "校正時の誤排出率目標",
        "Calibration Images": "校正画像数",
        "Canonical Checkpoint": "正規チェックポイント",
        "Conservative P95 FPS": "保守的な P95 FPS",
        "Create new revision": "新しい改訂を作成",
        "Decision Score Ranges": "判定スコア範囲",
        "Decision Score Semantic": "判定スコアの意味",
        "Defect Detection Evidence": "欠陥検出の根拠",
        "Deployment NG Score Threshold": "デプロイ NG スコアしきい値",
        "Disk Iterations": "ディスク処理回数",
        "Disk Radius": "ディスク半径",
        "Disk Morphological Opening": "ディスク形態学的オープニング",
        "Disk opening removes bright structures that are too thin to contain the selected disk. Select a disk larger than the expected fiber thickness but smaller than the smallest important defect.": "ディスクオープニングは、選択したディスクを含めないほど細い明るい構造を除去します。想定する繊維の太さより大きく、重要な最小欠陥より小さいディスクを選択してください。",
        "End-to-End Compute P50 / P95 / P99": "エンドツーエンド演算 P50 / P95 / P99",
        "Evaluation Duration": "評価時間",
        "Expected Maximum Fiber Thickness": "想定最大繊維径",
        "Expected Minimum Defect Diameter": "想定最小欠陥径",
        "Export Status": "出力ステータス",
        "Feature Layers": "特徴レイヤー",
        "File-Source End-to-End Time": "ファイル入力のエンドツーエンド時間",
        "Final Preprocessed ROI": "最終前処理済み ROI",
        "Folder Batch Wall Time": "フォルダーバッチ実時間",
        "Gaussian Kernel": "ガウシアンカーネル",
        "Gaussian Kernel Size": "ガウシアンカーネルサイズ",
        "Gaussian Sigma": "ガウシアンシグマ",
        "Gaussian blur suppresses fine texture and may weaken defects smaller than its blur scale.": "ガウシアンぼかしは細かな模様を抑制し、ぼかしのスケールより小さい欠陥を弱める可能性があります。",
        "Historical legacy preprocessing": "過去の互換前処理",
        "Image AUROC": "画像 AUROC",
        "Image F1": "画像 F1",
        "Image-folder project": "画像フォルダープロジェクト",
        "Industrial benchmark completed": "産業用ベンチマークが完了しました",
        "Industrial benchmark failed or cancelled": "産業用ベンチマークが失敗またはキャンセルされました",
        "Inference-Time Prediction": "推論時の判定",
        "Inference-Time Threshold": "推論時のしきい値",
        "Inference-time: OK 0, NG 0 | Displayed: OK 0, NG 0 | OK -> NG: 0 | NG -> OK: 0": "推論時: OK 0、NG 0 | 表示: OK 0、NG 0 | OK -> NG: 0 | NG -> OK: 0",
        "Mask": "マスク",
        "Measured FPS": "実測 FPS",
        "Measured Frames": "計測フレーム数",
        "Median Blur": "メディアンぼかし",
        "Median Kernel Size": "メディアンカーネルサイズ",
        "Mode": "モード",
        "Model Forward P50 / P95": "モデル順伝播 P50 / P95",
        "Model Pipeline P50 / P95": "モデルパイプライン P50 / P95",
        "Model Pipeline Time": "モデルパイプライン時間",
        "Model details are unavailable.": "モデル詳細を取得できません。",
        "Morphology": "形態学処理",
        "Morphology Border": "形態学処理の境界",
        "NG Test Images": "NG テスト画像数",
        "NO GENUINE NG TEST DATA.\nDEFECT-DETECTION PERFORMANCE HAS NOT BEEN VERIFIED.": "実際の NG テストデータがありません。\n欠陥検出性能は未検証です。",
        "No Additional Preprocessing": "追加前処理なし",
        "No benchmark image or folder selected": "ベンチマーク画像またはフォルダーが未選択です",
        "No completed training run selected": "完了した学習実行が未選択です",
        "No curated SuperADD backbone is available in the installed timm runtime.": "インストール済み timm 実行環境で利用できる SuperADD バックボーンがありません。",
        "No inference results are available for an image decision preview.": "画像判定をプレビューする推論結果がありません。",
        "No readable preview image is available.": "読み込めるプレビュー画像がありません。",
        "None": "なし",
        "Normal-only calibration selects a false-reject operating point. Defect-detection performance remains unverified without genuine NG data.": "正常品のみの校正は誤排出の動作点を選択します。実際の NG データがない場合、欠陥検出性能は未検証のままです。",
        "Normal-only conformal": "正常品のみの共形校正",
        "Normal-only maximum (legacy)": "正常品のみの最大値（互換）",
        "Not measured": "未計測",
        "Not recorded": "未記録",
        "OK Test Images": "OK テスト画像数",
        "Original with ROI": "ROI 付き元画像",
        "PASS / FAIL": "合格 / 不合格",
        "Peak VRAM": "VRAM 最大使用量",
        "Pixel inspection disabled": "ピクセル値の確認は無効です",
        "Pixel Mask Threshold": "ピクセルマスクしきい値",
        "Pixels per Millimetre": "1 ミリメートルあたりのピクセル数",
        "Precision": "精度",
        "Preserve RGB": "RGB を保持",
        "Prepared Input Size": "準備済み入力サイズ",
        "Preprocessed ROI": "前処理済み ROI",
        "Preprocessing Compute Time": "前処理演算時間",
        "Preprocessing P50 / P95": "前処理 P50 / P95",
        "Preset": "プリセット",
        "Preview uses persisted scores and does not run the model.": "プレビューは保存済みスコアを使用し、モデルを実行しません。",
        "Proposed Deployment NG Score Threshold": "提案デプロイ NG スコアしきい値",
        "Quality Status": "品質ステータス",
        "Raw Score Ranges": "生スコア範囲",
        "Recall": "再現率",
        "Reflect": "反射",
        "Replicate": "複製",
        "Revision": "改訂",
        "Run Date": "実行日時",
        "Run Name": "実行名",
        "Safety Reserve %": "安全余裕 %",
        "Saved Preprocessing": "保存済み前処理",
        "Score Semantic": "スコアの意味",
        "Select an available curated SuperADD backbone.": "利用可能な管理済み SuperADD バックボーンを選択してください。",
        "Select an inspection ROI or dataset image": "検査 ROI またはデータセット画像を選択してください",
        "Smoothing Border": "平滑化の境界",
        "Smoothing Filter": "平滑化フィルター",
        "SuperADD uses its native top-0.1% anomaly-map mean; this setting is not modified.": "SuperADD はネイティブの異常マップ上位 0.1% 平均を使用します。この設定は変更されません。",
        "Target FPS": "目標 FPS",
        "Target Frame Budget": "目標フレーム時間",
        "Threshold Method": "しきい値方式",
        "Threshold Revision": "しきい値改訂",
        "Threshold Source / Revision": "しきい値の出所 / 改訂",
        "Training Duration": "学習時間",
        "Starting training": "学習を開始しています",
        "Completed": "完了",
        "Failed": "失敗",
        "Retraining required": "再学習が必要です",
        "Running inference": "推論を実行中",
        "Inference failed": "推論に失敗しました",
        "Validating dataset": "データセットを検証中",
        "Preparing datamodule": "データモジュールを準備中",
        "Loading model": "モデルを読み込み中",
        "Extracting normal features": "正常特徴を抽出中",
        "Building anomaly model": "異常検出モデルを構築中",
        "Evaluating test images": "テスト画像を評価中",
        "Generating visualizations": "可視化を生成中",
        "Saving results": "結果を保存中",
        "Training model": "モデルを学習中",
        "Calibrating model": "モデルを校正中",
        "True Batch-One Latency": "真のバッチ 1 レイテンシー",
        "True NG": "真の NG",
        "True OK": "真の OK",
        "Warmup Frames": "ウォームアップフレーム数",
        "Box Blur": "ボックスぼかし",
        "Gaussian Blur (Recommended)": "ガウシアンぼかし（推奨）",
        "Grayscale + Disk Opening": "グレースケール + ディスクオープニング",
        "Grayscale + Gaussian": "グレースケール + ガウシアンぼかし",
        "Grayscale + Gaussian + Disk Opening": "グレースケール + ガウシアンぼかし + ディスクオープニング",
        "Grayscale + Median": "グレースケール + メディアンぼかし",
        "Grayscale Only": "グレースケールのみ",
        "Grayscale, replicated to 3 channels": "グレースケール、3 チャンネルに複製",
        "Labeled F1": "ラベル付き F1",
        "Labeled recall priority": "ラベル付き再現率優先",
        "Automatic from held-out calibration data": "保留校正データから自動設定",
        "Camera-equivalent": "カメラ相当",
        "File end-to-end": "ファイルのエンドツーエンド",
        "FP16 - CUDA only": "FP16 - CUDA のみ",
        "FP32": "FP32",
        "CPU": "CPU",
        "CUDA": "CUDA",
        "DINOv3 Small: Fastest candidate.": "DINOv3 Small: 最速候補。",
        "DINOv3 Small+: Recommended real-time candidate.": "DINOv3 Small+: 推奨リアルタイム候補。",
        "DINOv3 Base: Balanced quality/latency candidate.": "DINOv3 Base: 品質と速度のバランス候補。",
        "DINOv3 Large: Slow candidate.": "DINOv3 Large: 低速候補。",
        "DINOv3 Huge+: Current/reference configuration; expected to be very slow.": "DINOv3 Huge+: 現在の基準構成であり、非常に低速と見込まれます。",
        "No curated encoder is available in the installed timm runtime.": "インストール済み timm 実行環境で利用できる管理済みエンコーダーがありません。",
        "All curated encoders are available.": "すべての管理済みエンコーダーを利用できます。",
        "Trainer Settings (Not used for zero-shot evaluation)": "トレーナー設定（ゼロショット評価では未使用）",
        "Preview uses the same ROI and preprocessing implementation as training and inference.": "プレビューは学習および推論と同じ ROI と前処理実装を使用します。",
        "Move over a preview to inspect RGB values.": "プレビュー上にカーソルを移動して RGB 値を確認します。",
        "Decision preview unavailable until every inference result matches the loaded run's decision score semantic.": "すべての推論結果が読み込み済み実行の判定スコアの意味と一致するまで、判定プレビューは使用できません。",
        "Image-folder project | Train and evaluate | Training Validated | Training path is validated. Export remains unavailable until train/export/reload parity is recorded.": "画像フォルダープロジェクト | 学習と評価 | 学習経路検証済み | 学習経路は検証済みです。学習、出力、再読み込みの一致性が記録されるまで出力は利用できません。",
        "ROI or padding changes require retraining.": "ROI またはパディングを変更した場合は再学習が必要です。",
        "Legacy preprocessing-v2 is retained unchanged for compatibility with existing runs.": "既存の実行との互換性のため、互換前処理 v2 は変更されずに保持されます。",
        "Dinomaly uses its Anomalib trainer defaults and the configured step budget.": "Dinomaly は Anomalib トレーナーの既定値と設定済みステップ予算を使用します。",
        "This model uses one memory-bank collection pass.": "このモデルは 1 回のメモリーバンク収集パスを使用します。",
        "EfficientAD requires one-image training batches.": "EfficientAD は画像 1 枚の学習バッチを必要とします。",
        "PatchCore uses batches of eight images.": "PatchCore は 8 画像のバッチを使用します。",
        "Tiling is currently supported only by Dinomaly models.": "タイル処理は現在 Dinomaly モデルでのみサポートされています。",
        "Previous preview image": "前のプレビュー画像",
        "Next preview image": "次のプレビュー画像",
        "Choose the kernel size automatically from the Gaussian sigma.": "ガウシアンシグマからカーネルサイズを自動選択します。",
        "Available when Automatic odd kernel is turned off.": "奇数カーネルの自動選択を無効にした場合に使用できます。",
        "Not specified": "未指定",
    }

    _DYNAMIC_JAPANESE_FRAGMENTS = (
        ("Active preview source: ", "使用中のプレビュー元: "),
        ("Project Good Images", "プロジェクトの正常画像"),
        ("Custom Image", "カスタム画像"),
        ("Custom Folder", "カスタムフォルダー"),
        ("Inference-time: ", "推論時: "),
        ("Displayed: ", "表示: "),
        ("OK -> NG: ", "OK -> NG: "),
        ("NG -> OK: ", "NG -> OK: "),
        (" steps (", " ステップ（"),
        (" epochs)", " エポック）"),
        ("Image-folder project", "画像フォルダープロジェクト"),
        ("Video project required", "動画プロジェクトが必要"),
        ("Train and evaluate", "学習と評価"),
        ("Zero-shot evaluation", "ゼロショット評価"),
        ("Training Validated", "学習経路検証済み"),
        ("Torch Export Validated", "Torch 出力検証済み"),
        ("Experimental", "実験的"),
        ("Training path is validated. Export remains unavailable until train/export/reload parity is recorded.", "学習経路は検証済みです。学習、出力、再読み込みの一致性が記録されるまで出力は利用できません。"),
        ("Export formats remain unavailable until deployment parity validation is completed.", "デプロイ一致性の検証が完了するまで出力形式は利用できません。"),
        ("Unavailable: ", "利用不可: "),
        ("Existing ", "既存値 "),
        (" -> proposed ", " -> 提案値 "),
        ("OK->NG changes: ", "OK->NG 変更: "),
        ("NG->OK changes: ", "NG->OK 変更: "),
        ("False-reject rate: ", "誤排出率: "),
        ("NG recall: ", "NG 再現率: "),
        ("Warning: proposed threshold is outside observed calibration score range.", "警告: 提案しきい値は観測済み校正スコア範囲の外です。"),
        ("Warning: SuperADD scores are distance values, not probabilities.", "警告: SuperADD スコアは確率ではなく距離値です。"),
        ("Completed: ", "完了: "),
        ("Exported ", "出力しました: "),
        ("Results saved to ", "結果を保存しました: "),
    )

    def __init__(self) -> None:
        super().__init__()
        self._language = "en"

    @property
    def language(self) -> str:
        """Return the active two-letter UI language code."""
        return self._language

    def set_language(self, language: str) -> None:
        """Set English or Japanese and notify UI owners to retranslate visible controls."""
        if language not in {"en", "ja"}:
            raise ValueError(f"Unsupported UI language: {language}")
        if language == self._language:
            return
        self._language = language
        self.language_changed.emit()

    def text(self, english: str) -> str:
        """Translate an English UI source string for the active language."""
        if self._language != "ja":
            return english
        translated = self._JAPANESE.get(english)
        if translated is not None:
            return translated
        for source, replacement in self._DYNAMIC_JAPANESE_FRAGMENTS:
            english = english.replace(source, replacement)
        return english

    def set_button_text(self, button: QAbstractButton, english: str) -> None:
        """Set dynamic button text while retaining its English source for later language changes."""
        translated = self.text(english)
        button.setProperty(self._TEXT_SOURCE_PROPERTY, english)
        button.setProperty(f"{self._TEXT_SOURCE_PROPERTY}_rendered", translated)
        button.setText(translated)

    def set_label_text(self, label: QLabel, english: str) -> None:
        """Set dynamic label text while retaining its English source for later language changes."""
        translated = self.text(english)
        label.setProperty(self._TEXT_SOURCE_PROPERTY, english)
        label.setProperty(f"{self._TEXT_SOURCE_PROPERTY}_rendered", translated)
        label.setText(translated)

    def apply(self, root: QWidget) -> None:
        """Translate fixed child-widget text while leaving runtime data values intact."""
        self._translate_window_title(root)
        for widget in (root, *root.findChildren(QWidget)):
            self._translate_widget_text(widget)
            self._translate_tooltip(widget)
            if isinstance(widget, QComboBox):
                self._translate_combo_items(widget)
            if isinstance(widget, QTableWidget):
                self._translate_table_headers(widget)

    def _translate_window_title(self, widget: QWidget) -> None:
        title = widget.windowTitle()
        if title in self._JAPANESE:
            widget.setWindowTitle(self.text(title))

    def _translate_widget_text(self, widget: QWidget) -> None:
        if isinstance(widget, QLabel):
            self._translate_property(widget, self._TEXT_SOURCE_PROPERTY, widget.text, widget.setText)
        elif isinstance(widget, QAbstractButton):
            self._translate_property(widget, self._TEXT_SOURCE_PROPERTY, widget.text, widget.setText)
        elif isinstance(widget, QGroupBox):
            self._translate_property(widget, self._TEXT_SOURCE_PROPERTY, widget.title, widget.setTitle)
        elif isinstance(widget, QLineEdit):
            self._translate_property(
                widget,
                self._TEXT_SOURCE_PROPERTY,
                widget.placeholderText,
                widget.setPlaceholderText,
            )

    def _translate_tooltip(self, widget: QWidget) -> None:
        self._translate_property(widget, self._TOOLTIP_SOURCE_PROPERTY, widget.toolTip, widget.setToolTip)

    def _translate_property(self, widget: QWidget, property_name: str, getter: object, setter: object) -> None:
        current = str(getter())  # type: ignore[operator]
        source = widget.property(property_name)
        rendered_property = f"{property_name}_rendered"
        rendered = widget.property(rendered_property)
        if source is not None and rendered != current:
            if not self._is_translatable(current):
                widget.setProperty(property_name, None)
                widget.setProperty(rendered_property, None)
                return
            source = current
            widget.setProperty(property_name, source)
        if source is None:
            if not self._is_translatable(current):
                return
            source = current
            widget.setProperty(property_name, source)
        translated = self.text(str(source))
        setter(translated)  # type: ignore[operator]
        widget.setProperty(rendered_property, translated)

    def _is_translatable(self, value: str) -> bool:
        return value in self._JAPANESE or any(source in value for source, _replacement in self._DYNAMIC_JAPANESE_FRAGMENTS)

    def _translate_combo_items(self, combo: QComboBox) -> None:
        for index in range(combo.count()):
            source = combo.itemData(index, self._ITEM_SOURCE_ROLE)
            if source is None:
                current = combo.itemText(index)
                if current not in self._JAPANESE:
                    continue
                source = current
                combo.setItemData(index, source, self._ITEM_SOURCE_ROLE)
            combo.setItemText(index, self.text(str(source)))

    def _translate_table_headers(self, table: QTableWidget) -> None:
        for index in range(table.columnCount()):
            item = table.horizontalHeaderItem(index)
            if item is None:
                continue
            source = item.data(self._ITEM_SOURCE_ROLE)
            if source is None:
                current = item.text()
                if current not in self._JAPANESE:
                    continue
                source = current
                item.setData(self._ITEM_SOURCE_ROLE, source)
            item.setText(self.text(str(source)))