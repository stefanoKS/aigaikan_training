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
        "Dataset": "データセット",
        "Inspection Region": "検査領域",
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
        "Exports selected NG rows, or every NG detection when no rows are selected.": "選択した NG 行を出力します。行を選択しない場合は、すべての NG 検出を出力します。",
        "Inference Summary": "推論概要",
        "Training Run": "学習実行",
        "Input": "入力",
        "Anomaly Score": "異常スコア",
        "Predicted Result": "予測結果",
        "Training Threshold": "学習しきい値",
        "NG Export Threshold": "NG 出力しきい値",
        "Use custom export threshold": "カスタム出力しきい値を使用",
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
    }

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
        return self._JAPANESE.get(english, english) if self._language == "ja" else english

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
            if current not in self._JAPANESE:
                widget.setProperty(property_name, None)
                widget.setProperty(rendered_property, None)
                return
            source = current
            widget.setProperty(property_name, source)
        if source is None:
            if current not in self._JAPANESE:
                return
            source = current
            widget.setProperty(property_name, source)
        translated = self.text(str(source))
        setter(translated)  # type: ignore[operator]
        widget.setProperty(rendered_property, translated)

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