"""Application styles."""

APP_STYLE = """
QMainWindow, QWidget#AppShell {
    background: #0d1012;
    color: #e7eff0;
    font-family: "Bahnschrift", "Segoe UI", sans-serif;
    font-size: 13px;
}
QWidget#AppHeader {
    background: #151b1e;
    border-bottom: 1px solid #2c3d40;
}
QLabel#BrandLogo {
    background: transparent;
    border: none;
}
QLabel#BrandTitle {
    color: #f1f7f7;
    font-size: 20px;
    font-weight: 700;
}
QLabel#WorkspaceTitle {
    color: #87a0a2;
    font-size: 13px;
    font-weight: 600;
}
QLabel#ProjectIndicator {
    background: #1a292b;
    border: 1px solid #38565a;
    border-radius: 5px;
    color: #80f2e2;
    font-weight: 700;
    padding: 7px 12px;
}
QListWidget#Navigation {
    background: #101518;
    border: none;
    color: #a6babd;
    outline: 0;
    padding: 12px 10px;
}
QListWidget#Navigation::item {
    border-left: 3px solid transparent;
    border-radius: 4px;
    margin: 3px 0;
    padding: 11px 12px;
}
QListWidget#Navigation::item:hover {
    background: #1e4b50;
    color: #f2fffe;
}
QListWidget#Navigation::item:selected {
    background: #153438;
    border-left-color: #35ddcf;
    color: #dcfffb;
    font-weight: 700;
}
QListWidget#RecentProjectsList {
    background: #101518;
    border: 1px solid #2c3d40;
    border-radius: 4px;
    color: #d0e0e1;
    outline: 0;
}
QListWidget#RecentProjectsList::item {
    border-bottom: 1px solid #223236;
    padding: 8px;
}
QListWidget#RecentProjectsList::item:hover, QListWidget#RecentProjectsList::item:selected {
    background: #153438;
    color: #dcfffb;
}
QStackedWidget {
    background: #0d1012;
}
QScrollArea#PageScrollArea {
    border: none;
}
QScrollArea#PageScrollArea::viewport, QWidget#WorkspacePage {
    background: #0d1012;
}
QSplitter::handle {
    background: #223236;
    width: 1px;
}
QGroupBox {
    background: #151b1e;
    border: 1px solid #2c3d40;
    border-radius: 5px;
    color: #e7eff0;
    font-size: 13px;
    font-weight: 700;
    margin-top: 13px;
    padding: 14px 12px 10px 12px;
}
QGroupBox::title {
    background: #151b1e;
    color: #80f2e2;
    left: 13px;
    padding: 0 5px;
    subcontrol-origin: margin;
}
QLabel {
    color: #e7eff0;
}
QLabel#ModelSupport {
    background: #10272a;
    border-left: 3px solid #32d6c7;
    color: #b7eeea;
    padding: 7px 9px;
}
QLabel#DatasetThumbnail {
    background: #050708;
    border: 1px dashed #405a5e;
    color: #829396;
    padding: 4px;
}
QLabel#MaskFormat {
    background: #403720;
    border-left: 3px solid #ffcf70;
    color: #ffdc9a;
    padding: 7px 9px;
}
QDialog, QMessageBox, QMenu {
    background: #151b1e;
    border: 1px solid #38565a;
    color: #e7eff0;
}
QDialog QLabel, QMessageBox QLabel, QMenu::item {
    background: transparent;
    color: #e7eff0;
}
QDialogButtonBox {
    background: #151b1e;
}
QFileDialog QListView, QFileDialog QTreeView, QFileDialog QSidebar {
    background: #101518;
    border: 1px solid #38565a;
    color: #e7eff0;
    selection-background-color: #153438;
    selection-color: #dcfffb;
}
QToolTip {
    background: #151b1e;
    border: 1px solid #38565a;
    color: #e7eff0;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {
    background: #101518;
    border: 1px solid #38565a;
    border-radius: 4px;
    color: #e7eff0;
    min-height: 30px;
    padding: 3px 8px;
    selection-background-color: #1e7775;
    selection-color: #ffffff;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus, QPlainTextEdit:focus {
    border: 2px solid #35ddcf;
}
QComboBox::drop-down {
    border: 0;
    width: 28px;
}
QComboBox QAbstractItemView {
    background: #151b1e;
    border: 1px solid #38565a;
    color: #e7eff0;
    selection-background-color: #153438;
    selection-color: #dcfffb;
}
QPushButton {
    background: #1a292b;
    border: 1px solid #38565a;
    border-radius: 5px;
    color: #d0e0e1;
    font-weight: 600;
    min-height: 32px;
    padding: 4px 13px;
}
QPushButton:hover {
    background: #263b3f;
    border-color: #78eadc;
    color: #f2fffe;
}
QPushButton:pressed {
    background: #10272a;
}
QPushButton:disabled {
    background: #151a1c;
    border-color: #293437;
    color: #62767a;
}
QPushButton#PrimaryButton {
    background: #153438;
    border-color: #35ddcf;
    color: #dcfffb;
}
QPushButton#PrimaryButton:hover {
    background: #1e4b50;
    border-color: #78eadc;
    color: #f2fffe;
}
QPushButton#AlertButton {
    background: #28191b;
    border-color: #753f42;
    color: #ffb1ad;
}
QPushButton#AlertButton:hover {
    background: #452124;
    border-color: #ff716b;
    color: #fff4f3;
}
QTableWidget {
    alternate-background-color: #101518;
    background: #151b1e;
    border: 1px solid #2c3d40;
    border-radius: 4px;
    color: #e7eff0;
    gridline-color: #223236;
    selection-background-color: #153438;
    selection-color: #dcfffb;
}
QHeaderView::section {
    background: #1a292b;
    border: none;
    border-bottom: 1px solid #38565a;
    color: #b7d4d5;
    font-weight: 700;
    padding: 7px;
}
QProgressBar {
    background: #101518;
    border: 1px solid #38565a;
    border-radius: 4px;
    color: #e7eff0;
    min-height: 18px;
    text-align: center;
}
QProgressBar::chunk {
    background: #32d6c7;
    border-radius: 3px;
}
QCheckBox {
    color: #e7eff0;
    spacing: 7px;
}
QCheckBox::indicator {
    background: #101518;
    border: 1px solid #405a5e;
    border-radius: 3px;
    height: 16px;
    width: 16px;
}
QCheckBox::indicator:checked {
    background: #32d6c7;
    border-color: #32d6c7;
}
QScrollBar:vertical {
    background: transparent;
    margin: 2px;
    width: 10px;
}
QScrollBar::handle:vertical {
    background: #405a5e;
    border-radius: 4px;
    min-height: 24px;
}
"""

