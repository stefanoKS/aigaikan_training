"""Application styles."""

APP_STYLE = """
QMainWindow {
    background: #f4f6f8;
}
QListWidget {
    background: #1f2933;
    color: white;
    border: none;
    padding: 8px;
}
QListWidget::item {
    padding: 12px 10px;
    margin: 2px 0;
    border-radius: 6px;
}
QListWidget::item:selected {
    background: #3e6ae1;
}
QGroupBox {
    font-weight: bold;
    border: 1px solid #d9e2ec;
    border-radius: 8px;
    margin-top: 10px;
    background: white;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
}
"""

