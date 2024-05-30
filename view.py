# view.py
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog, QHBoxLayout, QTextEdit, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import Qt, QMimeData
from PyQt5.QtGui import QDragEnterEvent, QDropEvent


class MainView(QMainWindow):
    def __init__(self, view_model):
        super().__init__()
        self.view_model = view_model

        self.setWindowTitle("Defect Detection Application")
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()

        self.drop_area = QLabel("Перенесите файлы 3D моделей сюда или нажмите, чтобы открыть")
        self.drop_area.setAlignment(Qt.AlignCenter)
        self.drop_area.setStyleSheet("QLabel { background-color : lightgray; border: 2px dashed gray; }")
        self.drop_area.setFixedHeight(200)
        self.drop_area.mousePressEvent = self.open_file_dialog
        self.drop_area.setAcceptDrops(True)
        self.drop_area.installEventFilter(self)
        self.layout.addWidget(self.drop_area)

        self.info_label = QTextEdit()
        self.info_label.setReadOnly(True)
        self.info_label.setStyleSheet("QTextEdit { background-color : white; }")
        self.layout.addWidget(self.info_label)

        self.button_layout = QHBoxLayout()

        self.clear_button = QPushButton("Очистить")
        self.button_layout.addWidget(self.clear_button)

        self.analyze_button = QPushButton("Анализ")
        self.analyze_button.setEnabled(False)
        self.button_layout.addWidget(self.analyze_button)

        self.history_button = QPushButton("История анализов")
        self.button_layout.addWidget(self.history_button)

        self.layout.addLayout(self.button_layout)

        self.container = QWidget()
        self.container.setLayout(self.layout)
        self.setCentralWidget(self.container)

        self.clear_button.clicked.connect(self.clear_files)
        self.analyze_button.clicked.connect(self.analyze_files)
        self.history_button.clicked.connect(self.view_analysis_history)

        self.loaded_files = []

    def update_analyze_button_state(self):
        if self.loaded_files:
            self.analyze_button.setEnabled(True)
        else:
            self.analyze_button.setEnabled(False)

    def eventFilter(self, source, event):
        if event.type() == event.DragEnter and source == self.drop_area:
            if event.mimeData().hasUrls():
                event.accept()
            else:
                event.ignore()
            return True
        elif event.type() == event.Drop and source == self.drop_area:
            if event.mimeData().hasUrls():
                for url in event.mimeData().urls():
                    if url.isLocalFile() and url.toLocalFile().endswith('.stl'):
                        self.loaded_files.append(url.toLocalFile())
                self.update_drop_area()
                self.update_analyze_button_state()
                event.accept()
            else:
                event.ignore()
            return True
        return super().eventFilter(source, event)

    def open_file_dialog(self, event):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilters(["STL files (*.stl)"])
        if file_dialog.exec_():
            files = file_dialog.selectedFiles()
            self.loaded_files.extend(files)
            self.update_drop_area()
            self.update_analyze_button_state()

    def update_drop_area(self):
        if self.loaded_files:
            self.drop_area.setText(f"Загружено файлов: {len(self.loaded_files)}")
        else:
            self.drop_area.setText("Перенесите файлы 3D моделей сюда или нажмите, чтобы открыть")

    def clear_files(self):
        self.loaded_files = []
        self.update_drop_area()
        self.info_label.clear()
        self.update_analyze_button_state()

    def analyze_files(self):
        results = []
        for file in self.loaded_files:
            result = self.view_model.analyze_file(file)
            if result == "Defective":
                results.append(f"<font color='red'>File: {file}\nResult: {result}</font>")
            else:
                results.append(f"File: {file}\nResult: {result}")
        self.info_label.setHtml("<br><br>".join(results))

    def view_analysis_history(self):
        history = self.view_model.get_analysis_history()
        self.history_window = QMainWindow()
        self.history_window.setWindowTitle("История анализов")
        self.history_window.setGeometry(100, 100, 1000, 400)

        table_widget = QTableWidget()
        table_widget.setColumnCount(5)
        table_widget.setHorizontalHeaderLabels(['ID', 'File Name', 'Date', 'Result', 'Model Version'])

        table_widget.setRowCount(len(history))
        for row_idx, row_data in enumerate(history):
            for col_idx, col_data in enumerate(row_data):
                table_widget.setItem(row_idx, col_idx, QTableWidgetItem(str(col_data)))

        table_widget.setColumnWidth(0, 50)  # ID
        table_widget.setColumnWidth(1, 200)  # File Name
        table_widget.setColumnWidth(2, 150)  # Date
        table_widget.setColumnWidth(3, 100)  # Result
        table_widget.setColumnWidth(4, 150)  # Model Version

        table_widget.resizeColumnsToContents()

        layout = QVBoxLayout()
        layout.addWidget(table_widget)

        container = QWidget()
        container.setLayout(layout)
        self.history_window.setCentralWidget(container)

        self.history_window.show()
