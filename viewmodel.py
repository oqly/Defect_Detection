from PyQt5.QtCore import QObject
from model import DefectDetectionModel


class MainViewModel(QObject):
    def __init__(self):
        super().__init__()
        self.model = DefectDetectionModel()

    def analyze_file(self, file_path):
        return self.model.analyze(file_path)

    def get_analysis_history(self):
        self.model.cursor.execute('SELECT * FROM analysis_results')
        return self.model.cursor.fetchall()
