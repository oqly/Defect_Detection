# main.py
import sys
from PyQt5.QtWidgets import QApplication
from view import MainView
from viewmodel import MainViewModel

if __name__ == "__main__":
    app = QApplication(sys.argv)

    view_model = MainViewModel()
    view = MainView(view_model)

    view.show()
    sys.exit(app.exec_())
