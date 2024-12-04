import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QLabel, QLineEdit, QWidget
from PyQt5.QtCore import Qt
import subprocess


class LauncherWindow(QMainWindow):
    def __init__(self, parent=None):
        super(LauncherWindow, self).__init__(parent)
        self.setWindowTitle("Launcher UI")
        self.setGeometry(100, 100, 400, 200)

        # Number input section
        self.label = QLabel("Set Number:", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.number_input = QLineEdit(self)
        self.number_input.setPlaceholderText("Enter a number")
        self.number_input.setAlignment(Qt.AlignCenter)

        # Button to open main UI
        self.main_ui_button = QPushButton("Open Main UI", self)
        self.main_ui_button.clicked.connect(self.launch_main_ui)

        # Layout setup
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.number_input)
        layout.addWidget(self.main_ui_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def launch_main_ui(self):
        """Launch the main UI and close the launcher."""
        try:
            # Get the entered number
            entered_number = self.number_input.text()
            if not entered_number.isdigit():
                entered_number = "0"  # Default value if input is not valid

            # Launch the main UI and pass the number as an argument
            subprocess.Popen([sys.executable, "main.py", entered_number])

            # Close the launcher UI
            self.close()
        except Exception as e:
            print(f"Failed to launch main.py: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    launcher = LauncherWindow()
    launcher.show()
    sys.exit(app.exec_())
