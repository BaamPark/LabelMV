import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QLabel, QLineEdit, QWidget, QComboBox
from PyQt5.QtCore import Qt
import subprocess


class LauncherWindow(QMainWindow):
    def __init__(self, parent=None):
        super(LauncherWindow, self).__init__(parent)
        self.setWindowTitle("LabelMV Launcher")
        self.setGeometry(100, 100, 400, 300)

        # Number input section
        self.label_number = QLabel("Set Number of camera views:", self)
        self.label_number.setAlignment(Qt.AlignCenter)
        self.number_input = QLineEdit(self)
        self.number_input.setPlaceholderText("Enter a number")
        self.number_input.setAlignment(Qt.AlignCenter)

        # Resolution dropdown
        self.label_resolution = QLabel("Select Resolution:", self)
        self.label_resolution.setAlignment(Qt.AlignCenter)
        self.resolution_dropdown = QComboBox(self)
        self.resolution_dropdown.addItems([
            "800x600", "1024x768", "1280x720", 
            "1366x768", "1920x1080", "2560x1440"
        ])  # Add more options as needed

        # Button to open main UI
        self.main_ui_button = QPushButton("Open Main UI", self)
        self.main_ui_button.clicked.connect(self.launch_main_ui)

        # Layout setup
        layout = QVBoxLayout()
        layout.addWidget(self.label_number)
        layout.addWidget(self.number_input)
        layout.addWidget(self.label_resolution)
        layout.addWidget(self.resolution_dropdown)
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

            # Get the selected resolution
            selected_resolution = self.resolution_dropdown.currentText()

            # Launch the main UI and pass the number and resolution as arguments
            subprocess.Popen([sys.executable, "main.py", entered_number, selected_resolution])

            # Close the launcher UI
            self.close()
        except Exception as e:
            print(f"Failed to launch main.py: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    launcher = LauncherWindow()
    launcher.show()
    sys.exit(app.exec_())
