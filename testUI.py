import sys
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class ClickableLabel(QLabel):
    def __init__(self, image_path, scaled=False, clickable=True, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.scaled = scaled
        self.clickable = clickable
        self.set_image(scaled)

    def set_image(self, scaled):
        try:
            pixmap = QPixmap(self.image_path)
            if pixmap.isNull():
                raise ValueError(f"Failed to load image: {self.image_path}")
            if scaled:
                pixmap = pixmap.scaled(100, 100, Qt.KeepAspectRatio)
            self.setPixmap(pixmap)
            self.scaled = scaled
            print(f"Set image {self.image_path} scaled: {scaled}")
        except Exception as e:
            print(f"Error loading image: {e}")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.clickable:
            print(f"Image {self.image_path} clicked")
            self.parentWidget().swap_images(self)

class ImageSwapApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        try:
            self.original_image_path = 'samples/Beomseok 6 ppl 20 sec_above_0.png'
            self.scaled_image_paths = [
                'samples/Beomseok 6 ppl 20 sec_above_1.png',
                'samples/Beomseok 6 ppl 20 sec_above_2.png'
            ]

            self.layout = QVBoxLayout()
            self.setLayout(self.layout)

            self.image_layout = QHBoxLayout()
            self.layout.addLayout(self.image_layout)

            self.original_image_label = ClickableLabel(self.original_image_path, scaled=False, clickable=False, parent=self)
            self.scaled_image_labels = [
                ClickableLabel(self.scaled_image_paths[0], scaled=True, clickable=True, parent=self),
                ClickableLabel(self.scaled_image_paths[1], scaled=True, clickable=True, parent=self)
            ]

            self.image_layout.addWidget(self.scaled_image_labels[0])
            self.image_layout.addWidget(self.original_image_label)
            self.image_layout.addWidget(self.scaled_image_labels[1])

            self.setWindowTitle('Click Swap')
            self.show()
            print("UI initialized successfully")
        except Exception as e:
            print(f"Error initializing UI: {e}")

    def swap_images(self, clicked_label):
        try:
            # Swap image paths
            original_image_path = self.original_image_label.image_path
            clicked_image_path = clicked_label.image_path

            # Update image paths
            self.original_image_label.image_path = clicked_image_path
            clicked_label.image_path = original_image_path

            # Reload images from paths
            self.original_image_label.set_image(scaled=False)
            clicked_label.set_image(scaled=True)

            print("Images swapped")
        except Exception as e:
            print(f"Error swapping images: {e}")

if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        ex = ImageSwapApp()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error running the application: {e}")
