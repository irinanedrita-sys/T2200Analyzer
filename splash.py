# splash.py
import sys, os
from PyQt6.QtWidgets import QSplashScreen
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, QTimer

def resource_path(*parts):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, *parts)

def show_splash(app, duration=2000):
    """
    Показать сплэш.
    duration: миллисекунды. Если None — НЕ ставим автозакрытие (закроем через splash.finish(window)).
    Возвращает объект QSplashScreen.
    """
    pix = QPixmap(resource_path("resources", "splash.png"))
    splash = QSplashScreen(pix)
    splash.setWindowFlag(Qt.WindowType.FramelessWindowHint)
    splash.show()
    app.processEvents()

    if duration is not None:
        # гарантируем int, даже если передали строку
        QTimer.singleShot(int(duration), splash.close)

    return splash
