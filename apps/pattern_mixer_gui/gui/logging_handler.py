"""Qt logging handler for GUI log panel."""
from __future__ import annotations

import logging

from PySide6 import QtCore, QtWidgets


class QtLogEmitter(QtCore.QObject):
    """Qt signal emitter for log messages."""

    message_emitted = QtCore.Signal(str)

    @QtCore.Slot(str)
    def _emit_message(self, message: str) -> None:
        self.message_emitted.emit(message)


class QtLogHandler(logging.Handler):
    """Logging handler that emits log messages to a Qt widget."""

    def __init__(self, emitter: QtLogEmitter) -> None:
        super().__init__()
        self._emitter = emitter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:
            message = record.getMessage()
        QtCore.QMetaObject.invokeMethod(
            self._emitter,
            "_emit_message",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(str, message),
        )


class LogPanel(QtWidgets.QWidget):
    """Log panel widget with append-only text display."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._text = QtWidgets.QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(1000)
        self._text.setStyleSheet("font-family: Consolas, Monaco, monospace; font-size: 11px;")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._text)

    def append_line(self, message: str) -> None:
        self._text.appendPlainText(message)

    def clear(self) -> None:
        self._text.clear()
