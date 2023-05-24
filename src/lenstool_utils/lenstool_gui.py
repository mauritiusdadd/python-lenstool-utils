#!/usr/bin/env python
"""Just a simple lenstool wrapper based on QT."""

import os
import sys
import time
import platform
import subprocess
import argparse

from PyQt6 import QtWidgets, QtCore

from . import lenstool_wrapper


class LensToolPyQt():
    """
    PyQt6 wrapper for lenstool.

    Returns
    -------
    None.

    """

    def __init__(self):
        """
        Init the wrapper.

        Returns
        -------
        None.

        """
        self.lenstool = lenstool_wrapper.LensToolWrapper()

        self.qapp = QtWidgets.QApplication.instance()
        if self.qapp is None:
            # if it does not exist then a QApplication is created
            self.qapp = QtWidgets.QApplication([])

        self.main_wnd = QtWidgets.QMainWindow()
        self.main_wnd.setMinimumSize(800, 600)

        central_widget = QtWidgets.QWidget()
        self.main_wnd.setCentralWidget(central_widget)

        layout = QtWidgets.QVBoxLayout()

        self._options_gbox = QtWidgets.QGroupBox("Lenstool Options")
        options_gbox_layout = QtWidgets.QVBoxLayout()

        options_gbox_layout = QtWidgets.QGridLayout()
        self._lenstool_path_lable = QtWidgets.QLabel("lenstool path:")
        self._lenstool_path_txt = QtWidgets.QLineEdit()
        self._lenstool_path_chose = QtWidgets.QPushButton("Select")
        self._lenstool_path_chose.clicked.connect(self.openProgramFile)

        options_gbox_layout.addWidget(self._lenstool_path_lable, 0, 0)
        options_gbox_layout.addWidget(self._lenstool_path_txt, 0, 1)
        options_gbox_layout.addWidget(self._lenstool_path_chose, 0, 2)

        if self.lenstool.lenstool_exe:
            self._lenstool_path_txt.setText(self.lenstool.lenstool_exe)

        self._lenstool_input_lable = QtWidgets.QLabel("Param file")
        self._lenstool_input_txt = QtWidgets.QLineEdit()
        self._lenstool_input_chose = QtWidgets.QPushButton("Open")
        self._lenstool_input_chose.clicked.connect(self.openParamFile)

        options_gbox_layout.addWidget(self._lenstool_input_lable, 1, 0)
        options_gbox_layout.addWidget(self._lenstool_input_txt, 1, 1)
        options_gbox_layout.addWidget(self._lenstool_input_chose, 1, 2)

        self._options_gbox.setLayout(options_gbox_layout)
        layout.addWidget(self._options_gbox)

        self.prog_out = QtWidgets.QPlainTextEdit()
        self.prog_out.setReadOnly(True)
        self.prog_out.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )

        layout.addWidget(self.prog_out)

        pbar_layout = QtWidgets.QGridLayout()

        self._pbar_partprog_label = QtWidgets.QLabel("Partial progress")
        self._pbar_partial = QtWidgets.QProgressBar()
        self._pbar_partial.setMinimum(0)
        self._pbar_partial.setMaximum(1)
        self._pbar_partial.setValue(0)

        self._pbar_global_label = QtWidgets.QLabel("Total progress")
        self._pbar_global = QtWidgets.QProgressBar()
        self._pbar_global.setTextVisible(True)
        self._pbar_global.setMinimum(0)
        self._pbar_global.setMaximum(1)
        self._pbar_global.setValue(0)

        pbar_layout.addWidget(self._pbar_partprog_label, 0, 0)
        pbar_layout.addWidget(self._pbar_partial, 0, 1)
        pbar_layout.addWidget(self._pbar_global_label, 1, 0)
        pbar_layout.addWidget(self._pbar_global, 1, 1)

        layout.addLayout(pbar_layout)

        btn_run_layout = QtWidgets.QHBoxLayout()

        self._btn_run = QtWidgets.QPushButton("Run")
        self._btn_run.clicked.connect(self.doRun)

        self._btn_stop = QtWidgets.QPushButton("Stop")
        self._btn_stop.clicked.connect(self.doStop)
        self._btn_stop.setEnabled(False)

        btn_run_layout.addItem(
            QtWidgets.QSpacerItem(
                20, 20,
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Minimum
            )
        )

        btn_run_layout.addWidget(self._btn_stop)
        btn_run_layout.addWidget(self._btn_run)

        layout.addLayout(btn_run_layout)

        central_widget.setLayout(layout)

        statusbar = self.main_wnd.statusBar()
        statusbar.showMessage("Ready.", 5000)

        self._update_gui_timer = QtCore.QTimer()
        self._update_gui_timer.timeout.connect(self.parseProcessData)

    def parseProcessData(self):
        """
        Parse lenstool output and update gui.

        Returns
        -------
        None.

        """
        if self.lenstool._lenstool_popen is None:
            return
        elif self.lenstool._lenstool_popen.poll() is None:
            out = self.lenstool.parsedata()
            self._pbar_global.setValue(
                int(1000*self.lenstool.global_progress)
            )
            self._pbar_partial.setValue(
                int(1000*self.lenstool.partial_progress)
            )
            self.main_wnd.statusBar().showMessage(
                f"{self.lenstool.state} ETA: {self.lenstool.eta}",
            )

            if out:
                self.prog_out.appendPlainText(out)
        else:
            self.main_wnd.statusBar().showMessage("Ended", 5000)
            self.doStop()

    def doRun(self):
        """
        Start lenstool thread.

        Returns
        -------
        None.

        """
        self._btn_stop.setEnabled(True)
        self._btn_run.setEnabled(False)
        self._options_gbox.setEnabled(False)

        self._pbar_partial.setMaximum(0)
        self._pbar_global.setMaximum(0)

        self.prog_out.clear()
        model_par_file = self._lenstool_input_txt.text()

        if not (model_par_file or os.path.isfile(model_par_file)):
            msg = QtWidgets.QMessageBox()
            msg.setIcon(msg.Critical)
            msg.setText("Parameter file does not exist")
            msg.setDetailedText(
                "The selected parameter file does not exist or is corrupted. "
                "Check if the param file path is correct."
            )
            msg.setWindowTitle("No parameter file")
            msg.exec_()
            self.doStop()
            return

        self.lenstool.run(model_par_file)
        self._update_gui_timer.start(50)

        for x in range(10):
            if self.lenstool._lenstool_popen is None:
                time.sleep(0.1)
            else:
                break
        else:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(msg.Critical)
            msg.setText("Error in launching lenstool")
            msg.setWindowTitle("Error")
            msg.exec_()
            self.doStop()

        self._pbar_partial.setMaximum(1000)
        self._pbar_global.setMaximum(1000)

    def doStop(self):
        """
        Stop lenstool thread.

        Returns
        -------
        None.

        """
        self._update_gui_timer.stop()
        self._btn_stop.setEnabled(False)
        self._btn_run.setEnabled(True)
        self._options_gbox.setEnabled(True)

        self._pbar_partial.setValue(0)
        self._pbar_global.setValue(0)

        self._pbar_partial.setMaximum(1)
        self._pbar_global.setMaximum(1)

    def openProgramFile(self):
        """
        Get lenstool file path.

        Returns
        -------
        None.

        """
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.main_wnd,
            "Select lenstool executable",
            ".",
            "All Files (*)",
            options=options
        )
        if fileName:
            self._lenstool_path_txt.setText(fileName)

    def openParamFile(self):
        """
        Get the param file path.

        Returns
        -------
        None.

        """
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.main_wnd,
            "Select model parameter file",
            ".",
            "Param Files (*.par);;All Files (*)",
        )
        if fileName:
            self._lenstool_input_txt.setText(fileName)

    def exec(self):
        """
        Exec the main Qt loop.

        Returns
        -------
        None.

        """
        self.main_wnd.show()
        self.qapp.exec()


def main():
    """
    Run the main program.

    Returns
    -------
    None.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("paramfile", type=str, default=None, nargs='?')
    parser.add_argument('-r', '--run')

    args = parser.parse_args()

    lt_wrap = LensToolPyQt()

    if args.paramfile:
        lt_wrap._lenstool_input_txt.setText(args.paramfile)

    if args.run:
        lt_wrap.doRun()

    lt_wrap.exec()


if __name__ == '__main__':
    main()
