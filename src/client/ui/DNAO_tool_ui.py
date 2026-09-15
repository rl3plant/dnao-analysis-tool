# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'DNAO_tool.ui'
##
## Created by: Qt User Interface Compiler version 6.8.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QGroupBox, QHBoxLayout,
    QLabel, QListView, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QStackedWidget, QVBoxLayout,
    QWidget)

from .zoomable_image_label import ZoomableImageLabel

class Ui_StackedWidget(object):
    def setupUi(self, StackedWidget):
        if not StackedWidget.objectName():
            StackedWidget.setObjectName(u"StackedWidget")
        StackedWidget.resize(1493, 800)
        self.pg_main = QWidget()
        self.pg_main.setObjectName(u"pg_main")
        self.horizontalLayout = QHBoxLayout(self.pg_main)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.grp_session_selection = QGroupBox(self.pg_main)
        self.grp_session_selection.setObjectName(u"grp_session_selection")
        self.verticalLayout_2 = QVBoxLayout(self.grp_session_selection)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.grp_main_btns = QGroupBox(self.grp_session_selection)
        self.grp_main_btns.setObjectName(u"grp_main_btns")
        self.horizontalLayout_6 = QHBoxLayout(self.grp_main_btns)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.btn_create_session = QPushButton(self.grp_main_btns)
        self.btn_create_session.setObjectName(u"btn_create_session")

        self.horizontalLayout_6.addWidget(self.btn_create_session)

        self.btn_load_session = QPushButton(self.grp_main_btns)
        self.btn_load_session.setObjectName(u"btn_load_session")

        self.horizontalLayout_6.addWidget(self.btn_load_session)

        self.btn_delete_session = QPushButton(self.grp_main_btns)
        self.btn_delete_session.setObjectName(u"btn_delete_session")

        self.horizontalLayout_6.addWidget(self.btn_delete_session)


        self.verticalLayout_2.addWidget(self.grp_main_btns)

        self.lst_sessions = QListView(self.grp_session_selection)
        self.lst_sessions.setObjectName(u"lst_sessions")

        self.verticalLayout_2.addWidget(self.lst_sessions)


        self.horizontalLayout.addWidget(self.grp_session_selection)

        self.grp_details = QGroupBox(self.pg_main)
        self.grp_details.setObjectName(u"grp_details")
        self.verticalLayout = QVBoxLayout(self.grp_details)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.lst_session_stats = QListView(self.grp_details)
        self.lst_session_stats.setObjectName(u"lst_session_stats")

        self.verticalLayout.addWidget(self.lst_session_stats)


        self.horizontalLayout.addWidget(self.grp_details)

        StackedWidget.addWidget(self.pg_main)
        self.pg_session = QWidget()
        self.pg_session.setObjectName(u"pg_session")
        self.horizontalLayout_2 = QHBoxLayout(self.pg_session)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.grp_afm_imgs = QGroupBox(self.pg_session)
        self.grp_afm_imgs.setObjectName(u"grp_afm_imgs")
        self.verticalLayout_3 = QVBoxLayout(self.grp_afm_imgs)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.grp_session_btns = QGroupBox(self.grp_afm_imgs)
        self.grp_session_btns.setObjectName(u"grp_session_btns")
        self.horizontalLayout_5 = QHBoxLayout(self.grp_session_btns)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.btn_return_to_main = QPushButton(self.grp_session_btns)
        self.btn_return_to_main.setObjectName(u"btn_return_to_main")

        self.horizontalLayout_5.addWidget(self.btn_return_to_main)

        self.btn_load_img = QPushButton(self.grp_session_btns)
        self.btn_load_img.setObjectName(u"btn_load_img")

        self.horizontalLayout_5.addWidget(self.btn_load_img)

        self.btn_load_folder = QPushButton(self.grp_session_btns)
        self.btn_load_folder.setObjectName(u"btn_load_folder")

        self.horizontalLayout_5.addWidget(self.btn_load_folder)

        self.btn_start_annotation = QPushButton(self.grp_session_btns)
        self.btn_start_annotation.setObjectName(u"btn_start_annotation")

        self.horizontalLayout_5.addWidget(self.btn_start_annotation)

        self.btn_delete_annotation = QPushButton(self.grp_session_btns)
        self.btn_delete_annotation.setObjectName(u"btn_delete_annotation")

        self.horizontalLayout_5.addWidget(self.btn_delete_annotation)


        self.verticalLayout_3.addWidget(self.grp_session_btns)

        self.lst_session_img = QListView(self.grp_afm_imgs)
        self.lst_session_img.setObjectName(u"lst_session_img")
        self.lst_session_img.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        self.verticalLayout_3.addWidget(self.lst_session_img)

        self.progress_preprocessing = QProgressBar(self.grp_afm_imgs)
        self.progress_preprocessing.setObjectName(u"progress_preprocessing")
        self.progress_preprocessing.setValue(0)
        self.progress_preprocessing.setTextVisible(True)

        self.verticalLayout_3.addWidget(self.progress_preprocessing)

        self.lbl_preprocessing_status = QLabel(self.grp_afm_imgs)
        self.lbl_preprocessing_status.setObjectName(u"lbl_preprocessing_status")

        self.verticalLayout_3.addWidget(self.lbl_preprocessing_status)


        self.horizontalLayout_2.addWidget(self.grp_afm_imgs)

        self.grp_afm_results = QGroupBox(self.pg_session)
        self.grp_afm_results.setObjectName(u"grp_afm_results")
        self.verticalLayout_6 = QVBoxLayout(self.grp_afm_results)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.grp_results_preview = QGroupBox(self.grp_afm_results)
        self.grp_results_preview.setObjectName(u"grp_results_preview")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.grp_results_preview.sizePolicy().hasHeightForWidth())
        self.grp_results_preview.setSizePolicy(sizePolicy)
        self.horizontalLayout_10 = QHBoxLayout(self.grp_results_preview)
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.lbl_intact_preview = QLabel(self.grp_results_preview)
        self.lbl_intact_preview.setObjectName(u"lbl_intact_preview")

        self.horizontalLayout_10.addWidget(self.lbl_intact_preview)

        self.lbl_defect_preview = QLabel(self.grp_results_preview)
        self.lbl_defect_preview.setObjectName(u"lbl_defect_preview")

        self.horizontalLayout_10.addWidget(self.lbl_defect_preview)

        self.lbl_total_preview = QLabel(self.grp_results_preview)
        self.lbl_total_preview.setObjectName(u"lbl_total_preview")

        self.horizontalLayout_10.addWidget(self.lbl_total_preview)

        self.lbl_yield_preview = QLabel(self.grp_results_preview)
        self.lbl_yield_preview.setObjectName(u"lbl_yield_preview")

        self.horizontalLayout_10.addWidget(self.lbl_yield_preview)


        self.verticalLayout_6.addWidget(self.grp_results_preview)

        self.lbl_img_preview = QLabel(self.grp_afm_results)
        self.lbl_img_preview.setObjectName(u"lbl_img_preview")
        self.lbl_img_preview.setEnabled(True)
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy1.setHorizontalStretch(1)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.lbl_img_preview.sizePolicy().hasHeightForWidth())
        self.lbl_img_preview.setSizePolicy(sizePolicy1)
        self.lbl_img_preview.setMinimumSize(QSize(600, 400))
        self.lbl_img_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_6.addWidget(self.lbl_img_preview)


        self.horizontalLayout_2.addWidget(self.grp_afm_results)

        StackedWidget.addWidget(self.pg_session)
        self.pg_img = QWidget()
        self.pg_img.setObjectName(u"pg_img")
        self.horizontalLayout_3 = QHBoxLayout(self.pg_img)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.grp_annotation_menu = QGroupBox(self.pg_img)
        self.grp_annotation_menu.setObjectName(u"grp_annotation_menu")
        self.verticalLayout_4 = QVBoxLayout(self.grp_annotation_menu)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.groupBox = QGroupBox(self.grp_annotation_menu)
        self.groupBox.setObjectName(u"groupBox")
        self.horizontalLayout_4 = QHBoxLayout(self.groupBox)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.btn_return_session = QPushButton(self.groupBox)
        self.btn_return_session.setObjectName(u"btn_return_session")

        self.horizontalLayout_4.addWidget(self.btn_return_session)

        self.btn_save_annotation = QPushButton(self.groupBox)
        self.btn_save_annotation.setObjectName(u"btn_save_annotation")

        self.horizontalLayout_4.addWidget(self.btn_save_annotation)

        self.btn_export_annotation = QPushButton(self.groupBox)
        self.btn_export_annotation.setObjectName(u"btn_export_annotation")

        self.horizontalLayout_4.addWidget(self.btn_export_annotation)


        self.verticalLayout_4.addWidget(self.groupBox)

        self.lst_dnao = QListView(self.grp_annotation_menu)
        self.lst_dnao.setObjectName(u"lst_dnao")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.lst_dnao.sizePolicy().hasHeightForWidth())
        self.lst_dnao.setSizePolicy(sizePolicy2)

        self.verticalLayout_4.addWidget(self.lst_dnao)

        self.grp_navigation_btns = QGroupBox(self.grp_annotation_menu)
        self.grp_navigation_btns.setObjectName(u"grp_navigation_btns")
        self.horizontalLayout_8 = QHBoxLayout(self.grp_navigation_btns)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.btn_previous_dnao = QPushButton(self.grp_navigation_btns)
        self.btn_previous_dnao.setObjectName(u"btn_previous_dnao")

        self.horizontalLayout_8.addWidget(self.btn_previous_dnao)

        self.btn_next_dnao = QPushButton(self.grp_navigation_btns)
        self.btn_next_dnao.setObjectName(u"btn_next_dnao")

        self.horizontalLayout_8.addWidget(self.btn_next_dnao)

        self.btn_toggle_visualization = QPushButton(self.grp_navigation_btns)
        self.btn_toggle_visualization.setObjectName(u"btn_toggle_visualization")

        self.horizontalLayout_8.addWidget(self.btn_toggle_visualization)


        self.verticalLayout_4.addWidget(self.grp_navigation_btns)


        self.horizontalLayout_3.addWidget(self.grp_annotation_menu)

        self.grp_afm_img = QGroupBox(self.pg_img)
        self.grp_afm_img.setObjectName(u"grp_afm_img")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy3.setHorizontalStretch(2)
        sizePolicy3.setVerticalStretch(2)
        sizePolicy3.setHeightForWidth(self.grp_afm_img.sizePolicy().hasHeightForWidth())
        self.grp_afm_img.setSizePolicy(sizePolicy3)
        self.verticalLayout_5 = QVBoxLayout(self.grp_afm_img)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.scrollArea_afm_img = QScrollArea(self.grp_afm_img)
        self.scrollArea_afm_img.setObjectName(u"scrollArea_afm_img")
        self.scrollArea_afm_img.setWidgetResizable(True)
        self.widget = QWidget()
        self.widget.setObjectName(u"widget")
        self.widget.setGeometry(QRect(0, 0, 643, 753))
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy4)
        self.lbl_afm_img = ZoomableImageLabel(self.widget)
        self.lbl_afm_img.setObjectName(u"lbl_afm_img")
        self.lbl_afm_img.setGeometry(QRect(0, 0, 100, 100))
        sizePolicy4.setHeightForWidth(self.lbl_afm_img.sizePolicy().hasHeightForWidth())
        self.lbl_afm_img.setSizePolicy(sizePolicy4)
        self.lbl_afm_img.setMinimumSize(QSize(100, 100))
        self.lbl_afm_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scrollArea_afm_img.setWidget(self.widget)

        self.verticalLayout_5.addWidget(self.scrollArea_afm_img)


        self.horizontalLayout_3.addWidget(self.grp_afm_img)

        self.grp_afm_details = QGroupBox(self.pg_img)
        self.grp_afm_details.setObjectName(u"grp_afm_details")
        sizePolicy4.setHeightForWidth(self.grp_afm_details.sizePolicy().hasHeightForWidth())
        self.grp_afm_details.setSizePolicy(sizePolicy4)
        self.verticalLayout_7 = QVBoxLayout(self.grp_afm_details)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.grp_img_results = QGroupBox(self.grp_afm_details)
        self.grp_img_results.setObjectName(u"grp_img_results")
        sizePolicy.setHeightForWidth(self.grp_img_results.sizePolicy().hasHeightForWidth())
        self.grp_img_results.setSizePolicy(sizePolicy)
        self.horizontalLayout_9 = QHBoxLayout(self.grp_img_results)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.lbl_intact_dnao = QLabel(self.grp_img_results)
        self.lbl_intact_dnao.setObjectName(u"lbl_intact_dnao")

        self.horizontalLayout_9.addWidget(self.lbl_intact_dnao)

        self.lbl_defect_dnao = QLabel(self.grp_img_results)
        self.lbl_defect_dnao.setObjectName(u"lbl_defect_dnao")

        self.horizontalLayout_9.addWidget(self.lbl_defect_dnao)

        self.lbl_total_dnao = QLabel(self.grp_img_results)
        self.lbl_total_dnao.setObjectName(u"lbl_total_dnao")

        self.horizontalLayout_9.addWidget(self.lbl_total_dnao)

        self.lbl_yield_dnao = QLabel(self.grp_img_results)
        self.lbl_yield_dnao.setObjectName(u"lbl_yield_dnao")

        self.horizontalLayout_9.addWidget(self.lbl_yield_dnao)


        self.verticalLayout_7.addWidget(self.grp_img_results)

        self.lbl_dnao_closeup = QLabel(self.grp_afm_details)
        self.lbl_dnao_closeup.setObjectName(u"lbl_dnao_closeup")
        sizePolicy4.setHeightForWidth(self.lbl_dnao_closeup.sizePolicy().hasHeightForWidth())
        self.lbl_dnao_closeup.setSizePolicy(sizePolicy4)
        self.lbl_dnao_closeup.setMinimumSize(QSize(300, 200))

        self.verticalLayout_7.addWidget(self.lbl_dnao_closeup)

        self.grp_classification_btns = QGroupBox(self.grp_afm_details)
        self.grp_classification_btns.setObjectName(u"grp_classification_btns")
        self.horizontalLayout_7 = QHBoxLayout(self.grp_classification_btns)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.btn_set_intact = QPushButton(self.grp_classification_btns)
        self.btn_set_intact.setObjectName(u"btn_set_intact")

        self.horizontalLayout_7.addWidget(self.btn_set_intact)

        self.btn_set_defect = QPushButton(self.grp_classification_btns)
        self.btn_set_defect.setObjectName(u"btn_set_defect")

        self.horizontalLayout_7.addWidget(self.btn_set_defect)

        self.btn_set_invalid = QPushButton(self.grp_classification_btns)
        self.btn_set_invalid.setObjectName(u"btn_set_invalid")

        self.horizontalLayout_7.addWidget(self.btn_set_invalid)


        self.verticalLayout_7.addWidget(self.grp_classification_btns)


        self.horizontalLayout_3.addWidget(self.grp_afm_details)

        StackedWidget.addWidget(self.pg_img)

        self.retranslateUi(StackedWidget)

        StackedWidget.setCurrentIndex(2)


        QMetaObject.connectSlotsByName(StackedWidget)
    # setupUi

    def retranslateUi(self, StackedWidget):
        StackedWidget.setWindowTitle(QCoreApplication.translate("StackedWidget", u"DNAO Analysis Tool", None))
#if QT_CONFIG(accessibility)
        self.pg_main.setAccessibleName("")
#endif // QT_CONFIG(accessibility)
        self.grp_session_selection.setTitle(QCoreApplication.translate("StackedWidget", u"Sessions", None))
        self.grp_main_btns.setTitle("")
        self.btn_create_session.setText(QCoreApplication.translate("StackedWidget", u"Create Session", None))
        self.btn_load_session.setText(QCoreApplication.translate("StackedWidget", u"Load Session", None))
        self.btn_delete_session.setText(QCoreApplication.translate("StackedWidget", u"Delete Session", None))
        self.grp_details.setTitle(QCoreApplication.translate("StackedWidget", u"Details", None))
        self.grp_afm_imgs.setTitle(QCoreApplication.translate("StackedWidget", u"AFM Images", None))
        self.grp_session_btns.setTitle("")
        self.btn_return_to_main.setText(QCoreApplication.translate("StackedWidget", u"Return", None))
        self.btn_load_img.setText(QCoreApplication.translate("StackedWidget", u"Load Images", None))
        self.btn_load_folder.setText(QCoreApplication.translate("StackedWidget", u"Load Folder", None))
        self.btn_start_annotation.setText(QCoreApplication.translate("StackedWidget", u"Start/Continue Annotation", None))
        self.btn_delete_annotation.setText(QCoreApplication.translate("StackedWidget", u"Delete Annotation", None))
        self.lbl_preprocessing_status.setText(QCoreApplication.translate("StackedWidget", u"Preprocessing: 0/0 images", None))
        self.grp_afm_results.setTitle("")
        self.grp_results_preview.setTitle("")
        self.lbl_intact_preview.setText("")
        self.lbl_defect_preview.setText("")
        self.lbl_total_preview.setText("")
        self.lbl_yield_preview.setText("")
        self.lbl_img_preview.setText("")
        self.grp_annotation_menu.setTitle("")
        self.groupBox.setTitle("")
        self.btn_return_session.setText(QCoreApplication.translate("StackedWidget", u"Return to Session View (Esc)", None))
        self.btn_save_annotation.setText(QCoreApplication.translate("StackedWidget", u"Save Annotation (s)", None))
        self.btn_export_annotation.setText(QCoreApplication.translate("StackedWidget", u"Export Annotation (e)", None))
        self.grp_navigation_btns.setTitle("")
        self.btn_previous_dnao.setText(QCoreApplication.translate("StackedWidget", u"Previous (Left)", None))
        self.btn_next_dnao.setText(QCoreApplication.translate("StackedWidget", u"Next (Right)", None))
        self.btn_toggle_visualization.setText(QCoreApplication.translate("StackedWidget", u"Hide Annotations (t)", None))
        self.grp_afm_img.setTitle("")
        self.lbl_afm_img.setText("")
        self.grp_afm_details.setTitle("")
        self.grp_img_results.setTitle("")
        self.lbl_intact_dnao.setText("")
        self.lbl_defect_dnao.setText("")
        self.lbl_total_dnao.setText("")
        self.lbl_yield_dnao.setText("")
        self.lbl_dnao_closeup.setText("")
        self.grp_classification_btns.setTitle("")
        self.btn_set_intact.setText(QCoreApplication.translate("StackedWidget", u"Intact (1)", None))
        self.btn_set_defect.setText(QCoreApplication.translate("StackedWidget", u"Defect (2)", None))
        self.btn_set_invalid.setText(QCoreApplication.translate("StackedWidget", u"Invalid (3)", None))
    # retranslateUi

