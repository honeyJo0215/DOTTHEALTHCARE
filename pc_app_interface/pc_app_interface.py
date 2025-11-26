import sys
import io
import datetime
from typing import List, Dict, Any, Optional

import requests
import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QGroupBox,
    QLineEdit, QDoubleSpinBox, QMessageBox, QDialog, QDialogButtonBox,
    QComboBox, QFormLayout, QRadioButton, QButtonGroup, QSpacerItem,
    QSizePolicy
)
from PyQt6.QtCore import Qt

# ======================================
# 설정: food_camera 서버 주소
# ======================================
SERVER_URL = "http://127.0.0.1:8000"   # 나중에 온라인 서버로 바꾸면 여기만 수정


# ======================================
# Food 선택 팝업 다이얼로그
# ======================================

class FoodSelectionDialog(QDialog):
    """
    /predict 결과로 받은 candidates 리스트를 보여주고,
    유저가 선택하거나, 직접 입력할 수 있게 하는 팝업.
    반환값: (name, calories) 형태
    """
    def __init__(self, candidates: List[Dict[str, Any]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("음식 선택 / 칼로리 입력")
        self.setModal(True)

        self.candidates = candidates or []

        self.selected_name: Optional[str] = None
        self.selected_calories: Optional[float] = None

        main_layout = QVBoxLayout(self)

        info_label = QLabel("아래 후보 중에서 선택하거나, 직접 칼로리를 입력할 수 있어요.")
        main_layout.addWidget(info_label)

        # --- 후보 선택 영역 ---
        candidate_group = QGroupBox("Food Camera 인식 결과로 선택")
        cg_layout = QVBoxLayout(candidate_group)

        self.radio_use_candidate = QRadioButton("후보 리스트에서 선택")
        self.radio_use_manual = QRadioButton("직접 입력 사용")
        self.radio_use_candidate.setChecked(True)

        radio_group = QButtonGroup(self)
        radio_group.addButton(self.radio_use_candidate)
        radio_group.addButton(self.radio_use_manual)

        cg_layout.addWidget(self.radio_use_candidate)

        self.candidate_combo = QComboBox()
        if not self.candidates:
            self.candidate_combo.addItem("후보 없음 (직접 입력만 사용 가능)")
            self.candidate_combo.setEnabled(False)
            self.radio_use_manual.setChecked(True)
            self.radio_use_candidate.setEnabled(False)
        else:
            for cand in self.candidates:
                # name, calories 사용
                name = cand.get("name", "unknown")
                cal = cand.get("calories", 0.0)
                unit = cand.get("unit", "")
                serving_size = cand.get("serving_size", 1.0)
                text = f"{name}  ~ {cal:.0f} kcal  ({serving_size} {unit})"
                self.candidate_combo.addItem(text)
        cg_layout.addWidget(self.candidate_combo)

        main_layout.addWidget(candidate_group)

        # --- 직접 입력 영역 ---
        manual_group = QGroupBox("직접 음식 / 칼로리 입력")
        mg_layout = QFormLayout(manual_group)

        self.manual_name_edit = QLineEdit()
        self.manual_name_edit.setPlaceholderText("예: green onion chicken / 파닭 등")

        self.manual_cal_spin = QDoubleSpinBox()
        self.manual_cal_spin.setRange(0.0, 20000.0)
        self.manual_cal_spin.setDecimals(1)
        self.manual_cal_spin.setValue(500.0)

        mg_layout.addRow("음식 이름:", self.manual_name_edit)
        mg_layout.addRow("칼로리 (kcal):", self.manual_cal_spin)

        # 라디오 버튼 추가
        mg_layout.addRow(self.radio_use_manual)

        main_layout.addWidget(manual_group)

        # --- 버튼 영역 ---
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.on_accept)
        button_box.rejected.connect(self.reject)

        main_layout.addWidget(button_box)

    def on_accept(self):
        """
        OK 눌렀을 때 선택된 값 정리
        """
        if self.radio_use_candidate.isChecked() and self.candidates:
            idx = self.candidate_combo.currentIndex()
            cand = self.candidates[idx]
            self.selected_name = cand.get("name", "unknown")
            self.selected_calories = float(cand.get("calories", 0.0))
        else:
            # 직접 입력 사용
            name = self.manual_name_edit.text().strip()
            cal = float(self.manual_cal_spin.value())
            if not name:
                QMessageBox.warning(self, "입력 오류", "직접 입력 모드에서는 음식 이름을 입력해야 합니다.")
                return
            self.selected_name = name
            self.selected_calories = cal

        self.accept()

    @staticmethod
    def get_selection(candidates: List[Dict[str, Any]], parent=None):
        dlg = FoodSelectionDialog(candidates, parent)
        result = dlg.exec()
        if result == QDialog.DialogCode.Accepted:
            return dlg.selected_name, dlg.selected_calories
        return None, None


# ======================================
# 메인 윈도우
# ======================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("DOTTHEALTH - 칼로리 관리 (Food Camera 연동)")
        self.resize(900, 600)

        self.entries: List[Dict[str, Any]] = []  # {"time":..., "name":..., "calories":...}

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)

        # ---- 상단: 날짜 + 오늘 총 칼로리 ----
        top_layout = QHBoxLayout()

        today = datetime.date.today().strftime("%Y-%m-%d")
        self.date_label = QLabel(f"오늘 날짜: {today}")
        self.date_label.setStyleSheet("font-size: 14px;")

        self.total_cal_label = QLabel("오늘 섭취 칼로리: 0 kcal")
        self.total_cal_label.setStyleSheet("font-size: 16px; font-weight: bold;")

        top_layout.addWidget(self.date_label)
        top_layout.addStretch()
        top_layout.addWidget(self.total_cal_label)

        main_layout.addLayout(top_layout)

        # ---- Food Camera 버튼 영역 ----
        camera_layout = QHBoxLayout()

        self.food_camera_button = QPushButton("Food Camera")
        self.food_camera_button.setStyleSheet(
            "font-size: 14px; padding: 6px 12px; font-weight: 600;"
        )
        self.food_camera_button.clicked.connect(self.on_food_camera_clicked)

        beta_label = QLabel("beta")
        beta_label.setStyleSheet("color: gray; font-size: 10px; margin-left: 4px;")

        camera_layout.addWidget(self.food_camera_button)
        camera_layout.addWidget(beta_label)
        camera_layout.addStretch()

        main_layout.addLayout(camera_layout)

        # ---- 오늘 섭취 리스트 테이블 ----
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["시간", "음식 이름", "칼로리 (kcal)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        main_layout.addWidget(self.table)

        # ---- 아래: 직접 입력 영역 ----
        manual_group = QGroupBox("직접 음식 / 칼로리 입력")
        mg_layout = QHBoxLayout(manual_group)

        self.manual_name_edit = QLineEdit()
        self.manual_name_edit.setPlaceholderText("예: 샐러드, 고구마, 단백질 바 등")

        self.manual_cal_spin = QDoubleSpinBox()
        self.manual_cal_spin.setRange(0.0, 20000.0)
        self.manual_cal_spin.setDecimals(1)
        self.manual_cal_spin.setValue(300.0)

        self.manual_add_button = QPushButton("추가")
        self.manual_add_button.clicked.connect(self.on_add_manual_entry)

        mg_layout.addWidget(QLabel("음식 이름:"))
        mg_layout.addWidget(self.manual_name_edit, 2)
        mg_layout.addWidget(QLabel("칼로리:"))
        mg_layout.addWidget(self.manual_cal_spin)
        mg_layout.addWidget(QLabel("kcal"))
        mg_layout.addWidget(self.manual_add_button)

        main_layout.addWidget(manual_group)

        # ---- 상태 표시 라벨 ----
        self.status_label = QLabel("Food Camera 서버가 실행 중인지 확인해 주세요.")
        self.status_label.setStyleSheet("color: gray; font-size: 11px;")
        main_layout.addWidget(self.status_label)

        spacer = QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        main_layout.addItem(spacer)

    # ==========================
    # 유틸: 테이블/합계 업데이트
    # ==========================

    def add_entry(self, name: str, calories: float):
        now_str = datetime.datetime.now().strftime("%H:%M")
        entry = {
            "time": now_str,
            "name": name,
            "calories": float(calories),
        }
        self.entries.append(entry)

        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(entry["time"]))
        self.table.setItem(row, 1, QTableWidgetItem(entry["name"]))
        self.table.setItem(row, 2, QTableWidgetItem(f"{entry['calories']:.1f}"))

        self.update_total_calories()

    def update_total_calories(self):
        total = sum(e["calories"] for e in self.entries)
        self.total_cal_label.setText(f"오늘 섭취 칼로리: {total:.1f} kcal")

    # ==========================
    # 직접 입력 핸들러
    # ==========================

    def on_add_manual_entry(self):
        name = self.manual_name_edit.text().strip()
        cal = float(self.manual_cal_spin.value())

        if not name:
            QMessageBox.warning(self, "입력 오류", "음식 이름을 입력해 주세요.")
            return

        self.add_entry(name, cal)
        self.manual_name_edit.clear()
        self.manual_cal_spin.setValue(300.0)

    # ==========================
    # Food Camera 버튼 핸들러
    # ==========================

    def on_food_camera_clicked(self):
        """
        1) 카메라로 사진 촬영
        2) /predict 서버로 전송
        3) 후보들 중 선택 or 직접입력 → entries에 추가
        """
        # 1. 카메라 사진 촬영
        self.status_label.setText("카메라를 여는 중입니다... (창에서 스페이스로 촬영, q로 취소)")
        image_bytes = self.capture_image_from_camera()
        if image_bytes is None:
            self.status_label.setText("촬영이 취소되었거나 실패했습니다.")
            return

        # 2. 서버로 전송해서 분석
        self.status_label.setText("서버에 이미지를 보내는 중입니다...")
        try:
            name, cal = self.analyze_image_and_get_food(image_bytes)
        except Exception as e:
            QMessageBox.critical(self, "서버 오류", f"이미지 분석 중 오류가 발생했습니다:\n{e}")
            self.status_label.setText("이미지 분석 중 오류가 발생했습니다.")
            return

        if name is None or cal is None:
            # 사용자가 다이얼로그에서 취소
            self.status_label.setText("음식 선택이 취소되었습니다.")
            return

        # 3. 성공적으로 선택한 음식/칼로리를 오늘 섭취 목록에 추가
        self.add_entry(name, cal)
        self.status_label.setText(f"Food Camera로 '{name}' {cal:.1f} kcal 이(가) 추가되었습니다.")

    # ==========================
    # 카메라 캡처 부분 (OpenCV)
    # ==========================

    def capture_image_from_camera(self) -> Optional[bytes]:
        """
        OpenCV로 기본 카메라를 열어서 한 장 촬영.
        - 스페이스바: 촬영 후 캡처
        - q: 취소
        반환: JPEG 이미지 bytes 또는 None
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.critical(self, "카메라 오류", "카메라를 열 수 없습니다.")
            return None

        captured_frame = None
        cv2.namedWindow("Food Camera - 스페이스: 촬영 / q: 취소", cv2.WINDOW_NORMAL)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("Food Camera - 스페이스: 촬영 / q: 취소", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):  # 스페이스: 촬영
                captured_frame = frame.copy()
                break
            elif key == ord('q'):
                captured_frame = None
                break

        cap.release()
        cv2.destroyAllWindows()

        if captured_frame is None:
            return None

        # JPEG로 인메모리 인코딩
        success, buf = cv2.imencode(".jpg", captured_frame)
        if not success:
            QMessageBox.critical(self, "인코딩 오류", "이미지 인코딩에 실패했습니다.")
            return None

        return buf.tobytes()

    # ==========================
    # 서버 /predict 호출 + 선택 다이얼로그
    # ==========================

    def analyze_image_and_get_food(self, image_bytes: bytes):
        """
        /predict 서버에 이미지를 보내고,
        후보 리스트를 기반으로 음식/칼로리를 선택해서 반환.
        """
        url = f"{SERVER_URL}/predict"
        files = {
            "image": ("capture.jpg", image_bytes, "image/jpeg")
        }
        params = {
            "topk": 3,
            "food_topk": 5
        }

        resp = requests.post(url, files=files, params=params, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

        data = resp.json()
        candidates = data.get("candidates", [])

        # FoodSelectionDialog 로 후보 보여주기
        name, cal = FoodSelectionDialog.get_selection(candidates, parent=self)
        return name, cal


# ======================================
# 메인 엔트리 포인트
# ======================================

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
