import cv2
import csv
import os
import time
import numpy as np
import math

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_line(frame, pt1, pt2, color=(0, 0, 255)):
    cv2.line(frame, pt1, pt2, color, 1)


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    return number, mode


def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = "model/dataset.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return


def logging_image(number, mode, image):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        folder_path = f"image_dataset/{number}/"
        os.makedirs(folder_path, exist_ok=True)

        timestamp = int(time.time() * 1000)
        image_path = os.path.join(folder_path, f"{timestamp}.jpg")

        cv2.imwrite(image_path, image)


def draw_info(image, mode, number):
    mode_string = ["Normal", "Pengumpulan Dataset"]
    if mode == 0:
        cv2.putText(
            image,
            "MODE:" + mode_string[mode],
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    if mode == 1:
        cv2.putText(
            image,
            "MODE:" + mode_string[mode],
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        if 0 <= number <= 9:
            cv2.putText(
                image,
                "NUM:" + str(number),
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
    return image


def draw_predicted_infos(image, hand_class_predicted):
    # Atur posisi dan tampilan teks
    text = f"Prediksi: {hand_class_predicted}"
    position = (20, 40)  # Koordinat x, y untuk menampilkan teks
    font = cv2.FONT_HERSHEY_SIMPLEX  # Jenis font
    font_scale = 1  # Ukuran font
    color = (0, 255, 0)  # Warna teks (hijau)
    thickness = 2  # Ketebalan teks

    # Tambahkan teks ke gambar
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

    return image


def get_landmark_points(image, hands):
    landmark_list = []

    for hand in hands:
        landmarks = hand.landmark
        frame_height, frame_width, _ = image.shape

        for landmark in landmarks:
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            landmark_list.append(x)
            landmark_list.append(y)

    return landmark_list


def draw_hand_rect(image, hands, model, scaler, label_encoder):
    frame_height, frame_width, _ = image.shape
    for hand in hands:
        # Ambil koordinat landmark langsung sebagai array NumPy
        landmarks = np.array(
            [[int(l.x * frame_width), int(l.y * frame_height)] for l in hand.landmark]
        )

        # Dapatkan nilai min dan max
        min_x, min_y = np.min(landmarks, axis=0)
        max_x, max_y = np.max(landmarks, axis=0)

        # Normalisasi koordinat
        transformed_coordinates = landmarks - [min_x, min_y]
        flattened_data = transformed_coordinates.flatten().reshape(
            1, -1
        )  # Bentuk (1, 42)

        # Skalakan data dan prediksi kelas
        scaled_data = scaler.transform(flattened_data)
        predicted_label = model.predict(scaled_data)
        predicted_class = label_encoder.inverse_transform(predicted_label)[0]

        # Gambar bounding box
        cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

        # Tentukan posisi teks (letakkan di atas kiri kotak)
        text_position = (min_x, min_y - 10) if min_y > 20 else (min_x, min_y + 20)

        # Hitung ukuran teks
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_size, _ = cv2.getTextSize(
            f"{predicted_class}", font, font_scale, thickness
        )
        text_w, text_h = text_size

        # Gambar latar belakang teks (kotak hitam semi-transparan)
        cv2.rectangle(
            image,
            (text_position[0], text_position[1] - text_h),
            (text_position[0] + text_w + 5, text_position[1] + 5),
            (0, 0, 0),
            -1,
        )

        # Tambahkan teks kelas prediksi di dalam bounding box
        cv2.putText(
            image,
            f"{predicted_class}",
            text_position,
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def draw_landmark(image, hands):
    for hand in hands:
        landmarks = hand.landmark
        frame_height, frame_width, _ = image.shape

        palm_coor = (0, 0)

        base_thumb_coor = (0, 0)
        middle_thumb_coor = (0, 0)
        peak_thumb_coor = (0, 0)

        base_index_coor = (0, 0)
        middle_index_coor = (0, 0)
        peak_index_coor = (0, 0)

        base_middle_coor = (0, 0)
        middle_middle_coor = (0, 0)
        peak_middle_coor = (0, 0)

        base_ring_coor = (0, 0)
        middle_ring_coor = (0, 0)
        peak_ring_coor = (0, 0)

        base_little_coor = (0, 0)
        middle_little_coor = (0, 0)
        peak_little_coor = (0, 0)

        # iterasi setiap landmark pada tangan
        for idh, landmark in enumerate(landmarks):
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)

            # telapak tangan
            if idh == 0:
                cv2.circle(img=image, center=(x, y), radius=5, color=(0, 255, 255))
                palm_coor = (x, y)

            # ibu jari
            if idh == 1:  # pangkal ibu jari
                cv2.circle(img=image, center=(x, y), radius=5, color=(255, 0, 0))
                base_thumb_coor = (x, y)

            if idh == 2:  # tengah ibu jari
                cv2.circle(img=image, center=(x, y), radius=5, color=(0, 255, 255))
                middle_thumb_coor = (x, y)

            if idh == 4:  # ujung ibu jari
                cv2.circle(img=image, center=(x, y), radius=5, color=(255, 0, 0))
                peak_thumb_coor = (x, y)

            # telunjuk
            if idh == 5:  # pangkal telunjuk
                cv2.circle(img=image, center=(x, y), radius=5, color=(255, 0, 0))
                base_index_coor = (x, y)

            if idh == 6:  # tengah telunjuk
                cv2.circle(img=image, center=(x, y), radius=5, color=(0, 255, 255))
                middle_index_coor = (x, y)

            if idh == 8:  # ujung telunjuk
                cv2.circle(img=image, center=(x, y), radius=5, color=(255, 0, 0))
                peak_index_coor = (x, y)

            # jari tengah
            if idh == 9:  # pangkal jari tengah
                cv2.circle(img=image, center=(x, y), radius=5, color=(255, 0, 0))
                base_middle_coor = (x, y)

            if idh == 10:  # tengah jari tengah
                cv2.circle(img=image, center=(x, y), radius=5, color=(0, 255, 255))
                middle_middle_coor = (x, y)

            if idh == 12:  # ujung jari tengah
                cv2.circle(img=image, center=(x, y), radius=5, color=(255, 0, 0))
                peak_middle_coor = (x, y)

            # jari manis
            if idh == 13:  # pangkal jari manis
                cv2.circle(img=image, center=(x, y), radius=5, color=(255, 0, 0))
                base_ring_coor = (x, y)

            if idh == 14:  # tengah jari manis
                cv2.circle(img=image, center=(x, y), radius=5, color=(0, 255, 255))
                middle_ring_coor = (x, y)

            if idh == 16:  # ujung jari manis
                cv2.circle(img=image, center=(x, y), radius=5, color=(255, 0, 0))
                peak_ring_coor = (x, y)

            # jari kelingking
            if idh == 17:  # pangkal jari kelingking
                cv2.circle(img=image, center=(x, y), radius=5, color=(255, 0, 0))
                base_little_coor = (x, y)

            if idh == 18:  # tengah jari kelingking
                cv2.circle(img=image, center=(x, y), radius=5, color=(0, 255, 255))
                middle_little_coor = (x, y)

            if idh == 20:  # ujung jari kelingking
                cv2.circle(img=image, center=(x, y), radius=5, color=(255, 0, 0))
                peak_little_coor = (x, y)

        # Gambar Garis
        draw_line(image, palm_coor, peak_thumb_coor)
        draw_line(image, palm_coor, peak_index_coor)
        draw_line(image, palm_coor, peak_middle_coor)
        draw_line(image, palm_coor, peak_ring_coor)
        draw_line(image, palm_coor, peak_little_coor)

        draw_line(image, middle_thumb_coor, base_thumb_coor, (255, 200, 200))
        draw_line(image, middle_index_coor, base_index_coor, (255, 200, 200))
        draw_line(image, middle_middle_coor, base_middle_coor, (255, 200, 200))
        draw_line(image, middle_ring_coor, base_ring_coor, (255, 200, 200))
        draw_line(image, middle_little_coor, base_little_coor, (255, 200, 200))

        draw_line(image, middle_thumb_coor, peak_thumb_coor, (255, 200, 200))
        draw_line(image, middle_index_coor, peak_index_coor, (255, 200, 200))
        draw_line(image, middle_middle_coor, peak_middle_coor, (255, 200, 200))
        draw_line(image, middle_ring_coor, peak_ring_coor, (255, 200, 200))
        draw_line(image, middle_little_coor, peak_little_coor, (255, 200, 200))


def calculate_pedal_press(image, hand_landmarks):
    """
    Menghitung persentase himpitan/tekanan pedal berdasarkan rasio jarak
    antara Titik 4 (THUMB_TIP / Ujung Ibu Jari) dan Titik 6 (INDEX_FINGER_PIP / Sendi Tengah Telunjuk)
    terhadap ukuran telapak tangan (Titik 0 ke Titik 9).
    """
    h, w, _ = image.shape

    # Titik 0: Pergelangan Tangan (Wrist) - untuk acuan skala
    p0 = np.array([hand_landmarks.landmark[0].x * w, hand_landmarks.landmark[0].y * h])

    # Titik 4: Ujung Ibu Jari (THUMB_TIP)
    p4 = np.array([hand_landmarks.landmark[4].x * w, hand_landmarks.landmark[4].y * h])

    # Titik 6: Sendi Tengah Telunjuk (INDEX_FINGER_PIP)
    p6 = np.array([hand_landmarks.landmark[6].x * w, hand_landmarks.landmark[6].y * h])

    # Titik 9: Pangkal Jari Tengah (MIDDLE_FINGER_MCP) - untuk acuan skala
    p9 = np.array([hand_landmarks.landmark[9].x * w, hand_landmarks.landmark[9].y * h])

    # 1. Hitung Jarak antara Titik 4 (Ujung Ibu Jari) dan Titik 6 (Sendi Telunjuk)
    pinch_dist = np.linalg.norm(p4 - p6)

    # 2. Hitung Ukuran Telapak Tangan (Titik 0 ke 9) untuk Normalisasi Jarak Kamera
    palm_size = np.linalg.norm(p9 - p0)

    if palm_size == 0:
        return 0.0

    # 3. Hitung Rasio Relatif
    ratio = pinch_dist / palm_size

    # --- AMBANG KALIBRASI RASIO (P4 ke P6) ---
    # min_ratio: Saat ujung jempol menyentuh sendi telunjuk (Tekan 100%) ~0.15 - 0.22
    # max_ratio: Saat ujung jempol terbuka/jauh dari telunjuk (Tekan 0%) ~0.50 - 0.65
    min_ratio = 0.20
    max_ratio = 0.55

    # Skalakan rasio ke rentang 0.0 s.d. 1.0 (makin kecil rasio = makin tinggi tekannya)
    press_value = (max_ratio - ratio) / (max_ratio - min_ratio)
    press_value = float(np.clip(press_value, 0.0, 1.0))

    # --- VISUALISASI GARIS P4 (JEMPOL) KEPADA P6 (TELUNJUK) ---
    p4_int = (int(p4[0]), int(p4[1]))
    p6_int = (int(p6[0]), int(p6[1]))

    # Garis berwarna Magenta penghubung P4 & P6
    cv2.line(image, p4_int, p6_int, (255, 0, 255), 2)
    # Titik Kuning di P4 dan P6
    cv2.circle(image, p4_int, 5, (0, 255, 255), -1)
    cv2.circle(image, p6_int, 5, (0, 255, 255), -1)

    return press_value


def calculate_hands_line_angle(image, left_hand_landmarks, right_hand_landmarks):
    """
    Menghitung sudut kemiringan setir berdasarkan posisi FISIK tangan driver.
    """
    h, w, _ = image.shape

    left_wrist = np.array(
        [
            int(left_hand_landmarks.landmark[0].x * w),
            int(left_hand_landmarks.landmark[0].y * h),
        ]
    )
    right_wrist = np.array(
        [
            int(right_hand_landmarks.landmark[0].x * w),
            int(right_hand_landmarks.landmark[0].y * h),
        ]
    )

    dy_height = left_wrist[1] - right_wrist[1]
    dx_distance = abs(right_wrist[0] - left_wrist[0])

    if dx_distance > 0:
        angle_rad = math.atan(dy_height / dx_distance)
        angle_deg = math.degrees(angle_rad)
    else:
        angle_deg = 90.0 if dy_height > 0 else -90.0

    # Visualisasi Garis Setir
    cv2.line(image, tuple(left_wrist), tuple(right_wrist), (255, 100, 0), 2)
    siku_point = (right_wrist[0], left_wrist[1])
    cv2.line(image, tuple(left_wrist), siku_point, (128, 128, 255), 1, cv2.LINE_AA)
    cv2.line(image, tuple(right_wrist), siku_point, (128, 128, 255), 1, cv2.LINE_AA)

    return angle_deg


def draw_landmarks(image, hand_landmarks, hand_label):
    """Menggambar titik pergelangan tangan (ID 0)."""
    h, w, _ = image.shape
    wrist_x = int(hand_landmarks.landmark[0].x * w)
    wrist_y = int(hand_landmarks.landmark[0].y * h)

    color = (0, 255, 0) if hand_label == "Right" else (0, 100, 255)

    cv2.circle(image, (wrist_x, wrist_y), 8, color, 2)
    cv2.circle(image, (wrist_x, wrist_y), 3, color, -1)

    cv2.putText(
        image,
        hand_label,
        (wrist_x - 15, wrist_y - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA,
    )
