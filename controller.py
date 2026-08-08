import cv2
import mediapipe as mp
from utils.draw import (
    draw_landmarks,
    calculate_hands_line_angle,
    calculate_pedal_press,
)


def main():
    cap = cv2.VideoCapture(1)

    mp_hands = mp.solutions.hands
    hand_detector = mp_hands.Hands(
        max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5
    )

    while True:
        ret, image = cap.read()
        if not ret:
            break

        image = cv2.flip(image, 1)
        debug_image = image.copy()

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = hand_detector.process(image_rgb)

        hands_data = {}
        gas_val = 0.0
        brake_val = 0.0

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                hand_label = handedness.classification[0].label  # "Left" atau "Right"
                hands_data[hand_label] = hand_landmarks

                # Gambar titik dasar
                draw_landmarks(debug_image, hand_landmarks, hand_label)

                # Hitung Nilai Gas (Tangan Kanan) dan Rem (Tangan Kiri)
                if hand_label == "Right":
                    gas_val = calculate_pedal_press(debug_image, hand_landmarks)
                elif hand_label == "Left":
                    brake_val = calculate_pedal_press(debug_image, hand_landmarks)

            # Jika KEDUA TANGAN terdeteksi -> Hitung Sudut Kemudi
            if "Left" in hands_data and "Right" in hands_data:
                steering_angle = calculate_hands_line_angle(
                    debug_image, hands_data["Left"], hands_data["Right"]
                )

                # --- HUD DISPLAY (INFORMASI SIMULASI) ---
                # STATUS SETIR
                if steering_angle > 2.0:
                    steer_text = f"KIRI ({steering_angle:.1f} deg)"
                elif steering_angle < -2.0:
                    steer_text = f"KANAN ({steering_angle:.1f} deg)"
                else:
                    steer_text = f"LURUS ({steering_angle:.1f} deg)"

                cv2.putText(
                    debug_image,
                    f"Setir : {steer_text}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # HUD GAS (Tangan Kanan)
            cv2.putText(
                debug_image,
                f"GAS  : {gas_val * 100:.0f}%",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # HUD REM (Tangan Kiri)
            cv2.putText(
                debug_image,
                f"REM  : {brake_val * 100:.0f}%",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            # --- INDIKATOR VISUAL DENGAN PROGRESS BAR ---
            # Bar Gas (Hijau)
            cv2.rectangle(debug_image, (220, 65), (320, 85), (50, 50, 50), -1)
            cv2.rectangle(
                debug_image, (220, 65), (220 + int(gas_val * 100), 85), (0, 255, 0), -1
            )

            # Bar Rem (Merah)
            cv2.rectangle(debug_image, (220, 105), (320, 125), (50, 50, 50), -1)
            cv2.rectangle(
                debug_image,
                (220, 105),
                (220 + int(brake_val * 100), 125),
                (0, 0, 255),
                -1,
            )

        cv2.imshow("Simulator Kemudi - Setir, Gas, & Rem", debug_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
