import cv2
import csv
import os
import time

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
        csv_path = 'model/dataset.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return

def logging_image(number, mode, image):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        folder_path = f'image_dataset/{number}/'
        os.makedirs(folder_path, exist_ok=True)

        timestamp = int(time.time() * 1000)
        image_path = os.path.join(folder_path, f'{timestamp}.jpg')

        cv2.imwrite(image_path, image)


def draw_info(image, mode, number):
    mode_string = ['Normal', 'Pengumpulan Dataset']
    if mode == 0:
        cv2.putText(image, "MODE:" + mode_string[mode], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1,
                    cv2.LINE_AA)
    if mode == 1:
        cv2.putText(image, "MODE:" + mode_string[mode], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1,
                    cv2.LINE_AA)
        if 0 <= number <= 9:
            cv2.putText(image, "NUM:" + str(number), (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1,
                        cv2.LINE_AA)
    return image

def get_rectangle_landmark(image, hands):
    landmark_angle = []

    for hand in hands:
        landmarks = hand.landmark
        frame_height, frame_width, _ = image.shape

        min_x, min_y = frame_width, frame_height
        max_x, max_y = 0, 0

        # iterasi setiap landmark pada tangan
        for idh, landmark in enumerate(landmarks):
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)

            # Update min and max coordinates
            min_x, min_y = min(min_x, x), min(min_y, y)
            max_x, max_y = max(max_x, x), max(max_y, y)

        # Simpan hasil min dan max koordinat
        landmark_angle.append({'min': (min_x, min_y), 'max': (max_x, max_y)})

        # Gambar kotak pada tangan
        cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

    return landmark_angle

def get_landmarks(image, hands):
    landmark_coors = []

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

    return landmark_coors
