import cv2
import csv
import os
import mediapipe as mp

def main():
    hand_detector = mp.solutions.hands.Hands(
        max_num_hands=1
    )
    d_foldername = './image_dataset'
    csv_path = 'model/dataset.csv'

    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)

        for root, dirs, files in os.walk(d_foldername):
            for file in files:
                relative_path = os.path.relpath(os.path.join(root, file), d_foldername)
                pose = os.path.basename(root)
                pose = pose.split('-')[0]
                image_path = os.path.join(d_foldername, relative_path)

                image = cv2.imread(image_path)
                if image is None:
                    print(f"Skipping: {image_path} (could not load)")
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hand_detector.process(image_rgb)

                if results.multi_hand_landmarks:
                    landmark_list = get_landmarks(image, results.multi_hand_landmarks)
                    writer.writerow([pose] + landmark_list)

        print("A block of files has been transformed into the dataset")
    print("Completed")

def get_landmarks(image, hands):
    landmark_list = []

    for hand in hands:
        landmarks = hand.landmark
        frame_height, frame_width, _ = image.shape

        # Iterate through landmarks
        for landmark in landmarks:
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            landmark_list.append(x)
            landmark_list.append(y)

    return landmark_list

if __name__ == '__main__':
    main()
