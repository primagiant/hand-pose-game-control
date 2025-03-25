import copy
import mediapipe as mp
import joblib
import numpy
from utils.draw import *

FONT = cv2.FONT_HERSHEY_SIMPLEX
svm_model = joblib.load('./model/svm_model.pkl')
scaler = joblib.load('./model/scaler.pkl')
label_encoder = joblib.load('./model/label_encoder.pkl')


def main():
    cap = cv2.VideoCapture(0)
    hand_detector = mp.solutions.hands.Hands(
        max_num_hands=2
    )

    mode = 0

    while True:

        # Process Key (ESC: end)
        key = cv2.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        ret, image = cap.read()
        if not ret:
            break

        image = cv2.flip(image, 1)
        debug_image = copy.deepcopy(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hand_detector.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            draw_hand_rect(debug_image, results.multi_hand_landmarks, svm_model, scaler, label_encoder)

        cv2.imshow('Test Hand Pose Games', debug_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
