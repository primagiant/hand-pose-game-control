import copy
import mediapipe as mp
from utils.draw import *

FONT = cv2.FONT_HERSHEY_SIMPLEX


def main():
    cap = cv2.VideoCapture(0)
    hand_detector = mp.solutions.hands.Hands(
        max_num_hands=1
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
        saved_image = copy.deepcopy(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hand_detector.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            _ = get_landmarks(debug_image, results.multi_hand_landmarks)
            logging_image(number, mode, saved_image)

        debug_image = draw_info(debug_image, mode, number)
        cv2.imshow('Hand Pose Games Dataset Maker', debug_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
