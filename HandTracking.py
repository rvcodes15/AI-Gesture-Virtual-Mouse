import cv2
import mediapipe as mp
import math


class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        base_options = mp.tasks.BaseOptions(
            model_asset_path="hand_landmarker.task"
        )

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=maxHands,
            min_hand_detection_confidence=detectionCon,
            min_hand_presence_confidence=trackCon,
            min_tracking_confidence=trackCon
        )

        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
            options
        )

        self.mpDraw = mp.tasks.vision.HandLandmarksConnections
        self.tipIds = [4, 8, 12, 16, 20]
        self.lmList = []
        self.timestamp = 0

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=imgRGB
        )

        self.timestamp += 1

        self.results = self.landmarker.detect_for_video(
            mp_image,
            self.timestamp
        )

        if self.results.hand_landmarks and draw:
            for hand_landmarks in self.results.hand_landmarks:

                for connection in self.mpDraw.HAND_CONNECTIONS:
                    start = hand_landmarks[connection.start]
                    end = hand_landmarks[connection.end]

                    h, w, _ = img.shape

                    x1 = int(start.x * w)
                    y1 = int(start.y * h)
                    x2 = int(end.x * w)
                    y2 = int(end.y * h)

                    cv2.line(
                        img,
                        (x1, y1),
                        (x2, y2),
                        (255, 0, 255),
                        2
                    )

                for landmark in hand_landmarks:
                    h, w, _ = img.shape

                    cx = int(landmark.x * w)
                    cy = int(landmark.y * h)

                    cv2.circle(
                        img,
                        (cx, cy),
                        5,
                        (255, 0, 255),
                        cv2.FILLED
                    )

        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        bbox = []

        if not self.results.hand_landmarks:
            return self.lmList, bbox

        if handNo >= len(self.results.hand_landmarks):
            return self.lmList, bbox

        myHand = self.results.hand_landmarks[handNo]

        xList = []
        yList = []

        h, w, _ = img.shape

        for id, lm in enumerate(myHand):
            cx = int(lm.x * w)
            cy = int(lm.y * h)

            xList.append(cx)
            yList.append(cy)

            self.lmList.append([id, cx, cy])

            if draw:
                cv2.circle(
                    img,
                    (cx, cy),
                    5,
                    (255, 0, 255),
                    cv2.FILLED
                )

        xmin, xmax = min(xList), max(xList)
        ymin, ymax = min(yList), max(yList)

        bbox = (xmin, ymin, xmax, ymax)

        if draw:
            cv2.rectangle(
                img,
                (xmin - 20, ymin - 20),
                (xmax + 20, ymax + 20),
                (0, 255, 0),
                2
            )

        return self.lmList, bbox

    def fingersUp(self):
        fingers = []

        if not self.lmList:
            return [0, 0, 0, 0, 0]

        # Thumb
        if self.lmList[4][1] > self.lmList[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Four fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):

        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if draw:
            cv2.line(
                img,
                (x1, y1),
                (x2, y2),
                (255, 0, 255),
                t
            )

            cv2.circle(
                img,
                (x1, y1),
                r,
                (255, 0, 255),
                cv2.FILLED
            )

            cv2.circle(
                img,
                (x2, y2),
                r,
                (255, 0, 255),
                cv2.FILLED
            )

            cv2.circle(
                img,
                (cx, cy),
                r,
                (0, 0, 255),
                cv2.FILLED
            )

        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


def main():

    cap = cv2.VideoCapture(0)

    detector = handDetector(maxHands=1)

    while True:

        success, img = cap.read()

        if not success:
            print("Camera not detected.")
            break

        img = detector.findHands(img)

        lmList, bbox = detector.findPosition(img)

        if len(lmList) != 0:
            print(lmList[8])

        cv2.imshow("Hand Tracking", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()