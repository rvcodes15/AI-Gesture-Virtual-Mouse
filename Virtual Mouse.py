import cv2
import numpy as np
import time
import HandTracking as ht
import autopy
import pyautogui

# Variables Declaration
pTime = 0
width = 640
height = 480
frameR = 100
smoothening = 8

prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

detector = ht.handDetector(maxHands=1)

screen_width, screen_height = autopy.screen.size()

while True:
    success, img = cap.read()

    if not success:
        continue

    img = detector.findHands(img)
    lmlist, bbox = detector.findPosition(img)

    if len(lmlist) != 0:

        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]

        fingers = detector.fingersUp()

        cv2.rectangle(
            img,
            (frameR, frameR),
            (width - frameR, height - frameR),
            (255, 0, 255),
            2
        )

        # -------------------------
        # CURSOR MOVEMENT
        # Index finger only
        # -------------------------
        if fingers[1] == 1 and fingers[2] == 0:

            x3 = np.interp(
                x1,
                (frameR, width - frameR),
                (0, screen_width)
            )

            y3 = np.interp(
                y1,
                (frameR, height - frameR),
                (0, screen_height)
            )

            curr_x = prev_x + (x3 - prev_x) / smoothening
            curr_y = prev_y + (y3 - prev_y) / smoothening

            autopy.mouse.move(
                screen_width - curr_x,
                curr_y
            )

            cv2.circle(
                img,
                (x1, y1),
                7,
                (255, 0, 255),
                cv2.FILLED
            )

            prev_x, prev_y = curr_x, curr_y

        # -------------------------
        # SCROLL MODE
        # Three fingers up
        # -------------------------
        if (
            fingers[1] == 1
            and fingers[2] == 1
            and fingers[3] == 1
        ):

            scroll_y = lmlist[12][2] - lmlist[9][2]

            if scroll_y > 20:
                pyautogui.scroll(3)

            elif scroll_y < -20:
                pyautogui.scroll(-3)

            cv2.putText(
                img,
                "SCROLL MODE",
                (20, 100),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 255, 0),
                2
            )

        # -------------------------
        # LEFT CLICK
        # Index + Middle finger
        # but NOT three fingers
        # -------------------------
        if (
            fingers[1] == 1
            and fingers[2] == 1
            and fingers[3] == 0
        ):

            length, img, lineInfo = detector.findDistance(
                8, 12, img
            )

            if length < 40:

                cv2.circle(
                    img,
                    (lineInfo[4], lineInfo[5]),
                    15,
                    (0, 255, 0),
                    cv2.FILLED
                )

                autopy.mouse.click()

    # -------------------------
    # FPS
    # -------------------------

    cTime = time.time()

    fps = 1 / (cTime - pTime)

    pTime = cTime

    cv2.putText(
        img,
        str(int(fps)),
        (20, 50),
        cv2.FONT_HERSHEY_PLAIN,
        3,
        (255, 0, 0),
        3
    )

    cv2.imshow("AI Gesture Virtual Mouse", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()