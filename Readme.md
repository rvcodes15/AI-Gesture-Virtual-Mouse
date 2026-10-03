# AI Gesture Virtual Mouse 🖱️

A real-time AI-powered virtual mouse that allows users to control the computer cursor using hand gestures through a webcam.

The project uses **OpenCV** for webcam processing, **MediaPipe Tasks** for hand landmark detection, and **PyAutoGUI/AutoPy** for mouse control.

---

## 🚀 Features

- Real-time hand tracking using webcam
- AI-based hand landmark detection
- Cursor movement using the index finger
- Click using index finger + middle finger gesture
- Scroll control using hand gestures
- Cursor movement smoothing
- Real-time FPS display
- No physical mouse required

---

## 🖐️ Gesture Controls

| Gesture | Action |
|---|---|
| ☝️ Index finger up | Move cursor |
| ☝️ + 🖕 fingers together | Left click |
| ☝️ + 🖕 + scroll gesture | Scroll |
| No hand detected | Cursor remains unchanged |

---

## 🛠️ Technologies Used

- Python 3.12
- OpenCV
- MediaPipe Tasks
- NumPy
- PyAutoGUI
- AutoPy

---

## 📁 Project Structure

```text
AI-Gesture-Virtual-Mouse/
│
├── HandTracking.py
├── Virtual Mouse.py
├── hand_landmarker.task
├── requirements.txt
├── Virtual Mouse.gif
├── LICENSE
├── .gitignore
└── Readme.md