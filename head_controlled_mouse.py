import cv2
import mediapipe as mp
import pyautogui

class HeadControlledMouse:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.screen_w, self.screen_h = pyautogui.size()

    def run(self):
        while True:
            success, frame = self.cam.read()
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output = self.face_mesh.process(rgb_frame)
            landmark_points = output.multi_face_landmarks

            if landmark_points:
                self.process_landmarks(landmark_points[0].landmark, frame)

            cv2.imshow('Eye Controlled Mouse', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def process_landmarks(self, landmarks, frame):
        frame_h, frame_w, _ = frame.shape

        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))
            if id == 1:
                screen_x = self.screen_w * landmark.x
                screen_y = self.screen_h * landmark.y
                pyautogui.moveTo(screen_x, screen_y)

        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))
        if (left[0].y - left[1].y) < 0.004:
            pyautogui.click()
            pyautogui.sleep(1)
