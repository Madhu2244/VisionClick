import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from typing import Optional, List

CV2_VIDEO_FEED = 0
CV2_FRAME_FLIP = 1

class HeadControlledMouse:
    """A class to control the mouse cursor using head movements and eye blinking detected through the webcam."""

    def __init__(self) -> None:
        """Initializes the head-controlled mouse system."""
        self.cam = cv2.VideoCapture(CV2_VIDEO_FEED)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.screen_w, self.screen_h = pyautogui.size()
        self.running: bool = True

    def run(self) -> None:
        """Main loop to capture webcam frames and process head movements and eye blinks to control the mouse."""
        while self.running:
            success, frame = self.cam.read()
            if not success:
                continue  # If the frame is not successfully captured, skip this iteration
            
            frame = cv2.flip(frame, CV2_FRAME_FLIP)

            landmark_points = self.get_landmark_points(frame)

            if landmark_points:
                self.process_landmarks(landmark_points[0].landmark, frame)

            cv2.imshow('Eye Controlled Mouse', frame)

            self.check_quit()

    def get_landmark_points(self, frame: np.ndarray) -> Optional[List[any]]:
        """
        Processes the given frame to detect facial landmarks.

        Args:
            frame (np.ndarray): The current frame captured from the webcam.

        Returns:
            Optional[List[mp.framework.formats.landmark_pb2.NormalizedLandmarkList]]: A list of detected facial landmarks. None if no face is detected.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = self.face_mesh.process(rgb_frame)
        return output.multi_face_landmarks
    
    def check_quit(self) -> None:
        """Checks if the 'q' key is pressed to quit the application."""
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.running = False

    def process_landmarks(self, landmarks: List[any], frame: np.ndarray) -> None:
        """
        Processes detected landmarks to control mouse movement and clicks.

        Args:
            landmarks (List[mp.framework.formats.landmark_pb2.NormalizedLandmark]): Detected landmarks for the largest face in the frame.
            frame (np.ndarray): The current frame for drawing debug information.
        """
        frame_h, frame_w, _ = frame.shape

        # Process eye landmarks for mouse movement
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))
            if id == 1:
                screen_x = self.screen_w * landmark.x
                screen_y = self.screen_h * landmark.y
                pyautogui.moveTo(screen_x, screen_y)

        # Process eyelid landmarks for click action
        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))
        if (left[0].y - left[1].y) < 0.004:
            pyautogui.click()
            pyautogui.sleep(1)