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

    def process_eye_landmarks(self, landmarks, frame, frame_w, frame_h, eye_indices, color=(0, 255, 0), move_mouse = False):
        """
        Processes and visualizes landmarks for an eye.

        Args:
            landmarks: The detected facial landmarks.
            frame: The current frame from the webcam.
            frame_w: The width of the frame.
            frame_h: The height of the frame.
            eye_indices: The indices of landmarks for the eye being processed.
            color: The color for the visualization of landmarks. Default is green.
        """
        for id, landmark in enumerate([landmarks[i] for i in eye_indices]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, color)  # Visual feedback

            # Example of moving the mouse based on one landmark, adjust according to your needs
            if move_mouse and id == 1:  # Adjust this condition as needed
                screen_x = self.screen_w * landmark.x
                screen_y = self.screen_h * landmark.y
                pyautogui.moveTo(screen_x, screen_y)

    def process_landmarks(self, landmarks: List[any], frame: np.ndarray) -> None:
        """
        Processes detected landmarks to control mouse movement and clicks.

        Args:
            landmarks (List[mp.framework.formats.landmark_pb2.NormalizedLandmark]): Detected landmarks for the largest face in the frame.
            frame (np.ndarray): The current frame for drawing debug information.
        """
        frame_h, frame_w, _ = frame.shape
        
        # Define landmark ranges for both eyes, adjust these based on your needs
        right_eye_indices = range(474, 478)  # Assuming these are for the right eye
        left_eye_indices = range(468, 472)  # Placeholder, adjust with correct left eye indices

        self.process_eye_landmarks(landmarks, frame, frame_w, frame_h, right_eye_indices, color=(0, 255, 0), move_mouse = True)

        self.process_eye_landmarks(landmarks, frame, frame_w, frame_h, left_eye_indices, color=(0, 255, 0), move_mouse = False)