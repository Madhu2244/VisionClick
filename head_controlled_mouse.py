import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from typing import Optional, List

class CameraManager:
    """Manages the camera feed for capturing frames."""

    def __init__(self, video_feed=0):
        """ Initializes the CameraManager. """
        self.video_feed = video_feed
        self.cam = cv2.VideoCapture(self.video_feed)
    
    def read_frame(self):
        """ Reads a frame from the camera feed. """
        success, frame = self.cam.read()
        if success:
            return cv2.flip(frame, 1)
        return None
    
    def release(self):
        """Releases the camera."""
        self.cam.release()

class LandmarkManager:
    """Manages landmark detection on frames."""

    def __init__(self, screen_w, screen_h):
        """ Initializes the LandmarkManager. """
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.screen_w = screen_w
        self.screen_h = screen_h
    
    def process_frame(self, frame: np.ndarray):
        """ Processes a frame to detect landmarks. """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.face_mesh.process(rgb_frame).multi_face_landmarks
    
    def process_eye_landmarks(self, landmarks, frame, frame_w, frame_h, eye_indices, mouse_controller, color=(0, 255, 0), move_mouse = False):
        """ Processes landmarks related to eyes. """
        for id, landmark in enumerate([landmarks[i] for i in eye_indices]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, color)  # Visual feedback

            # Example of moving the mouse based on one landmark, adjust according to your needs
            if move_mouse and id == 1:  # Adjust this condition as needed
                screen_x = self.screen_w * landmark.x
                screen_y = self.screen_h * landmark.y
                mouse_controller.move_mouse_smoothly(screen_x, screen_y, damping=0.1)
                # pyautogui.moveTo(screen_x, screen_y)

    def process_landmarks(self, landmarks: List[any], frame: np.ndarray, mouse_controller) -> None:
        """ Sets up the landmarks related to eyes. """
        frame_h, frame_w, _ = frame.shape
        
        # Define landmark ranges for both eyes, adjust these based on your needs
        right_eye_indices = range(474, 478)  # Assuming these are for the right eye
        left_eye_indices = range(468, 472)  # Placeholder, adjust with correct left eye indices

        self.process_eye_landmarks(landmarks, frame, frame_w, frame_h, right_eye_indices, mouse_controller, color=(0, 255, 0), move_mouse = True)
        self.process_eye_landmarks(landmarks, frame, frame_w, frame_h, left_eye_indices, mouse_controller, color=(0, 255, 0), move_mouse = False)

class MouseController:
    """Controls mouse movements."""

    def __init__(self):
        """Initializes the MouseController."""
        self.screen_w, self.screen_h = pyautogui.size()

    def lerp(self, start, end, alpha):
        """Performs linear interpolation."""
        return start + (end - start) * alpha
    
    def move_mouse_smoothly(self, target_x, target_y, damping=0.1):
        """Moves the mouse cursor smoothly to the target position."""
        current_x, current_y = pyautogui.position()
        new_x = self.lerp(current_x, target_x, damping)
        new_y = self.lerp(current_y, target_y, damping)
        pyautogui.moveTo(new_x, new_y)

class HeadControlledMouse:
    """A class to control the mouse cursor using head movements and eye blinking detected through the webcam."""

    def __init__(self) -> None:
        """Initializes the head-controlled mouse system."""
        screen_w, screen_h = pyautogui.size()
        self.camera_manager = CameraManager()
        self.landmark_manager = LandmarkManager(screen_w, screen_h)
        self.mouse_controller = MouseController()
        self.running = True

    def run(self) -> None:
        """Main loop to capture webcam frames and process head movements and eye blinks to control the mouse."""
        while self.running:
            frame = self.camera_manager.read_frame()
            if frame is None:
                continue
            
            landmark_points = self.landmark_manager.process_frame(frame)
            if landmark_points:
                self.landmark_manager.process_landmarks(landmark_points[0].landmark, frame, self.mouse_controller)
            
            cv2.imshow('Head Controlled Mouse', frame)
            self.check_quit()
    
    def check_quit(self) -> None:
        """Checks if the 'q' key is pressed to quit the application."""
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.running = False