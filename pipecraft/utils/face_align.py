import cv2
import numpy as np
from einops import rearrange
from typing import List, Union, Tuple, Dict, Optional
import os
import threading
from tqdm import tqdm
import logging
import traceback
from matplotlib import pyplot as plt

try:
    import mediapipe as mp
except ImportError:
    # Install MediaPipe if not already installed
    os.system("pip install mediapipe")
    import mediapipe as mp

class FaceAlignmentProcessor:
    """
    A class for processing videos, detecting facial landmarks,
    and performing face alignment using MediaPipe.
    """
    
    def __init__(self, 
                 resolution: int = 256, 
                 device: str = "cpu",
                 static_image_mode: bool = False,
                 max_num_faces: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 use_mediapipe: bool = True,
                 ):
        """
        Initialize the FaceAlignmentProcessor.
        
        Args:
            resolution: Output resolution for aligned faces.
            device: Device to run processing on ('cpu' or 'cuda:X').
            static_image_mode: Whether to treat each frame as a static image.
            max_num_faces: Maximum number of faces to detect.
            min_detection_confidence: Minimum confidence for face detection.
            min_tracking_confidence: Minimum confidence for landmark tracking.
        """
        self.resolution = resolution
        self.device = device
        
        self.use_mediapipe = use_mediapipe
        
        if use_mediapipe:
            # Initialize MediaPipe face mesh
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=static_image_mode,
                max_num_faces=max_num_faces,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                refine_landmarks=True,
            )
        else:
            import face_alignment
            self.face_mesh = face_alignment.FaceAlignment(
                    face_alignment.LandmarksType.TWO_D, flip_input=False, device=device
                )
        
        # Initialize laplacian smoother and affine restorer
        self.smoother = laplacianSmooth()
        self.restorer = AlignRestore()
        
        # Mapping for landmark conversion
        self.landmark_points_68 = [
            162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 
            378, 365, 397, 288, 323, 454, 389, 71, 63, 105, 
            66, 107, 336, 296, 334, 293, 301, 168, 197, 5, 
            4, 75, 97, 2, 326, 305, 33, 160, 158, 133, 153, 
            144, 362, 385, 387, 263, 373, 380, 61, 39, 37, 
            0, 267, 269, 291, 405, 314, 17, 84, 181, 78, 82, 
            13, 312, 308, 317, 14, 87
        ]
        self.lock = threading.Lock()
        
    def __detect_landmarks__(self, image: np.ndarray, return_2d, retries: int = 4):
        height, width = image.shape[:2]
        if self.use_mediapipe:
            results = self.face_mesh.process(image)
            detected_faces = results.multi_face_landmarks
        else:
            detected_faces = self.face_mesh.get_landmarks(image)
            
        if not detected_faces:  # Face not detected
            if retries > 0:
                # Retry detection
                return self.__detect_landmarks__(image, return_2d, retries - 1)
            # else:
                # No face detected after retries
                # logging.warning("Face not detected after retries.")
                # # cv2.imwrite("face_not_detected.jpg", image)
                # plt.imshow(image)
                # plt.axis('off')
                # plt.show()
            raise RuntimeError("Face not detected")
        
        face_landmarks = detected_faces[0]  # Only use the first face
        if self.use_mediapipe:
        
            # Extract landmark coordinates
            landmark_coordinates = []
            for landmark in face_landmarks.landmark:
                x = landmark.x * width
                y = landmark.y * height
                z = landmark.z * width  # Scale Z relative to image width
                
                if return_2d:
                    landmark_coordinates.append((x, y))
                else:
                    landmark_coordinates.append((x, y, z))
                    
            landmark_coordinates = np.array(landmark_coordinates)
            
            # Convert to face_alignment format if requested
            face_landmarks = self.convert_to_face_alignment_format(landmark_coordinates)
        return face_landmarks
        
    def detect_face(self,
                     image: Union[np.ndarray]) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in the given image.
        """
        height, width = image.shape[:2]
        face_landmarks = self.__detect_landmarks__(image)
        # Extract bounding box coordinates
        x_min = int(min([lm.x for lm in face_landmarks.landmark]) * width)
        x_max = int(max([lm.x for lm in face_landmarks.landmark]) * width)
        y_min = int(min([lm.y for lm in face_landmarks.landmark]) * height)
        y_max = int(max([lm.y for lm in face_landmarks.landmark]) * height)
        # Return bounding box coordinates
        return [(x_min, y_min, x_max, y_max)]
    
    def detect_landmarks(self, 
                         image: Union[np.ndarray], 
                         return_2d: bool = True) -> Optional[np.ndarray]:
        """
        Detect facial landmarks in the given image.
        
        Args:
            image: Input image as numpy array
            return_2d: Whether to return 2D landmarks only.
            convert_to_fa_format: Whether to convert to face_alignment format (68 points).
            
        Returns:
            Array of landmark coordinates or None if no face detected.
        """
        landmark_coordinates = self.__detect_landmarks__(image, return_2d)
            
        return landmark_coordinates
    
    def convert_to_face_alignment_format(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Convert MediaPipe's 478 landmarks to face_alignment's 68 landmarks format.
        
        Args:
            landmarks: MediaPipe facial landmarks array [478, 2] or [478, 3]
            
        Returns:
            Array of 68 landmark points in face_alignment format.
        """
        landmarks_extracted = []
        for index in self.landmark_points_68:
            x = landmarks[index][0]
            y = landmarks[index][1]
            landmarks_extracted.append((x, y))
        return np.array(landmarks_extracted)
        
    def get_aligned_faces(self, 
                        image: Union[np.ndarray],
                        old_state: Optional[Dict] = {},
                        return_box: bool = False,
                        return_matrix: bool = False,
                        upscale_interpolation=cv2.INTER_CUBIC,
                        downscale_interpolation=cv2.INTER_AREA,
                        debug_visualization: bool = True,
                        debug_save_path: Optional[str] = None,
                        ) -> Union[np.ndarray, Tuple]:
        """
        Detect face and return aligned face.
        
        Args:
            image: Input image
            old_state: Previous state for smooth tracking
            return_box: Whether to return the face bounding box
            return_matrix: Whether to return the affine transformation matrix
            upscale_interpolation: Interpolation method for upscaling
            downscale_interpolation: Interpolation method for downscaling
            debug_visualization: Whether to visualize landmarks for debugging
            debug_save_path: Path to save the debug visualization
            
        Returns:
            Aligned face image, and optionally bounding box and affine matrix
        """
        image = cv2.resize(image, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)
        # Get landmarks in face_alignment format
        landmarks = self.detect_landmarks(image, return_2d=True)
        
        if landmarks is None:
            raise RuntimeError("Face not detected")
        
        # Apply Laplacian smoothing to landmarks
        points, pts_last = self.smoother.smooth(landmarks, old_state.get("pts_last", None))
        
        # Calculate reference points for alignment
        lmk3_ = np.zeros((3, 2))
        # lmk3_[0] = points[17:22].mean(0)  # Left eye region
        # lmk3_[1] = points[22:27].mean(0)  # Right eye region
        # lmk3_[2] = points[27:36].mean(0)  # Nose region
        
        lmk3_[0] = landmarks[36:42].mean(0)  # Left eye region
        lmk3_[1] = landmarks[42:48].mean(0)  # Right eye region
        lmk3_[2] = landmarks[27:36].mean(0)  # Nose region
        
        # Debug visualization if requested
        if debug_visualization:
            image = self.visualize_alignment_landmarks(image, points, save_path=debug_save_path)
        
        # Align and warp the face
        face, affine_matrix, p_bias = self.restorer.align_warp_face(
            image, 
            p_bias=old_state.get("p_bias", None),
            lmks3=lmk3_, smooth=True, border_mode="constant"
        )
        
        # Get bounding box
        box = [0, 0, face.shape[1], face.shape[0]]  # x1, y1, x2, y2
        
        # Calculate original dimensions
        H, W, C = image.shape
        is_downscaling = max(H, W) > self.resolution
        interpolation = downscale_interpolation if is_downscaling else upscale_interpolation
        
        # Resize to target resolution
        face = cv2.resize(face, (self.resolution, self.resolution), interpolation=interpolation)
        
        state = {
            "pts_last": pts_last,
            "p_bias": p_bias
        }
        
        # Determine what to return
        if return_box and return_matrix:
            return face, box, affine_matrix, state
        elif return_box:
            return face, box, state
        elif return_matrix:
            return face, affine_matrix, state
        else:
            return face, state
            
    def process_frames(self, frames: List[np.ndarray], break_on_error: bool = False) -> List[np.ndarray]:
        """
        Process a list of frames and extract aligned faces.
        
        Args:
            frames: List of frames (numpy arrays)
            
        Returns:
            List of aligned face images
        """
        with self.lock:
            aligned_faces = []
            state = {}
            for frame in frames:
                try:
                    # print(f"Processing frame of shape: {frame.shape}")
                    aligned_face, state = self.get_aligned_faces(frame, old_state=state)
                    # print(f"Aligned face shape: {aligned_face.shape}")
                    aligned_faces.append(aligned_face)
                except RuntimeError as e:
                    # print(f"Error processing frame: {e}. Only processed {len(aligned_faces)} frames.")
                    # traceback.print_exc()
                    if break_on_error:
                        break
                    continue
            return aligned_faces
        
    def detect_faces(self, frames: List[np.ndarray], break_on_error: bool = False) -> List[np.ndarray]:
        """
        Detect faces in a list of frames.
        
        Args:
            frames: List of frames (numpy arrays)
            
        Returns:
            List of bounding boxes for detected faces
        """
        with self.lock:
            face_boxes = []
            for frame in frames:
                try:
                    boxes = self.detect_face(frame)
                    face_boxes.append(boxes)
                except Exception as e:
                    # print(f"Error detecting faces: {e}. Only processed {len(face_boxes)} frames.")
                    # traceback.print_exc()
                    if break_on_error:
                        break
                    continue
            return face_boxes
    
    def reset(self):
        """
        Reset the processor state.
        """
        self.smoother.pts_last = None
        self.restorer.p_bias = None
    
    def close(self):
        """
        Release resources.
        """
        if self.face_mesh:
            self.face_mesh.close()

    def visualize_alignment_landmarks(self, image, landmarks, save_path=None, show=True):
        """
        Visualize the specific landmarks used for face alignment.
        
        Args:
            image: Input image
            landmarks: Full set of facial landmarks
            save_path: Optional path to save the visualization
            show: Whether to display the visualization
            
        Returns:
            Visualization image with landmarks drawn
        """
        # Create a copy of the image for drawing
        vis_image = image.copy()
    
        # # Scale landmarks to the image size
        # height, width = vis_image.shape[:2]
        # landmarks = landmarks * np.array([width, height])
        # landmarks = landmarks.astype(int)
        
        # Draw all landmarks as small dots
        for i, (x, y) in enumerate(landmarks):
            cv2.circle(vis_image, (int(x), int(y)), 1, (0, 255, 0), -1)
            # Label every landmark with its index
            cv2.putText(vis_image, str(i), (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Extract the specific landmark regions used for alignment
        # left_eye_region = landmarks[17:22]  # Left eye region
        # right_eye_region = landmarks[22:27]  # Right eye region
        # nose_region = landmarks[27:36]  # Nose region
        
        left_eye_region = landmarks[36:42]  # Left eye region
        right_eye_region = landmarks[42:48]  # Right eye region
        nose_region = landmarks[27:36]  # Nose region
        
        # Calculate the mean points used for alignment
        left_eye_center = np.mean(left_eye_region, axis=0).astype(int)
        right_eye_center = np.mean(right_eye_region, axis=0).astype(int)
        nose_center = np.mean(nose_region, axis=0).astype(int)
        
        # # Draw the specific regions with different colors
        for point in left_eye_region:
            cv2.circle(vis_image, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1)  # Blue for left eye
        
        for point in right_eye_region:
            cv2.circle(vis_image, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)  # Red for right eye
        
        for point in nose_region:
            cv2.circle(vis_image, (int(point[0]), int(point[1])), 2, (0, 255, 255), -1)  # Yellow for nose
        
        # Draw the mean points (centers) used for alignment with larger circles
        cv2.circle(vis_image, tuple(left_eye_center), 5, (255, 0, 0), 2)  # Blue circle for left eye center
        cv2.circle(vis_image, tuple(right_eye_center), 5, (0, 0, 255), 2)  # Red circle for right eye center
        cv2.circle(vis_image, tuple(nose_center), 5, (0, 255, 255), 2)  # Yellow circle for nose center
        
        # # Draw lines connecting the alignment points to show the triangle
        cv2.line(vis_image, tuple(left_eye_center), tuple(right_eye_center), (255, 255, 0), 2)
        cv2.line(vis_image, tuple(left_eye_center), tuple(nose_center), (255, 255, 0), 2)
        cv2.line(vis_image, tuple(right_eye_center), tuple(nose_center), (255, 255, 0), 2)
        
        # Add text labels
        cv2.putText(vis_image, "Left Eye", (left_eye_center[0]-20, left_eye_center[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(vis_image, "Right Eye", (right_eye_center[0]-20, right_eye_center[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(vis_image, "Nose", (nose_center[0]-20, nose_center[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Show the image if requested
        if show:
            plt.figure(figsize=(12, 10))
            plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title('Face Alignment Landmarks')
            plt.show()
        
        # # Save the image if a path is provided
        # if save_path:
        #     cv2.imwrite(save_path, vis_image)
        return vis_image
    

import numpy as np

import cv2

def transformation_from_points(points1, points0, smooth=True, p_bias=None):
    points2 = np.array(points0, dtype=np.float64)
    points1 = points1.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = np.linalg.svd(np.matmul(points1.T, points2))
    R = (np.matmul(U, Vt)).T
    sR = (s2 / s1) * R
    T = c2.reshape(2, 1) - (s2 / s1) * np.matmul(R, c1.reshape(2, 1))
    M = np.concatenate((sR, T), axis=1)
    if smooth:
        bias = points2[2] - points1[2]
        if p_bias is None:
            p_bias = bias
        else:
            bias = p_bias * 0.2 + bias * 0.8
        p_bias = bias
        M[:, 2] = M[:, 2] + bias
    return M, p_bias

class AlignRestore(object):
    def __init__(self, align_points=3):
        if align_points == 3:
            self.upscale_factor = 1
            ratio = 2.8
            self.crop_ratio = (ratio, ratio)
            self.face_template = np.array([[19 - 2, 30 - 10], [56 + 2, 30 - 10], [37.5, 45 - 5]])
            self.face_template = self.face_template * ratio
            self.face_size = (int(75 * self.crop_ratio[0]), int(100 * self.crop_ratio[1]))
            # self.p_bias = None

    def process(self, img, p_bias, lmk_align=None, smooth=True, align_points=3):
        # Removed debug writes for efficiency
        aligned_face, affine_matrix, p_bias = self.align_warp_face(img, lmk_align, p_bias=p_bias, smooth=smooth)
        restored_img = self.restore_img(img, aligned_face, affine_matrix)
        return aligned_face, restored_img

    def align_warp_face(self, img, lmks3, p_bias, smooth=True, border_mode="constant"):
        affine_matrix, p_bias = transformation_from_points(lmks3, self.face_template, smooth, p_bias)
        border_mode = {"constant": cv2.BORDER_CONSTANT,
                       "reflect101": cv2.BORDER_REFLECT101,
                       "reflect": cv2.BORDER_REFLECT}.get(border_mode, cv2.BORDER_CONSTANT)
        cropped_face = cv2.warpAffine(
            img,
            affine_matrix,
            self.face_size,
            flags=cv2.INTER_LINEAR,  # faster interpolation
            borderMode=border_mode,
            borderValue=[127, 127, 127],
        )
        return cropped_face, affine_matrix, p_bias

    def align_warp_face2(self, img, landmark, border_mode="constant"):
        affine_matrix = cv2.estimateAffinePartial2D(landmark, self.face_template)[0]
        border_mode = {"constant": cv2.BORDER_CONSTANT,
                       "reflect101": cv2.BORDER_REFLECT101,
                       "reflect": cv2.BORDER_REFLECT}.get(border_mode, cv2.BORDER_CONSTANT)
        cropped_face = cv2.warpAffine(
            img, affine_matrix, self.face_size, borderMode=border_mode, borderValue=(135, 133, 132)
        )
        return cropped_face, affine_matrix

    def restore_img(self, input_img, face, affine_matrix):
        h, w, _ = input_img.shape
        h_up, w_up = int(h * self.upscale_factor), int(w * self.upscale_factor)
        upsample_img = cv2.resize(input_img, (w_up, h_up), interpolation=cv2.INTER_LINEAR)
        inverse_affine = cv2.invertAffineTransform(affine_matrix)
        inverse_affine *= self.upscale_factor
        extra_offset = 0.5 * self.upscale_factor if self.upscale_factor > 1 else 0
        inverse_affine[:, 2] += extra_offset
        inv_restored = cv2.warpAffine(face, inverse_affine, (w_up, h_up), flags=cv2.INTER_LINEAR)
        mask = np.ones((self.face_size[1], self.face_size[0]), dtype=np.float32)
        inv_mask = cv2.warpAffine(mask, inverse_affine, (w_up, h_up))
        inv_mask_erosion = cv2.erode(
            inv_mask, np.ones((max(1, int(2 * self.upscale_factor)), max(1, int(2 * self.upscale_factor))), np.uint8)
        )
        pasted_face = inv_mask_erosion[:, :, None] * inv_restored
        total_face_area = np.sum(inv_mask_erosion)
        w_edge = int(total_face_area**0.5) // 20
        erosion_radius = max(1, w_edge * 2)
        inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
        blur_size = max(1, w_edge * 2)
        inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
        inv_soft_mask = inv_soft_mask[:, :, None]
        upsample_img = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * upsample_img
        return upsample_img.astype(np.uint8)

class laplacianSmooth:
    def __init__(self, smoothAlpha=0.3):
        self.smoothAlpha = smoothAlpha
        # self.pts_last = None

    def smooth(self, pts_cur, pts_last=None):
        if pts_last is None:
            pts_last = pts_cur.copy()
            return pts_cur.copy(), pts_last
        x1 = np.min(pts_cur[:, 0])
        x2 = np.max(pts_cur[:, 0])
        width = x2 - x1
        pts_update = []
        for i in range(len(pts_cur)):
            x_new, y_new = pts_cur[i]
            x_old, y_old = pts_last[i]
            tmp = (x_new - x_old) ** 2 + (y_new - y_old) ** 2
            w = np.exp(-tmp / (width * self.smoothAlpha + 1e-6))
            x = x_old * w + x_new * (1 - w)
            y = y_old * w + y_new * (1 - w)
            pts_update.append([x, y])
        pts_update = np.array(pts_update)
        pts_last = pts_update.copy()
        return pts_update, pts_last

# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = FaceAlignmentProcessor(resolution=512)
    
    # Process a list of videos
    video_paths = [
        "path/to/video1.mp4",
        "path/to/video2.mp4"
    ]
    
    # Process with frame saving
    results = processor.process_videos(
        video_paths=video_paths,
        output_dir="aligned_faces",
        max_frames_per_video=100,
        frame_step=5,
        save_frames=True
    )
    
    # Clean up
    processor.close()