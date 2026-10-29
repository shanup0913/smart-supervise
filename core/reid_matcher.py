import os
import cv2
import threading
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class TargetAngleProfile:
    """Stores biometric and visual features for a specific reference angle/photo."""
    def __init__(self, face_feat, face_box, upper_color, lower_color, deep_emb, thumbnail, angle_label="Front"):
        self.face_feature = face_feat
        self.face_box = face_box
        self.has_face = (face_feat is not None)
        self.upper_color = upper_color
        self.lower_color = lower_color
        self.deep_emb = deep_emb
        self.thumbnail = thumbnail
        self.angle_label = angle_label

class TargetMatcher:
    """
    TargetMatcher: High-Accuracy Multi-Angle Biometric Face & Clothing Re-ID Engine.
    Supports CPU multi-core parallel execution and automatic NVIDIA CUDA GPU acceleration.
    """
    def __init__(self, match_threshold=0.55):
        self.match_threshold = match_threshold
        # Auto-detect NVIDIA CUDA GPU; fall back to CPU seamlessly
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.RLock()
        
        # ── 1. Deep Feature Extractor (MobileNetV3) ──────────────────────────
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        mobilenet = models.mobilenet_v3_small(weights=weights)
        self.feature_extractor = mobilenet.features
        self.pool = mobilenet.avgpool
        self.feature_extractor.eval()
        self.feature_extractor.to(self.device)
        
        self.transform = transforms.Compose([
            transforms.Resize((128, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # ── 2. Face Detector (YuNet) & Recognizer (SFace) ───────────────────
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
        os.makedirs(models_dir, exist_ok=True)
        yunet_path = os.path.join(models_dir, "face_detection_yunet.onnx")
        sface_path = os.path.join(models_dir, "face_recognition_sface.onnx")
        
        import urllib.request
        if not os.path.exists(yunet_path):
            try:
                urllib.request.urlretrieve(
                    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
                    yunet_path
                )
            except Exception as e:
                print(f"[Matcher Warning] Could not download YuNet: {e}")
                
        if not os.path.exists(sface_path):
            try:
                urllib.request.urlretrieve(
                    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
                    sface_path
                )
            except Exception as e:
                print(f"[Matcher Warning] Could not download SFace: {e}")

        self.face_detector = None
        self.face_recognizer = None
        
        if os.path.exists(yunet_path) and os.path.exists(sface_path):
            try:
                self.face_detector = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320), 0.3, 0.3)
                self.face_recognizer = cv2.FaceRecognizerSF.create(sface_path, "")
            except Exception as e:
                print(f"[Matcher Warning] Could not load face models: {e}")

        # Secondary fallback: OpenCV built-in Haar Cascade
        self.haar_cascade = None
        haar_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if os.path.exists(haar_path):
            try:
                self.haar_cascade = cv2.CascadeClassifier(haar_path)
            except Exception:
                pass
        
        # ── Target State (Multi-Angle Gallery) ────────────────────────────────
        self.target_name = "Target Person"
        self.target_profiles = [] # list of TargetAngleProfile
        self.target_thumbnail = None

    def _enhance_lighting_if_dark(self, image_bgr):
        """Enhances contrast and illumination for dark, cinematic, or shadowy video crops."""
        if image_bgr is None or image_bgr.size == 0:
            return image_bgr
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        mean_lum = float(np.mean(gray))
        if mean_lum < 85.0: # dark room, night scene, or cinema lighting
            lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            enhanced_lab = cv2.merge([l_enhanced, a, b])
            return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return image_bgr

    def _detect_face_and_feature(self, image_bgr):
        """
        Multi-Stage Face Detection with lighting adaptation (Thread-Safe):
        1. YuNet at native scale.
        2. YuNet on head region (upper 55% of crop) at 2x upscale.
        3. Haar Cascade as robust secondary fallback.
        Returns: (face_feature, face_box) or (None, None)
        """
        if self.face_detector is None or self.face_recognizer is None or image_bgr is None:
            return None, None
            
        h, w = image_bgr.shape[:2]
        if h < 20 or w < 20:
            return None, None

        # Pre-enhance crop if lighting is dark
        enhanced = self._enhance_lighting_if_dark(image_bgr)
            
        with self._lock:
            # Stage 1: YuNet on native crop
            try:
                self.face_detector.setInputSize((w, h))
                _, faces = self.face_detector.detect(enhanced)
                if faces is not None and len(faces) > 0:
                    # Filter to upper 65% of person body with confidence >= 0.40 (prevents false stomach/shirt fold detections)
                    valid_faces = [f for f in faces if f[-1] >= 0.40 and f[1] < h * 0.65]
                    if valid_faces:
                        best_face = max(valid_faces, key=lambda f: f[-1])
                        aligned = self.face_recognizer.alignCrop(enhanced, best_face)
                        feat = self.face_recognizer.feature(aligned)
                        box = (int(best_face[0]), int(best_face[1]), int(best_face[2]), int(best_face[3]))
                        return feat, box
            except Exception:
                pass

            # Stage 2: YuNet on head region (upper 55%) at 2x upscale
            try:
                head_h = int(h * 0.55)
                if head_h >= 20:
                    head_crop = enhanced[:head_h, :]
                    scaled_head = cv2.resize(head_crop, (w * 2, head_h * 2))
                    self.face_detector.setInputSize((w * 2, head_h * 2))
                    _, faces = self.face_detector.detect(scaled_head)
                    if faces is not None and len(faces) > 0:
                        valid_faces = [f for f in faces if f[-1] >= 0.40]
                        if valid_faces:
                            best_face = max(valid_faces, key=lambda f: f[-1])
                            aligned = self.face_recognizer.alignCrop(scaled_head, best_face)
                            feat = self.face_recognizer.feature(aligned)
                            box = (int(best_face[0]/2), int(best_face[1]/2), int(best_face[2]/2), int(best_face[3]/2))
                            return feat, box
            except Exception:
                pass

            # Stage 3: Haar Cascade fallback
            if self.haar_cascade is not None:
                try:
                    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
                    haar_faces = self.haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(24, 24))
                    if len(haar_faces) > 0:
                        hx, hy, hw, hh = max(haar_faces, key=lambda b: b[2] * b[3])
                        f_arr = np.array([
                            hx, hy, hw, hh,
                            hx + hw * 0.3, hy + hh * 0.35,
                            hx + hw * 0.7, hy + hh * 0.35,
                            hx + hw * 0.5, hy + hh * 0.55,
                            hx + hw * 0.35, hy + hh * 0.75,
                            hx + hw * 0.65, hy + hh * 0.75,
                            0.90
                        ], dtype=np.float32)
                        aligned = self.face_recognizer.alignCrop(enhanced, f_arr)
                        feat = self.face_recognizer.feature(aligned)
                        return feat, (hx, hy, hw, hh)
                except Exception:
                    pass

        return None, None

    def _extract_clothing_colors(self, image_bgr, face_box=None):
        """Extracts LAB color histograms for clothing below the chin."""
        h, w = image_bgr.shape[:2]
        
        if face_box is not None:
            fx, fy, fw, fh = face_box
            chin_y = min(h, fy + fh)
            torso_crop = image_bgr[chin_y:h, :]
            if torso_crop.shape[0] > 10:
                th = torso_crop.shape[0]
                upper = torso_crop[:int(th * 0.55), :]
                lower = torso_crop[int(th * 0.55):, :]
            else:
                upper = image_bgr[int(h * 0.25):int(h * 0.65), :]
                lower = image_bgr[int(h * 0.65):, :]
        else:
            upper = image_bgr[int(h * 0.15):int(h * 0.55), :]
            lower = image_bgr[int(h * 0.55):int(h * 0.95), :]

        def get_clothing_hist(crop):
            if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
                return np.zeros(256, dtype=np.float32)
            
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            mask = (gray < 235).astype(np.uint8) * 255
            if cv2.countNonZero(mask) < 15:
                mask = None

            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            hist = cv2.calcHist([lab], [1, 2], mask, [16, 16], [0, 256, 0, 256])
            # Gaussian blur the 16x16 histogram grid so color variations from shadows/lighting overlap smoothly
            hist = cv2.GaussianBlur(hist, (5, 5), 1.0)
            norm = np.linalg.norm(hist)
            return (hist / norm).flatten() if norm > 0 else np.zeros(256, dtype=np.float32)

        return get_clothing_hist(upper), get_clothing_hist(lower)

    def _extract_deep_features(self, image_pil):
        """Extracts 576-dimensional deep feature vector (Thread-Safe)."""
        with self._lock:
            tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.feature_extractor(tensor)
                pooled = self.pool(features)
                emb = torch.flatten(pooled, 1).cpu().numpy()[0]
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb

    def add_target_photo(self, image_path_or_crop, angle_label=None):
        """
        Adds a reference photo/angle to the target's multi-angle gallery.
        Example angle_labels: 'Front', 'Side Profile', 'Back / Walking Away'.
        """
        if isinstance(image_path_or_crop, str):
            image_bgr = cv2.imread(image_path_or_crop)
            if image_bgr is None:
                raise ValueError(f"Could not load image: {image_path_or_crop}")
        else:
            image_bgr = image_path_or_crop

        if angle_label is None:
            idx = len(self.target_profiles) + 1
            if idx == 1:
                angle_label = "Front View"
            elif idx == 2:
                angle_label = "Side Profile"
            elif idx == 3:
                angle_label = "Back / Walking"
            else:
                angle_label = f"Angle {idx}"

        # 1. Biometric Face Analysis
        face_feat, face_box = self._detect_face_and_feature(image_bgr)
        
        # 2. Clothing Color Signatures
        upper_color, lower_color = self._extract_clothing_colors(image_bgr, face_box=face_box)
        
        # 3. Deep Appearance Representation
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        deep_emb = self._extract_deep_features(Image.fromarray(image_rgb))
        
        # 4. Thumbnail
        thumbnail = cv2.resize(image_bgr, (96, 128))

        profile = TargetAngleProfile(
            face_feat=face_feat,
            face_box=face_box,
            upper_color=upper_color,
            lower_color=lower_color,
            deep_emb=deep_emb,
            thumbnail=thumbnail,
            angle_label=angle_label
        )
        with self._lock:
            if len(self.target_profiles) == 0:
                self.target_thumbnail = thumbnail
            self.target_profiles.append(profile)
        return True

    def set_target_photo(self, image_path_or_crop, target_name="Target Person", angle_label="Front View"):
        """Clears gallery and sets the primary reference photo."""
        self.clear_target()
        self.target_name = target_name
        return self.add_target_photo(image_path_or_crop, angle_label=angle_label)

    def compare(self, candidate_bgr_crop):
        """
        Compares candidate against all stored reference angles in the gallery.
        Returns: (is_match: bool, best_similarity_score: float)
        """
        with self._lock:
            if len(self.target_profiles) == 0 or candidate_bgr_crop.size == 0:
                return False, 0.0
            profiles_snapshot = list(self.target_profiles)

        if candidate_bgr_crop.shape[0] < 20 or candidate_bgr_crop.shape[1] < 10:
            return False, 0.0

        try:
            # Pre-enhance candidate if lighting is dark / cinema style
            enhanced_cand = self._enhance_lighting_if_dark(candidate_bgr_crop)

            # 1. Candidate Face Biometrics
            cand_face_feat, cand_face_box = self._detect_face_and_feature(enhanced_cand)
            
            # 2. Candidate Clothing Color
            cand_upper, cand_lower = self._extract_clothing_colors(enhanced_cand, face_box=cand_face_box)

            # 3. Candidate Deep Features
            cand_rgb = cv2.cvtColor(enhanced_cand, cv2.COLOR_BGR2RGB)
            cand_deep = self._extract_deep_features(Image.fromarray(cand_rgb))

            # Evaluate candidate against EVERY reference angle profile
            scores = []
            for profile in profiles_snapshot:
                sim_upper = max(0.0, float(np.dot(profile.upper_color, cand_upper)))
                sim_lower = max(0.0, float(np.dot(profile.lower_color, cand_lower)))
                color_score = (0.55 * sim_upper) + (0.45 * sim_lower)

                raw_deep_sim = float(np.dot(profile.deep_emb, cand_deep))
                sim_deep = max(0.0, min(1.0, (raw_deep_sim - 0.25) / 0.60))

                if profile.has_face and self.face_recognizer is not None and cand_face_feat is not None:
                    # Both faces visible: Biometric SFace comparison (Thread-safe)
                    with self._lock:
                        raw_face_cos = self.face_recognizer.match(
                            profile.face_feature, cand_face_feat, cv2.FaceRecognizerSF_FR_COSINE
                        )
                    if raw_face_cos < 0.363:
                        # Confirmed DIFFERENT person -> capped <= 15%
                        score = max(0.0, (raw_face_cos / 0.363) * 0.15) if raw_face_cos > 0 else 0.02
                    else:
                        # Confirmed SAME person -> 70% to 98%
                        face_calibrated = 0.70 + ((raw_face_cos - 0.363) / (1.0 - 0.363)) * 0.30
                        score = (0.75 * face_calibrated) + (0.15 * color_score) + (0.10 * sim_deep)
                else:
                    # Face turned away / sunglasses / walking backwards -> evaluate clothing + appearance against this angle
                    score = (0.55 * color_score) + (0.45 * sim_deep)

                scores.append(max(0.0, min(1.0, score)))

            best_score = max(scores) if scores else 0.0
            is_match = best_score >= self.match_threshold

            return is_match, round(best_score, 3)
        except Exception:
            return False, 0.0

    def clear_target(self):
        """Clears all stored target profiles (Thread-Safe)."""
        with self._lock:
            self.target_name = "Target Person"
            self.target_profiles.clear()
            self.target_thumbnail = None
