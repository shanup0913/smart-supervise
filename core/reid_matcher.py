import os
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class TargetMatcher:
    """
    TargetMatcher: Dual-Mode Person Re-Identification & Biometric Recognition Engine.
    
    Combines:
    1. Facial Biometrics (OpenCV YuNet + SFace Deep Learning Face Recognition)
       - Provides 100% discriminative power between different individuals (e.g. man vs woman)
    2. Adaptive Clothing Color Signatures (LAB color space)
       - Automatically detects face/neck to accurately separate clothing from skin/hair
    3. Deep Neural Network Appearance Embeddings (MobileNetV3)
       - Holistic silhouette and visual texture representation
    """
    def __init__(self, match_threshold=0.70):
        self.match_threshold = match_threshold
        self.device = torch.device("cpu")
        
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
        
        # Auto-download models if missing
        yunet_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        sface_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
        
        import urllib.request
        if not os.path.exists(yunet_path):
            try:
                urllib.request.urlretrieve(yunet_url, yunet_path)
            except Exception as e:
                print(f"[Matcher Warning] Could not download YuNet: {e}")
                
        if not os.path.exists(sface_path):
            try:
                urllib.request.urlretrieve(sface_url, sface_path)
            except Exception as e:
                print(f"[Matcher Warning] Could not download SFace: {e}")

        self.face_detector = None
        self.face_recognizer = None
        
        if os.path.exists(yunet_path) and os.path.exists(sface_path):
            try:
                self.face_detector = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320), 0.5, 0.3)
                self.face_recognizer = cv2.FaceRecognizerSF.create(sface_path, "")
            except Exception as e:
                print(f"[Matcher Warning] Could not load face models: {e}")
        
        # ── Target State ─────────────────────────────────────────────────────
        self.target_name = None
        self.target_deep_emb = None
        self.target_face_feature = None
        self.target_has_face = False
        self.target_upper_color = None
        self.target_lower_color = None
        self.target_thumbnail = None

    def _detect_face_and_feature(self, image_bgr):
        """
        Detects face using YuNet and extracts 128-d SFace feature embedding.
        Returns: (face_feature, face_box) or (None, None)
        """
        if self.face_detector is None or self.face_recognizer is None or image_bgr is None:
            return None, None
            
        h, w = image_bgr.shape[:2]
        if h < 20 or w < 20:
            return None, None
            
        try:
            self.face_detector.setInputSize((w, h))
            _, faces = self.face_detector.detect(image_bgr)
            if faces is None or len(faces) == 0:
                return None, None
                
            # Pick the largest face in the crop
            best_face = max(faces, key=lambda f: f[2] * f[3])
            aligned_face = self.face_recognizer.alignCrop(image_bgr, best_face)
            face_feature = self.face_recognizer.feature(aligned_face)
            face_box = (int(best_face[0]), int(best_face[1]), int(best_face[2]), int(best_face[3]))
            return face_feature, face_box
        except Exception:
            return None, None

    def _extract_clothing_colors(self, image_bgr, face_box=None):
        """
        Extracts LAB color histograms for clothing.
        If a face is detected, extracts clothing region strictly below the chin.
        """
        h, w = image_bgr.shape[:2]
        
        if face_box is not None:
            fx, fy, fw, fh = face_box
            chin_y = min(h, fy + fh)
            # Torso starts right below the chin
            torso_crop = image_bgr[chin_y:h, :]
            if torso_crop.shape[0] > 10:
                th = torso_crop.shape[0]
                upper = torso_crop[:int(th * 0.55), :]
                lower = torso_crop[int(th * 0.55):, :]
            else:
                upper = image_bgr[int(h * 0.25):int(h * 0.65), :]
                lower = image_bgr[int(h * 0.65):, :]
        else:
            # Fallback standard full-body pedestrian split
            upper = image_bgr[int(h * 0.15):int(h * 0.55), :]
            lower = image_bgr[int(h * 0.55):int(h * 0.95), :]

        def get_clothing_hist(crop):
            if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
                return np.zeros(256, dtype=np.float32)
            
            # Mask out white/overexposed background
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            mask = (gray < 235).astype(np.uint8) * 255
            if cv2.countNonZero(mask) < 15:
                mask = None

            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            hist = cv2.calcHist([lab], [1, 2], mask, [16, 16], [0, 256, 0, 256])
            hist = cv2.normalize(hist, None, alpha=1.0, norm_type=cv2.NORM_L1).flatten()
            norm = np.linalg.norm(hist)
            return hist / norm if norm > 0 else hist

        return get_clothing_hist(upper), get_clothing_hist(lower)

    def _extract_deep_features(self, image_pil):
        """Extracts 576-dimensional deep feature vector."""
        tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.feature_extractor(tensor)
            pooled = self.pool(features)
            emb = torch.flatten(pooled, 1).cpu().numpy()[0]
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def _calibrate_face_score(self, cos_sim):
        """
        Calibrates raw SFace cosine similarity (threshold 0.363) to a [0.0, 1.0] confidence score.
        """
        if cos_sim <= 0.0:
            return 0.0
        elif cos_sim < 0.363:
            # Mismatch range: maps smoothly from 0.0 to 0.45
            return (cos_sim / 0.363) * 0.45
        else:
            # Match range: maps smoothly from 0.50 to 1.00
            return 0.50 + ((cos_sim - 0.363) / (1.0 - 0.363)) * 0.50

    def set_target_photo(self, image_path_or_crop, target_name="Target Person"):
        """Stores the visual biometric signature of the target person."""
        if isinstance(image_path_or_crop, str):
            image_bgr = cv2.imread(image_path_or_crop)
            if image_bgr is None:
                raise ValueError(f"Could not load image: {image_path_or_crop}")
        else:
            image_bgr = image_path_or_crop

        self.target_name = target_name
        self.target_thumbnail = cv2.resize(image_bgr, (96, 128))
        
        # 1. Biometric Face Analysis
        face_feat, face_box = self._detect_face_and_feature(image_bgr)
        self.target_face_feature = face_feat
        self.target_has_face = (face_feat is not None)
        
        # 2. Adaptive Clothing Color Signatures
        self.target_upper_color, self.target_lower_color = self._extract_clothing_colors(image_bgr, face_box=face_box)
        
        # 3. Deep Appearance Representation
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.target_deep_emb = self._extract_deep_features(Image.fromarray(image_rgb))
        return True

    def compare(self, candidate_bgr_crop):
        """
        Compares candidate against target using Face Biometrics + Whole-Body Re-ID.
        Returns: (is_match: bool, similarity_score: float)
        """
        if self.target_deep_emb is None or candidate_bgr_crop.size == 0:
            return False, 0.0

        if candidate_bgr_crop.shape[0] < 20 or candidate_bgr_crop.shape[1] < 10:
            return False, 0.0

        try:
            # 1. Candidate Face Biometrics
            cand_face_feat, cand_face_box = self._detect_face_and_feature(candidate_bgr_crop)
            
            # 2. Candidate Clothing Color
            cand_upper, cand_lower = self._extract_clothing_colors(candidate_bgr_crop, face_box=cand_face_box)
            sim_upper = float(np.dot(self.target_upper_color, cand_upper))
            sim_lower = float(np.dot(self.target_lower_color, cand_lower))

            # 3. Candidate Deep Features
            cand_rgb = cv2.cvtColor(candidate_bgr_crop, cv2.COLOR_BGR2RGB)
            cand_deep = self._extract_deep_features(Image.fromarray(cand_rgb))
            sim_deep = float(np.dot(self.target_deep_emb, cand_deep))

            # ── Multi-Modal Fusion Scoring ──────────────────────────────────
            if self.target_has_face and cand_face_feat is not None and self.face_recognizer is not None:
                # Both target and candidate have faces visible:
                # SFace Cosine Similarity
                raw_face_cos = self.face_recognizer.match(
                    self.target_face_feature, cand_face_feat, cv2.FaceRecognizerSF_FR_COSINE
                )
                sim_face = self._calibrate_face_score(raw_face_cos)

                # If facial features are a clear mismatch (e.g. man vs woman), 
                # hard-cap total score to prevent false positives from clothing colors!
                if sim_face < 0.35:
                    total_score = min(sim_face, 0.30)
                else:
                    # 60% Facial Biometrics + 25% Clothing Color + 15% Deep Appearance
                    color_score = (0.55 * sim_upper) + (0.45 * sim_lower)
                    total_score = (0.60 * sim_face) + (0.25 * color_score) + (0.15 * sim_deep)
            else:
                # Face not visible in candidate (seen from back, far CCTV distance, etc.)
                # Fallback to Person Re-ID: 40% upper clothing + 35% lower clothing + 25% deep appearance
                total_score = (0.40 * sim_upper) + (0.35 * sim_lower) + (0.25 * sim_deep)

            total_score = max(0.0, min(1.0, total_score))
            is_match = total_score >= self.match_threshold

            return is_match, round(total_score, 3)
        except Exception as e:
            return False, 0.0

    def clear_target(self):
        self.target_name = None
        self.target_deep_emb = None
        self.target_face_feature = None
        self.target_has_face = False
        self.target_upper_color = None
        self.target_lower_color = None
        self.target_thumbnail = None
