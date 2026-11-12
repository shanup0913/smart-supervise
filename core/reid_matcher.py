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
    def __init__(
        self,
        face_feat,
        face_box,
        upper_color,
        lower_color,
        deep_emb,
        thumbnail,
        angle_label="Front",
        deep_emb_flip=None,
        is_upper_only=False,
    ):
        self.face_feature = face_feat
        self.face_box = face_box
        self.has_face = (face_feat is not None)
        self.upper_color = upper_color
        self.lower_color = lower_color
        self.deep_emb = deep_emb
        self.deep_emb_flip = deep_emb_flip
        self.thumbnail = thumbnail
        self.angle_label = angle_label
        self.is_upper_only = is_upper_only

class TargetMatcher:
    """
    TargetMatcher: High-Accuracy Multi-Angle Biometric Face & Clothing Re-ID Engine.
    Supports CPU multi-core parallel execution and automatic NVIDIA CUDA GPU acceleration.
    """
    def __init__(self, match_threshold=0.68):
        self.match_threshold = match_threshold
        # Auto-detect NVIDIA CUDA GPU; fall back to CPU seamlessly
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.RLock()
        self._face_lock = threading.RLock()
        self._deep_lock = threading.RLock()
        
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

    def _detect_face_and_feature(self, image_bgr, pre_enhanced=False, fast_mode=False):
        """
        Multi-Stage Face Detection with lighting adaptation (Thread-Safe):
        1. YuNet at native scale.
        2. YuNet on head region (upper 55% of crop) at 2x upscale.
        3. Haar Cascade as robust secondary fallback (skipped in fast_mode).
        Returns: (face_feature, face_box) or (None, None)
        """
        if self.face_detector is None or self.face_recognizer is None or image_bgr is None:
            return None, None
            
        h, w = image_bgr.shape[:2]
        if h < 20 or w < 20:
            return None, None

        # Pre-enhance crop if lighting is dark (avoid running twice if caller already enhanced)
        enhanced = image_bgr if pre_enhanced else self._enhance_lighting_if_dark(image_bgr)
            
        with self._face_lock:
            # Stage 1: YuNet on native crop
            try:
                self.face_detector.setInputSize((w, h))
                _, faces = self.face_detector.detect(enhanced)
                if faces is not None and len(faces) > 0:
                    # Filter to upper 60% of person body, realistic min face size >= 12px, and confidence >= 0.46
                    valid_faces = [
                        f for f in faces
                        if f[-1] >= 0.46 and f[1] < h * 0.60 and f[2] >= 12 and f[3] >= 12
                    ]
                    if valid_faces:
                        best_face = max(valid_faces, key=lambda f: f[-1])
                        aligned = self.face_recognizer.alignCrop(enhanced, best_face)
                        feat = self.face_recognizer.feature(aligned)
                        box = (int(best_face[0]), int(best_face[1]), int(best_face[2]), int(best_face[3]))
                        return feat, box
            except Exception:
                pass

            # Stage 2: YuNet on head region (upper 55%) at 2x upscale for medium/distant CCTV crops
            try:
                head_h = int(h * 0.55)
                if head_h >= 20 and w < 260:
                    head_crop = enhanced[:head_h, :]
                    scaled_head = cv2.resize(head_crop, (w * 2, head_h * 2))
                    self.face_detector.setInputSize((w * 2, head_h * 2))
                    _, faces = self.face_detector.detect(scaled_head)
                    if faces is not None and len(faces) > 0:
                        valid_faces = [f for f in faces if f[-1] >= 0.45 and f[2] >= 12 and f[3] >= 12]
                        if valid_faces:
                            best_face = max(valid_faces, key=lambda f: f[-1])
                            aligned = self.face_recognizer.alignCrop(scaled_head, best_face)
                            feat = self.face_recognizer.feature(aligned)
                            box = (int(best_face[0]/2), int(best_face[1]/2), int(best_face[2]/2), int(best_face[3]/2))
                            return feat, box
            except Exception:
                pass

            # Stage 3: Haar Cascade fallback restricted to upper head region (skipped in fast_mode)
            if not fast_mode and self.haar_cascade is not None:
                try:
                    head_h = max(24, int(h * 0.55))
                    head_gray = cv2.cvtColor(enhanced[:head_h, :], cv2.COLOR_BGR2GRAY)
                    haar_faces = self.haar_cascade.detectMultiScale(
                        head_gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24)
                    )
                    if len(haar_faces) > 0:
                        hx, hy, hw, hh = [int(v) for v in max(haar_faces, key=lambda b: b[2] * b[3])]
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
        """Extracts LAB chroma-weighted color signatures from the center body corridor."""
        h, w = image_bgr.shape[:2]
        # Focus on center 60% width when crop is wide enough to reject edge background wall/floor pixels
        if w >= 26:
            cx0, cx1 = int(w * 0.20), int(w * 0.80)
            body = image_bgr[:, cx0:cx1]
        else:
            body = image_bgr

        if face_box is not None:
            fx, fy, fw, fh = face_box
            chin_y = min(h, fy + fh)
            torso_crop = body[chin_y:h, :]
            if torso_crop.shape[0] > 12:
                th = torso_crop.shape[0]
                upper = torso_crop[:int(th * 0.52), :]
                lower = torso_crop[int(th * 0.52):max(int(th * 0.52) + 2, int(th * 0.92)), :]
            else:
                upper = body[int(h * 0.22):int(h * 0.56), :]
                lower = body[int(h * 0.56):max(int(h * 0.56) + 2, int(h * 0.92)), :]
        else:
            upper = body[int(h * 0.18):int(h * 0.52), :]
            lower = body[int(h * 0.54):max(int(h * 0.54) + 2, int(h * 0.92)), :]

        def get_clothing_hist(crop):
            if crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 6:
                return np.zeros(48, dtype=np.float32)

            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            hl = cv2.calcHist([lab], [0], None, [16], [0, 256])
            ha = cv2.calcHist([lab], [1], None, [16], [0, 256])
            hb = cv2.calcHist([lab], [2], None, [16], [0, 256])

            hl = cv2.GaussianBlur(hl, (1, 3), 0.8)
            ha = cv2.GaussianBlur(ha, (1, 3), 0.8)
            hb = cv2.GaussianBlur(hb, (1, 3), 0.8)

            # Emphasize illumination-invariant color chroma (A/B) over raw brightness (L)
            hl = (hl / (np.linalg.norm(hl) + 1e-6)) * 0.55
            ha = (ha / (np.linalg.norm(ha) + 1e-6)) * 1.0
            hb = (hb / (np.linalg.norm(hb) + 1e-6)) * 1.0

            sig = np.concatenate([hl.flatten(), ha.flatten(), hb.flatten()])
            norm = np.linalg.norm(sig)
            return (sig / norm).astype(np.float32) if norm > 0 else np.zeros(48, dtype=np.float32)

        return get_clothing_hist(upper), get_clothing_hist(lower)

    @staticmethod
    def _color_hist_similarity(h_ref, h_cand):
        """Chroma-aware clothing similarity blending full LAB and chroma-only (A/B) agreement."""
        full_cos = max(0.0, float(np.dot(h_ref, h_cand)))
        ab_ref = h_ref[16:]
        ab_cand = h_cand[16:]
        ab_denom = (float(np.linalg.norm(ab_ref)) * float(np.linalg.norm(ab_cand))) + 1e-6
        ab_cos = max(0.0, float(np.dot(ab_ref, ab_cand)) / ab_denom)
        return (0.55 * full_cos) + (0.45 * ab_cos)

    def _extract_deep_features(self, image_pil):
        """Extracts 576-dimensional deep feature vector (Thread-Safe)."""
        with self._deep_lock:
            tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.feature_extractor(tensor)
                pooled = self.pool(features)
                emb = torch.flatten(pooled, 1).cpu().numpy()[0]
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb

    @staticmethod
    def _load_image_bgr(image_path):
        """Loads an image from disk safely on Windows, including Unicode file paths."""
        if not image_path or not os.path.isfile(image_path):
            return None
        try:
            data = np.fromfile(image_path, dtype=np.uint8)
            if data.size > 0:
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if img is not None:
                    return img
        except Exception:
            pass
        return cv2.imread(image_path)

    def add_target_photo(self, image_path_or_crop, angle_label=None):
        """
        Adds a reference photo/angle to the target's multi-angle gallery.
        Example angle_labels: 'Front', 'Side Profile', 'Back / Walking Away'.
        """
        if isinstance(image_path_or_crop, str):
            image_bgr = self._load_image_bgr(image_path_or_crop)
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

        # Normalize lighting gently on reference photo before feature extraction
        work_ref = self._enhance_lighting_if_dark(image_bgr)

        # 1. Biometric Face Analysis
        face_feat, face_box = self._detect_face_and_feature(work_ref, pre_enhanced=True)

        # Detect close-up / upper-body portraits where bottom half is not pants/legs
        ref_h = work_ref.shape[0]
        is_upper_only = bool(
            face_box is not None and (
                (face_box[1] + face_box[3]) >= ref_h * 0.45 or face_box[3] >= ref_h * 0.24
            )
        )

        # 2. Clothing Color Signatures
        upper_color, lower_color = self._extract_clothing_colors(work_ref, face_box=face_box)

        # 3. Deep Appearance Representation (both native and horizontally mirrored for walk-direction invariance)
        image_rgb = cv2.cvtColor(work_ref, cv2.COLOR_BGR2RGB)
        deep_emb = self._extract_deep_features(Image.fromarray(image_rgb))
        deep_emb_flip = self._extract_deep_features(Image.fromarray(cv2.flip(image_rgb, 1)))

        # 4. Thumbnail
        thumbnail = cv2.resize(image_bgr, (96, 128))

        profile = TargetAngleProfile(
            face_feat=face_feat,
            face_box=face_box,
            upper_color=upper_color,
            lower_color=lower_color,
            deep_emb=deep_emb,
            thumbnail=thumbnail,
            angle_label=angle_label,
            deep_emb_flip=deep_emb_flip,
            is_upper_only=is_upper_only,
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

    def compare(self, candidate_bgr_crop, fast_mode=False):
        """
        Compares candidate against all stored reference angles in the gallery.
        Uses close-up downscale cap and lazy MobileNetV3 deep embedding so clear mismatches
        short-circuit without burning neural network CPU cycles.
        Returns: (is_match: bool, best_similarity_score: float)
        """
        with self._lock:
            if len(self.target_profiles) == 0 or candidate_bgr_crop.size == 0:
                return False, 0.0
            profiles_snapshot = list(self.target_profiles)

        # Stricter candidate size check: ignore tiny limb/hand fragments
        ch, cw = candidate_bgr_crop.shape[:2]
        if ch < 50 or cw < 20:
            return False, 0.0

        try:
            # Cap appearance crop to 320px tall for fast MobileNet/LAB features, while keeping up to 600px head detail for YuNet/SFace
            if ch > 320:
                scale = 320.0 / ch
                work_crop = cv2.resize(
                    candidate_bgr_crop,
                    (max(20, int(cw * scale)), 320),
                    interpolation=cv2.INTER_LINEAR
                )
            else:
                work_crop = candidate_bgr_crop

            enhanced_cand = self._enhance_lighting_if_dark(work_crop)

            # 1. Candidate Face Biometrics on higher-res head crop when available
            if ch > 320:
                if ch > 600:
                    f_scale = 600.0 / ch
                    face_body = cv2.resize(
                        candidate_bgr_crop,
                        (max(20, int(cw * f_scale)), 600),
                        interpolation=cv2.INTER_LINEAR,
                    )
                else:
                    face_body = candidate_bgr_crop
                head_slice = self._enhance_lighting_if_dark(face_body[:max(24, int(face_body.shape[0] * 0.62)), :])
                cand_face_feat, raw_face_box = self._detect_face_and_feature(
                    head_slice, pre_enhanced=True, fast_mode=fast_mode
                )
                if raw_face_box is not None:
                    rx, ry, rw, rh = raw_face_box
                    cand_face_min_dim = min(rw, rh)
                    sx = work_crop.shape[1] / max(1, face_body.shape[1])
                    sy = work_crop.shape[0] / max(1, face_body.shape[0])
                    cand_face_box = (int(rx * sx), int(ry * sy), max(1, int(rw * sx)), max(1, int(rh * sy)))
                else:
                    cand_face_box = None
                    cand_face_min_dim = 0
            else:
                cand_face_feat, cand_face_box = self._detect_face_and_feature(
                    enhanced_cand, pre_enhanced=True, fast_mode=fast_mode
                )
                cand_face_min_dim = min(cand_face_box[2], cand_face_box[3]) if cand_face_box else 0

            # 2. Candidate Clothing Color (~0.3ms fast signature)
            cand_upper, cand_lower = self._extract_clothing_colors(enhanced_cand, face_box=cand_face_box)

            # 3. Lazy Deep Features cache (only computed if actually needed by matching branch)
            cand_deep = None

            def _get_deep_sim(profile):
                nonlocal cand_deep
                if cand_deep is None:
                    cand_rgb = cv2.cvtColor(enhanced_cand, cv2.COLOR_BGR2RGB)
                    cand_deep = self._extract_deep_features(Image.fromarray(cand_rgb))
                raw_deep_sim = float(np.dot(profile.deep_emb, cand_deep))
                flip_emb = getattr(profile, "deep_emb_flip", None)
                if flip_emb is not None:
                    raw_deep_sim = max(raw_deep_sim, float(np.dot(flip_emb, cand_deep)))
                if raw_deep_sim <= 0.50:
                    return max(0.0, (raw_deep_sim / 0.50) * 0.18)
                if raw_deep_sim <= 0.72:
                    return 0.18 + ((raw_deep_sim - 0.50) / 0.22) * 0.48
                return min(1.0, 0.66 + ((raw_deep_sim - 0.72) / 0.22) * 0.32)

            # Evaluate candidate against EVERY reference angle profile
            scores = []
            for profile in profiles_snapshot:
                sim_upper = self._color_hist_similarity(profile.upper_color, cand_upper)
                sim_lower = self._color_hist_similarity(profile.lower_color, cand_lower)

                # Dual-zone conjunction: if either shirt OR pants is wrong color, apply quadratic penalty
                min_color_sim = min(sim_upper, sim_lower)
                avg_color_sim = (0.55 * sim_upper) + (0.45 * sim_lower)
                if min_color_sim < 0.42:
                    penalty = max(0.0, min_color_sim / 0.42)
                    color_score = avg_color_sim * (penalty ** 2.0)
                else:
                    color_score = avg_color_sim

                if getattr(profile, "is_upper_only", False):
                    eff_color = sim_upper
                    eff_min_color = sim_upper
                else:
                    eff_color = color_score
                    eff_min_color = min_color_sim

                if profile.has_face and self.face_recognizer is not None and cand_face_feat is not None:
                    # Both faces visible: Biometric SFace comparison (Thread-safe)
                    with self._face_lock:
                        raw_face_cos = self.face_recognizer.match(
                            profile.face_feature, cand_face_feat, cv2.FaceRecognizerSF_FR_COSINE
                        )
                    if raw_face_cos >= 0.363:
                        # Confirmed SAME person -> 74% to 99%
                        face_calibrated = 0.74 + ((raw_face_cos - 0.363) / (1.0 - 0.363)) * 0.25
                        sim_deep = _get_deep_sim(profile)
                        score = (0.72 * face_calibrated) + (0.16 * eff_color) + (0.12 * sim_deep)
                    elif raw_face_cos >= 0.20:
                        # Borderline CCTV / 3/4 side angle: fuse biometric hint with clothing & deep Re-ID
                        if eff_color >= 0.56 and eff_min_color >= 0.40:
                            sim_deep = _get_deep_sim(profile)
                            face_mid = 0.42 + ((raw_face_cos - 0.20) / 0.163) * 0.32
                            score = min(0.78, (0.38 * face_mid) + (0.34 * eff_color) + (0.28 * sim_deep))
                        else:
                            score = max(0.0, min(0.32, eff_color * 0.45))
                    elif cand_face_min_dim < 44 and raw_face_cos >= 0.08 and eff_color >= 0.72 and eff_min_color >= 0.50:
                        # Small/tilted blurry CCTV face: verify via strong clothing + MobileNet appearance
                        sim_deep = _get_deep_sim(profile)
                        if sim_deep >= 0.48:
                            score = min(0.75, ((0.54 * eff_color) + (0.46 * sim_deep)) * 0.94)
                        else:
                            score = max(0.0, min(0.28, eff_color * 0.35))
                    else:
                        # Confirmed DIFFERENT person -> capped <= 14% (MISMATCH, skip MobileNet!)
                        score = max(0.0, (raw_face_cos / 0.20) * 0.14) if raw_face_cos > 0 else 0.02

                elif profile.has_face and cand_face_feat is None:
                    # Reference profile has face, but candidate has NO visible face (turned away / walking back)
                    if eff_color >= 0.58 and eff_min_color >= 0.42:
                        sim_deep = _get_deep_sim(profile)
                        combined = (0.54 * eff_color) + (0.46 * sim_deep)
                        if sim_deep >= 0.40:
                            score = min(0.84, combined)
                        else:
                            score = max(0.0, min(0.45, combined))
                    else:
                        # Clearly mismatched clothing without visible face -> skip MobileNet!
                        score = max(0.0, min(0.35, eff_color * 0.50))

                else:
                    # Reference profile itself has no face (e.g. explicitly uploaded Back/Walking profile)
                    sim_deep = _get_deep_sim(profile)
                    score = (0.50 * eff_color) + (0.50 * sim_deep)

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
