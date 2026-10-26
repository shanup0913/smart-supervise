import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class TargetMatcher:
    """
    TargetMatcher: High-accuracy person search & re-identification engine.
    
    Combines:
    1. Upper-body clothing color (LAB color space)
    2. Lower-body clothing color (LAB color space)
    3. Deep neural network visual appearance embedding (MobileNetV3)
    """
    def __init__(self, match_threshold=0.72):
        self.match_threshold = match_threshold
        self.device = torch.device("cpu")
        
        # Load lightweight MobileNetV3 feature extractor
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
        
        self.target_name = None
        self.target_deep_emb = None
        self.target_upper_color = None
        self.target_lower_color = None
        self.target_thumbnail = None

    def _extract_clothing_colors(self, image_bgr):
        """Extracts LAB color histograms for upper and lower clothing, ignoring blank background."""
        h, w = image_bgr.shape[:2]
        
        # Upper body: 15% to 55% of height
        upper = image_bgr[int(h * 0.15):int(h * 0.55), :]
        # Lower body: 55% to 95% of height
        lower = image_bgr[int(h * 0.55):int(h * 0.95), :]

        def get_clothing_hist(crop):
            if crop.size == 0:
                return np.zeros(256, dtype=np.float32)
            
            # Mask out near-white or extreme overexposed background
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            mask = (gray < 235).astype(np.uint8) * 255
            if cv2.countNonZero(mask) < 20:
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

    def set_target_photo(self, image_path_or_crop, target_name="Target Person"):
        """Stores the visual signature of the target person."""
        if isinstance(image_path_or_crop, str):
            image_bgr = cv2.imread(image_path_or_crop)
            if image_bgr is None:
                raise ValueError(f"Could not load image: {image_path_or_crop}")
        else:
            image_bgr = image_path_or_crop

        self.target_name = target_name
        self.target_thumbnail = cv2.resize(image_bgr, (96, 128))
        self.target_upper_color, self.target_lower_color = self._extract_clothing_colors(image_bgr)
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.target_deep_emb = self._extract_deep_features(Image.fromarray(image_rgb))
        return True

    def compare(self, candidate_bgr_crop):
        """
        Compares candidate against target.
        Returns: (is_match: bool, similarity_score: float)
        """
        if self.target_deep_emb is None or candidate_bgr_crop.size == 0:
            return False, 0.0

        if candidate_bgr_crop.shape[0] < 20 or candidate_bgr_crop.shape[1] < 10:
            return False, 0.0

        try:
            cand_upper, cand_lower = self._extract_clothing_colors(candidate_bgr_crop)
            sim_upper = float(np.dot(self.target_upper_color, cand_upper))
            sim_lower = float(np.dot(self.target_lower_color, cand_lower))

            cand_rgb = cv2.cvtColor(candidate_bgr_crop, cv2.COLOR_BGR2RGB)
            cand_deep = self._extract_deep_features(Image.fromarray(cand_rgb))
            sim_deep = float(np.dot(self.target_deep_emb, cand_deep))

            # Weighting: 40% upper clothing + 35% lower clothing + 25% deep appearance
            total_score = (0.40 * sim_upper) + (0.35 * sim_lower) + (0.25 * sim_deep)
            total_score = max(0.0, min(1.0, total_score))
            is_match = total_score >= self.match_threshold

            return is_match, round(total_score, 3)
        except Exception:
            return False, 0.0

    def clear_target(self):
        self.target_name = None
        self.target_deep_emb = None
        self.target_upper_color = None
        self.target_lower_color = None
        self.target_thumbnail = None
