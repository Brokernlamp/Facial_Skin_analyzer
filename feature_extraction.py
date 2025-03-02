import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

class FeatureExtractor:
    def __init__(self):
        """Initialize feature extractor for skin analysis"""
        pass
    
    def extract_features(self, image):
        """
        Extract features from a skin image for classification
        
        Args:
            image: BGR image
            
        Returns:
            list: Features vector
        """
        if image is None or image.size == 0:
            return None
        
        # Resize for consistency
        image = cv2.resize(image, (128, 128))
        
        # Convert to appropriate color spaces
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        else:
            gray = image
            hsv = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE for better feature extraction
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_enhanced = clahe.apply(gray)
        
        # Basic statistics
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Color features from different channels
        mean_hue = np.mean(hsv[:,:,0])
        mean_saturation = np.mean(hsv[:,:,1])
        std_saturation = np.std(hsv[:,:,1])
        mean_value = np.mean(hsv[:,:,2])
        
        # LAB color space features (good for skin tones)
        mean_l = np.mean(lab[:,:,0])
        mean_a = np.mean(lab[:,:,1])  # Redness/greenness
        mean_b = np.mean(lab[:,:,2])  # Yellowness/blueness
        std_a = np.std(lab[:,:,1])
        std_b = np.std(lab[:,:,2])
        
        # Texture features
        # Local Binary Pattern for texture analysis
        radius = 2
        n_points = 8 * radius
        lbp = local_binary_pattern(gray_enhanced, n_points, radius, method='uniform')
        n_bins = n_points + 2
        lbp_hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
        
        # Calculate GLCM properties for multiple distances and angles
        distances = [1, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(gray_enhanced, distances, angles, 256, symmetric=True, normed=True)
        
        # Extract multiple GLCM properties
        contrast = graycoprops(glcm, 'contrast').flatten().mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').flatten().mean()
        homogeneity = graycoprops(glcm, 'homogeneity').flatten().mean()
        energy = graycoprops(glcm, 'energy').flatten().mean()
        correlation = graycoprops(glcm, 'correlation').flatten().mean()
        
        # Edge/detail detection
        laplacian = cv2.Laplacian(gray_enhanced, cv2.CV_64F)
        laplacian_var = np.var(laplacian)
        
        # Frequency domain features using DCT (Discrete Cosine Transform)
        dct = cv2.dct(np.float32(gray_enhanced))
        # Take the mean of different frequency bands
        low_freq = np.mean(dct[0:8, 0:8])
        mid_freq = np.mean(dct[8:32, 8:32])
        high_freq = np.mean(dct[32:, 32:])
        
        # Color histogram features
        hist_b = cv2.calcHist([image], [0], None, [16], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [16], [0, 256])
        hist_r = cv2.calcHist([image], [2], None, [16], [0, 256])
        
        # Normalize histograms
        hist_b = cv2.normalize(hist_b, hist_b).flatten()
        hist_g = cv2.normalize(hist_g, hist_g).flatten()
        hist_r = cv2.normalize(hist_r, hist_r).flatten()
        
        # Combine all features into a single vector
        features = [
            mean_brightness, std_brightness,
            mean_hue, mean_saturation, std_saturation, mean_value,
            mean_l, mean_a, mean_b, std_a, std_b,
            contrast, dissimilarity, homogeneity, energy, correlation,
            laplacian_var, low_freq, mid_freq, high_freq
        ]
        
        # Add histogram features
        features.extend(hist_b)
        features.extend(hist_g)
        features.extend(hist_r)
        
        # Add LBP histogram
        features.extend(lbp_hist)
        
        return features