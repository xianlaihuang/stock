import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


class SellDTWDetector:
    def __init__(self, templates, window=20, default_threshold=0.3):
        self.templates = templates
        self.window = window
        self.default_threshold = default_threshold

    @staticmethod
    def normalize(series):
        arr = np.array(series, dtype=float)
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10:
            return np.zeros_like(arr)
        return 2.0 * (arr - mn) / (mx - mn) - 1.0

    def match(self, close_series, threshold=None):
        if len(close_series) < self.window:
            return []
        window_data = close_series[-self.window:]
        normalized = self.normalize(window_data)
        matched = []
        for name, template in self.templates.items():
            tmpl = np.array(template, dtype=float)
            dist, _ = fastdtw(normalized.reshape(-1, 1), tmpl.reshape(-1, 1), dist=euclidean)
            norm_dist = float(dist) / np.sqrt(self.window * len(tmpl))
            effective_threshold = threshold if threshold is not None else self.default_threshold
            if norm_dist <= effective_threshold:
                matched.append({'pattern': name, 'distance': round(norm_dist, 4)})
        return matched

    def compute_dynamic_threshold(self, close_series, percentile=10, min_matches=5):
        if len(close_series) < self.window:
            return self.default_threshold
        all_distances = {name: [] for name in self.templates}
        for i in range(self.window, len(close_series) + 1):
            window_data = close_series[i - self.window:i]
            normalized = self.normalize(window_data)
            for name, template in self.templates.items():
                tmpl = np.array(template, dtype=float)
                dist, _ = fastdtw(normalized.reshape(-1, 1), tmpl.reshape(-1, 1), dist=euclidean)
                norm_dist = float(dist) / np.sqrt(self.window * len(tmpl))
                all_distances[name].append(norm_dist)
        effective_patterns = []
        for name, dists in all_distances.items():
            if len(dists) >= min_matches:
                effective_patterns.append(name)
        if not effective_patterns:
            return self.default_threshold, list(self.templates.keys())
        all_vals = []
        for name in effective_patterns:
            all_vals.extend(all_distances[name])
        if not all_vals:
            return self.default_threshold, effective_patterns
        threshold = float(np.percentile(all_vals, percentile))
        threshold = max(threshold, 0.05)
        return round(threshold, 3), effective_patterns
