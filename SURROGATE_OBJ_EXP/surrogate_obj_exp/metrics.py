from __future__ import annotations

import numpy as np


def coverage_metrics(xys, bins: int = 50):
    if len(xys) == 0:
        return {'xy_coverage_bins': 0, 'xy_path_length': 0.0, 'xy_bbox_area': 0.0}
    arr = np.asarray(xys, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return {'xy_coverage_bins': 0, 'xy_path_length': 0.0, 'xy_bbox_area': 0.0}
    x, y = arr[:, 0], arr[:, 1]
    hist, _, _ = np.histogram2d(x, y, bins=bins)
    diffs = np.diff(arr, axis=0)
    path_length = np.linalg.norm(diffs, axis=1).sum() if len(diffs) else 0.0
    bbox_area = max(0.0, (x.max() - x.min()) * (y.max() - y.min()))
    return {
        'xy_coverage_bins': int((hist > 0).sum()),
        'xy_path_length': float(path_length),
        'xy_bbox_area': float(bbox_area),
        'xy_x_min': float(x.min()),
        'xy_x_max': float(x.max()),
        'xy_y_min': float(y.min()),
        'xy_y_max': float(y.max()),
    }
