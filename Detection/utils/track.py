import numpy as np

def _iou(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if not inter:
        return 0.0
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (areaA + areaB - inter)

class _Track:
    _next_id = 1
    def __init__(self, det, min_hits):
        self.id = _Track._next_id
        _Track._next_id += 1
        self.box  = det[:4]
        self.conf = det[4]
        self.cls  = int(det[5])
        self.hits = 1
        self.age  = 0
        self.min_hits  = min_hits
        self.confirmed = (min_hits <= 1)

    def update(self, det):
        self.box  = det[:4]
        self.conf = det[4]
        self.cls  = int(det[5])
        self.hits += 1
        self.age   = 0
        if self.hits >= self.min_hits:
            self.confirmed = True

    def to_out(self):
        x1, y1, x2, y2 = [int(v) for v in self.box]
        return [x1, y1, x2, y2, self.id, self.cls, round(self.conf, 3)]

class SimpleTracker:
    def __init__(self, max_age=30, min_hits=2, iou_threshold=0.3):
        self.max_age  = max_age
        self.min_hits = min_hits
        self.iou_thr  = iou_threshold
        self.tracks   = []

    def update(self, detections):
        for t in self.tracks:
            t.age += 1
        unmatched = list(range(len(detections)))
        if self.tracks and detections:
            mat = np.array([
                [_iou(d[:4], t.box) for t in self.tracks]
                for d in detections
            ])
            while True:
                di, ti = divmod(np.argmax(mat), len(self.tracks))
                if mat[di, ti] < self.iou_thr:
                    break
                self.tracks[ti].update(detections[di])
                if di in unmatched:
                    unmatched.remove(di)
                mat[di, :] = -1
                mat[:, ti] = -1
        for i in unmatched:
            self.tracks.append(_Track(detections[i], self.min_hits))
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        return [t.to_out() for t in self.tracks if t.confirmed]