from collections import defaultdict

class Analytics:
    def __init__(self, line_y_fraction=0.5):
        self.line_y_fraction = line_y_fraction
        self._prev    = {}
        self._seen    = defaultdict(set)
        self._entries = defaultdict(int)
        self._exits   = defaultdict(int)

    def update(self, track_id, label, cx, cy, line_y):
        self._seen[label].add(track_id)
        event = None
        prev  = self._prev.get(track_id)
        if prev is not None:
            if prev < line_y <= cy:
                self._entries[label] += 1
                event = "ENTRY"
            elif prev >= line_y > cy:
                self._exits[label] += 1
                event = "EXIT"
        self._prev[track_id] = cy
        return event

    def get_stats(self):
        classes = set(self._seen) | set(self._entries)
        cs = {}
        for c in classes:
            cs[c] = {
                "total":   len(self._seen.get(c, set())),
                "entries": self._entries.get(c, 0),
                "exits":   self._exits.get(c, 0),
            }
        return {"class_stats": cs,
                "grand_total": sum(len(v) for v in self._seen.values())}