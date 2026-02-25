import cv2

def _label(frame, text, pos, fg, bg):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, 0.5, 1)
    x, y = pos
    cv2.rectangle(frame, (x - 2, y - th - 4), (x + tw + 4, y + bl + 2), bg, -1)
    cv2.putText(frame, text, pos, font, 0.5, fg, 1, cv2.LINE_AA)

def draw_frame(frame, tracks, class_map, colors, stats, line_y, fps, fid):
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = frame.shape[:2]

    # Counting line
    cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)
    cv2.putText(frame, "Counting Line", (8, line_y - 6), font, 0.4, (0, 255, 255), 1)

    # Bounding boxes + labels
    for trk in tracks:
        x1, y1, x2, y2, tid, cls, conf = trk
        lbl = class_map.get(cls, "?")
        col = colors.get(lbl, (200, 200, 200))
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        _label(frame, f"{lbl} #{tid}  {conf:.2f}", (x1, y1 - 2), (255, 255, 255), col)
        cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 4, col, -1)

    # Analytics HUD (top-right)
    px = w - 220; py = 10; lh = 22
    n  = len(stats["class_stats"])
    ov = frame.copy()
    cv2.rectangle(ov, (px - 8, py - 4), (w - 4, py + 30 + lh * (n + 1)), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, "ANALYTICS", (px, py + 14), font, 0.45, (255, 200, 0), 1)
    row = py + 14 + lh
    for cls_, d in stats["class_stats"].items():
        col = colors.get(cls_, (200, 200, 200))
        txt = f"{cls_[:7]:7s} T:{d['total']:3d} IN:{d['entries']:2d} OUT:{d['exits']:2d}"
        cv2.putText(frame, txt, (px, row), font, 0.38, col, 1)
        row += lh
    cv2.putText(frame, f"Total: {stats['grand_total']}", (px, row), font, 0.4, (255, 255, 255), 1)

    # FPS counter (bottom-left)
    _label(frame, f"FPS:{fps:5.1f}  Frame:{fid:05d}", (8, h - 10), (255, 255, 255), (30, 30, 30))
    return frame