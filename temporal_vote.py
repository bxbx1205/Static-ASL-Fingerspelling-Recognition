"""Small, explainable prediction smoother shared by the webcam demonstration."""
from collections import Counter, deque


class TemporalVote:
    """Majority vote over the last `window` per-frame labels.

    A label is reported only when at least `required` of the recent frames
    agree, which stops the displayed letter flickering when one frame is
    misread. The cost is a short delay before a new letter appears.
    """

    def __init__(self, window=7, required=5):
        if not 1 <= required <= window:
            raise ValueError('required must be between 1 and window')
        self.required = required
        self.history = deque(maxlen=window)

    def clear(self):
        self.history.clear()

    def update(self, label):
        """Add this frame's label; return the stable label or None."""
        # No-hand/low-score/nothing clears stale predictions immediately.
        if label is None or label == 'nothing':
            self.clear()
            return None
        self.history.append(label)
        winner, count = Counter(self.history).most_common(1)[0]
        return winner if count >= self.required else None
