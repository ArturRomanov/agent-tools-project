from __future__ import annotations


class MemoryScorer:
    def score(self, memory_type: str, content: str) -> tuple[float, float]:
        base_salience = 0.45
        base_confidence = 0.60

        if memory_type == "preference":
            base_salience = 0.70
            base_confidence = 0.75
        elif memory_type == "fact":
            base_salience = 0.78
            base_confidence = 0.80
        elif memory_type == "task":
            base_salience = 0.85
            base_confidence = 0.70

        if len(content) > 160:
            base_confidence -= 0.08

        return (max(0.0, min(base_salience, 1.0)), max(0.0, min(base_confidence, 1.0)))
