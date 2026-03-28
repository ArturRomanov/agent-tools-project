from __future__ import annotations

import re


class MemoryExtractor:
    def extract(self, user_text: str, assistant_text: str) -> list[tuple[str, str]]:
        """Return (type, content) pairs for durable long-term memory."""
        candidates: list[tuple[str, str]] = []
        user = user_text.strip()
        assistant = assistant_text.strip()

        pref_patterns = [
            r"\bi prefer\b([^\.!?]+)",
            r"\bmy preference is\b([^\.!?]+)",
        ]
        fact_patterns = [
            r"\bmy name is\b([^\.!?]+)",
            r"\bi am\b([^\.!?]+)",
        ]

        for pattern in pref_patterns:
            for match in re.finditer(pattern, user, flags=re.IGNORECASE):
                value = match.group(0).strip()
                if value:
                    candidates.append(("preference", value))

        for pattern in fact_patterns:
            for match in re.finditer(pattern, user, flags=re.IGNORECASE):
                value = match.group(0).strip()
                if value:
                    candidates.append(("fact", value))

        if "remember" in user.lower():
            candidates.append(("task", user[:280]))

        if assistant:
            sentence = assistant.split(".")[0].strip()
            if sentence:
                candidates.append(("assistant_summary", sentence[:280]))

        deduped: list[tuple[str, str]] = []
        seen: set[str] = set()
        for memory_type, content in candidates:
            key = f"{memory_type}:{content.lower()}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append((memory_type, content))
        return deduped[:10]
