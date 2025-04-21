from typing import Protocol, List


class PathShortener(Protocol):
    def shorten_path(self, all_features: List[str],  current_features: List[str], path: str) -> str:
        ...
