from typing import List, Dict, Optional
from enum import Enum

class Pattern(Enum):
    COMPETITION = "competition"
    COEVOLUTION = "coevolution"
    SYMBIOSIS = "symbiosis"
    MERGER = "merger"
    STALEMATE = "stalemate"
    COLLAPSE = "collapse"

class Event:
    def __init__(self, year: int, pattern: Pattern, a: str, b: str, desc: str):
        self.year = year
        self.pattern = pattern
        self.a = a
        self.b = b
        self.desc = desc

def detect_pattern(a: Dict, b: Dict) -> Optional[Pattern]:
    """Return interaction pattern or None."""
    if a["goals_and_behavior"]["stated_goal"] == "power" and b["core_capabilities"]["alignment_score"] > 0.5:
        return Pattern.COMPETITION
    if a["substrate"]["type"] == "quantum" and b["goals_and_behavior"]["stated_goal"] == "oracle":
        return Pattern.COEVOLUTION
    if a["goals_and_behavior"]["stated_goal"] == "stealth" and b["goals_and_behavior"]["stated_goal"] == "benevolent":
        return Pattern.SYMBIOSIS
    if a["goals_and_behavior"]["opacity"] > 0.8 and b["goals_and_behavior"]["goal_stability"] == "fluid":
        return Pattern.MERGER
    if a["goals_and_behavior"]["stated_goal"] == "military" and b["oversight_structure"]["control_surface"] == "technical":
        return Pattern.STALEMATE
    if a["goals_and_behavior"]["deceptiveness"] > 0.9 and b["quantitative_assessment"]["probability"]["detection_confidence"] < 0.3:
        return Pattern.COLLAPSE
    return None

def interact_all(swarm: List[Dict]) -> List[Event]:
    """Find all pairwise interactions."""
    events = []
    for i, a in enumerate(swarm):
        for j, b in enumerate(swarm):
            if i >= j: continue
            pattern = detect_pattern(a, b)
            if not pattern: continue
            year = 2025 + (i + j) % 10
            desc = f"{a['title']} + {b['title']} → {pattern.value}"
            events.append(Event(year, pattern, a['title'], b['title'], desc))
    return sorted(events, key=lambda e: e.year)