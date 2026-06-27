"""Rule-based semantic parsing and prompt-action policy for WorldArena prompts."""

from __future__ import annotations

import csv
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


PROMPT_SETS = (
    ("base", "instruction_path"),
    ("variant_1", "instruction_1_path"),
    ("variant_2", "instruction_2_path"),
)

POLICIES = {
    "SAME_ACTION_OK",
    "ACTION_MAYBE_OK",
    "TARGET_CHANGED",
    "VERB_CHANGED",
    "AMBIGUOUS",
}

PREFIX_PATTERNS = [
    re.compile(
        r"^\s*In a fixed robotic workspace,\s*generate a rigid, physically consistent embodied robotic arm\.\s*"
        r"The arm maintains high stability with no deformation and enters the frame to\s*",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*In a fixed robotic workspace,.*?enters the frame to\s*", re.IGNORECASE),
]

VERB_ALIASES = {
    "pick": ["pick", "grab", "grasp", "take", "collect"],
    "place": ["place", "put", "set", "deposit", "drop", "lower", "position"],
    "move": ["move", "shift", "slide", "carry", "transfer", "bring", "glide"],
    "open": ["open", "pull open"],
    "close": ["close", "shut"],
    "press": ["press", "push", "click", "activate", "tap"],
    "stack": ["stack", "pile", "arrange", "rank", "sort"],
    "lift": ["lift", "raise"],
    "hang": ["hang", "hook"],
    "rotate": ["rotate", "turn", "twist", "orient"],
    "scan": ["scan", "qrcode", "qr code"],
    "shake": ["shake"],
    "dump": ["dump", "pour", "toss", "throw"],
    "strike": ["strike", "hit", "hammer"],
    "hide": ["hide", "store"],
}

OBJECT_KEYWORDS = [
    "bottle", "can", "block", "cube", "bowl", "tray", "basket", "bin", "dustbin",
    "container", "bucket", "box", "hamburg", "hamburger", "fries", "pack", "card",
    "playing cards", "microwave", "fridge", "cabinet", "drawer", "door", "laptop",
    "alarm-clock", "alarm clock", "button", "bell", "calendar", "stopwatch", "stapler",
    "pen", "marker", "hammer", "tool", "scissors", "cup", "mug", "plate", "ball",
    "cylinder", "rod", "rope", "hook", "qrcode", "qr code", "phone", "remote", "book",
    "folder", "shelf", "table", "counter", "container", "object", "item", "cap",
    "handle", "knob", "switch", "panel", "lid", "bag", "basket", "basketball",
]

RECEPTACLE_KEYWORDS = [
    "bin", "dustbin", "basket", "container", "tray", "bucket", "box", "cabinet", "drawer",
    "fridge", "microwave", "laptop", "shelf", "table", "counter", "plate", "holder",
    "corner", "wall", "surface", "desk", "floor", "basket", "blue container", "red bin",
]

SPATIAL_PATTERNS = [
    "left", "right", "center", "middle", "corner", "edge", "top", "under", "beneath",
    "behind", "front", "inside", "into", "onto", "on top", "beside", "near", "far",
    "towards", "away", "across", "above", "below", "between", "against", "around",
]

TASK_FAMILY_PATTERNS = [
    ("scanning_qrcode", ["qrcode", "qr code", "scan"]),
    ("button_press_click", ["press", "push", "click", "button", "bell", "activate", "tap"]),
    ("articulated_open_close", ["open", "close", "shut", "drawer", "door", "cabinet", "microwave", "fridge", "laptop", "handle"]),
    ("stacking", ["stack", "on top", "biggest to smallest", "smallest", "pile"]),
    ("ranking_arrangement", ["rank", "ranking", "arrange", "ascending", "descending", "sort", "order"]),
    ("dumping_pouring", ["dump", "pour", "toss", "throw"]),
    ("shaking", ["shake"]),
    ("rotation_orientation", ["rotate", "turn", "twist", "orient"]),
    ("hanging", ["hang", "hook"]),
    ("handover", ["handover", "hand over", "pass to", "give"]),
    ("tool_use", ["hammer", "strike", "hit", "tool", "scissors", "stapler"]),
    ("object_to_container", ["into the", "in the", "inside", "bin", "basket", "container", "tray", "bucket", "box", "dustbin"]),
    ("lifting", ["lift", "raise"]),
    ("pick_place", ["pick", "grab", "place", "put", "set", "move"]),
]

INTERACTION_PATTERNS = [
    ("open_close", ["open", "close", "shut"]),
    ("press_click", ["press", "push", "click", "button", "activate"]),
    ("pick_place", ["pick", "grab", "place", "put", "drop", "move"]),
    ("stack_arrange", ["stack", "arrange", "rank", "sort"]),
    ("tool_use", ["hammer", "strike", "hit", "tool"]),
    ("rotation", ["rotate", "turn", "twist"]),
]

STOP_WORDS = {
    "the", "a", "an", "with", "using", "use", "arms", "arm", "left", "right", "both",
    "then", "and", "to", "it", "them", "this", "that", "from", "for", "of", "by",
}


@dataclass
class PromptSemantic:
    split: str
    episode_id: int
    prompt_set: str
    raw_prompt: str
    prefix_removed_prompt: str
    main_verbs: list[str]
    main_objects: list[str]
    receptacles_or_targets: list[str]
    spatial_relations: list[str]
    task_family: str
    interaction_type: str
    bimanual_likelihood: float
    articulated_object_likelihood: float
    physical_difficulty_score: float
    semantic_parse_confidence: float

    def to_row(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "episode_id": self.episode_id,
            "prompt_set": self.prompt_set,
            "raw_prompt": self.raw_prompt,
            "prefix_removed_prompt": self.prefix_removed_prompt,
            "main_verbs": ";".join(self.main_verbs),
            "main_objects": ";".join(self.main_objects),
            "receptacles_or_targets": ";".join(self.receptacles_or_targets),
            "spatial_relations": ";".join(self.spatial_relations),
            "task_family": self.task_family,
            "interaction_type": self.interaction_type,
            "bimanual_likelihood": round(self.bimanual_likelihood, 4),
            "articulated_object_likelihood": round(self.articulated_object_likelihood, 4),
            "physical_difficulty_score": round(self.physical_difficulty_score, 4),
            "semantic_parse_confidence": round(self.semantic_parse_confidence, 4),
        }


def load_instruction(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return str(data.get("instruction", ""))


def remove_prompt_prefix(text: str) -> str:
    stripped = text.strip()
    for pattern in PREFIX_PATTERNS:
        stripped = pattern.sub("", stripped).strip()
    return stripped


def unique_in_order(values: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        value = value.strip().lower()
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def find_aliases(text: str, aliases: dict[str, list[str]]) -> list[str]:
    lower = text.lower()
    found = []
    for canonical, words in aliases.items():
        for word in words:
            pattern = r"\b" + re.escape(word) + r"\b"
            if re.search(pattern, lower):
                found.append(canonical)
                break
    return unique_in_order(found)


def find_keywords(text: str, keywords: list[str]) -> list[str]:
    lower = text.lower()
    found = []
    for keyword in keywords:
        pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
        if re.search(pattern, lower):
            found.append(keyword.lower())
    return unique_in_order(found)


def extract_noun_phrases(text: str) -> list[str]:
    lower = text.lower()
    phrases = []
    for match in re.finditer(r"\b(?:the|a|an)\s+([a-z0-9\- ]{2,80}?)(?=\s+(?:with|using|by|into|in|on|under|beneath|behind|beside|near|from|to|and|then|,|\.|$))", lower):
        phrase = re.sub(r"\s+", " ", match.group(1).strip())
        tokens = [tok for tok in phrase.split() if tok not in STOP_WORDS]
        if tokens:
            phrases.append(" ".join(tokens[-4:]))
    return unique_in_order(phrases)


def infer_task_family(text: str, verbs: list[str], objects: list[str]) -> str:
    lower = text.lower()
    haystack = " ".join([lower, " ".join(verbs), " ".join(objects)])
    for family, patterns in TASK_FAMILY_PATTERNS:
        if any(pattern in haystack for pattern in patterns):
            return family
    return "unknown"


def infer_interaction_type(text: str) -> str:
    lower = text.lower()
    for interaction, patterns in INTERACTION_PATTERNS:
        if any(pattern in lower for pattern in patterns):
            return interaction
    return "unknown"


def score_bimanual(text: str) -> float:
    lower = text.lower()
    score = 0.0
    if "both arms" in lower or "using both" in lower or "two arms" in lower:
        score += 0.7
    if "left arm" in lower and "right arm" in lower:
        score += 0.8
    if "another arm" in lower or "other arm" in lower:
        score += 0.4
    return min(score, 1.0)


def score_articulated(text: str, objects: list[str], verbs: list[str]) -> float:
    lower = text.lower()
    articulated_terms = ["microwave", "fridge", "cabinet", "drawer", "door", "laptop", "handle", "knob", "lid"]
    score = 0.0
    if any(term in lower for term in articulated_terms):
        score += 0.65
    if "open" in verbs or "close" in verbs:
        score += 0.35
    return min(score, 1.0)


def score_difficulty(text: str, verbs: list[str], spatial: list[str], bimanual: float, articulated: float) -> float:
    score = 1.0
    score += min(len(verbs), 5) * 0.45
    score += min(len(spatial), 6) * 0.22
    score += bimanual * 1.2
    score += articulated * 0.8
    if len(text) > 260:
        score += 0.6
    if len(text) > 420:
        score += 0.6
    return round(min(score, 5.0), 4)


def score_confidence(task_family: str, verbs: list[str], objects: list[str], text: str) -> float:
    score = 0.35
    if task_family != "unknown":
        score += 0.25
    if verbs:
        score += 0.2
    if objects:
        score += 0.15
    if len(text) > 20:
        score += 0.05
    return min(score, 1.0)


def parse_prompt(split: str, episode_id: int, prompt_set: str, raw_prompt: str) -> PromptSemantic:
    core = remove_prompt_prefix(raw_prompt)
    verbs = find_aliases(core, VERB_ALIASES)
    keyword_objects = find_keywords(core, OBJECT_KEYWORDS)
    phrase_objects = extract_noun_phrases(core)
    objects = unique_in_order([*keyword_objects, *phrase_objects])[:20]
    targets = find_keywords(core, RECEPTACLE_KEYWORDS)[:15]
    spatial = find_keywords(core, SPATIAL_PATTERNS)[:15]
    task_family = infer_task_family(core, verbs, objects)
    interaction_type = infer_interaction_type(core)
    bimanual = score_bimanual(core)
    articulated = score_articulated(core, objects, verbs)
    difficulty = score_difficulty(core, verbs, spatial, bimanual, articulated)
    confidence = score_confidence(task_family, verbs, objects, core)
    return PromptSemantic(
        split=split,
        episode_id=episode_id,
        prompt_set=prompt_set,
        raw_prompt=raw_prompt,
        prefix_removed_prompt=core,
        main_verbs=verbs,
        main_objects=objects,
        receptacles_or_targets=targets,
        spatial_relations=spatial,
        task_family=task_family,
        interaction_type=interaction_type,
        bimanual_likelihood=bimanual,
        articulated_object_likelihood=articulated,
        physical_difficulty_score=difficulty,
        semantic_parse_confidence=confidence,
    )


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a = set(left)
    b = set(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def sequence_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right).ratio()


def try_tfidf_pair_similarity(left: str, right: str) -> tuple[float, str]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

        matrix = TfidfVectorizer(ngram_range=(1, 2), lowercase=True).fit_transform([left, right])
        return float(cosine_similarity(matrix[0], matrix[1])[0][0]), "sklearn_tfidf"
    except Exception:
        return sequence_similarity(left, right), "sequence_fallback"


def infer_reuse_policy(
    base: PromptSemantic,
    variant: PromptSemantic,
    text_similarity: float,
    seq_similarity: float,
    verb_overlap: float,
    object_overlap: float,
    receptacle_overlap: float,
) -> tuple[str, float, str]:
    task_family_same = base.task_family == variant.task_family
    main_verb_changed = verb_overlap < 0.5
    main_object_changed = object_overlap < 0.45
    target_changed = receptacle_overlap < 0.45

    reasons = []
    if not task_family_same:
        reasons.append(f"task_family changed {base.task_family}->{variant.task_family}")
    if main_verb_changed:
        reasons.append(f"verb overlap low ({verb_overlap:.2f})")
    if main_object_changed:
        reasons.append(f"object overlap low ({object_overlap:.2f})")
    if target_changed:
        reasons.append(f"target/receptacle overlap low ({receptacle_overlap:.2f})")
    if seq_similarity < 0.58:
        reasons.append(f"text similarity low ({seq_similarity:.2f})")

    if not reasons and text_similarity >= 0.82 and verb_overlap >= 0.75 and object_overlap >= 0.65:
        return "SAME_ACTION_OK", 0.82, "same task family with high text, verb, and object similarity"
    if target_changed and not main_verb_changed:
        return "TARGET_CHANGED", min(0.9, 0.55 + (1.0 - receptacle_overlap) * 0.35), "; ".join(reasons)
    if main_verb_changed:
        return "VERB_CHANGED", min(0.9, 0.55 + (1.0 - verb_overlap) * 0.35), "; ".join(reasons)
    if task_family_same and text_similarity >= 0.62 and object_overlap >= 0.45:
        return "ACTION_MAYBE_OK", 0.62, "; ".join(reasons or ["same task family but prompt differs enough to require manual check"])
    return "AMBIGUOUS", 0.55, "; ".join(reasons or ["mixed semantic signals; manual inspection recommended"])


def compare_prompt_pair(base: PromptSemantic, variant: PromptSemantic, similarity_label: str) -> dict[str, Any]:
    tfidf_similarity, method = try_tfidf_pair_similarity(base.prefix_removed_prompt, variant.prefix_removed_prompt)
    seq_similarity = sequence_similarity(base.prefix_removed_prompt, variant.prefix_removed_prompt)
    verb_overlap = jaccard(base.main_verbs, variant.main_verbs)
    object_overlap = jaccard(base.main_objects, variant.main_objects)
    receptacle_overlap = jaccard(base.receptacles_or_targets, variant.receptacles_or_targets)
    policy, confidence, reason = infer_reuse_policy(
        base,
        variant,
        tfidf_similarity,
        seq_similarity,
        verb_overlap,
        object_overlap,
        receptacle_overlap,
    )
    if method == "sequence_fallback":
        reason = f"TF-IDF unavailable; used SequenceMatcher fallback. {reason}"
    return {
        "split": base.split,
        "episode_id": base.episode_id,
        "comparison": similarity_label,
        "base_prompt_set": base.prompt_set,
        "variant_prompt_set": variant.prompt_set,
        "text_similarity_tfidf": round(tfidf_similarity, 4),
        "text_similarity_method": method,
        "text_similarity_sequence_matcher": round(seq_similarity, 4),
        "verb_overlap": round(verb_overlap, 4),
        "object_overlap": round(object_overlap, 4),
        "receptacle_overlap": round(receptacle_overlap, 4),
        "task_family_base": base.task_family,
        "task_family_variant": variant.task_family,
        "task_family_same": base.task_family == variant.task_family,
        "main_verb_changed": verb_overlap < 0.5,
        "main_object_changed": object_overlap < 0.45,
        "target_receptacle_changed": receptacle_overlap < 0.45,
        "estimated_action_reuse_policy": policy,
        "policy_confidence": round(confidence, 4),
        "reason": reason,
    }


def read_episode_level(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_dataset_path(root: Path, rel_or_abs: str) -> Path:
    path = Path(rel_or_abs)
    if path.is_absolute():
        return path
    return root / path


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_semantic_prompt_analysis(root: Path, out_dir: Path, episode_csv: Path, logger: logging.Logger) -> dict[str, Any]:
    logger.info("Running semantic prompt analysis from %s", episode_csv)
    episode_rows = read_episode_level(episode_csv)
    semantics: list[PromptSemantic] = []
    by_episode: dict[tuple[str, int], dict[str, PromptSemantic]] = defaultdict(dict)
    errors = []

    for row in episode_rows:
        split = row["dataset"]
        episode_id = int(row["episode_id"])
        for prompt_set, path_field in PROMPT_SETS:
            path_text = row.get(path_field, "")
            if not path_text:
                errors.append(f"{split}/episode{episode_id}: missing {path_field}")
                continue
            path = resolve_dataset_path(root, path_text)
            try:
                raw = load_instruction(path)
                parsed = parse_prompt(split, episode_id, prompt_set, raw)
                semantics.append(parsed)
                by_episode[(split, episode_id)][prompt_set] = parsed
            except Exception as exc:
                errors.append(f"{path}: {exc}")

    semantic_rows = [item.to_row() for item in semantics]
    semantic_fields = [
        "split", "episode_id", "prompt_set", "raw_prompt", "prefix_removed_prompt",
        "main_verbs", "main_objects", "receptacles_or_targets", "spatial_relations",
        "task_family", "interaction_type", "bimanual_likelihood",
        "articulated_object_likelihood", "physical_difficulty_score", "semantic_parse_confidence",
    ]
    semantic_csv = out_dir / "prompt_semantics.csv"
    write_csv(semantic_csv, semantic_rows, semantic_fields)

    diff_rows = []
    policy_rows = []
    for (split, episode_id), prompts in sorted(by_episode.items()):
        base = prompts.get("base")
        if base is None:
            continue
        for variant_name, label in (("variant_1", "base_vs_instruction_1"), ("variant_2", "base_vs_instruction_2")):
            variant = prompts.get(variant_name)
            if variant is None:
                continue
            policy_row = compare_prompt_pair(base, variant, label)
            policy_rows.append(policy_row)
            diff_rows.append({
                "split": split,
                "episode_id": episode_id,
                "comparison": label,
                "base_prompt": base.prefix_removed_prompt,
                "variant_prompt": variant.prefix_removed_prompt,
                "base_task_family": base.task_family,
                "variant_task_family": variant.task_family,
                "base_main_verbs": ";".join(base.main_verbs),
                "variant_main_verbs": ";".join(variant.main_verbs),
                "base_main_objects": ";".join(base.main_objects),
                "variant_main_objects": ";".join(variant.main_objects),
                "base_receptacles_or_targets": ";".join(base.receptacles_or_targets),
                "variant_receptacles_or_targets": ";".join(variant.receptacles_or_targets),
                "text_similarity_tfidf": policy_row["text_similarity_tfidf"],
                "text_similarity_sequence_matcher": policy_row["text_similarity_sequence_matcher"],
                "task_family_same": policy_row["task_family_same"],
                "main_verb_changed": policy_row["main_verb_changed"],
                "main_object_changed": policy_row["main_object_changed"],
                "target_receptacle_changed": policy_row["target_receptacle_changed"],
            })

    diff_csv = out_dir / "prompt_variant_diff.csv"
    write_csv(diff_csv, diff_rows, [
        "split", "episode_id", "comparison", "base_prompt", "variant_prompt",
        "base_task_family", "variant_task_family", "base_main_verbs", "variant_main_verbs",
        "base_main_objects", "variant_main_objects", "base_receptacles_or_targets",
        "variant_receptacles_or_targets", "text_similarity_tfidf", "text_similarity_sequence_matcher",
        "task_family_same", "main_verb_changed", "main_object_changed", "target_receptacle_changed",
    ])

    policy_csv = out_dir / "prompt_action_policy.csv"
    write_csv(policy_csv, policy_rows, [
        "split", "episode_id", "comparison", "base_prompt_set", "variant_prompt_set",
        "text_similarity_tfidf", "text_similarity_method", "text_similarity_sequence_matcher",
        "verb_overlap", "object_overlap", "receptacle_overlap", "task_family_base",
        "task_family_variant", "task_family_same", "main_verb_changed", "main_object_changed",
        "target_receptacle_changed", "estimated_action_reuse_policy", "policy_confidence", "reason",
    ])

    summary = summarize_semantic_outputs(semantic_rows, policy_rows, errors)
    logger.info("Wrote %s", semantic_csv)
    logger.info("Wrote %s", diff_csv)
    logger.info("Wrote %s", policy_csv)
    if errors:
        logger.warning("Semantic analysis completed with %d prompt read/parse errors", len(errors))
    return {
        "prompt_semantics_csv": str(semantic_csv),
        "prompt_variant_diff_csv": str(diff_csv),
        "prompt_action_policy_csv": str(policy_csv),
        "semantic_summary": summary,
    }


def summarize_semantic_outputs(semantic_rows: list[dict[str, Any]], policy_rows: list[dict[str, Any]], errors: list[str]) -> dict[str, Any]:
    family_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    verb_counts: Counter[str] = Counter()
    object_counts: Counter[str] = Counter()
    policy_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    similarity_values: list[float] = []

    for row in semantic_rows:
        family_by_split[row["split"]][row["task_family"]] += 1
        for verb in str(row["main_verbs"]).split(";"):
            if verb:
                verb_counts[verb] += 1
        for obj in str(row["main_objects"]).split(";"):
            if obj:
                object_counts[obj] += 1
    for row in policy_rows:
        policy_by_split[row["split"]][row["estimated_action_reuse_policy"]] += 1
        similarity_values.append(float(row["text_similarity_sequence_matcher"]))

    avg_similarity = sum(similarity_values) / len(similarity_values) if similarity_values else 0.0
    return {
        "semantic_rows": len(semantic_rows),
        "policy_rows": len(policy_rows),
        "task_family_by_split": {split: dict(counter.most_common()) for split, counter in family_by_split.items()},
        "top_verbs": dict(verb_counts.most_common(30)),
        "top_objects": dict(object_counts.most_common(30)),
        "policy_by_split": {split: dict(counter.most_common()) for split, counter in policy_by_split.items()},
        "avg_sequence_similarity": round(avg_similarity, 4),
        "errors": errors[:50],
    }
