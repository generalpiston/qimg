from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import sqlite3
from typing import Any

from qimg.fts import tokenize
from qimg.util.time import now_iso


EXIF_SOURCE = "exif_pillow@1.0"
CAPTION_SOURCE = "heuristic_caption@1.0"
LABEL_SOURCE = "heuristic_labels@1.0"
OBJECT_SOURCE = "object_detector@1.0"
OCR_SOURCE = "ocr_tesseract@0.1"
OBJECT_MODEL = "qimg-object-detector-detr-resnet-50"

_OBJECT_DETECTOR: Any | None = None
_OBJECT_DETECTOR_INIT_FAILED = False


@dataclass(slots=True)
class FactsExtractStats:
    processed: int = 0
    captions: int = 0
    tags: int = 0
    objects: int = 0
    ocr: int = 0
    errors: int = 0
    ocr_runtime_available: bool = True
    ocr_skipped_unavailable: int = 0
    object_runtime_available: bool = True
    object_skipped_unavailable: int = 0


def _insert_fact(
    conn: sqlite3.Connection,
    asset_id: str,
    fact_type: str,
    key: str | None,
    value_text: str | None,
    value_json: str | None,
    confidence: float | None,
    source: str,
) -> None:
    if value_text is not None and not value_text.strip() and value_json is None:
        return
    ts = now_iso()
    conn.execute(
        """
        INSERT INTO facts(asset_id, fact_type, key, value_text, value_json, confidence, source, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (asset_id, fact_type, key, value_text, value_json, confidence, source, ts, ts),
    )


def replace_exif_fact(conn: sqlite3.Connection, asset_id: str, exif_json: str | None, source: str = EXIF_SOURCE) -> None:
    conn.execute(
        "DELETE FROM facts WHERE asset_id = ? AND fact_type = 'exif' AND source = ?",
        (asset_id, source),
    )
    if not exif_json:
        return
    try:
        parsed = json.loads(exif_json)
    except json.JSONDecodeError:
        return
    if not isinstance(parsed, dict) or not parsed:
        return
    _insert_fact(conn, asset_id, "exif", "all", None, json.dumps(parsed), None, source)


def list_facts(conn: sqlite3.Connection, asset_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT id, asset_id, fact_type, key, value_text, value_json, confidence, source, created_at, updated_at
        FROM facts
        WHERE asset_id = ?
        ORDER BY fact_type, id
        """,
        (asset_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def list_all_facts(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
          f.id,
          f.asset_id,
          a.rel_path,
          f.fact_type,
          f.key,
          f.value_text,
          f.value_json,
          f.confidence,
          f.source,
          f.created_at,
          f.updated_at
        FROM facts f
        JOIN assets a ON a.id = f.asset_id
        WHERE a.is_deleted = 0
        ORDER BY f.asset_id, f.fact_type, f.id
        """
    ).fetchall()
    return [dict(r) for r in rows]


def remove_facts_by_source(conn: sqlite3.Connection, source: str, asset_id: str | None = None) -> int:
    if asset_id:
        return int(conn.execute("DELETE FROM facts WHERE source = ? AND asset_id = ?", (source, asset_id)).rowcount)
    return int(conn.execute("DELETE FROM facts WHERE source = ?", (source,)).rowcount)


def _label_candidates(rel_path: str, limit: int = 8) -> list[str]:
    # Filename/folder tokens are a deterministic local fallback when no detector is configured.
    normalized = rel_path.replace("\\", "/")
    stem = Path(normalized).stem
    folder = normalized.rsplit("/", 1)[0] if "/" in normalized else ""
    raw = f"{folder} {stem}"
    toks = [t for t in tokenize(raw).split() if len(t) > 2]

    dedup: list[str] = []
    seen: set[str] = set()
    for tok in toks:
        if tok in seen:
            continue
        seen.add(tok)
        dedup.append(tok)
        if len(dedup) >= limit:
            break
    return dedup


def _caption_from_rel_path(rel_path: str) -> str:
    stem = Path(rel_path).stem.replace("_", " ").replace("-", " ")
    phrase = re.sub(r"\s+", " ", stem).strip()
    if not phrase:
        phrase = "untitled image"
    return f"Image likely related to {phrase}."


def _extract_ocr_text(path: Path) -> str | None:
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
    except Exception:
        return None

    try:
        with Image.open(path) as img:
            text = str(pytesseract.image_to_string(img)).strip()
            return text or None
    except Exception:
        return None


def _get_object_detector() -> Any | None:
    global _OBJECT_DETECTOR, _OBJECT_DETECTOR_INIT_FAILED
    if _OBJECT_DETECTOR is not None:
        return _OBJECT_DETECTOR
    if _OBJECT_DETECTOR_INIT_FAILED:
        return None

    try:
        from qimg.detector import LocalObjectDetector

        _OBJECT_DETECTOR = LocalObjectDetector(model=OBJECT_MODEL)
    except Exception:
        _OBJECT_DETECTOR_INIT_FAILED = True
        return None
    return _OBJECT_DETECTOR


def object_runtime_available() -> bool:
    return _get_object_detector() is not None


def ocr_runtime_available() -> bool:
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
    except Exception:
        return False
    return pytesseract is not None and Image is not None


def extract_facts(
    conn: sqlite3.Connection,
    asset_ids: list[str] | None,
    extract_caption: bool,
    extract_tags: bool,
    extract_objects: bool,
    extract_ocr: bool,
) -> FactsExtractStats:
    stats = FactsExtractStats()
    ocr_available = ocr_runtime_available() if extract_ocr else True
    object_available = object_runtime_available() if extract_objects else True
    if extract_ocr and not ocr_available:
        stats.ocr_runtime_available = False
    if extract_objects and not object_available:
        stats.object_runtime_available = False

    if asset_ids:
        placeholders = ",".join("?" for _ in asset_ids)
        rows = conn.execute(
            f"SELECT id, rel_path, abs_path FROM assets WHERE is_deleted = 0 AND id IN ({placeholders})",
            tuple(asset_ids),
        ).fetchall()
    else:
        rows = conn.execute("SELECT id, rel_path, abs_path FROM assets WHERE is_deleted = 0").fetchall()

    for row in rows:
        asset_id = str(row["id"])
        rel_path = str(row["rel_path"])
        abs_path = Path(str(row["abs_path"]))
        stats.processed += 1

        try:
            if extract_caption:
                conn.execute(
                    "DELETE FROM facts WHERE asset_id = ? AND fact_type = 'caption' AND source = ?",
                    (asset_id, CAPTION_SOURCE),
                )
                _insert_fact(
                    conn,
                    asset_id,
                    "caption",
                    None,
                    _caption_from_rel_path(rel_path),
                    None,
                    None,
                    CAPTION_SOURCE,
                )
                stats.captions += 1

            if extract_tags:
                labels = _label_candidates(rel_path)

                conn.execute(
                    "DELETE FROM facts WHERE asset_id = ? AND fact_type = 'tag' AND source = ?",
                    (asset_id, LABEL_SOURCE),
                )
                for idx, label in enumerate(labels):
                    conf = max(0.1, 0.9 - (idx * 0.1))
                    _insert_fact(conn, asset_id, "tag", label, label, None, conf, LABEL_SOURCE)
                    stats.tags += 1

            if extract_objects:
                conn.execute(
                    "DELETE FROM facts WHERE asset_id = ? AND fact_type = 'object'",
                    (asset_id,),
                )
                detector = _get_object_detector() if object_available else None
                if detector is None:
                    stats.object_skipped_unavailable += 1
                else:
                    source = str(getattr(detector, "source", OBJECT_SOURCE))
                    for obj in detector.detect(abs_path):
                        label = str(getattr(obj, "label", "")).strip().lower()
                        if not label:
                            continue
                        conf_raw = getattr(obj, "confidence", None)
                        conf = float(conf_raw) if conf_raw is not None else None
                        bbox_raw = getattr(obj, "bbox_xyxy", None)
                        value_json: str | None = None
                        if isinstance(bbox_raw, (tuple, list)) and len(bbox_raw) == 4:
                            bbox = [float(x) for x in bbox_raw]
                            value_json = json.dumps({"bbox_xyxy": bbox})
                        _insert_fact(conn, asset_id, "object", label, label, value_json, conf, source)
                        stats.objects += 1

            if extract_ocr and ocr_available:
                conn.execute(
                    "DELETE FROM facts WHERE asset_id = ? AND fact_type = 'ocr' AND source = ?",
                    (asset_id, OCR_SOURCE),
                )
                text = _extract_ocr_text(abs_path)
                if text:
                    _insert_fact(conn, asset_id, "ocr", None, text, None, None, OCR_SOURCE)
                    stats.ocr += 1
            elif extract_ocr and not ocr_available:
                stats.ocr_skipped_unavailable += 1
        except Exception:
            stats.errors += 1

    return stats


def summarize_facts(conn: sqlite3.Connection, asset_id: str, tag_limit: int = 8, object_limit: int = 8) -> dict[str, Any]:
    rows = conn.execute(
        """
        SELECT fact_type, key, value_text, value_json, confidence, source, updated_at, id
        FROM facts
        WHERE asset_id = ?
        ORDER BY updated_at DESC, id DESC
        """,
        (asset_id,),
    ).fetchall()

    caption: str | None = None
    ocr_text: str | None = None
    tags: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []
    exif: dict[str, Any] = {}
    derived: dict[str, Any] = {}

    for row in rows:
        fact_type = str(row["fact_type"])
        value_text = (row["value_text"] or "").strip() or None
        value_json = row["value_json"]
        key = (row["key"] or "").strip() or None
        confidence = row["confidence"]

        if fact_type == "caption" and caption is None:
            caption = value_text
            continue

        if fact_type == "ocr" and ocr_text is None:
            ocr_text = value_text
            continue

        if fact_type in {"tag", "object"}:
            label = value_text or key
            if not label:
                continue
            item = {
                "label": label,
                "conf": float(confidence) if confidence is not None else None,
                "source": str(row["source"]),
            }
            if fact_type == "tag":
                tags.append(item)
            else:
                objects.append(item)
            continue

        if fact_type in {"exif", "derived"}:
            target = exif if fact_type == "exif" else derived
            if value_json:
                try:
                    parsed = json.loads(str(value_json))
                    if isinstance(parsed, dict):
                        target.update(parsed)
                        continue
                except json.JSONDecodeError:
                    pass
            if key and value_text is not None:
                target[key] = value_text

    tags.sort(key=lambda x: x["conf"] if x["conf"] is not None else -1.0, reverse=True)
    objects.sort(key=lambda x: x["conf"] if x["conf"] is not None else -1.0, reverse=True)

    return {
        "caption": caption,
        "tags": tags[:tag_limit],
        "objects": objects[:object_limit],
        "ocr": ocr_text,
        "exif": exif,
        "derived": derived,
    }
