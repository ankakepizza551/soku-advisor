"""
char_advisor.py — キャラクター別アドバイス

char_data.json を参照し、自分/相手キャラと入力傾向から
キャラ固有のヒントを生成する。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

_CHAR_CACHE: dict | None = None

# char_data.json の tips / opp_watch は内容が未精査のため、レポートへの出力を止めている。
# 中身を書き直したら True に戻す（キャラ名の表示・選択には影響しない）。
CHAR_ADVICE_ENABLED = False


def _data_path() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "char_data.json"
    return Path(__file__).parent / "char_data.json"


def _load_char_data() -> dict:
    global _CHAR_CACHE
    if _CHAR_CACHE is None:
        _CHAR_CACHE = json.loads(_data_path().read_text(encoding="utf-8"))
    return _CHAR_CACHE


def char_id_choices() -> list[str]:
    """GUI用: 'aya — 射命丸文' 形式のリスト"""
    data = _load_char_data()
    return [f"{cid} — {info['name']}" for cid, info in sorted(data.items(), key=lambda x: x[1]["name"])]


def parse_char_choice(value: str) -> Optional[str]:
    if not value or value.startswith("（"):
        return None
    raw = value.split("—")[0].strip()
    return normalize_char_id(raw)


def normalize_char_id(raw: str) -> Optional[str]:
    """ローマ字IDを正規化（aya, Aya, AYA → aya）"""
    if not raw:
        return None
    key = raw.strip().lower()
    data = _load_char_data()
    if key in data:
        return key
    # 日本語名から逆引き
    for cid, info in data.items():
        if info.get("name") == raw.strip():
            return cid
    return None


def char_display_name(char_id: Optional[str]) -> str:
    if not char_id:
        return "不明"
    info = _load_char_data().get(char_id)
    return info["name"] if info else char_id


def generate_char_advice(
    self_char: Optional[str],
    opp_char: Optional[str],
    self_stats=None,
    damage_summary: Optional[str] = None,
) -> list[str]:
    """
    キャラクター情報に基づくアドバイスを返す。
    self_stats: LiveStats（任意）— 入力傾向で tip を選別
    """
    advice: list[str] = []
    if not CHAR_ADVICE_ENABLED:
        return advice
    data = _load_char_data()
    self_id = normalize_char_id(self_char) if self_char else None
    opp_id = normalize_char_id(opp_char) if opp_char else None

    if not self_id and not opp_id:
        return advice

    if self_id and self_id in data:
        info = data[self_id]
        advice.append(f"🎭 使用キャラ: {info['name']} — {info['style']}")

        tips = list(info.get("tips", []))
        # 入力傾向で tip を並べ替え（該当しそうなものを優先）
        if self_stats is not None:
            if self_stats.hold_pct("d") < 3.0:
                tips.sort(key=lambda t: 0 if "D" in t or "ダッシュ" in t else 1)
            if self_stats.hold_pct("b") < 1.0:
                tips.sort(key=lambda t: 0 if "B" in t or "強" in t else 1)
            combo_total = sum(self_stats.combo_counts.values())
            if combo_total < 5:
                tips.sort(key=lambda t: 0 if "236" in t or "214" in t or "623" in t else 1)

        for tip in tips[:2]:
            advice.append(f"  💡 {tip}")

    if opp_id and opp_id in data:
        opp_info = data[opp_id]
        advice.append(f"🎯 相手キャラ: {opp_info['name']} — 注意点")
        for note in opp_info.get("opp_watch", [])[:3]:
            advice.append(f"  ⚠️ {note}")

    if damage_summary and opp_id:
        advice.append(
            f"  📌 被弾分析と合わせて、{data[opp_id]['name']}の "
            f"{data[opp_id]['opp_watch'][0] if data[opp_id].get('opp_watch') else '基本攻撃'} "
            f"への対策を意識してみてください。"
        )

    return advice
