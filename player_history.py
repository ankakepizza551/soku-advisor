"""
player_history.py — プレイヤー履歴の蓄積・傾向分析

複数セッションの解析結果を JSON ファイルに蓄積し、
AI コーチングの文脈として傾向サマリーを提供する。

保存先: ~/Videos/.soku_advisor_history/<player_name>.json
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

HISTORY_DIR = Path.home() / "Videos" / ".soku_advisor_history"
MAX_SESSIONS = 50


def _history_path(player_name: str) -> Path:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(c for c in player_name if c.isalnum() or c in " _-")[:40].strip() or "unknown"
    return HISTORY_DIR / f"{safe}.json"


def load_history(player_name: str) -> dict:
    path = _history_path(player_name)
    if not path.exists():
        return {"player_name": player_name, "sessions": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"player_name": player_name, "sessions": []}


def save_session(
    player_name: str,
    *,
    self_char: Optional[str],
    opp_char: Optional[str],
    round_wins: int,
    round_losses: int,
    stats: Optional[dict],
    damage_events_count: int,
    avg_damage_pct: float,
    notable_advice: list[str],
    source: Optional[str] = None,
) -> None:
    """1セッションの解析結果を履歴に追記・保存する。

    source は解析元ファイルのパス。同じファイルからレポートを作り直した時は
    前の分を置き換える（同じ対戦を二重に数えない）。
    """
    history = load_history(player_name)
    entry: dict = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "self_char": self_char or "unknown",
        "opp_char": opp_char or "unknown",
        "round_wins": round_wins,
        "round_losses": round_losses,
        "total_rounds": round_wins + round_losses,
        "damage_events_count": damage_events_count,
        "avg_damage_pct": round(avg_damage_pct, 1),
        "notable_advice": notable_advice[:5],
    }
    if stats:
        entry.update({
            "active_ratio": stats.get("active_ratio_pct", 0),
            "a_hold_pct":   stats.get("a_hold_pct", 0),
            "b_hold_pct":   stats.get("b_hold_pct", 0),
            "c_hold_pct":   stats.get("c_hold_pct", 0),
            "d_hold_pct":   stats.get("d_hold_pct", 0),
            "combo_total":  stats.get("combo_total", 0),
        })

    sessions: list = history.setdefault("sessions", [])
    if source:
        entry["source"] = source
        sessions[:] = [s for s in sessions if s.get("source") != source]
    sessions.append(entry)
    if len(sessions) > MAX_SESSIONS:
        sessions[:] = sessions[-MAX_SESSIONS:]

    _history_path(player_name).write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  [履歴] {player_name} の履歴を保存（累計 {len(sessions)} セッション）")


def build_trend_context(player_name: str, last_n: int = 10) -> Optional[dict]:
    """AI に渡す傾向サマリーを構築する。セッションが2件未満なら None。"""
    history = load_history(player_name)
    sessions = history.get("sessions", [])
    if len(sessions) < 2:
        return None

    recent = sessions[-last_n:]

    total_rounds = sum(s.get("total_rounds", 0) for s in recent)
    total_wins   = sum(s.get("round_wins", 0) for s in recent)
    win_rate = (total_wins / total_rounds * 100) if total_rounds > 0 else 0.0

    has_stats = [s for s in recent if "active_ratio" in s]

    def avg(key: str) -> Optional[float]:
        if not has_stats:
            return None
        return sum(s.get(key, 0) for s in has_stats) / len(has_stats)

    avg_active = avg("active_ratio")
    avg_a      = avg("a_hold_pct")
    avg_b      = avg("b_hold_pct")
    avg_c      = avg("c_hold_pct")
    avg_d      = avg("d_hold_pct")

    # ボタン押下率は行動の回数と一致しないので、高い低いの評価はせず数値のまま渡す
    consistent_patterns: list[str] = []

    # ダメージ受けやすさの傾向
    dmg_sessions = [s for s in recent if s.get("damage_events_count", 0) > 0]
    avg_dmg_count = (
        sum(s["damage_events_count"] for s in dmg_sessions) / len(dmg_sessions)
        if dmg_sessions else 0.0
    )
    if avg_dmg_count > 15:
        consistent_patterns.append(f"被弾回数が多い傾向がある（直近平均 {avg_dmg_count:.0f} 回/セッション）")

    # 勝率トレンド（前半 vs 後半）
    trend = "unknown"
    if len(recent) >= 4:
        half = len(recent) // 2
        def wr(sl: list) -> float:
            r = sum(s.get("total_rounds", 0) for s in sl)
            w = sum(s.get("round_wins", 0) for s in sl)
            return w / r * 100 if r else 0.0
        wr1, wr2 = wr(recent[:half]), wr(recent[half:])
        if wr2 - wr1 > 15:
            trend = "improving"
        elif wr1 - wr2 > 15:
            trend = "declining"
        else:
            trend = "stable"

    # よく使うキャラ
    char_counts: dict[str, int] = {}
    for s in recent:
        c = s.get("self_char", "unknown")
        char_counts[c] = char_counts.get(c, 0) + 1
    most_used_char = max(char_counts, key=char_counts.get) if char_counts else None

    # 過去アドバイスから繰り返し出てきている指摘（重複を集計）
    # 文中の数値（15.3% など）は毎回変わるので、数値を伏せた形で同じ指摘かどうかを見る。
    # 1セッション内で同じ指摘が複数あっても1回と数え、表示には最新の文面を使う。
    advice_freq: dict[str, int] = {}
    advice_latest: dict[str, str] = {}
    for s in sessions[-10:]:
        seen: set[str] = set()
        for a in s.get("notable_advice", []):
            key = re.sub(r"\d+(?:\.\d+)?", "N", a)
            advice_latest[key] = a
            if key not in seen:
                seen.add(key)
                advice_freq[key] = advice_freq.get(key, 0) + 1
    repeated_advice = [
        advice_latest[k] for k, cnt in sorted(advice_freq.items(), key=lambda x: -x[1]) if cnt >= 2
    ][:5]

    return {
        "player": player_name,
        "total_sessions_recorded": len(sessions),
        "recent_sessions_analyzed": len(recent),
        "overall_round_win_rate_pct": round(win_rate, 1),
        "trend": trend,
        "most_used_char": most_used_char,
        "avg_active_ratio_pct": round(avg_active, 1) if avg_active is not None else None,
        "avg_button_hold_pct": (
            {k: round(v, 1) for k, v in (("a", avg_a), ("b", avg_b), ("c", avg_c), ("d", avg_d))}
            if avg_a is not None else None
        ),
        "avg_damage_events_per_session": round(avg_dmg_count, 1),
        "consistent_patterns": consistent_patterns,
        "repeatedly_flagged_issues": repeated_advice,
    }


def get_session_count(player_name: str) -> int:
    history = load_history(player_name)
    return len(history.get("sessions", []))


def clear_history(player_name: str) -> None:
    path = _history_path(player_name)
    if path.exists():
        path.unlink()
