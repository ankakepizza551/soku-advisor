"""
ai_advisor.py — LLM による自然文コーチング

OpenAI 互換 API（OpenAI / Ollama 等）に解析サマリーを渡し、
自然文のアドバイスを生成する。API キー未設定時はスキップ。

環境変数:
  SOKU_AI_API_KEY   — API キー（Ollama ローカルなら "ollama" 等任意）
  SOKU_AI_BASE_URL  — デフォルト https://api.openai.com/v1
  SOKU_AI_MODEL     — デフォルト gpt-4o-mini
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Optional


def _ai_config() -> tuple[str, str, str]:
    api_key = os.environ.get("SOKU_AI_API_KEY", "").strip()
    base_url = os.environ.get("SOKU_AI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("SOKU_AI_MODEL", "gpt-4o-mini")
    return api_key, base_url, model


def ai_available() -> bool:
    return bool(_ai_config()[0])


def build_ai_payload(
    *,
    self_name: str,
    opp_name: str,
    self_char: Optional[str],
    opp_char: Optional[str],
    viewpoint: int,
    rule_advice: list[str],
    match_summaries: list[dict],
    self_stats_summary: Optional[dict] = None,
    history_context: Optional[dict] = None,
) -> dict[str, Any]:
    from char_advisor import char_display_name

    payload: dict[str, Any] = {
        "self": {"name": self_name, "char": char_display_name(self_char), "side": f"P{viewpoint}"},
        "opponent": {"name": opp_name, "char": char_display_name(opp_char)},
        "matches": match_summaries,
        "self_input": self_stats_summary or {},
        "existing_advice": rule_advice[:20],
    }
    if history_context:
        payload["player_history"] = history_context
    return payload


def _stats_summary(stats) -> dict:
    if stats is None:
        return {}
    return {
        "active_ratio_pct": round(stats.active_ratio(), 1),
        "a_hold_pct": round(stats.hold_pct("a"), 1),
        "b_hold_pct": round(stats.hold_pct("b"), 1),
        "c_hold_pct": round(stats.hold_pct("c"), 1),
        "d_hold_pct": round(stats.hold_pct("d"), 1),
        "action_counts": dict(getattr(stats, "act_counts", {})),
        "combo_total": sum(stats.combo_counts.values()),
        "top_combos": sorted(stats.combo_counts.items(), key=lambda x: -x[1])[:5],
        "spirit": stats.spirit_summary() if getattr(stats, "sp_frames", 0) else None,
    }


def generate_ai_advice(payload: dict[str, Any], timeout_sec: int = 60) -> str:
    """LLM に解析結果を渡して自然文コーチングを生成。失敗時は空文字。"""
    api_key, base_url, model = _ai_config()
    if not api_key:
        return ""

    has_history = bool(payload.get("player_history"))
    history_instruction = (
        "payload に player_history キーがある場合は過去セッションの傾向も踏まえ、"
        "「以前から繰り返し出ている課題」と「今回特有の問題」を区別して言及してください。"
        "repeatedly_flagged_issues に同じ指摘が並んでいる場合は特に強調してください。"
        if has_history else ""
    )
    system = (
        "あなたは東方非想天則（2D対戦格闘ゲーム）のコーチです。"
        "渡された対戦データをもとに、プレイヤーへ具体的で励ましも含む日本語アドバイスを書いてください。"
        "3〜6段落、箇条書き可。憶測は控えめに。データにないことは断定しない。"
        "「あなた」は解説視点のプレイヤーを指す。"
        + (f" {history_instruction}" if history_instruction else "")
    )
    user = (
        "以下のJSONは対戦解析結果です。このプレイヤー（self）向けにコーチング文を書いてください。\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )

    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.7,
        "max_tokens": 1200,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"].strip()
    except (OSError, KeyError, IndexError, json.JSONDecodeError) as exc:
        print(f"  [AI] 生成失敗: {exc}")
        return ""
