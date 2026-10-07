"""
soku_advisor/analyzer.py
東方非想天則 リプレイ動画アドバイザー

Usage:
    python analyzer.py <video.mp4> [replay.rep] [-o output.html]
    python analyzer.py <video.mp4> --live recording.json [-o output.html]
    python analyzer.py --live recording.json [-o output.html]   # 動画なしモード

.rep ファイルを省略した場合、動画と同ディレクトリ・同ベース名の .rep を自動検索する。
soku_live_reader.py で記録した JSON ファイルを --live で渡すと両プレイヤーの入力を分析できる。
"""

import argparse
import html
import json
import re
import struct
import sys
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import quote

import cv2
import numpy as np

from char_advisor import generate_char_advice, normalize_char_id, char_display_name
from ai_advisor import ai_available, build_ai_payload, generate_ai_advice, _stats_summary

class AnalyzeError(Exception):
    """解析を続行できないエラー（入力ファイルが開けない等）。"""


MATCHUP_RE = re.compile(
    r"(\d{4})_(.+)\(([^)]+)\)\s*vs\s*(.+)\(([^)]+)\)",
    re.IGNORECASE,
)


def parse_matchup(stem: str) -> tuple[str, str, Optional[str], Optional[str]]:
    """ファイル名からプレイヤー名・キャラIDを抽出"""
    m = MATCHUP_RE.match(stem)
    if not m:
        return "P1", "P2", None, None
    return (
        m.group(2).strip(),
        m.group(4).strip(),
        normalize_char_id(m.group(3)),
        normalize_char_id(m.group(5)),
    )

# ────────────────────────────────────────────
# HP バー検出定数  (640×480 固定解像度)
# ────────────────────────────────────────────
FRAME_W, FRAME_H = 640, 480           # 下の座標はこの解像度が前提
HP_Y = 39
P1_X_START, P1_X_END = 10, 261       # P1 バー左端〜右端
P2_X_START, P2_X_END = 378, 629      # P2 バー左端〜右端
HP_BAR_FULL = 251                     # フル HP 時のピクセル幅

HP_SAMPLE_SEC = 0.1                  # 動画から HP を読む間隔（秒）
HP_MEDIAN_WINDOW = 7                 # 読んだ値をならす幅（サンプル数。0.7秒）

# HP バーの黄色/オレンジ判定条件（BGR）
def _is_hp_color(row: np.ndarray) -> np.ndarray:
    """BGRの行配列を受け取り、HPバー色かどうかのboolマスクを返す"""
    r, g, b = row[:, 2], row[:, 1], row[:, 0]
    return (r > 180) & (g > 80) & (b < 160)


def detect_hp(frame: np.ndarray) -> tuple[float, float]:
    """フレームからP1/P2のHP割合(0.0〜1.0)を返す"""
    row = frame[HP_Y]
    p1_count = int(np.sum(_is_hp_color(row[P1_X_START:P1_X_END + 1])))
    p2_count = int(np.sum(_is_hp_color(row[P2_X_START:P2_X_END + 1])))
    return min(p1_count / HP_BAR_FULL, 1.0), min(p2_count / HP_BAR_FULL, 1.0)


def smooth_hp(samples: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """動画から読んだ HP を前後の中央値でならす。

    暗転やフラッシュなどの演出で、バーが 0.1〜0.3 秒だけ違う値（0% になることもある）に
    読めることがある。そのままだとラウンドの区切りや最低HPを誤るので取り除く。
    """
    half = HP_MEDIAN_WINDOW // 2
    p1 = np.array([s[1] for s in samples])
    p2 = np.array([s[2] for s in samples])
    out = []
    for i, (sec, _, _) in enumerate(samples):
        a, b = max(0, i - half), i + half + 1
        out.append((sec, float(np.median(p1[a:b])), float(np.median(p2[a:b]))))
    return out


# ────────────────────────────────────────────
# ラウンド分割
# ────────────────────────────────────────────
DEATH_HP          = 0.03    # この値以下を「ラウンド終了（死亡）」と見なす
RESET_HP           = 0.85    # 死亡後にこの値以上に戻ったらラウンドリセット（両者同時）
ROUND_MIN_DURATION = 10.0    # これより短いラウンドは無効（ローディング等を除外）
ROUND_ACTIVE_THRESHOLD = 0.15  # 一度この値を下回ったラウンドを「有効」と見なす

@dataclass
class RoundData:
    round_num: int
    start_sec: float
    end_sec: float = 0.0
    hp_timeline: list = field(default_factory=list)  # [(sec, p1_hp, p2_hp), ...]
    p1_won: Optional[bool] = None  # True=P1勝ち, False=P2勝ち, None=不明
    incomplete: bool = False       # 決着前に記録が終わった
    ko_sample: Optional[tuple] = None  # 決着した時の (sec, p1_hp, p2_hp)

    def final_hps(self) -> tuple[float, float]:
        """決着時点（決着していなければ最後）の HP"""
        if not self.hp_timeline:
            return (0.0, 0.0)
        s = self.ko_sample or self.hp_timeline[-1]
        return s[1], s[2]

    def min_hp(self) -> tuple[float, float]:
        if not self.hp_timeline:
            return (1.0, 1.0)
        # 決着後（バーが消えた結果画面など）は数えない
        end = self.ko_sample[0] if self.ko_sample else self.hp_timeline[-1][0]
        p1s = [t[1] for t in self.hp_timeline if t[0] <= end]
        p2s = [t[2] for t in self.hp_timeline if t[0] <= end]
        return min(p1s), min(p2s)

    def duration(self) -> float:
        return self.end_sec - self.start_sec


def split_rounds(hp_samples: list[tuple[float, float, float]]) -> list[RoundData]:
    """
    [(sec, p1, p2)] → RoundData のリスト

    検出方針:
    1. どちらかの HP が DEATH_HP 以下 → had_death フラグON
    2. had_death ON 状態で両者 HP が RESET_HP 以上 → ラウンドリセット
    天候回復(凪等)はどちらかのHPが0になることはないのでフラグが立たない。
    """
    rounds: list[RoundData] = []
    current: Optional[RoundData] = None
    had_death = False

    for sec, p1, p2 in hp_samples:
        if current is None:
            current = RoundData(round_num=1, start_sec=sec)

        # ラウンド終了イベント: どちらかが瀕死
        if min(p1, p2) <= DEATH_HP:
            had_death = True

        # ラウンドリセットイベント: 死亡後に両HP復活
        if had_death and p1 >= RESET_HP and p2 >= RESET_HP:
            had_death = False
            _finalize_round(current)
            if (current.duration() >= ROUND_MIN_DURATION
                    and any(min(t[1], t[2]) < ROUND_ACTIVE_THRESHOLD for t in current.hp_timeline)):
                rounds.append(current)
            current = RoundData(round_num=len(rounds) + 1, start_sec=sec)

        current.hp_timeline.append((sec, p1, p2))

    if current and current.hp_timeline:
        _finalize_round(current)
        if not had_death:
            # 決着前に記録が終わったラウンド。その時点のHP差で勝敗を付けない
            current.p1_won = None
            current.incomplete = True
        if (current.duration() >= ROUND_MIN_DURATION
                and any(min(t[1], t[2]) < ROUND_ACTIVE_THRESHOLD for t in current.hp_timeline)):
            rounds.append(current)

    return rounds


def _finalize_round(r: RoundData) -> None:
    if not r.hp_timeline:
        return
    # 勝敗は決着の瞬間で見る。その後は結果画面などで両方のバーが消えるので、
    # 最後のサンプルでは判定できない。
    # 片方だけが 0 になった最初のサンプルを決着とする（残り数%から逆転した側を負けにしない）。
    # 動画でバーが 0 まで読めなかった時は、DEATH_HP 以下になった最初のサンプルで代用する。
    r.ko_sample = next(
        (s for s in r.hp_timeline if (s[1] <= 0) != (s[2] <= 0)),
        next((s for s in r.hp_timeline if min(s[1], s[2]) <= DEATH_HP), None),
    )
    r.end_sec = (r.ko_sample or r.hp_timeline[-1])[0]
    p1f, p2f = r.final_hps()
    if (p1f <= 0) != (p2f <= 0):
        # 片方だけ 0: 残っている側の勝ち（残りがわずかでも）
        r.p1_won = p1f > 0
    elif p1f > p2f + 0.05:
        r.p1_won = True
    elif p2f > p1f + 0.05:
        r.p1_won = False
    else:
        r.p1_won = None  # 引き分け or 不明


def _self_names(
    p1_name: str,
    p2_name: str,
    viewpoint: int,
) -> tuple[str, str, str, str]:
    """視点に応じて (自分名, 相手名, 自分side, 相手side) を返す"""
    if viewpoint == 2:
        return p2_name, p1_name, "p2", "p1"
    return p1_name, p2_name, "p1", "p2"


def _self_won_from_p1_result(p1_won: Optional[bool], viewpoint: int) -> Optional[bool]:
    if p1_won is None:
        return None
    return p1_won if viewpoint == 1 else not p1_won


def _viewpoint_label(p1_name: str, p2_name: str, viewpoint: int) -> str:
    self_name, _, _, _ = _self_names(p1_name, p2_name, viewpoint)
    slot = "P1・左" if viewpoint == 1 else "P2・右"
    return f" &nbsp;·&nbsp; 解説視点: {self_name}（{slot}）"


# ────────────────────────────────────────────
# ダメージイベント検出（ライブ記録専用）
# ────────────────────────────────────────────
DAMAGE_THRESHOLD = 0.05   # 一連の被弾で合計5%以上減ったらダメージイベント
DAMAGE_WINDOW_SEC = 1.5   # ダメージ直前 N 秒を「直前行動」として分析
BIG_DAMAGE = 0.10         # これ以上の被弾は「大ダメージ」として時刻を出す
COMBO_GAP_SEC = 1.0     # 次のヒットまでこの秒数以内なら同じ被弾（コンボ）とみなす
WAKEUP_HIT_FRAMES = 30    # ダウン（ACT_DOWN）が終わってからこのフレーム数以内の被弾は「起き上がり直後」
OPP_STARTER_TOP = 5       # 相手の始動技ごとの被ダメージを上位いくつまで出すか
SESSION_TREND_TOP = 3     # 連戦で、試合ごとの推移を出す始動技の数

# 位置（記録 version 4 以降の px / py / face がある時のみ）。
# 実記録で確認: 開幕は P1=480, P2=800。ステージ端は 40 / 1240（そこで止まる）。
# 距離の区切りは実記録（文 vs 紫）から決めた: キャラどうしは 39 より近づかない。
# 打撃始動が当たった距離は 39〜214（中央値 101）。開幕の距離（320）は中距離に入る。
STAGE_LEFT, STAGE_RIGHT = 40, 1240
CORNER_DIST = 150         # 端からこの距離以内で、相手が中央側にいる時を「画面端に追い込まれている」とする
DIST_BOUNDS = (60, 220, 450)     # 相手との横の距離の区切り。DIST_LABELS の境目
DIST_LABELS = ("密着", "近距離", "中距離", "遠距離")


def is_cornered(me: dict, opp: dict) -> bool:
    """me が画面端を背負っているか（端の近くにいて、相手が中央側にいる）"""
    x, ox = me["px"], opp["px"]
    return (x - STAGE_LEFT <= CORNER_DIST and ox > x) or (STAGE_RIGHT - x <= CORNER_DIST and ox < x)


def distance_band(distance: float) -> int:
    """DIST_LABELS の何番目か"""
    return sum(1 for bound in DIST_BOUNDS if distance >= bound)

@dataclass
class DamageEvent:
    time_sec: float             # 最初のヒットの時刻
    hp_before: float
    hp_after: float
    damage_pct: float           # 一連の被弾で食らったダメージ合計 (0.0〜1.0)
    prev_inputs: list[dict]     # 最初のヒット直前1.5秒分のフレームデータ
    target: str                 # "p1" or "p2"
    hits: int = 1               # HPが減ったフレーム数（ヒット数の目安）
    spirit_before: Optional[int] = None  # 最初のヒット直前の霊力（ライブ記録のみ）
    estimated: bool = False     # prev_inputs が .rep を動画に時刻合わせして得た推定か
    opp_act: Optional[int] = None   # 最初のヒットの時点で相手が最後に出していた攻撃のアクションID（ライブ記録のみ）
    opp_move: str = ""          # opp_act の表示名（「JA」「236B」「スキル(570)」など）
    after_down: bool = False    # 最初のヒットがダウン明け WAKEUP_HIT_FRAMES 以内か（ライブ記録のみ）
    video_sec: Optional[float] = None   # 動画の再生位置（ライブ記録を動画に時刻合わせできた時のみ）
    distance: Optional[int] = None  # 最初のヒットの時の相手との横の距離（位置のある記録のみ）
    cornered: bool = False      # 最初のヒットの時に画面端に追い込まれていたか（位置のある記録のみ）

    def what_was_doing(self) -> str:
        """直前行動のサマリー文字列を返す"""
        if not self.prev_inputs:
            return "不明"
        if self.estimated:
            return self._from_rep_inputs()
        last_act = self.prev_inputs[-1].get("act")
        cat = act_category(last_act)
        own = ACT_LABELS[cat] + "中" if cat in ("melee", "bullet", "skill", "spell", "move", "guard") else ""
        if last_act == ACT_GUARD_CRUSH:
            own = "ガードクラッシュ中"
        # ダウン明けの被弾は、やられ状態が残っていても追撃ではない
        if self.after_down:
            return "起き上がり直後" + (f"（{own}）" if own else "")
        # 被弾直前のアクションが技なら、それが一番確かな答え
        if own:
            return own
        # 前の被弾から COMBO_GAP_SEC 以上空いて、やられ状態のまま次を食らった場合。
        # この間の入力は行動になっていないので、ボタンからは推測しない
        if cat == "hit":
            return "やられ中（追撃）"
        # 技を出していなければ、最後の30フレームでどのボタンが多かったか
        recent = self.prev_inputs[-30:]
        a_held = sum(1 for f in recent if f.get("a"))
        b_held = sum(1 for f in recent if f.get("b"))
        c_held = sum(1 for f in recent if f.get("c"))
        d_held = sum(1 for f in recent if f.get("d"))
        left   = sum(1 for f in recent if f.get("x", 0) < 0)
        right  = sum(1 for f in recent if f.get("x", 0) > 0)
        combos = [_combo_key(f["combo"]) for f in recent if f.get("combo", 0)]

        parts = []
        if a_held > 10:  parts.append("A（打撃）中")
        if b_held > 10:  parts.append("B（射撃）中")
        if c_held > 10:  parts.append("C（射撃）中")
        if d_held > 5:   parts.append("D（飛翔/ダッシュ）中")
        if combos:
            code = max(set(combos), key=combos.count)
            parts.append(COMBO_NAMES.get(code, f"コマンド({code})") + "入力中")
        if not parts:
            # 向きは記録していないので前進/後退は断定しない
            if left > right:  parts.append("←入力中")
            elif right > left: parts.append("→入力中")
            else:             parts.append("ニュートラル")
        return "・".join(parts)


    def _from_rep_inputs(self) -> str:
        """.rep の入力だけからの推定。アクションIDが無いので、押したボタンで答える"""
        recent = self.prev_inputs[-45:]
        # 攻撃ボタンは数フレームしか押さないので、押していた時間ではなく最後に押したものを見る。
        # （窓を 1.5 秒ぶん全部に広げると、関係ない古い入力を拾って正解率が下がった）
        for f in reversed(recent):
            for key, label in (("c", "C（射撃）"), ("b", "B（射撃）"), ("a", "A（打撃）")):
                if f.get(key):
                    return label + "を押した後"
        if sum(1 for f in recent if f.get("d")) > 5:
            return "D（飛翔/ダッシュ）中"
        left = sum(1 for f in recent if f.get("x", 0) < 0)
        right = sum(1 for f in recent if f.get("x", 0) > 0)
        if left > right:
            return "←入力中"
        if right > left:
            return "→入力中"
        return "ニュートラル"


def detect_damage_events(frames: list[dict], fps: int = 60) -> list[DamageEvent]:
    """
    ライブ記録フレームリストからダメージイベントを検出する。
    COMBO_GAP_SEC 以内に続くHP減少を1回の被弾（コンボ）にまとめ、
    合計が DAMAGE_THRESHOLD 以上のものに、最初のヒット直前
    DAMAGE_WINDOW_SEC 秒の入力状況を添付する。
    """
    events: list[DamageEvent] = []
    if not frames:
        return events
    window = int(DAMAGE_WINDOW_SEC * fps)
    skill_names = skill_command_names(frames)

    for side in ("p1", "p2"):
        other = "p2" if side == "p1" else "p1"
        # 記録開始時点のHPを基準にする（ラウンド途中から記録しても誤検出しない）
        prev = frames[0][side]["hp"] / HP_MAX_SOKU
        start_i: Optional[int] = None   # 進行中の被弾の開始フレーム
        hp_before = prev
        last_hit_t = 0.0
        hits = 0

        def close(hp_after: float) -> None:
            if start_i is None or hp_before - hp_after < DAMAGE_THRESHOLD:
                return
            # 最初のヒットから遡って、相手が最後に出していた攻撃（射撃は出した後に当たることがある）
            opp_act = next(
                (frames[j][other].get("act") for j in range(start_i, max(0, start_i - window) - 1, -1)
                 if act_category(frames[j][other].get("act")) in ATTACK_CATEGORIES),
                None,
            )
            after_down = any(
                frames[j][side].get("act") in ACT_DOWN
                for j in range(max(0, start_i - WAKEUP_HIT_FRAMES), start_i)
            )
            me, opp = frames[start_i][side], frames[start_i][other]
            has_pos = "px" in me and "px" in opp
            events.append(DamageEvent(
                distance=abs(me["px"] - opp["px"]) if has_pos else None,
                cornered=is_cornered(me, opp) if has_pos else False,
                time_sec=frames[start_i]["t"],
                hp_before=hp_before,
                hp_after=hp_after,
                damage_pct=hp_before - hp_after,
                prev_inputs=[frames[j][side] for j in range(max(0, start_i - window), start_i)],
                target=side,
                hits=hits,
                spirit_before=frames[start_i - 1][side].get("sp") if start_i > 0 else None,
                opp_act=opp_act,
                opp_move=_act_name(opp_act, skill_names.get((other, opp_act))) if opp_act is not None else "",
                after_down=after_down,
            ))

        for i, frame in enumerate(frames):
            t = frame["t"]
            hp = frame[side]["hp"] / HP_MAX_SOKU

            # 間が空いた / HPが増えた（ラウンドリセット・回復）→ 被弾を確定
            if start_i is not None and (t - last_hit_t > COMBO_GAP_SEC or hp > prev):
                close(prev)
                start_i = None

            if hp < prev:
                if start_i is None:
                    start_i = i
                    hp_before = prev
                    hits = 0
                hits += 1
                last_hit_t = t

            prev = hp

        close(prev)

    events.sort(key=lambda e: e.time_sec)
    return events


def detect_damage_events_from_hp(
    rounds: list[RoundData],
    rep_sync: Optional["RepSync"] = None,
) -> list[DamageEvent]:
    """
    動画から読んだ HP だけでダメージイベントを検出する（ライブ記録が無い時用）。
    決着後はバーが消えて HP が 0 に読めるので、各ラウンドの決着時点までしか見ない。
    rep_sync（.rep を動画に時刻合わせした結果）があれば、被弾直前の入力を .rep から付ける。
    無ければ直前行動は付かない。
    """
    events: list[DamageEvent] = []
    for round_index, r in enumerate(rounds):
        timeline = [s for s in r.hp_timeline if s[0] <= r.end_sec]
        if not timeline:
            continue
        for idx, side in ((1, "p1"), (2, "p2")):
            prev = timeline[0][idx]
            start_t: Optional[float] = None
            hp_before = prev
            last_hit_t = 0.0

            def close(hp_after: float) -> None:
                if start_t is None or hp_before - hp_after < DAMAGE_THRESHOLD:
                    return
                prev_inputs = rep_sync.inputs_before(round_index, side, start_t) if rep_sync else []
                events.append(DamageEvent(
                    time_sec=start_t,
                    hp_before=hp_before,
                    hp_after=hp_after,
                    damage_pct=hp_before - hp_after,
                    prev_inputs=prev_inputs,
                    target=side,
                    hits=0,
                    estimated=bool(prev_inputs),
                ))

            for sample in timeline:
                t, hp = sample[0], sample[idx]
                if start_t is not None and (t - last_hit_t > COMBO_GAP_SEC or hp > prev):
                    close(prev)
                    start_t = None
                if hp < prev:
                    if start_t is None:
                        start_t = t
                        hp_before = prev
                    last_hit_t = t
                prev = hp
            close(prev)

    events.sort(key=lambda e: e.time_sec)
    return events


def _starter_totals(events: list[DamageEvent]) -> list[tuple[str, list[float]]]:
    """食らわせた側の始動技ごとのダメージの並びを、合計の多い順に OPP_STARTER_TOP 件まで返す"""
    by_move: dict[str, list[float]] = {}
    for e in events:
        if e.opp_move:
            by_move.setdefault(e.opp_move, []).append(e.damage_pct)
    return sorted(by_move.items(), key=lambda x: -sum(x[1]))[:OPP_STARTER_TOP]


def _starter_summary(events: list[DamageEvent]) -> str:
    return "、".join(f"{name} {sum(d)*100:.0f}%・{len(d)}回" for name, d in _starter_totals(events))


def session_starter_lines(
    matches: list["MatchData"],
    self_side: str,
    self_name: str,
) -> list[str]:
    """連戦全体での始動技の集計と、被ダメージの多い始動技の試合ごとの推移（ライブ記録のみ）。

    同じ番号でもキャラが違えば別の技なので、途中でキャラを変えた連戦では、
    一番多かった組み合わせの試合だけを集計する。
    """
    opp_side = "p2" if self_side == "p1" else "p1"
    matchup = Counter((m.p1_char, m.p2_char) for m in matches).most_common(1)[0][0]
    group = [m for m in matches if (m.p1_char, m.p2_char) == matchup]
    if len(group) < 2:
        return []
    if len(group) == len(matches):
        scope = f"全{len(group)}試合"
    else:
        scope = (f"{char_display_name(matchup[0])} vs {char_display_name(matchup[1])} の"
                 f"{len(group)}試合（試合{'・'.join(str(m.match_num) for m in group)}）")

    taken = [e for m in group for e in m.damage_events if e.target == self_side]
    dealt = [e for m in group for e in m.damage_events if e.target == opp_side]
    lines: list[str] = []
    totals = _starter_totals(taken)
    if totals:
        lines.append(f"⚠️ {scope}での相手の始動技ごとの被ダメージ: {_starter_summary(taken)}")
        for name, _ in totals[:SESSION_TREND_TOP]:
            per_match = [
                sum(e.damage_pct for e in m.damage_events if e.target == self_side and e.opp_move == name)
                for m in group
            ]
            lines.append(f"  {name} 始動の被ダメージの推移（試合順）: "
                         + " → ".join(f"{d*100:.0f}%" for d in per_match))
    if _starter_totals(dealt):
        lines.append(f"🗡️ {scope}での {self_name} の始動技ごとの与ダメージ: {_starter_summary(dealt)}")
    return lines


def analyze_offense(events: list[DamageEvent], opp_side: str, self_name: str) -> list[str]:
    """自分の始動技ごとの与ダメージ（相手の被弾を攻撃側から見たもの。ライブ記録のみ）"""
    starters = _starter_summary([e for e in events if e.target == opp_side])
    return [f"🗡️ {self_name} の始動技ごとの与ダメージ: {starters}"] if starters else []


def _fmt_time(sec: float) -> str:
    """「106.3秒（1:46）」形式。動画で探しやすいよう分:秒も付ける"""
    if sec < 60:
        return f"{sec:.1f}秒"
    return f"{sec:.1f}秒（{int(sec // 60)}:{int(sec % 60):02d}）"


def analyze_damage_patterns(
    events: list[DamageEvent],
    target: str,
    p_name: str,
) -> list[str]:
    """
    ダメージイベントのパターンを分析して文章にする。
    target: "p1" or "p2"
    """
    my_events = [e for e in events if e.target == target]
    if not my_events:
        return [f"✅ {p_name} への有効なダメージイベントは検出されませんでした。"]

    results = []
    total_dmg = sum(e.damage_pct for e in my_events)
    avg_dmg   = total_dmg / len(my_events)

    results.append(
        f"⚠️ {p_name} は {len(my_events)} 回被弾しました"
        f"（コンボは1回として集計。平均 {avg_dmg*100:.0f}% / 回、合計 {total_dmg*100:.0f}%）"
    )

    # 霊力（ライブ記録のみ）: 1玉未満だと射撃・スキルが出せない
    with_spirit = [e for e in my_events if e.spirit_before is not None]
    if with_spirit:
        low = sum(1 for e in with_spirit if e.spirit_before < SPIRIT_ORB)
        results.append(f"  被弾{len(with_spirit)}回のうち、霊力1玉未満の時に食らったのは{low}回")

    # 位置（version 4 以降の記録のみ）: どの距離で、画面端で、どれだけ食らったか
    with_pos = [e for e in my_events if e.distance is not None]
    if with_pos:
        bands = []
        for k, label in enumerate(DIST_LABELS):
            hit = [e for e in with_pos if distance_band(e.distance) == k]
            bands.append(f"{label} {len(hit)}回・{sum(e.damage_pct for e in hit)*100:.0f}%")
        results.append(f"  被弾した時の相手との距離: {' / '.join(bands)}")
        cornered = [e for e in with_pos if e.cornered]
        results.append(
            f"  被弾{len(with_pos)}回のうち、画面端に追い込まれていた時が{len(cornered)}回"
            f"（合計 {sum(e.damage_pct for e in cornered)*100:.0f}%）"
        )

    # 直前行動の集計（入力のある記録のみ。動画だけの時は付かない）
    action_counts: dict[str, int] = {}
    for e in my_events:
        if not e.prev_inputs:
            continue
        action = e.what_was_doing()
        action_counts[action] = action_counts.get(action, 0) + 1

    top_actions = sorted(action_counts.items(), key=lambda x: -x[1])[:3]
    if top_actions:
        desc = "、".join(f"「{a}」({c}回)" for a, c in top_actions)
        results.append(f"  ダメージを受けた直前の行動: {desc}")

    # 相手の始動技ごとの被ダメージ（ライブ記録のみ）
    starters = _starter_summary(my_events)
    if starters:
        results.append(f"  相手の始動技ごとの被ダメージ: {starters}")

    # 特に大きいダメージ
    # 10%以上は全部、時刻順に出す（動画で順に見返せるように）
    for e in my_events:
        if e.damage_pct >= BIG_DAMAGE:
            when = _fmt_time(e.time_sec) if e.video_sec is None else f"動画 {_fmt_time(e.video_sec)}"
            line = f"  大ダメージ: {when} / {e.damage_pct*100:.0f}%被弾"
            if e.hits:
                line += f"（{e.hits}ヒット）"
            if e.prev_inputs:
                line += f" / 直前→{e.what_was_doing()}"
            if e.opp_move:
                line += f" / 相手→{e.opp_move}"
            if e.distance is not None:
                line += f" / {DIST_LABELS[distance_band(e.distance)]}" + ("・画面端" if e.cornered else "")
            results.append(line)

    if any(e.video_sec is not None for e in my_events):
        results.append(
            "  ※ 「動画」と付いた時刻は動画の再生位置です"
            "（ライブ記録と動画の HP の減り方を突き合わせて出しています）"
        )

    if any(e.estimated for e in my_events):
        results.append(
            "  ※ 直前の行動は .rep の入力を動画に時刻合わせして出した推定です"
            "（ずれは 0.3 秒程度まで。技の種類までは分かりません）"
        )

    return results


# ────────────────────────────────────────────
# .rep 入力解析
# ────────────────────────────────────────────
# .rep の並び（実ファイル約1500本と、同じ試合のライブ記録との突き合わせで確認）:
#   0x0C        対戦の種類。1=対CPU、3=対人（ネット対戦を含む）
#   0x0E        P1 キャラ番号、0x0F カラー、0x10 デッキ枚数(u32)、続いてカード(u16×枚数)
#   3バイト空けて P2 も同じ並び
#   P2 のカードの後 10 バイト空けて 入力数(u32)、続いて入力(u16×入力数) がファイル末尾まで
# 入力の下位8ビットが INPUT_LABELS。対人は P1, P2 の順に1フレームずつ交互に並び、
# 対CPU は P1 だけが1フレームに1個並ぶ（CPU の入力は入っていない）。
# 対人で P1 が先なのは約6000本の集計で確認した: 開幕の最初の横入力が、先の列は ← が8割、
# 後の列は → が8割（どちらも後ろ下がり。P1 は左側、P2 は右側から始まる）。
# 同じプレイヤーのボタン比率も、P1 の時は先の列、P2 の時は後の列に出る。
REP_MODE_OFS = 0x0C
REP_MODE_VS_COM = 1
REP_P1_OFS = 0x0E
REP_DECK_MAX = 40   # これより多いデッキ枚数は読み間違いとみなす
REP_CHAR_IDS = [
    "reimu", "marisa", "sakuya", "alice", "patchouli", "youmu", "remilia",
    "yuyuko", "yukari", "suika", "reisen", "aya", "komachi", "iku", "tenshi",
    "sanae", "cirno", "meiling", "utsuho", "suwako",
]

INPUT_LABELS = {
    0x01: "↑",  0x02: "↓",  0x04: "←",  0x08: "→",
    0x10: "A",   0x20: "B",   0x40: "C",   0x80: "D",
}
DIRECTIONS = {0x01, 0x02, 0x04, 0x08}
BUTTONS    = {0x10, 0x20, 0x40, 0x80}

@dataclass
class RepStats:
    total_frames: int = 0
    button_counts: dict = field(default_factory=lambda: {k: 0 for k in INPUT_LABELS})
    neutral_frames: int = 0
    p1_name: str = "P1"
    p2_name: str = "P2"
    active_player: int = 2   # 1 or 2 — この統計がどちらのプレイヤーのものか
    p1_char: Optional[str] = None  # ヘッダから読んだキャラID（対CPU戦の P2 も入っている）
    p2_char: Optional[str] = None

    def button_pct(self, bit: int) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.button_counts.get(bit, 0) / self.total_frames * 100

    def active_ratio(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return (self.total_frames - self.neutral_frames) / self.total_frames * 100


# ────────────────────────────────────────────
# ライブ記録 JSON 解析 (soku_live_reader.py 出力)
# ────────────────────────────────────────────
# コマンドコードはビットフラグ。4ビットごとに1コマンドで、その中の並びが A/B/C/D。
# 実記録で確認できたこと:
#   - B と C は隣のビット（512→アクション520 / 1024→521、8192→540 / 16384→541）
#   - 22 は ↓ だけを入れて B で 10 回中 10 回 536870912（→アクション560）
#   - 2 と 514(=2|512) はどちらもアクション500。複数立っていたら下位のコマンドを採る
# 623 と 421 は入力方向からの推定（623 は →↓ を入れた時に出た。421 はCPU側でしか出ていない）。
# キーは各コマンドの B のビット。
# ビット16〜27 のコマンド（実記録で 524288 などが出ている）は対応が分かっていないので、生の値のまま扱う。
COMBO_NAMES = {
    2:         "236",
    32:        "214",
    512:       "623",
    8192:      "421",
    536870912: "22",
}

# 歩き・ダッシュのアクションID。前後は向きに合わせてゲーム側が決めるので、
# 左右の入力から推測するより確か（画面右側にいる時は ← が前進になる）。
ACT_FORWARD = (4, 200)   # 前歩き / 前ダッシュ
ACT_BACKWARD = (5, 201)  # 後ろ歩き / バックステップ
ACT_DOWN = (197, 198, 199)   # ダウン〜起き上がり（下の 170〜199 の注記を参照）
ACT_GUARD_CRUSH = 143        # 地上のガードクラッシュ（同じ試合の動画のコマで確認）

# 通常技のアクションID → 技名。キャラ共通。4キャラ（文・幽々子・妖夢・パチュリー）の記録で、
# その行動が始まった時のボタン・方向・空中かどうかと突き合わせて確かめたものだけ載せている。
# 前後の区別が要る 330 / 408 / 418 は、向きの入った記録（文 vs 紫）で確かめた:
# 330 は後ろ+A、408 と 418 は前ダッシュ中の B / C。載っていない番号は「打撃(310)」のように出る。
NORMAL_NAMES = {
    300: "近A", 301: "遠A", 302: "6A", 303: "2A", 304: "3A", 305: "DA",
    306: "JA", 307: "J6A", 308: "J2A", 309: "J8A",
    320: "AA", 321: "AAA", 322: "AAAA", 323: "AAAAA", 330: "4A",
    400: "B", 401: "6B", 402: "2B", 404: "JB", 406: "J2B", 408: "DB",
    410: "C", 411: "6C", 412: "2C", 414: "JC", 415: "J6C", 416: "J2C", 418: "DC",
}
SKILL_NAME_WINDOW = 12      # スキルが始まる何フレーム前までのコマンド入力を見るか
SKILL_NAME_MIN_SHARE = 0.6  # 同じコマンドがこの割合以上で揃った時だけ技名にする


# アクションID（act）の番号帯 → 行動カテゴリ。実記録で確認した対応:
#   HPが減るフレームは 50〜99、霊力だけ減るのは 150 番台、500 番台はコマンド入力の直後。
# 690 以降など表に無い帯は集計しない。
# 170〜199 も集計しない。実記録では 197〜199 が 98（やられ）の直後に約50フレーム、
# 180/181 が 71（空中やられ）の直後に約20フレーム続く（ダウン〜起き上がり・着地と思われる）。
ACT_CATEGORIES = [
    ("hit",    50, 150, "被弾"),
    ("guard", 150, 170, "ガード"),
    ("move",  200, 300, "ダッシュ・飛翔"),
    ("melee", 300, 400, "打撃"),
    ("bullet", 400, 500, "射撃"),
    ("skill", 500, 600, "スキル"),
    ("spell", 600, 690, "スペルカード"),
]
ACT_LABELS = {key: label for key, _, _, label in ACT_CATEGORIES}
# レポートに並べる順（自分から出した行動 → 防御）
ACT_REPORT_ORDER = ["melee", "bullet", "skill", "spell", "move", "guard"]
ATTACK_CATEGORIES = ("melee", "bullet", "skill", "spell")


def act_category(act: Optional[int]) -> Optional[str]:
    if act is None:
        return None
    for key, lo, hi, _ in ACT_CATEGORIES:
        if lo <= act < hi:
            return key
    return None


def _act_name(act: int, skill_name: Optional[str] = None) -> str:
    """技名。分からないものは「打撃(330)」のようにカテゴリと番号で出す"""
    if act in NORMAL_NAMES:
        return NORMAL_NAMES[act]
    if skill_name:
        return skill_name
    cat = act_category(act)
    return f"{ACT_LABELS[cat]}({act})" if cat else f"行動({act})"


def skill_command_names(frames: list[dict]) -> dict[tuple[str, int], str]:
    """スキルのアクションID → 「236B」のようなコマンド名。キーは (side, act)。

    スキルの番号とコマンドの対応はキャラごとに違う（同じ 22 でも文は 540、妖夢は 560）ので、
    表は持たず、その記録の中でスキルが始まった時に入っていたコマンドから決める。
    派生や追加入力で始まる番号はコマンドが揃わないので、名前を付けない。
    """
    names: dict[tuple[str, int], str] = {}
    for side in ("p1", "p2"):
        seen: dict[int, list[str]] = {}
        prev = None
        for i, frame in enumerate(frames):
            act = frame[side].get("act")
            if act != prev and act_category(act) == "skill":
                recent = [frames[j][side] for j in range(max(0, i - SKILL_NAME_WINDOW), i + 1)]
                cmds = [COMBO_NAMES.get(_combo_key(f["combo"])) for f in recent if f.get("combo")]
                cmd = next((c for c in reversed(cmds) if c), "")
                b = any(f.get("b") for f in recent)
                c = any(f.get("c") for f in recent)
                # 空中で出した時は J を付ける（高さは位置のある記録にしか無い）
                air = "J" if frame[side].get("py", 0) > 0 else ""
                seen.setdefault(act, []).append(
                    air + cmd + ("B" if b and not c else "C" if c and not b else "") if cmd else "")
            prev = act
        for act, found in seen.items():
            name, count = Counter(found).most_common(1)[0]
            if name and count / len(found) >= SKILL_NAME_MIN_SHARE:
                names[(side, act)] = name
    return names


def _combo_key(code: int) -> int:
    """生のコマンドコードを集計用のキー（COMBO_NAMES のキー、無ければ生値）にまとめる"""
    for bit in COMBO_NAMES:
        if code & ((bit >> 1) * 0xF):  # そのコマンドの A/B/C/D どれか
            return bit
    return code

HP_MAX_SOKU = 10000  # 天則の最大HP値
SPIRIT_ORB = 200     # 霊力1玉ぶん（最大 1000 = 5玉）。射撃・スキル1回で1玉使う
SPIRIT_GUARD_DROP = 100  # ガード中に1フレームでこれ以上減ったら「削られた」と数える

@dataclass
class LiveStats:
    """soku_live_reader.py のJSONから生成するプレイヤー入力統計"""
    total_frames: int = 0
    neutral_frames: int = 0
    # ボタン押し回数（「pressed」の遷移0→1をカウント）
    press_a: int = 0
    press_b: int = 0
    press_c: int = 0
    press_d: int = 0
    # ホールドフレーム数
    held_up: int = 0
    held_down: int = 0
    held_left: int = 0
    held_right: int = 0
    held_a: int = 0
    held_b: int = 0
    held_c: int = 0
    held_d: int = 0
    # コマンド入力（遷移カウント）
    combo_counts: dict = field(default_factory=dict)
    p1_name: str = "P1"
    p2_name: str = "P2"
    has_axis: bool = True   # 方向入力（x/y）が記録されているか
    # 行動カテゴリ（ACT_CATEGORIES のキー）ごとの回数。アクションが切り替わった回数で数える
    act_counts: dict = field(default_factory=dict)
    # 前進・後退していたフレーム数（ACT_FORWARD / ACT_BACKWARD）
    fwd_frames: int = 0
    back_frames: int = 0
    # 霊力（記録に sp がある時のみ）
    sp_frames: int = 0       # 霊力を記録したフレーム数
    sp_sum: int = 0          # 霊力の合計（平均用）
    sp_low_frames: int = 0   # 1玉未満だったフレーム数
    sp_guard_drops: int = 0  # ガード中に霊力を削られた回数
    # 位置（記録に px がある時のみ）
    pos_frames: int = 0         # 位置を記録したフレーム数
    cornered_frames: int = 0    # 画面端に追い込まれていたフレーム数
    cornering_frames: int = 0   # 相手を画面端に追い込んでいたフレーム数
    dist_frames: list = field(default_factory=lambda: [0] * len(DIST_LABELS))   # DIST_LABELS ごとのフレーム数

    def position_summary(self) -> str:
        """「画面端に追い込まれていた時間 12.3% / …」形式。位置の記録が無ければ空文字"""
        if not self.pos_frames:
            return ""
        n = self.pos_frames
        bands = " / ".join(f"{label} {c / n * 100:.0f}%" for label, c in zip(DIST_LABELS, self.dist_frames))
        return (
            f"画面端に追い込まれていた時間 {self.cornered_frames / n * 100:.1f}% / "
            f"相手を追い込んでいた時間 {self.cornering_frames / n * 100:.1f}% / 距離の内訳: {bands}"
        )

    def spirit_summary(self) -> str:
        """「平均 4.2玉 / 1玉未満の時間 1.2% / ガードで削られた回数 1回」形式。記録が無ければ空文字"""
        if not self.sp_frames:
            return ""
        return (
            f"平均 {self.sp_sum / self.sp_frames / SPIRIT_ORB:.1f}玉 / "
            f"1玉未満の時間 {self.sp_low_frames / self.sp_frames * 100:.1f}% / "
            f"ガードで削られた回数 {self.sp_guard_drops}回"
        )

    def fwd_pct(self) -> float:
        return self.fwd_frames / self.total_frames * 100 if self.total_frames else 0.0

    def back_pct(self) -> float:
        return self.back_frames / self.total_frames * 100 if self.total_frames else 0.0

    def act_summary(self) -> str:
        """「打撃×58、射撃×19」形式。行動データが無ければ空文字"""
        return "、".join(
            f"{ACT_LABELS[k]}×{self.act_counts[k]}"
            for k in ACT_REPORT_ORDER if self.act_counts.get(k)
        )

    def active_ratio(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return (self.total_frames - self.neutral_frames) / self.total_frames * 100

    def hold_pct(self, key: str) -> float:
        if self.total_frames == 0:
            return 0.0
        val = getattr(self, f"held_{key}", 0)
        return val / self.total_frames * 100

    def press_pct(self, key: str) -> float:
        if self.total_frames == 0:
            return 0.0
        val = getattr(self, f"press_{key}", 0)
        return val / self.total_frames * 100


def parse_live_json(
    live_path: Path,
    p1_name: str = "P1",
    p2_name: str = "P2",
) -> tuple[list[tuple[float, float, float]], "LiveStats", "LiveStats", list[dict], dict]:
    """
    live_reader.py が出力した JSON を読み込む。
    Returns:
        hp_samples: [(sec, p1_hp_frac, p2_hp_frac), ...]
        p1_stats: LiveStats for P1
        p2_stats: LiveStats for P2
        frames: 生のフレームリスト
        meta: 記録時のメタ情報（version, matches など。古い記録では一部のみ）
    """
    raw = json.loads(live_path.read_text(encoding="utf-8-sig"))
    frames = raw.get("frames", [])
    hp_samples = frames_to_hp_samples(frames)
    p1, p2 = parse_live_stats_from_frames(frames, p1_name, p2_name)
    return hp_samples, p1, p2, frames, raw.get("meta", {})


# ────────────────────────────────────────────
# 連戦（複数試合）分割
# ────────────────────────────────────────────
MATCH_GAP_SEC = 5.0  # 試合間の記録空白（秒）。これより長いと別試合とみなす


@dataclass
class MatchData:
    match_num: int
    start_sec: float
    end_sec: float
    frames: list[dict]
    hp_samples: list[tuple[float, float, float]]
    rounds: list[RoundData]
    live_p1: LiveStats
    live_p2: LiveStats
    damage_events: list[DamageEvent]
    p1_won: Optional[bool] = None
    p1_char: Optional[str] = None
    p2_char: Optional[str] = None


def frames_to_hp_samples(frames: list[dict]) -> list[tuple[float, float, float]]:
    samples = []
    for frame in frames:
        t = frame["t"]
        hp1 = max(0.0, min(1.0, frame["p1"]["hp"] / HP_MAX_SOKU))
        hp2 = max(0.0, min(1.0, frame["p2"]["hp"] / HP_MAX_SOKU))
        samples.append((t, hp1, hp2))
    return samples


def parse_live_stats_from_frames(
    frames: list[dict],
    p1_name: str = "P1",
    p2_name: str = "P2",
) -> tuple[LiveStats, LiveStats]:
    p1 = LiveStats(p1_name=p1_name, p2_name=p2_name)
    p2 = LiveStats(p1_name=p1_name, p2_name=p2_name)

    # a/b/c/d は「押し続けているフレーム数」のカウンタ（押していなければ 0）。
    # 方向は x/y（同じくカウンタ、符号が向き）。x/y は記録 version 3 以降のみで、
    # それ以前の記録には先行入力バッファ（dir/hx/hy）しか無く、方向は集計できない。
    for side, ls in (("p1", p1), ("p2", p2)):
        other = "p2" if side == "p1" else "p1"
        prev: dict = {}
        for frame in frames:
            fd = frame[side]
            ls.total_frames += 1
            od = frame[other]
            if "px" in fd and "px" in od:
                ls.pos_frames += 1
                ls.cornered_frames += is_cornered(fd, od)
                ls.cornering_frames += is_cornered(od, fd)
                ls.dist_frames[distance_band(abs(fd["px"] - od["px"]))] += 1
            has_axis = "x" in fd
            if not has_axis:
                ls.has_axis = False
            x = fd.get("x", 0)
            y = fd.get("y", 0)
            if y < 0: ls.held_up    += 1
            if y > 0: ls.held_down  += 1
            if x < 0: ls.held_left  += 1
            if x > 0: ls.held_right += 1
            for key in "abcd":
                if fd.get(key):
                    setattr(ls, f"held_{key}", getattr(ls, f"held_{key}") + 1)
                    if not prev.get(key):
                        setattr(ls, f"press_{key}", getattr(ls, f"press_{key}") + 1)
            if not (x or y or any(fd.get(key) for key in "abcd")):
                ls.neutral_frames += 1
            combo = _combo_key(fd.get("combo", 0))
            if combo and not prev.get("combo"):
                ls.combo_counts[combo] = ls.combo_counts.get(combo, 0) + 1
            act = fd.get("act")
            if act in ACT_FORWARD:
                ls.fwd_frames += 1
            elif act in ACT_BACKWARD:
                ls.back_frames += 1
            if act != prev.get("act"):
                cat = act_category(act)
                # 被弾・ガードは1回の中でIDが次々変わる（のけぞり→ダウン→起き上がり）ので、
                # その状態に入った時だけ数える
                if cat in ("hit", "guard") and act_category(prev.get("act")) == cat:
                    cat = None
                if cat:
                    ls.act_counts[cat] = ls.act_counts.get(cat, 0) + 1
            sp = fd.get("sp")
            if sp is not None:
                ls.sp_frames += 1
                ls.sp_sum += sp
                if sp < SPIRIT_ORB:
                    ls.sp_low_frames += 1
                if (prev.get("sp", 0) - sp >= SPIRIT_GUARD_DROP
                        and act_category(act) == "guard"):
                    ls.sp_guard_drops += 1
            prev = fd

    return p1, p2


def split_live_frames_into_matches(
    frames: list[dict],
    gap_sec: float = MATCH_GAP_SEC,
) -> list[list[dict]]:
    """ライブ記録フレームを試合ごとに分割する。"""
    if not frames:
        return []

    if any("match" in f for f in frames):
        groups: dict[int, list[dict]] = {}
        for frame in frames:
            mid = frame.get("match", 1)
            groups.setdefault(mid, []).append(frame)
        return [groups[k] for k in sorted(groups.keys())]

    groups: list[list[dict]] = [[frames[0]]]
    for frame in frames[1:]:
        if frame["t"] - groups[-1][-1]["t"] > gap_sec:
            groups.append([frame])
        else:
            groups[-1].append(frame)
    return groups


def match_winner_from_rounds(rounds: list[RoundData]) -> Optional[bool]:
    p1_wins = sum(1 for r in rounds if r.p1_won is True)
    p2_wins = sum(1 for r in rounds if r.p1_won is False)
    if p1_wins > p2_wins:
        return True
    if p2_wins > p1_wins:
        return False
    return None


def build_match_data(
    match_num: int,
    frames: list[dict],
    p1_name: str,
    p2_name: str,
) -> MatchData:
    hp_samples = frames_to_hp_samples(frames)
    rounds = split_rounds(hp_samples)
    live_p1, live_p2 = parse_live_stats_from_frames(frames, p1_name, p2_name)
    damage_events = detect_damage_events(frames)
    return MatchData(
        match_num=match_num,
        start_sec=frames[0]["t"],
        end_sec=frames[-1]["t"],
        frames=frames,
        hp_samples=hp_samples,
        rounds=rounds,
        live_p1=live_p1,
        live_p2=live_p2,
        damage_events=damage_events,
        p1_won=match_winner_from_rounds(rounds),
    )


def generate_match_advice(
    match: MatchData,
    p1_name: str,
    p2_name: str,
    viewpoint: int = 1,
) -> list[str]:
    """1試合分のアドバイス（セッション比較は含めない）"""
    advice: list[str] = []
    prefix = f"【試合{match.match_num}】"
    self_name, opp_name, self_side, _ = _self_names(p1_name, p2_name, viewpoint)

    wins = sum(1 for r in match.rounds if _self_won_from_p1_result(r.p1_won, viewpoint) is True)
    losses = sum(1 for r in match.rounds if _self_won_from_p1_result(r.p1_won, viewpoint) is False)
    self_won = _self_won_from_p1_result(match.p1_won, viewpoint)
    if self_won is True:
        result = f"{self_name} の勝ち"
    elif self_won is False:
        result = f"{opp_name} の勝ち"
    else:
        result = "引き分け/不明"
    advice.append(f"{prefix} {result}（ラウンド {wins}勝{losses}敗）")

    if match.rounds:
        si = 0 if viewpoint == 1 else 1
        oi = 1 - si
        avg_min_self = sum(r.min_hp()[si] for r in match.rounds) / len(match.rounds)
        avg_min_opp = sum(r.min_hp()[oi] for r in match.rounds) / len(match.rounds)
        if avg_min_self < 0.2:
            advice.append(f"{prefix} HPが20%以下まで削られたラウンドが多いです。")
        if avg_min_opp > 0.5:
            advice.append(f"{prefix} 相手HPを半分以上残すラウンドが多いです。")

    for line in analyze_damage_patterns(match.damage_events, self_side, self_name):
        advice.append(f"{prefix} {line}")
    for line in analyze_offense(match.damage_events, "p2" if self_side == "p1" else "p1", self_name):
        advice.append(f"{prefix} {line}")

    self_stats = match.live_p1 if viewpoint == 1 else match.live_p2
    sub: list[str] = []
    _advice_from_live(sub, self_stats)
    for line in sub:
        advice.append(f"{prefix} {line}")

    return advice


def generate_session_advice(
    matches: list[MatchData],
    live_p1: Optional[LiveStats],
    live_p2: Optional[LiveStats],
    p1_name: str,
    p2_name: str,
    viewpoint: int = 1,
) -> list[str]:
    """連戦セッション全体のアドバイス"""
    advice: list[str] = []
    self_name, opp_name, self_side, _ = _self_names(p1_name, p2_name, viewpoint)
    match_wins = sum(1 for m in matches if _self_won_from_p1_result(m.p1_won, viewpoint) is True)
    match_losses = sum(1 for m in matches if _self_won_from_p1_result(m.p1_won, viewpoint) is False)
    advice.append(f"📋 セッション: {len(matches)}試合 {match_wins}勝{match_losses}敗（{self_name}視点）")
    advice.extend(session_starter_lines(matches, self_side, self_name))

    for match in matches:
        advice.extend(generate_match_advice(match, p1_name, p2_name, viewpoint))

    self_stats = live_p1 if viewpoint == 1 else live_p2
    opp_stats = live_p2 if viewpoint == 1 else live_p1
    if self_stats is not None and opp_stats is not None:
        advice.append(f"--- {self_name} vs {opp_name}（セッション全体） ---")
        _advice_from_comparison(advice, self_stats, opp_stats, self_name, opp_name)

    return advice


def _read_rep_players(data: bytes) -> tuple[list[Optional[str]], list[list[int]], int]:
    """.rep のヘッダから ([P1キャラ, P2キャラ], [P1デッキ, P2デッキ], 入力数の位置) を読む。"""
    chars: list[Optional[str]] = []
    decks: list[list[int]] = []
    pos = REP_P1_OFS
    for _ in range(2):
        char_no = data[pos]
        chars.append(REP_CHAR_IDS[char_no] if char_no < len(REP_CHAR_IDS) else None)
        deck = struct.unpack_from("<I", data, pos + 2)[0]
        if deck > REP_DECK_MAX:
            raise ValueError("deck size")
        decks.append(list(struct.unpack_from(f"<{deck}H", data, pos + 6)))
        pos += 6 + deck * 2 + 3
    return chars, decks, pos + 10 - 3


def read_rep_decks(rep_path: Path) -> list[list[int]]:
    """.rep から [P1デッキ, P2デッキ]（カード番号の並び）を読む。"""
    try:
        return _read_rep_players(rep_path.read_bytes())[1]
    except (IndexError, ValueError, struct.error):
        raise AnalyzeError(f".rep の形式が想定と違うため読めませんでした: {rep_path.name}")


def read_rep_inputs(rep_path: Path) -> tuple[list[tuple[int, ...]], list[Optional[str]]]:
    """.rep から (プレイヤーごとの入力列, [P1キャラ, P2キャラ]) を読む。
    入力列は対人なら [P1, P2]、対CPU なら [P1] だけ。"""
    data = rep_path.read_bytes()
    try:
        chars, _, pos = _read_rep_players(data)
        count = struct.unpack_from("<I", data, pos)[0]
        if len(data) - (pos + 4) != count * 2:
            raise ValueError("input count")
        inputs = struct.unpack_from(f"<{count}H", data, pos + 4)
    except (IndexError, ValueError, struct.error):
        raise AnalyzeError(f".rep の形式が想定と違うため読めませんでした: {rep_path.name}")

    if data[REP_MODE_OFS] == REP_MODE_VS_COM:
        return [inputs], chars
    return [inputs[0::2], inputs[1::2]], chars


def parse_rep(rep_path: Path, p1_name: str = "P1", p2_name: str = "P2") -> list[RepStats]:
    """.rep を読み、入力が入っているプレイヤーごとの RepStats を返す（対CPU なら P1 だけ）。"""
    streams, chars = read_rep_inputs(rep_path)
    result = []
    for player, stream in enumerate(streams, start=1):
        counts = {k: 0 for k in INPUT_LABELS}
        neutral = 0
        for v in stream:
            if v == 0:
                neutral += 1
            else:
                for bit in INPUT_LABELS:
                    if v & bit:
                        counts[bit] += 1
        result.append(RepStats(
            total_frames=len(stream),
            button_counts=counts,
            neutral_frames=neutral,
            p1_name=p1_name,
            p2_name=p2_name,
            active_player=player,
            p1_char=chars[0],
            p2_char=chars[1],
        ))
    return result


# ────────────────────────────────────────────
# カード（デッキ構成と使用回数）
# ────────────────────────────────────────────
# カード番号は 0〜99 がシステムカード（全キャラ共通）、100〜199 がスキルカード、200〜 がスペルカード。
# 名前は card_data.json から引く（ゲームの data/csv/<キャラ>/spellcard.csv から抜き出したもの）。
CARD_FILE = "card_data.json"
CARD_KINDS = [("spell", "スペルカード"), ("skill", "スキルカード"), ("system", "システムカード")]
# スペルカード 200〜219 を使った時のアクションID は 600〜619（実記録で 607↔207、602↔202 を確認）
ACT_SPELL_USE = range(600, 620)
# 1回使った後、このフレーム数のあいだは手札が減っても数えない
# （スペルカードのコストぶんのカードが、記録の上で1フレーム遅れて消えても二重に数えないため）
CARD_USE_GAP = 10

_CARD_CACHE: Optional[dict] = None


def _card_data() -> dict:
    global _CARD_CACHE
    if _CARD_CACHE is None:
        base = Path(sys._MEIPASS) if getattr(sys, "frozen", False) else Path(__file__).parent
        try:
            _CARD_CACHE = json.loads((base / CARD_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _CARD_CACHE = {}
    return _CARD_CACHE


def card_kind(card_id: int) -> str:
    return "system" if card_id < 100 else "skill" if card_id < 200 else "spell"


def card_name(char_id: Optional[str], card_id: int) -> str:
    """カード名。表に無い番号は「カード(123)」のように出す"""
    table = _card_data().get("common" if card_id < 100 else char_id or "", {})
    return table.get(str(card_id)) or f"カード({card_id})"


def count_card_uses(frames: list[dict]) -> Optional[dict[str, Counter]]:
    """両者が使ったカードの回数（キーはカード番号）。手札の記録（version 5 以降）が無ければ None。

    手札は先頭が次に使うカード。使うと先頭から消え、スペルカードはコストぶんのカードも
    同時に消える（実記録で確認）。なので手札の枚数が減った時の、直前の先頭のカードを数える。
    カードを送るだけの時は並びが回るだけ、引いた時は後ろに増えるだけで、どちらも数えない。
    天候で手札が入れ替わる時（実記録では両者同時に中身が変わった）も枚数は同じなので数えない。
    伊吹瓢は使った20フレームほど後に2枚引くので、使った時には1枚減る。
    """
    if not any("hand" in f["p1"] or "hand" in f["p2"] for f in frames):
        return None
    uses = {"p1": Counter(), "p2": Counter()}
    for side in ("p1", "p2"):
        prev: Optional[list] = None
        prev_match = None
        last_use = -CARD_USE_GAP
        for i, frame in enumerate(frames):
            hand = frame[side].get("hand")
            if frame.get("match") != prev_match:
                prev, prev_match = None, frame.get("match")
            if hand is None:
                continue
            if prev and len(hand) < len(prev) and i - last_use >= CARD_USE_GAP:
                # 被弾中に減るのは、相手の符蝕薬で先頭のカードを壊された時
                if act_category(frame[side].get("act")) != "hit":
                    uses[side][prev[0]] += 1
                    last_use = i
            prev = hand
    return uses


def count_spell_uses_from_acts(frames: list[dict]) -> dict[str, Counter]:
    """手札の入っていない古い記録用。アクションID から、使ったスペルカードだけ数える。"""
    uses = {"p1": Counter(), "p2": Counter()}
    for side in ("p1", "p2"):
        prev = None
        for frame in frames:
            act = frame[side].get("act")
            if act != prev and act in ACT_SPELL_USE:
                uses[side][act - ACT_SPELL_USE.start + 200] += 1
            prev = act
    return uses


@dataclass
class CardSummary:
    """レポートの「カード」欄1つぶん（1試合、またはデッキが同じ連戦の合計）"""
    chars: tuple                      # (P1キャラ, P2キャラ)
    decks: tuple = (None, None)       # (P1デッキ, P2デッキ)。分からない側は None
    uses: Optional[tuple] = None      # (P1, P2) の使用回数 Counter。分からない記録では None
    spells_only: bool = False         # 使用回数がスペルカードだけ（手札の無い古い記録）
    title: str = ""                   # 連戦で試合ごとに分ける時の見出し

    def has_data(self) -> bool:
        return any(self.decks) or bool(self.uses and any(self.uses))


def card_summaries_from_live(
    match_groups: list[list[dict]],
    recorded_matches: list[dict],
    match_chars: list[tuple],
) -> list[CardSummary]:
    """ライブ記録の試合ごとのカード欄。キャラもデッキも同じ連戦は、1つにまとめて合計を出す。"""
    rec_by_id = {m.get("match"): m for m in recorded_matches}
    summaries = []
    for n, (group, chars) in enumerate(zip(match_groups, match_chars), start=1):
        rec = rec_by_id.get(group[0].get("match"), {})
        uses = count_card_uses(group)
        spells_only = uses is None
        if uses is None:
            uses = count_spell_uses_from_acts(group)
        summaries.append(CardSummary(
            chars=chars,
            decks=(rec.get("p1_deck"), rec.get("p2_deck")),
            uses=(uses["p1"], uses["p2"]),
            spells_only=spells_only,
            title=f"試合{n}",
        ))
    if not summaries:
        return []

    def key(cs: CardSummary):
        return cs.chars, tuple(tuple(sorted(d)) if d else None for d in cs.decks), cs.spells_only

    if len({key(cs) for cs in summaries}) == 1:
        first = summaries[0]
        total = tuple(sum((cs.uses[i] for cs in summaries), Counter()) for i in range(2))
        summaries = [CardSummary(
            chars=first.chars, decks=first.decks, uses=total, spells_only=first.spells_only,
            title=f"全{len(summaries)}試合の合計" if len(summaries) > 1 else "",
        )]
    return [cs for cs in summaries if cs.has_data()]


# ────────────────────────────────────────────
# .rep と動画の時刻合わせ
# ────────────────────────────────────────────
# .rep には時刻も HP も無く、動画には入力が無い。そこで「攻撃ボタンを押した少し後に
# 相手の HP が減る」ことを手がかりに、ラウンドごとに一番よく重なるずれを探す。
#   動画の時刻(1/60秒単位) = rate × .rep のフレーム番号 + off
# 実記録で分かっていること:
#   - rate は 1.01〜1.02（.rep の 60 フレームは実時間で 1 秒より少し長い）
#   - .rep は対戦画面が出て約1〜2秒後から始まり、KO のたびに 70〜100 フレームぶん飛ぶ
#     （なので off はラウンドごとに増える）
#   - 同じ試合のライブ記録（入力がフレーム単位で突き合わせられる）で確かめた誤差は
#     平均 0.13 秒、最大 0.25 秒
SYNC_RATES = [1.0 + 0.002 * i for i in range(13)]   # 1.000〜1.024
SYNC_LAG_MIN, SYNC_LAG_MAX = -120, 1200             # off を探す範囲（1/60秒単位）
SYNC_MIN_Z = 2.5        # 重なりの山がこれより低いラウンドがあれば、合わせられなかったとみなす
SYNC_MARGIN = 15        # 被弾時刻のこのフレーム数手前までを「直前の入力」とする（誤差ぶんの余裕）
ATTACK_BITS = 0x70      # A / B / C


@dataclass
class RepSync:
    streams: list            # プレイヤーごとの入力列
    rate: float
    offsets: list[int]       # ラウンドごとの off
    scores: list[float]      # ラウンドごとの山の高さ（z 値）

    def rep_frame(self, round_index: int, sec: float) -> int:
        return int((sec * 60 - self.offsets[round_index]) / self.rate)

    def inputs_before(self, round_index: int, side: str, sec: float) -> list[dict]:
        """被弾時刻 sec の直前 DAMAGE_WINDOW_SEC ぶんの入力を、ライブ記録と同じ形の辞書で返す"""
        player = 0 if side == "p1" else 1
        if player >= len(self.streams):
            return []   # 対CPU戦の P2
        stream = self.streams[player]
        end = self.rep_frame(round_index, sec) - SYNC_MARGIN
        start = max(0, end - int(DAMAGE_WINDOW_SEC * 60))
        return [
            {
                "a": v & 0x10, "b": v & 0x20, "c": v & 0x40, "d": v & 0x80,
                "x": -1 if v & 0x04 else (1 if v & 0x08 else 0),
                "y": -1 if v & 0x01 else (1 if v & 0x02 else 0),
                "combo": 0,
            }
            for v in stream[start:max(start, min(end, len(stream)))]
        ]


def _gauss_smooth(x: np.ndarray, width: int = 5) -> np.ndarray:
    kernel = np.exp(-0.5 * (np.arange(-3 * width, 3 * width + 1) / width) ** 2)
    return np.convolve(x, kernel / kernel.sum(), mode="same")


def _damage_series(hp_samples: list[tuple[float, float, float]], col: int, n: int) -> np.ndarray:
    """hp_samples の col 番目（1=P1, 2=P2）の HP の減りを 1/60 秒単位の列にする"""
    d = np.zeros(n)
    for a, b in zip(hp_samples, hp_samples[1:]):
        drop = a[col] - b[col]
        if 0 < drop < 0.6:
            i0 = int(a[0] * 60)
            i1 = max(i0 + 1, int(b[0] * 60))
            d[i0:i1] += drop / (i1 - i0)
    return d


def align_rep_to_video(
    hp_samples: list[tuple[float, float, float]],
    rounds: list[RoundData],
    streams: list,
) -> Optional[RepSync]:
    """.rep の入力列を動画の時刻に合わせる。自信が持てない時は None。"""
    if not rounds or not streams or len(hp_samples) < 2:
        return None
    n = int(hp_samples[-1][0] * 60) + 60
    # (攻撃する側の入力列の番号, 食らう側の hp_samples 内の位置)
    pairs = [(0, 2)] + ([(1, 1)] if len(streams) > 1 else [])

    damage = {victim: _damage_series(hp_samples, victim, n) for _, victim in pairs}

    # 攻撃ボタンを押した瞬間の列
    attack = {}
    for attacker, _ in pairs:
        st = np.asarray(streams[attacker], dtype=np.int64)
        pressed = st & ATTACK_BITS
        attack[attacker] = ((pressed & ~np.concatenate(([0], pressed[:-1]))) != 0).astype(float)

    size = 1
    while size < n + int(len(streams[0]) * SYNC_RATES[-1]) + 1:
        size *= 2
    lags = np.arange(SYNC_LAG_MIN, SYNC_LAG_MAX)

    # ラウンドごと・向きごとの HP の減り（rate によらないので先に変換しておく）
    round_damage = []
    for r in rounds:
        s, e = int(r.start_sec * 60), int(r.end_sec * 60)
        per_pair = {}
        for _, victim in pairs:
            y = np.zeros(n)
            y[s:e] = damage[victim][s:e]
            if y.sum() > 0:
                per_pair[victim] = np.fft.rfft(_gauss_smooth(y) / y.sum(), size)
        round_damage.append(per_pair)

    best = None
    for rate in SYNC_RATES:
        m = int(len(streams[0]) * rate)
        index = (np.arange(m) / rate).astype(int)
        xs = {}
        for attacker, _ in pairs:
            a = attack[attacker]
            x = _gauss_smooth(a[np.minimum(index, len(a) - 1)])
            xs[attacker] = np.conj(np.fft.rfft(x / (x.mean() + 1e-9), size))
        total, offsets, scores = 0.0, [], []
        for per_pair in round_damage:
            curve = np.zeros(len(lags))
            for attacker, victim in pairs:
                if victim in per_pair:
                    # corr[lag] = Σ x[f]·y[f + lag]
                    corr = np.fft.irfft(per_pair[victim] * xs[attacker], size)
                    curve += corr[lags % size]
            k = int(curve.argmax())
            total += curve[k]
            offsets.append(int(lags[k]))
            scores.append(float((curve[k] - curve.mean()) / (curve.std() + 1e-9)))
        if best is None or total > best[0]:
            best = (total, rate, offsets, scores)

    _, rate, offsets, scores = best
    # 山が低い、または off がラウンドを追って増えていない時は、合っていないとみなす
    if min(scores) < SYNC_MIN_Z or any(b < a for a, b in zip(offsets, offsets[1:])):
        return None
    return RepSync(streams=streams, rate=rate, offsets=offsets, scores=scores)


# ────────────────────────────────────────────
# ライブ記録と動画の時刻合わせ
# ────────────────────────────────────────────
# ライブ記録と動画は別々に撮るので開始位置が合わない（動画は .rep を後から再生して
# 録ることもある）。どちらにも HP があるので、ラウンドごとに HP の減り方が一番よく
# 重なるずれを探す。
#   動画の時刻(1/60秒単位) = rate × ライブ記録の時刻(1/60秒単位) + off
# 動画に写っていないラウンド（連戦の記録に 1 試合ぶんの動画、など）は合わせない。
LIVE_SYNC_RATES = [0.99 + 0.002 * i for i in range(11)]   # 0.990〜1.010
LIVE_SYNC_MIN_CORR = 0.5    # 重なりの相関がこれ未満のラウンドは、動画に写っていないとみなす


@dataclass
class LiveVideoSync:
    rate: float
    starts: list[float]              # ラウンドごとの開始時刻（ライブ記録の秒）
    offsets: list[Optional[int]]     # ラウンドごとの off。合わせられなかったラウンドは None
    scores: list[float]              # ラウンドごとの相関（0〜1）

    def video_sec(self, live_sec: float) -> Optional[float]:
        """ライブ記録の時刻を動画の再生位置に直す。合わせられなかったラウンドなら None"""
        k = max(0, sum(1 for s in self.starts if s <= live_sec) - 1)
        off = self.offsets[k]
        if off is None:
            return None
        return (live_sec * 60 * self.rate + off) / 60


def align_live_to_video(
    live_hp: list[tuple[float, float, float]],
    video_hp: list[tuple[float, float, float]],
    rounds: list[RoundData],
) -> Optional[LiveVideoSync]:
    """ライブ記録の時刻を動画に合わせる。どのラウンドも合わなければ None。"""
    if not rounds or len(live_hp) < 2 or len(video_hp) < 2:
        return None
    n_live = int(live_hp[-1][0] * 60) + 60
    n_video = int(video_hp[-1][0] * 60) + 60
    size = 1
    while size < n_video + int(n_live * LIVE_SYNC_RATES[-1]) + 1:
        size *= 2

    cols = (1, 2)
    live = {c: _damage_series(live_hp, c, n_live) for c in cols}
    video = {c: _gauss_smooth(_damage_series(video_hp, c, n_video)) for c in cols}
    video_fft = {c: np.fft.rfft(video[c], size) for c in cols}

    best = None
    for rate in LIVE_SYNC_RATES:
        m = int(n_live * rate)
        index = np.minimum((np.arange(m) / rate).astype(int), n_live - 1)
        offsets: list[Optional[int]] = []
        scores: list[float] = []
        for r in rounds:
            s, e = int(r.start_sec * 60 * rate), min(m, int(r.end_sec * 60 * rate) + 1)
            xs = {}
            for c in cols:
                x = np.zeros(m)
                x[s:e] = live[c][index[s:e]]
                xs[c] = _gauss_smooth(x)
            x_norm = np.sqrt(sum(float((xs[c] ** 2).sum()) for c in cols))
            if x_norm == 0:
                offsets.append(None)
                scores.append(0.0)
                continue
            # corr[lag] = Σ x[f]·y[f + lag]
            corr = sum(np.fft.irfft(video_fft[c] * np.conj(np.fft.rfft(xs[c], size)), size) for c in cols)
            k = int(corr.argmax())
            lag = k if k <= n_video else k - size
            # 重なった区間どうしの相関（1 に近いほど同じ減り方）
            a, b = max(0, s + lag), min(n_video, e + lag)
            y_norm = np.sqrt(sum(float((video[c][a:b] ** 2).sum()) for c in cols)) if b > a else 0.0
            score = float(corr[k] / (x_norm * y_norm)) if y_norm > 0 else 0.0
            offsets.append(lag if score >= LIVE_SYNC_MIN_CORR else None)
            scores.append(score)
        total = sum(sc for sc, off in zip(scores, offsets) if off is not None)
        if best is None or total > best[0]:
            best = (total, rate, offsets, scores)

    _, rate, offsets, scores = best
    if all(off is None for off in offsets):
        return None
    return LiveVideoSync(rate=rate, starts=[r.start_sec for r in rounds], offsets=offsets, scores=scores)


def generate_advice(
    rounds: list[RoundData],
    rep: Optional[RepStats],
    live_p1: Optional["LiveStats"] = None,
    live_p2: Optional["LiveStats"] = None,
    damage_events: Optional[list[DamageEvent]] = None,
    viewpoint: int = 1,
    p1_name: str = "P1",
    p2_name: str = "P2",
) -> list[str]:
    """解析結果から日本語アドバイスを生成"""
    advice = []
    self_name, opp_name, self_side, _ = _self_names(p1_name, p2_name, viewpoint)

    wins = sum(1 for r in rounds if _self_won_from_p1_result(r.p1_won, viewpoint) is True)
    losses = sum(1 for r in rounds if _self_won_from_p1_result(r.p1_won, viewpoint) is False)
    if wins + losses > 0:
        advice.append(f"ラウンド結果: {wins}勝{losses}敗（{self_name}視点）")

    if rounds:
        si = 0 if viewpoint == 1 else 1
        oi = 1 - si
        avg_min_self = sum(r.min_hp()[si] for r in rounds) / len(rounds)
        avg_min_opp = sum(r.min_hp()[oi] for r in rounds) / len(rounds)
        if avg_min_self < 0.2:
            advice.append("⚠️ 自分のHPが何度も20%以下まで削られています。被ダメージを抑える立ち回りを意識しましょう。")
        if avg_min_opp > 0.5:
            advice.append("💡 相手のHPを50%以上残しているラウンドが多いです。攻めの手を増やせると勝率が上がりそうです。")

    if damage_events:
        for line in analyze_damage_patterns(damage_events, self_side, self_name):
            advice.append(line)
        advice.extend(analyze_offense(damage_events, "p2" if self_side == "p1" else "p1", self_name))

    self_stats = live_p1 if viewpoint == 1 else live_p2
    opp_stats = live_p2 if viewpoint == 1 else live_p1
    if self_stats is not None:
        _advice_from_live(advice, self_stats)
        if opp_stats is not None:
            _advice_from_comparison(advice, self_stats, opp_stats, self_name, opp_name)
        return advice

    if rep is None:
        advice.append("📝 .rep / ライブ記録ファイルが見つからなかったため、入力分析はスキップしました。")
        return advice

    # 対CPU戦の .rep には P1 の入力しか入っていない
    if rep.active_player != viewpoint:
        recorded = p2_name if rep.active_player == 2 else p1_name
        advice.append(
            f"📝 この .rep には {recorded} 側の入力しか入っていません。"
            f"視点を {recorded}（{'P2・右' if rep.active_player == 2 else 'P1・左'}）にすると入力分析できます。"
        )
        return advice

    # ── 攻撃ボタン使用比率
    z_pct = rep.button_pct(0x10)
    x_pct = rep.button_pct(0x20)
    c_pct = rep.button_pct(0x40)
    a_pct = rep.button_pct(0x80)

    # 押下率は行動の回数と一致しないので、評価はせず数値だけ出す
    advice.append(
        f"🎮 入力発生率 {rep.active_ratio():.1f}% / "
        f"ボタン押下率: A {z_pct:.1f}% / B {x_pct:.1f}% / C {c_pct:.1f}% / D {a_pct:.1f}%"
    )

    return advice


def _advice_from_live(advice: list[str], ls: "LiveStats") -> None:
    """ライブ記録データ（P1）に基づくアドバイスを advice リストに追加"""
    tf = ls.total_frames
    if tf == 0:
        return

    a_pct  = ls.hold_pct("a")
    b_pct  = ls.hold_pct("b")
    c_pct  = ls.hold_pct("c")
    d_pct  = ls.hold_pct("d")
    active_pct = ls.active_ratio()

    # ボタンを押している時間の割合は、行動の回数とは一致しない
    # （D をほとんど押さずにダッシュ・飛翔を何十回もしている記録がある）。
    # 多い少ないの評価はせず、数値だけ出す。
    advice.append(
        f"🎮 入力発生率 {active_pct:.1f}% / "
        f"ボタン押下率: A {a_pct:.1f}% / B {b_pct:.1f}% / C {c_pct:.1f}% / D {d_pct:.1f}%"
    )

    moving = ls.fwd_frames + ls.back_frames
    if moving >= 60 and ls.back_frames / moving > 0.65:
        advice.append(f"💡 歩き・ダッシュのうち後退が{ls.back_frames / moving * 100:.0f}%です。下がりすぎて画面端を背負っていないか確認しましょう。")

    if ls.sp_frames:
        advice.append(f"🔷 霊力: {ls.spirit_summary()}")

    if ls.pos_frames:
        advice.append(f"📍 位置: {ls.position_summary()}")

    # 行動内訳（アクションIDがある記録では、コマンドコードより確かなのでこちらを使う）
    if ls.act_counts:
        advice.append(f"🎯 行動内訳: {ls.act_summary()}")
        melee = ls.act_counts.get("melee", 0)
        bullet = ls.act_counts.get("bullet", 0)
        if melee + bullet >= 20:
            if bullet < melee * 0.2:
                advice.append(f"💡 打撃{melee}回に対して射撃が{bullet}回です。射撃での牽制を混ぜると近づきやすくなります。")
            elif melee < bullet * 0.2:
                advice.append(f"💡 射撃{bullet}回に対して打撃が{melee}回です。射撃を当てた後の攻め込みを増やせるか見てみましょう。")
        return

    # コマンド入力
    combos = ls.combo_counts
    if combos:
        total_combos = sum(combos.values())
        top = sorted(combos.items(), key=lambda x: -x[1])[:3]
        combo_desc = "、".join(
            f"{COMBO_NAMES.get(k, f'コマンド({k})')}×{v}"
            for k, v in top
        )
        advice.append(f"🎯 特殊コマンド入力: {combo_desc}（合計{total_combos}回）")

        # 236(波動) vs 214(逆波動) バランス
        fwd = combos.get(2, 0)    # 236
        rev = combos.get(32, 0)   # 214
        if fwd + rev > 10:
            ratio_f = fwd / (fwd + rev)
            if ratio_f > 0.7:
                advice.append(f"💡 前方コマンド（236）が多め（{fwd}回 vs 逆{rev}回）。逆方向の技も状況に応じて使えると幅が広がります。")
            elif ratio_f < 0.3:
                advice.append(f"💡 逆方向コマンド（214）が多め（{rev}回 vs 前{fwd}回）。前方向コマンドも混ぜてみましょう。")


def _advice_from_comparison(
    advice: list[str],
    self_ls: "LiveStats",
    opp_ls: "LiveStats",
    self_name: str,
    opp_name: str,
) -> None:
    """自分 vs 相手 の入力を比較してアドバイスを追加"""
    # 自分の数値は _advice_from_live で出している。相手の分も評価はせず数値だけ出す
    advice.append(
        f"📊 {opp_name} の入力発生率 {opp_ls.active_ratio():.1f}% / ボタン押下率: "
        + " / ".join(f"{k.upper()} {opp_ls.hold_pct(k):.1f}%" for k in "abcd")
    )

    c_self = sum(self_ls.combo_counts.values())
    c_opp = sum(opp_ls.combo_counts.values())
    if opp_ls.sp_frames:
        advice.append(f"📊 {opp_name} の霊力: {opp_ls.spirit_summary()}")
    if self_ls.act_counts and opp_ls.act_counts:
        advice.append(f"📊 相手の行動内訳: {opp_ls.act_summary()}")
        g_self = self_ls.act_counts.get("guard", 0)
        h_self = self_ls.act_counts.get("hit", 0)
        if g_self + h_self >= 5 and g_self < h_self * 0.3:
            advice.append(f"⚠️ ガード{g_self}回に対して、やられ状態に{h_self}回なっています。ガードできる場面が無いか見直してみましょう。")
    elif c_self + c_opp > 0:
        advice.append(f"🎯 コマンド入力数: {self_name} {c_self}回 vs {opp_name} {c_opp}回")
        if c_opp > c_self * 1.5 and c_opp > 5:
            advice.append("  相手のコマンド技が多め。相手の特殊技に対する対策を考えてみましょう。")

    move_self = self_ls.fwd_frames + self_ls.back_frames
    move_opp = opp_ls.fwd_frames + opp_ls.back_frames
    if move_self >= 60 and move_opp >= 60:
        fwd_self = self_ls.fwd_frames / move_self
        fwd_opp = opp_ls.fwd_frames / move_opp
        if fwd_opp - fwd_self > 0.2:
            advice.append(
                f"📊 相手({opp_name})はより積極的に前進しています。"
                f"間合い管理で先手を取られやすい状況です。"
            )


# ────────────────────────────────────────────
# HTML レポート生成
# ────────────────────────────────────────────
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>Soku Advisor — {title}</title>
{chartjs_tag}
<style>
:root{{
  --bg:#0f0f1a; --surface:#1a1a2e; --accent:#e8c060; --text:#e0e0e0;
  --p1:#4a9eff; --p2:#ff6b6b; --win:#4caf50; --lose:#f44336;
}}
*{{box-sizing:border-box; margin:0; padding:0}}
body{{background:var(--bg); color:var(--text); font-family:'Segoe UI',sans-serif; padding:24px}}
h1{{color:var(--accent); font-size:1.6rem; margin-bottom:4px}}
.subtitle{{color:#888; font-size:.9rem; margin-bottom:24px}}
.grid{{display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:24px}}
.card{{background:var(--surface); border-radius:10px; padding:16px}}
.card h2{{font-size:1rem; color:var(--accent); margin-bottom:12px; border-bottom:1px solid #333; padding-bottom:6px}}
.round-badge{{display:inline-block; padding:3px 10px; border-radius:20px; font-size:.8rem; font-weight:bold}}
.win{{background:var(--win); color:#fff}}
.lose{{background:var(--lose); color:#fff}}
.draw{{background:#888; color:#fff}}
.round-row{{display:flex; align-items:center; gap:10px; margin-bottom:8px; font-size:.85rem}}
.round-chart{{height:140px; margin-bottom:12px}}
canvas{{max-height:200px}}
.advice-list{{list-style:none; padding:0}}
.advice-list li{{background:#16162a; border-left:3px solid var(--accent); padding:10px 14px;
                 margin-bottom:8px; border-radius:0 6px 6px 0; font-size:.88rem; line-height:1.5}}
.stat-grid{{display:grid; grid-template-columns:repeat(4,1fr); gap:8px}}
.stat-box{{background:#16162a; border-radius:6px; padding:10px; text-align:center}}
.stat-val{{font-size:1.4rem; font-weight:bold; color:var(--accent)}}
.stat-lbl{{font-size:.7rem; color:#888; margin-top:2px}}
.bar-wrap{{margin-bottom:6px}}
.bar-label{{font-size:.75rem; color:#aaa; display:flex; justify-content:space-between; margin-bottom:2px}}
.bar-bg{{background:#222; border-radius:3px; height:10px; overflow:hidden}}
.bar-fill{{height:100%; border-radius:3px; transition:width .3s}}
.p1-color{{background:var(--p1)}}
.p2-color{{background:var(--p2)}}
table th, table td{{padding:4px 8px; border-bottom:1px solid #222}}
.match-section{{margin-bottom:20px;padding-top:8px;border-top:1px solid #333}}
.match-section h3{{color:var(--accent);font-size:.95rem;margin-bottom:8px}}
.session-table{{width:100%;font-size:.85rem;border-collapse:collapse;margin-top:8px}}
.ai-text p{{margin-bottom:12px;line-height:1.65;font-size:.9rem;color:#ddd}}
.ai-text ul{{margin:8px 0 8px 20px;font-size:.88rem;color:#ccc}}
.video-card{{position:sticky; top:0; z-index:5; margin-bottom:16px; text-align:center; box-shadow:0 4px 12px rgba(0,0,0,.5)}}
.video-card video{{max-width:100%; max-height:42vh; background:#000; border-radius:6px}}
.video-card .hint{{color:#888; font-size:.75rem; margin-top:4px}}
a.seek{{color:var(--accent); cursor:pointer; text-decoration:underline dotted}}
a.seek:hover{{color:#fff}}
@media(max-width:700px){{.grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<h1>🎮 Soku Advisor</h1>
<div class="subtitle">{title} &nbsp;·&nbsp; {duration}{session_label}{viewpoint_label}</div>

{video_section}

<div class="grid">
<div class="card">
<h2>{summary_title}</h2>
{round_rows}
</div>
<div class="card">
<h2>💡 アドバイス</h2>
<ul class="advice-list">
{advice_items}
</ul>
</div>
</div>

{hp_chart_section}

{ai_section}

{input_section}

<script>
// 動画をその再生位置の少し手前から再生する（動画が無いレポートでは何もしない）
function seekTo(sec) {{
  const v = document.getElementById('matchVideo');
  if (!v) return;
  v.currentTime = Math.max(0, sec - {seek_lead});
  v.play();
}}
document.querySelectorAll('a.seek').forEach(a => {{
  a.addEventListener('click', () => seekTo(parseFloat(a.dataset.t)));
}});
{chart_scripts}
</script>
</body>
</html>
"""

def _build_card_table(cs: CardSummary, player: int, name: str) -> str:
    """1プレイヤー分のカード表。name はエスケープ済み"""
    char = cs.chars[player]
    deck = Counter(cs.decks[player] or [])
    uses = cs.uses[player] if cs.uses else Counter()
    show_deck, show_uses = bool(deck), cs.uses is not None
    head = '<th style="text-align:left">カード</th>'
    head += "<th>デッキ</th>" if show_deck else ""
    head += "<th>使用</th>" if show_uses else ""
    rows = ""
    for kind, label in CARD_KINDS:
        ids = sorted(c for c in set(deck) | set(uses) if card_kind(c) == kind)
        if not ids:
            continue
        rows += f'<tr><td colspan="3" style="color:#aaa">{label}</td></tr>'
        for c in ids:
            rows += f"<tr><td>{html.escape(card_name(char, c))}</td>"
            if show_deck:
                count = f"{deck[c]}枚" if deck[c] else "—"
                rows += f'<td style="text-align:center">{count}</td>'
            if show_uses:
                used = f"<b>{uses[c]}回</b>" if uses[c] else '<span style="color:#555">0回</span>'
                rows += f'<td style="text-align:center">{used}</td>'
            rows += "</tr>"
    if not rows:
        rows = '<tr><td colspan="3" style="color:#888">記録がありません</td></tr>'
    total = f" — 使用 計{sum(uses.values())}回" if show_uses else ""
    return (
        f'<div><div style="font-size:.85rem;margin-bottom:4px">{name}'
        f'（{html.escape(char_display_name(char))}）{total}</div>'
        f'<table class="session-table"><tr>{head}</tr>{rows}</table></div>'
    )


def _build_card_section(
    summaries: list[CardSummary], p1_name: str, p2_name: str, viewpoint: int,
) -> str:
    """デッキ構成とカードの使用回数。名前はエスケープ済みのものを受け取る"""
    if not summaries:
        return ""
    order = (0, 1) if viewpoint == 1 else (1, 0)   # 自分を左に出す
    names = (p1_name, p2_name)
    body = ""
    for cs in summaries:
        if cs.title:
            body += f'<div class="match-section"><h3>{html.escape(cs.title)}</h3>'
        body += '<div class="grid" style="margin-bottom:0">'
        body += "".join(_build_card_table(cs, i, names[i]) for i in order)
        body += "</div>"
        if cs.title:
            body += "</div>"
    notes = []
    if any(cs.uses is None for cs in summaries):
        notes.append(".rep にはデッキしか入っていないため、使用回数は出ません（ライブ記録を使った解析で出ます）")
    if any(cs.spells_only for cs in summaries):
        notes.append("この記録には手札が入っていないため、使用回数はスペルカードだけ・デッキは無しです"
                     "（新しい版で取ったライブ記録なら全部出ます）")
    note = "".join(f'<div style="color:#888;font-size:.75rem;margin-top:8px">{n}</div>' for n in notes)
    return f"""
<div class="card" style="margin-top:16px">
<h2>🃏 カード（デッキ構成と使用回数）</h2>
{body}{note}
</div>"""


def _build_live_input_section(ls: "LiveStats", name: str, color_class: str) -> str:
    """ライブ記録の1プレイヤー分の入力セクションHTML"""
    tf = ls.total_frames
    dur = tf // 60

    bars = (
        _pct_bar("A（打撃）",      ls.hold_pct("a"),     color_class) +
        _pct_bar("B（弱射撃）",    ls.hold_pct("b"),     color_class) +
        _pct_bar("C（強射撃）",    ls.hold_pct("c"),     color_class) +
        _pct_bar("D（飛翔）",      ls.hold_pct("d"),     color_class)
    )
    if ls.has_axis:
        bars += (
            _pct_bar("↑",              ls.hold_pct("up"),    "p2-color") +
            _pct_bar("↓",              ls.hold_pct("down"),  "p2-color") +
            _pct_bar("←",              ls.hold_pct("left"),  "p2-color") +
            _pct_bar("→",              ls.hold_pct("right"), "p2-color")
        )

    # コマンド入力テーブル
    combo_rows = ""
    if ls.combo_counts:
        top_combos = sorted(ls.combo_counts.items(), key=lambda x: -x[1])[:8]
        for code, count in top_combos:
            cname = COMBO_NAMES.get(code, f"コマンド({code})")
            combo_rows += f'<tr><td>{cname}</td><td style="text-align:right;color:var(--accent)">{count}回</td></tr>'
        combo_html = f"""
<div style="margin-top:12px">
<div style="font-size:.8rem;color:#aaa;margin-bottom:6px">特殊コマンド入力</div>
<table style="width:100%;font-size:.82rem;border-collapse:collapse">
{combo_rows}
</table>
</div>"""
    else:
        combo_html = '<div style="font-size:.8rem;color:#888;margin-top:8px">コマンド入力データなし</div>'

    # 行動回数テーブル（アクションIDがある記録ではコマンド表の代わりに出す）
    if ls.act_counts:
        act_rows = "".join(
            f'<tr><td>{ACT_LABELS[k]}</td>'
            f'<td style="text-align:right;color:var(--accent)">{ls.act_counts.get(k, 0)}回</td></tr>'
            for k in ACT_REPORT_ORDER
        )
        combo_html = f"""
<div style="margin-top:12px">
<div style="font-size:.8rem;color:#aaa;margin-bottom:6px">行動回数</div>
<table style="width:100%;font-size:.82rem;border-collapse:collapse">
{act_rows}
</table>
</div>"""

    return f"""
<div class="card" style="margin-top:16px">
<h2>🎮 入力分析（{name} — ライブ記録）</h2>
<div class="stat-grid" style="margin-bottom:16px">
  <div class="stat-box"><div class="stat-val">{dur}s</div><div class="stat-lbl">記録時間</div></div>
  <div class="stat-box"><div class="stat-val">{ls.active_ratio():.0f}%</div><div class="stat-lbl">入力発生率</div></div>
  <div class="stat-box"><div class="stat-val">{ls.press_a}</div><div class="stat-lbl">A（打撃）回数</div></div>
  <div class="stat-box"><div class="stat-val">{ls.act_counts.get("skill", 0) if ls.act_counts else sum(ls.combo_counts.values())}</div><div class="stat-lbl">{"スキル回数" if ls.act_counts else "コマンド総数"}</div></div>
</div>
{bars}
{combo_html}
</div>"""


def _pct_bar(label: str, pct: float, css_class: str) -> str:
    width = min(pct * 3.5, 100)  # scale for visual
    return (f'<div class="bar-wrap"><div class="bar-label"><span>{label}</span>'
            f'<span>{pct:.1f}%</span></div>'
            f'<div class="bar-bg"><div class="bar-fill {css_class}" style="width:{width:.0f}%"></div></div></div>')


def _dual_bar(label: str, v1: float, v2: float, max_val: float = 50.0) -> str:
    """P1とP2を並べた比較バー"""
    w1 = min(v1 / max_val * 100, 100)
    w2 = min(v2 / max_val * 100, 100)
    return (
        f'<div class="bar-wrap">'
        f'<div class="bar-label"><span>{label}</span>'
        f'<span style="color:var(--p1)">{v1:.1f}%</span>'
        f'&nbsp;/&nbsp;<span style="color:var(--p2)">{v2:.1f}%</span></div>'
        f'<div style="display:flex;gap:4px">'
        f'<div style="flex:1"><div class="bar-bg"><div class="bar-fill p1-color" style="width:{w1:.0f}%"></div></div></div>'
        f'<div style="flex:1"><div class="bar-bg"><div class="bar-fill p2-color" style="width:{w2:.0f}%"></div></div></div>'
        f'</div></div>'
    )


def _build_comparison_section(
    self_ls: "LiveStats",
    opp_ls: "LiveStats",
    self_name: str,
    opp_name: str,
    viewpoint: int = 1,
) -> str:
    """自分 vs 相手 入力比較セクションHTML"""
    self_fwd, self_back = self_ls.fwd_pct(), self_ls.back_pct()
    opp_fwd, opp_back = opp_ls.fwd_pct(), opp_ls.back_pct()

    rows = (
        _dual_bar("A（打撃）",     self_ls.hold_pct("a"), opp_ls.hold_pct("a")) +
        _dual_bar("B（弱射撃）",   self_ls.hold_pct("b"), opp_ls.hold_pct("b")) +
        _dual_bar("C（強射撃）",   self_ls.hold_pct("c"), opp_ls.hold_pct("c")) +
        _dual_bar("D（飛翔）",     self_ls.hold_pct("d"), opp_ls.hold_pct("d")) +
        _dual_bar("入力発生率",    self_ls.active_ratio(), opp_ls.active_ratio(), max_val=100.0)
    )
    if self_ls.act_counts and opp_ls.act_counts:
        rows += (
            _dual_bar("前進",          self_fwd, opp_fwd) +
            _dual_bar("後退",          self_back, opp_back)
        )

    all_codes = set(self_ls.combo_counts) | set(opp_ls.combo_counts)
    combo_rows = ""
    for code in sorted(all_codes, key=lambda c: -(self_ls.combo_counts.get(c, 0) + opp_ls.combo_counts.get(c, 0))):
        cname = COMBO_NAMES.get(code, f"コマンド({code})")
        c_self = self_ls.combo_counts.get(code, 0)
        c_opp = opp_ls.combo_counts.get(code, 0)
        combo_rows += (
            f'<tr><td>{cname}</td>'
            f'<td style="text-align:right;color:var(--p1)">{c_self}回</td>'
            f'<td style="text-align:right;color:var(--p2)">{c_opp}回</td></tr>'
        )

    table_title, first_col = "特殊コマンド比較", "コマンド"
    if self_ls.act_counts and opp_ls.act_counts:
        # アクションIDがある記録では行動回数で比べる
        table_title, first_col = "行動回数比較", "行動"
        combo_rows = "".join(
            f'<tr><td>{ACT_LABELS[k]}</td>'
            f'<td style="text-align:right;color:var(--p1)">{self_ls.act_counts.get(k, 0)}回</td>'
            f'<td style="text-align:right;color:var(--p2)">{opp_ls.act_counts.get(k, 0)}回</td></tr>'
            for k in ACT_REPORT_ORDER
        )

    combo_table = ""
    if combo_rows:
        combo_table = f"""
<div style="margin-top:14px">
<div style="font-size:.8rem;color:#aaa;margin-bottom:6px">{table_title}</div>
<table style="width:100%;font-size:.82rem;border-collapse:collapse">
<tr><th style="text-align:left;color:#888">{first_col}</th>
<th style="text-align:right;color:var(--p1)">{self_name}（あなた）</th>
<th style="text-align:right;color:var(--p2)">{opp_name}（相手）</th></tr>
{combo_rows}
</table>
</div>"""

    return f"""
<div class="card" style="margin-top:16px">
<h2>📊 入力比較（あなた vs 相手）</h2>
<div style="font-size:.78rem;color:#888;margin-bottom:10px">
  <span style="color:var(--p1)">■ {self_name}（あなた）</span>&nbsp;&nbsp;
  <span style="color:var(--p2)">■ {opp_name}（相手）</span>
</div>
{rows}
{combo_table}
</div>"""


def _result_badge(
    p1_won: Optional[bool],
    p1_name: str,
    p2_name: str,
    viewpoint: int,
) -> str:
    self_won = _self_won_from_p1_result(p1_won, viewpoint)
    self_name, opp_name, _, _ = _self_names(p1_name, p2_name, viewpoint)
    if self_won is True:
        return f'<span class="round-badge win">{self_name} 勝ち</span>'
    if self_won is False:
        return f'<span class="round-badge lose">{opp_name} 勝ち</span>'
    return '<span class="round-badge draw">不明</span>'


SEEK_LEAD_SEC = 2.0     # レポートから動画へ飛ぶ時、その場面の何秒手前から再生するか

CHARTJS_FILE = "chart.umd.min.js"   # 同梱している Chart.js（MIT License）
CHARTJS_CDN = "https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"


def _chartjs_tag() -> str:
    """グラフ用の Chart.js を読み込む <script>。

    ネットにつながっていなくてもグラフが出るよう、同梱のファイルをレポートに埋め込む。
    ファイルが見つからない時だけ、ネット上のものを読みに行く。
    """
    base = Path(sys._MEIPASS) if getattr(sys, "frozen", False) else Path(__file__).parent
    try:
        code = (base / CHARTJS_FILE).read_text(encoding="utf-8")
    except OSError:
        return f'<script src="{CHARTJS_CDN}"></script>'
    code = re.sub(r"\n?//# sourceMappingURL=\S+\s*$", "", code)
    return f"<script>{code}</script>"

# 記録の時刻 → 動画の再生位置（秒）。動画に写っていない時刻は None。
# 動画だけの解析ではそのまま返し、動画が無い・時刻合わせできなかった時は関数自体が None。
ToVideo = Optional[Callable[[float], Optional[float]]]


def _fmt_clock(sec: float) -> str:
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def _seek_link(video_sec: float, text: str) -> str:
    return f'<a class="seek" data-t="{video_sec:.1f}">{text}</a>'


def _video_src(video_path: Path, out_path: Path) -> str:
    """レポートから見た動画の場所。同じドライブなら相対パス（フォルダごと動かしても切れない）"""
    try:
        rel = os.path.relpath(video_path.resolve(), out_path.resolve().parent)
        return quote(rel.replace(os.sep, "/"))
    except ValueError:      # ドライブが違う
        return video_path.resolve().as_uri()


def _build_round_rows_html(
    rounds: list[RoundData],
    p1_name: str,
    p2_name: str,
    viewpoint: int = 1,
    to_video: ToVideo = None,
) -> str:
    html = ""
    for r in rounds:
        dur = r.end_sec - r.start_sec
        p1f, p2f = r.final_hps()
        if r.incomplete:
            badge = '<span class="round-badge draw">途中まで</span>'
        else:
            badge = _result_badge(r.p1_won, p1_name, p2_name, viewpoint)
        start = to_video(r.start_sec) if to_video else None
        # ラウンドの頭はそのまま見たいので、手前に戻るぶんを足しておく
        jump = "" if start is None else " " + _seek_link(start + SEEK_LEAD_SEC, f"動画 {_fmt_clock(start)}〜")
        html += (
            f'<div class="round-row">'
            f'<b>R{r.round_num}</b> {badge}'
            f'<span style="color:#888;font-size:.8rem">{dur:.0f}秒 / '
            f'{p1_name} {p1f*100:.0f}% vs {p2_name} {p2f*100:.0f}%{jump}</span>'
            f'</div>\n'
        )
    return html or '<div style="color:#888;font-size:.85rem">ラウンドなし</div>'


def _hp_data_to_json(
    all_hp: list[tuple[float, float, float]],
    max_points: int = 600,
    to_video: ToVideo = None,
) -> tuple[str, bool]:
    """グラフ用の [時刻, P1, P2] の並びと、その時刻が動画の再生位置かどうかを返す。

    動画に写っていない区間が混ざる時は、横軸が途中で変わらないよう記録の時刻のままにする。
    """
    step = max(1, len(all_hp) // max_points)
    sampled = all_hp[::step]
    video_time = False
    if to_video:
        times = [to_video(t) for t, _, _ in sampled]
        if all(v is not None for v in times):
            sampled = [(v, p1, p2) for v, (_, p1, p2) in zip(times, sampled)]
            video_time = True
    data = json.dumps([[round(t, 1), round(p1, 3), round(p2, 3)] for t, p1, p2 in sampled])
    return data, video_time


def _js_str(label: str) -> str:
    """HTMLエスケープ済みの名前を <script> 内に埋め込めるJS文字列リテラルにする"""
    return json.dumps(html.unescape(label), ensure_ascii=False).replace("</", "<\\/")


def _build_hp_chart_script(
    chart_id: str,
    hp_json: str,
    p1_label: str,
    p2_label: str,
    video_time: bool = False,
) -> str:
    # 横軸が動画の再生位置の時は 分:秒 で出し、グラフをクリックするとその場面へ飛ぶ
    if video_time:
        label_js = "d => Math.floor(d[0] / 60) + ':' + String(Math.floor(d[0] % 60)).padStart(2, '0')"
        click_js = """
      onClick: (evt, _els, chart) => {
        const hit = chart.getElementsAtEventForMode(evt, 'index', {intersect: false}, false);
        if (hit.length) seekTo(hpData[hit[0].index][0]);
      },"""
    else:
        label_js = "d => d[0].toFixed(1) + 's'"
        click_js = ""
    return f"""
(function() {{
  const el = document.getElementById('{chart_id}');
  if (!el) return;
  const hpData = {hp_json};
  new Chart(el.getContext('2d'), {{
    type: 'line',
    data: {{
      labels: hpData.map({label_js}),
      datasets: [
        {{label: {_js_str(p1_label)}, data: hpData.map(d => (d[1]*100).toFixed(1)),
          borderColor: '#4a9eff', backgroundColor: 'rgba(74,158,255,.1)',
          tension: .3, pointRadius: 0, fill: true}},
        {{label: {_js_str(p2_label)}, data: hpData.map(d => (d[2]*100).toFixed(1)),
          borderColor: '#ff6b6b', backgroundColor: 'rgba(255,107,107,.1)',
          tension: .3, pointRadius: 0, fill: true}},
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,{click_js}
      scales: {{
        x: {{ticks: {{maxTicksLimit: 10, color: '#888'}}, grid: {{color:'#222'}}}},
        y: {{min: 0, max: 100, ticks: {{color:'#888', callback: v => v+'%'}}, grid: {{color:'#222'}}}}
      }},
      plugins: {{legend: {{labels: {{color:'#e0e0e0'}}}}}}
    }}
  }});
}})();
"""


def _build_session_summary_html(
    matches: list[MatchData],
    p1_name: str,
    p2_name: str,
    viewpoint: int = 1,
) -> str:
    rows = ""
    for m in matches:
        dur = m.end_sec - m.start_sec
        rw = sum(1 for r in m.rounds if _self_won_from_p1_result(r.p1_won, viewpoint) is True)
        rl = sum(1 for r in m.rounds if _self_won_from_p1_result(r.p1_won, viewpoint) is False)
        badge = _result_badge(m.p1_won, p1_name, p2_name, viewpoint)
        rows += (
            f"<tr><td>試合{m.match_num}</td><td>{badge}</td>"
            f"<td>{rw}勝{rl}敗</td><td>{dur:.0f}秒</td>"
            f"<td>{m.start_sec:.0f}s〜{m.end_sec:.0f}s</td></tr>"
        )
    return f"""
<table class="session-table">
<tr><th>#</th><th>結果</th><th>ラウンド</th><th>時間</th><th>記録位置</th></tr>
{rows}
</table>"""


def _build_multi_match_chart_section(
    matches: list[MatchData],
    p1_name: str,
    p2_name: str,
    viewpoint: int = 1,
    to_video: ToVideo = None,
) -> tuple[str, str]:
    section = '<div class="card" style="margin-bottom:16px"><h2>❤️ 試合別 HP 推移</h2>'
    scripts = ""
    self_name, opp_name, _, _ = _self_names(p1_name, p2_name, viewpoint)
    for m in matches:
        chart_id = f"hpChart{m.match_num}"
        self_won = _self_won_from_p1_result(m.p1_won, viewpoint)
        if self_won is True:
            result = f"{self_name} 勝ち"
        elif self_won is False:
            result = f"{opp_name} 勝ち"
        else:
            result = "不明"
        if m.p1_char or m.p2_char:
            result += html.escape(
                f" / {char_display_name(m.p1_char)} vs {char_display_name(m.p2_char)}"
            )
        section += (
            f'<div class="match-section">'
            f'<h3>試合 {m.match_num} — {result} '
            f'({m.start_sec:.0f}s〜{m.end_sec:.0f}s)</h3>'
            f'{_build_round_rows_html(m.rounds, p1_name, p2_name, viewpoint, to_video)}'
            f'<canvas id="{chart_id}" style="max-height:150px;margin-top:10px"></canvas>'
            f'</div>'
        )
        hp_json, video_time = _hp_data_to_json(m.hp_samples, max_points=300, to_video=to_video)
        scripts += _build_hp_chart_script(chart_id, hp_json, p1_name, p2_name, video_time)
    section += "</div>"
    return section, scripts


def build_html(
    video_path: Optional[Path],
    rounds: list[RoundData],
    all_hp: list[tuple[float, float, float]],
    rep: Optional[RepStats],
    advice: list[str],
    p1_name: str,
    p2_name: str,
    live_p1: Optional["LiveStats"] = None,
    live_p2: Optional["LiveStats"] = None,
    matches: Optional[list[MatchData]] = None,
    viewpoint: int = 1,
    ai_text: str = "",
    char_label: str = "",
    rep_opp: Optional[RepStats] = None,
    video_src: str = "",
    to_video: ToVideo = None,
    card_summaries: Optional[list[CardSummary]] = None,
) -> str:
    """video_src（レポートから見た動画の場所）があれば動画を埋め込み、to_video で
    再生位置が分かる時刻をクリックで飛べるようにする。"""
    if not video_src:
        to_video = None
    # 以降の HTML 断片はすべてエスケープ済みの名前で組み立てる
    p1_name, p2_name = html.escape(p1_name), html.escape(p2_name)
    title = html.escape(video_path.stem) if video_path else "ライブ記録分析"
    total_sec = all_hp[-1][0] if all_hp else 0
    duration_str = f"{int(total_sec//60)}分{int(total_sec%60)}秒"
    multi = matches is not None and len(matches) > 1
    self_name, opp_name, _, _ = _self_names(p1_name, p2_name, viewpoint)
    viewpoint_label = _viewpoint_label(p1_name, p2_name, viewpoint)
    if char_label:
        viewpoint_label += f" &nbsp;·&nbsp; {html.escape(char_label)}"
    self_stats = live_p1 if viewpoint == 1 else live_p2
    opp_stats = live_p2 if viewpoint == 1 else live_p1

    if multi:
        session_label = f" &nbsp;·&nbsp; {len(matches)}試合"
        summary_title = "📋 セッション結果"
        round_rows_html = _build_session_summary_html(matches, p1_name, p2_name, viewpoint)
        hp_chart_section, chart_scripts = _build_multi_match_chart_section(
            matches, p1_name, p2_name, viewpoint, to_video,
        )
    else:
        session_label = ""
        summary_title = "📊 ラウンド結果"
        round_rows_html = _build_round_rows_html(rounds, p1_name, p2_name, viewpoint, to_video)
        hp_json, video_time = _hp_data_to_json(all_hp, to_video=to_video)
        axis_note = "（横軸は動画の再生位置。クリックでその場面へ）" if video_time else ""
        hp_chart_section = (
            '<div class="card" style="margin-bottom:16px">'
            f'<h2>❤️ HP 推移{axis_note}</h2>'
            '<canvas id="hpChart" style="max-height:200px"></canvas></div>'
        )
        chart_scripts = _build_hp_chart_script("hpChart", hp_json, p1_name, p2_name, video_time)

    # 大ダメージの行の時刻を、動画のその場面へ飛ぶリンクにする。
    # 「動画」と付いた時刻は再生位置。ライブ記録が無い解析では、付いていなくても動画の時刻
    video_only = live_p1 is None and live_p2 is None and not multi

    def _linkify(line: str) -> str:
        escaped = html.escape(line)
        if not video_src:
            return escaped
        m = re.search(r"大ダメージ: (動画 )?((\d+\.\d)秒(?:（\d+:\d\d）)?)", escaped)
        if not m or not (m.group(1) or video_only):
            return escaped
        return escaped[:m.start(2)] + _seek_link(float(m.group(3)), m.group(2)) + escaped[m.end(2):]

    advice_html = "\n".join(f"<li>{_linkify(a)}</li>" for a in advice)

    video_section = ""
    if video_src:
        video_section = (
            '<div class="card video-card">'
            f'<video id="matchVideo" src="{html.escape(video_src, quote=True)}" controls preload="metadata"></video>'
            '<div class="hint">点線の付いた時刻や HP グラフをクリックすると、その場面の'
            f'{SEEK_LEAD_SEC:.0f}秒手前から再生します</div></div>'
        )

    # ── カード → 入力セクション
    input_section = _build_card_section(card_summaries or [], p1_name, p2_name, viewpoint)
    for rs, color in ((rep, "p1-color"), (rep_opp, "p2-color")):
        if not rs:
            continue
        active_name = p1_name if rs.active_player == 1 else p2_name
        a_pct = rs.button_pct(0x10)
        d_pct = rs.button_pct(0x80)
        bars = (
            _pct_bar("A（打撃）",   a_pct, color) +
            _pct_bar("B（弱射撃）", rs.button_pct(0x20), color) +
            _pct_bar("C（強射撃）", rs.button_pct(0x40), color) +
            _pct_bar("D（飛翔）",   d_pct, color) +
            _pct_bar("↑",           rs.button_pct(0x01), "p2-color") +
            _pct_bar("↓",           rs.button_pct(0x02), "p2-color") +
            _pct_bar("←",           rs.button_pct(0x04), "p2-color") +
            _pct_bar("→",           rs.button_pct(0x08), "p2-color")
        )

        input_section += f"""
<div class="card" style="margin-top:16px">
<h2>🎮 入力分析（{active_name} — .rep）</h2>
<div class="stat-grid" style="margin-bottom:16px">
  <div class="stat-box"><div class="stat-val">{rs.total_frames//60}s</div><div class="stat-lbl">記録時間</div></div>
  <div class="stat-box"><div class="stat-val">{rs.active_ratio():.0f}%</div><div class="stat-lbl">入力発生率</div></div>
  <div class="stat-box"><div class="stat-val">{a_pct:.1f}%</div><div class="stat-lbl">A使用率</div></div>
  <div class="stat-box"><div class="stat-val">{d_pct:.1f}%</div><div class="stat-lbl">D使用率</div></div>
</div>
{bars}
</div>"""

    # ── ライブ記録データがある場合は比較セクション + 各プレイヤー詳細を追加
    if live_p1 is not None and live_p2 is not None:
        input_section += _build_comparison_section(
            self_stats, opp_stats, self_name, opp_name, viewpoint,
        )
    if self_stats is not None:
        input_section += _build_live_input_section(
            self_stats, f"{self_name}（あなた）", "p1-color",
        )
    if opp_stats is not None:
        input_section += _build_live_input_section(
            opp_stats, f"{opp_name}（相手）", "p2-color",
        )

    ai_section = _format_ai_section(ai_text)

    return HTML_TEMPLATE.format(
        title=title,
        duration=duration_str,
        session_label=session_label,
        viewpoint_label=viewpoint_label,
        summary_title=summary_title,
        chartjs_tag=_chartjs_tag(),
        video_section=video_section,
        seek_lead=SEEK_LEAD_SEC,
        round_rows=round_rows_html,
        advice_items=advice_html,
        hp_chart_section=hp_chart_section,
        ai_section=ai_section,
        chart_scripts=chart_scripts,
        input_section=input_section,
    )


def _format_ai_section(text: str) -> str:
    if not text.strip():
        return ""
    parts = []
    for block in html.escape(text.strip()).split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if block.startswith("- ") or block.startswith("* "):
            items = "".join(f"<li>{ln[2:].strip()}</li>" for ln in block.splitlines() if ln.strip())
            parts.append(f"<ul>{items}</ul>")
        else:
            parts.append(f"<p>{block.replace(chr(10), '<br>')}</p>")
    body = "\n".join(parts)
    return f"""
<div class="card" style="margin-bottom:16px">
<h2>🤖 AIコーチング</h2>
<div class="ai-text">{body}</div>
</div>"""


# ────────────────────────────────────────────
# メイン処理
# ────────────────────────────────────────────
def read_video_hp(video_path: Path) -> list[tuple[float, float, float]]:
    """動画から HP_SAMPLE_SEC ごとに両者の HP を読む"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise AnalyzeError(f"動画を開けませんでした: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # HPバーの座標は 640×480 前提。違う解像度をそのまま読むと、でたらめな値になる
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    need_resize = (width, height) != (FRAME_W, FRAME_H)
    if need_resize:
        if height == 0 or abs(width / height - FRAME_W / FRAME_H) > 0.02:
            cap.release()
            raise AnalyzeError(
                f"動画の解像度 {width}×{height} には対応していません"
                f"（4:3 の動画が必要です。{FRAME_W}×{FRAME_H} 推奨）"
            )
        print(f"  解像度 {width}×{height} → {FRAME_W}×{FRAME_H} に縮小して解析します"
              f"（HPの読み取り精度が落ちる場合があります）")
    print(f"  動画: {fps:.1f}fps / {total_frames}フレーム ({total_frames/fps:.1f}秒)")

    hp: list[tuple[float, float, float]] = []
    frame_idx = 0
    sample_every = max(1, round(fps * HP_SAMPLE_SEC))

    print("  HPバー解析中...", end="", flush=True)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every == 0:
            if need_resize:
                frame = cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
            p1_hp, p2_hp = detect_hp(frame)
            # フレーム間隔が一定でない動画がある（rep_movier の録画で、番号÷fps とは最大 0.5 秒ずれた）
            # ので、フレームの時刻を使う。取れない時だけ番号から計算する
            msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            sec = msec / 1000 if msec > 0 or frame_idx == 0 else frame_idx / fps
            hp.append((sec, p1_hp, p2_hp))
        frame_idx += 1
        if frame_idx % (sample_every * 100) == 0:
            pct = frame_idx / total_frames * 100
            print(f"\r  HPバー解析中... {pct:.0f}%", end="", flush=True)
    cap.release()
    hp = smooth_hp(hp)
    print(f"\r  HPバー解析完了: {len(hp)}サンプル")
    return hp


def analyze(
    video_path: Optional[Path],
    rep_path: Optional[Path],
    out_path: Path,
    live_path: Optional[Path] = None,
    viewpoint: int = 1,
    p1_char: Optional[str] = None,
    p2_char: Optional[str] = None,
    use_ai: bool = False,
    player_name: Optional[str] = None,
    use_history: bool = False,
) -> None:
    # ── プレイヤー名・キャラの決定
    explicit_p1_char, explicit_p2_char = p1_char, p2_char
    p1_name, p2_name = "P1", "P2"
    if video_path:
        p1_name, p2_name, fc1, fc2 = parse_matchup(video_path.stem)
        p1_char = p1_char or fc1
        p2_char = p2_char or fc2
        print(f"[解析開始] {video_path.name}")
    elif live_path:
        p1_name, p2_name, fc1, fc2 = parse_matchup(live_path.stem)
        p1_char = p1_char or fc1
        p2_char = p2_char or fc2
        print(f"[解析開始] ライブ記録モード: {live_path.name}")
    print(f"  P1: {p1_name}  /  P2: {p2_name}")
    self_name, opp_name, _, _ = _self_names(p1_name, p2_name, viewpoint)
    self_char = p1_char if viewpoint == 1 else p2_char
    opp_char = p2_char if viewpoint == 1 else p1_char
    print(f"  解説視点: {self_name}（{'P1・左' if viewpoint == 1 else 'P2・右'}）")
    if self_char or opp_char:
        print(f"  キャラ: {char_display_name(self_char)} vs {char_display_name(opp_char)}")

    all_hp: list[tuple[float, float, float]] = []
    live_p1_stats: Optional[LiveStats] = None
    live_p2_stats: Optional[LiveStats] = None

    # ── ライブJSON 解析（HPデータも取得）
    live_frames: list[dict] = []
    recorded_matches: list[dict] = []
    if live_path and live_path.exists():
        print(f"  ライブ記録解析: {live_path.name}")
        all_hp, live_p1_stats, live_p2_stats, live_frames, live_meta = parse_live_json(live_path, p1_name, p2_name)
        # 記録時にメモリから読んだキャラ。手動指定が無ければファイル名からの推定より優先する。
        # 連戦の途中でキャラを変えた時は、全体としては一番多く使ったキャラを採る
        recorded_matches = live_meta.get("matches") or []
        rec_p1 = [m["p1_char"] for m in recorded_matches if m.get("p1_char")]
        rec_p2 = [m["p2_char"] for m in recorded_matches if m.get("p2_char")]
        if not explicit_p1_char and rec_p1:
            p1_char = Counter(rec_p1).most_common(1)[0][0]
        if not explicit_p2_char and rec_p2:
            p2_char = Counter(rec_p2).most_common(1)[0][0]
        self_char = p1_char if viewpoint == 1 else p2_char
        opp_char = p2_char if viewpoint == 1 else p1_char
        if rec_p1 or rec_p2:
            label = "キャラ（手動指定を優先）" if explicit_p1_char or explicit_p2_char else "記録キャラ"
            print(f"    {label}: {char_display_name(p1_char)} vs {char_display_name(p2_char)}")
        print(f"    フレーム数: {live_p1_stats.total_frames} ({live_p1_stats.total_frames/60:.1f}秒)")
        print(f"    P1入力発生率: {live_p1_stats.active_ratio():.1f}%")
        print(f"    P2入力発生率: {live_p2_stats.active_ratio():.1f}%")

    # ── 動画 HP サンプリング
    video_hp: list[tuple[float, float, float]] = []
    if video_path and not all_hp:
        all_hp = read_video_hp(video_path)
    elif video_path and all_hp:
        # ライブデータのHP + 動画の両方ある場合はライブデータ優先。
        # 動画の HP は、レポートに動画の再生位置を出すための時刻合わせにだけ使う
        print(f"  動画は {video_path.name} を参照（HPはライブデータ優先）")
        try:
            video_hp = read_video_hp(video_path)
        except AnalyzeError as e:
            print(f"  動画を読めなかったため、動画の再生位置は出しません: {e}")

    if not all_hp:
        raise AnalyzeError("HPデータが取得できませんでした（動画かライブJSONが必要です）")

    # ── 試合分割（連戦対応）
    match_groups = split_live_frames_into_matches(live_frames) if live_frames else []
    matches: list[MatchData] = []
    if len(match_groups) > 1:
        matches = [
            build_match_data(i + 1, group, p1_name, p2_name)
            for i, group in enumerate(match_groups)
        ]
        # 試合ごとのキャラ（手動指定 > その試合の記録 > 全体の値）
        rec_by_id = {m.get("match"): m for m in recorded_matches}
        for m, group in zip(matches, match_groups):
            rec = rec_by_id.get(group[0].get("match"), {})
            m.p1_char = explicit_p1_char or rec.get("p1_char") or p1_char
            m.p2_char = explicit_p2_char or rec.get("p2_char") or p2_char
        print(f"  試合検出: {len(matches)}試合")
        for m in matches:
            result = "勝" if m.p1_won else ("負" if m.p1_won is False else "?")
            rw = sum(1 for r in m.rounds if r.p1_won is True)
            rl = sum(1 for r in m.rounds if r.p1_won is False)
            print(f"    試合{m.match_num}: {m.start_sec:.1f}s〜{m.end_sec:.1f}s "
                  f"({result}, ラウンド{rw}勝{rl}敗)")

    # ── ラウンド分割（セッション全体）
    rounds = split_rounds(all_hp)
    print(f"  ラウンド検出: {len(rounds)}ラウンド")
    for r in rounds:
        result = "勝" if r.p1_won else ("負" if r.p1_won is False else "?")
        print(f"    R{r.round_num}: {r.start_sec:.1f}s〜{r.end_sec:.1f}s ({result})")

    # ── .rep 解析（ライブデータがない場合のフォールバック）
    rep: Optional[RepStats] = None
    rep_opp: Optional[RepStats] = None
    rep_streams: list = []
    if rep_path and rep_path.exists() and live_p1_stats is None:
        print(f"  .rep 解析: {rep_path.name}")
        reps = parse_rep(rep_path, p1_name, p2_name)
        rep_streams, _ = read_rep_inputs(rep_path)
        for rs in reps:
            print(f"    P{rs.active_player} 入力発生率: {rs.active_ratio():.1f}%（{rs.total_frames}フレーム）")
        if len(reps) == 1:
            print("    対CPU戦の記録のため、P2 の入力はありません")
        # 視点側を rep に、もう一方を rep_opp に。視点側が無ければ（対CPUでP2視点）ある方を rep に
        rep = next((rs for rs in reps if rs.active_player == viewpoint), reps[0])
        rep_opp = next((rs for rs in reps if rs is not rep), None)
        # キャラは .rep のヘッダが確実。手動指定が無ければファイル名からの推定より優先する
        if not explicit_p1_char and rep.p1_char:
            p1_char = rep.p1_char
        if not explicit_p2_char and rep.p2_char:
            p2_char = rep.p2_char
        print(f"    記録キャラ: {char_display_name(p1_char)} vs {char_display_name(p2_char)}")
        self_char = p1_char if viewpoint == 1 else p2_char
        opp_char = p2_char if viewpoint == 1 else p1_char

    # ── カード（デッキ構成と使用回数）
    card_summaries: list[CardSummary] = []
    if live_frames:
        card_summaries = card_summaries_from_live(
            match_groups, recorded_matches,
            [(m.p1_char, m.p2_char) for m in matches] or [(p1_char, p2_char)],
        )
    elif rep_streams:
        card_summaries = [CardSummary(chars=(p1_char, p2_char), decks=tuple(read_rep_decks(rep_path)))]
    for cs in card_summaries:
        if cs.uses:
            label = f"{cs.title}: " if cs.title else ""
            kind = "スペルカード" if cs.spells_only else "カード"
            print(f"  {label}{kind}使用: P1 {sum(cs.uses[0].values())}回 / P2 {sum(cs.uses[1].values())}回")

    # ── ダメージイベント検出（ライブデータ時のみ）
    damage_events: list[DamageEvent] = []
    if live_frames and len(matches) <= 1:
        damage_events = detect_damage_events(live_frames)
        print(f"  ダメージイベント検出: P1被弾{sum(1 for e in damage_events if e.target=='p1')}件 / P2被弾{sum(1 for e in damage_events if e.target=='p2')}件")
    elif not live_frames:
        # 動画だけの時: HP の減り方から被弾の時刻と大きさだけ出す
        rep_sync = align_rep_to_video(all_hp, rounds, rep_streams) if rep_streams else None
        if rep_sync:
            print(f"  .rep と動画の時刻合わせ: rate {rep_sync.rate:.3f} / "
                  f"ラウンドごとのずれ {[round(o / 60, 1) for o in rep_sync.offsets]}秒")
        elif rep_streams:
            print("  .rep と動画の時刻合わせ: 合わせられなかったため、被弾直前の入力は出しません")
        damage_events = detect_damage_events_from_hp(rounds, rep_sync)
        print(f"  ダメージイベント検出（動画のHPから）: P1被弾{sum(1 for e in damage_events if e.target=='p1')}件 / P2被弾{sum(1 for e in damage_events if e.target=='p2')}件")
    elif matches:
        total_p1 = sum(sum(1 for e in m.damage_events if e.target == "p1") for m in matches)
        total_p2 = sum(sum(1 for e in m.damage_events if e.target == "p2") for m in matches)
        print(f"  ダメージイベント検出: P1被弾{total_p1}件 / P2被弾{total_p2}件")

    # ── ライブ記録と動画の時刻合わせ（両方ある時のみ）
    # to_video: 記録の時刻 → 動画の再生位置。動画だけの解析では記録の時刻がそのまま再生位置
    to_video: ToVideo = (lambda sec: sec) if video_path and not live_frames else None
    if video_hp:
        live_sync = align_live_to_video(all_hp, video_hp, rounds)
        if live_sync:
            to_video = live_sync.video_sec
            print(f"  ライブ記録と動画の時刻合わせ: rate {live_sync.rate:.3f} / ラウンドごとのずれ "
                  f"{[None if o is None else round(o / 60, 1) for o in live_sync.offsets]}秒")
            for e in damage_events + [e for m in matches for e in m.damage_events]:
                e.video_sec = live_sync.video_sec(e.time_sec)
        else:
            print("  ライブ記録と動画の時刻合わせ: 合わせられなかったため、動画の再生位置は出しません")

    # ── アドバイス生成
    if len(matches) > 1:
        advice = generate_session_advice(
            matches, live_p1_stats, live_p2_stats, p1_name, p2_name, viewpoint,
        )
    else:
        advice = generate_advice(
            rounds, rep, live_p1_stats, live_p2_stats, damage_events,
            viewpoint=viewpoint, p1_name=p1_name, p2_name=p2_name,
        )

    # ── キャラ別アドバイス
    self_stats = live_p1_stats if viewpoint == 1 else live_p2_stats
    char_lines = generate_char_advice(self_char, opp_char, self_stats)
    if char_lines:
        advice.append("--- キャラ別アドバイス ---")
        advice.extend(char_lines)

    # ── 履歴への保存 / 傾向コンテキストの取得
    history_context: Optional[dict] = None
    if use_history:
        from player_history import save_session, build_trend_context
        _hist_name = player_name or self_name

        # 今回のセッションのラウンド勝敗集計
        if matches and len(matches) > 1:
            _rw = sum(
                sum(1 for r in m.rounds if _self_won_from_p1_result(r.p1_won, viewpoint) is True)
                for m in matches
            )
            _rl = sum(
                sum(1 for r in m.rounds if _self_won_from_p1_result(r.p1_won, viewpoint) is False)
                for m in matches
            )
        else:
            _rw = sum(1 for r in rounds if _self_won_from_p1_result(r.p1_won, viewpoint) is True)
            _rl = sum(1 for r in rounds if _self_won_from_p1_result(r.p1_won, viewpoint) is False)

        # 被弾イベント
        _self_side = "p1" if viewpoint == 1 else "p2"
        _all_dmg = damage_events + [e for m in matches for e in m.damage_events] if matches else damage_events
        _self_dmg = [e for e in _all_dmg if e.target == _self_side]
        _avg_dmg = (
            sum(e.damage_pct for e in _self_dmg) / len(_self_dmg) * 100
            if _self_dmg else 0.0
        )

        # 連戦では各行の頭に「【試合N】 」が付くので、外してから種類を見る
        _plain = [re.sub(r"^【試合\d+】\s*", "", a) for a in advice]
        _notable = [a for a in _plain if a.startswith(("⚠️", "💡"))][:5]

        save_session(
            _hist_name,
            source=str((live_path or video_path).resolve()),
            self_char=self_char,
            opp_char=opp_char,
            round_wins=_rw,
            round_losses=_rl,
            stats=_stats_summary(self_stats) if self_stats else None,
            damage_events_count=len(_self_dmg),
            avg_damage_pct=_avg_dmg,
            notable_advice=_notable,
        )

        if use_ai:
            history_context = build_trend_context(_hist_name)
            if history_context:
                print(f"  [履歴] {history_context['total_sessions_recorded']} セッション分の傾向を参照")
            else:
                print("  [履歴] 蓄積セッションが少ないため傾向分析はスキップ（2回以上で有効）")

    # ── AI 自然文解説（任意）
    ai_text = ""
    if use_ai:
        if not ai_available():
            print("  [AI] SOKU_AI_API_KEY が未設定のためスキップ")
            advice.append("🤖 AI解説: APIキー（SOKU_AI_API_KEY）が未設定です")
        else:
            print("  [AI] 自然文コーチング生成中...")
            match_payloads = []
            if matches:
                for m in matches:
                    sw = _self_won_from_p1_result(m.p1_won, viewpoint)
                    match_payloads.append({
                        "match": m.match_num,
                        "result": "win" if sw else ("lose" if sw is False else "unknown"),
                        "rounds": len(m.rounds),
                        "self_char": char_display_name(m.p1_char if viewpoint == 1 else m.p2_char),
                        "opp_char": char_display_name(m.p2_char if viewpoint == 1 else m.p1_char),
                        "duration_sec": round(m.end_sec - m.start_sec, 1),
                    })
            else:
                rw = sum(1 for r in rounds if _self_won_from_p1_result(r.p1_won, viewpoint) is True)
                rl = sum(1 for r in rounds if _self_won_from_p1_result(r.p1_won, viewpoint) is False)
                match_payloads.append({"rounds_wl": f"{rw}勝{rl}敗", "round_count": len(rounds)})

            payload = build_ai_payload(
                self_name=self_name,
                opp_name=opp_name,
                self_char=self_char,
                opp_char=opp_char,
                viewpoint=viewpoint,
                rule_advice=advice,
                match_summaries=match_payloads,
                self_stats_summary=_stats_summary(self_stats),
                history_context=history_context,
            )
            ai_text = generate_ai_advice(payload)
            if ai_text:
                print("  [AI] 生成完了")
            else:
                advice.append("🤖 AI解説: 生成に失敗しました（接続/APIを確認）")

    # ── HTML 出力
    html = build_html(
        video_path, rounds, all_hp, rep, advice, p1_name, p2_name,
        live_p1=live_p1_stats, live_p2=live_p2_stats,
        matches=matches if len(matches) > 1 else None,
        viewpoint=viewpoint,
        ai_text=ai_text,
        rep_opp=rep_opp,
        video_src=_video_src(video_path, out_path) if video_path else "",
        to_video=to_video,
        card_summaries=card_summaries,
        # 試合ごとにキャラが違う時は、見出しには出さず各試合の欄に出す
        char_label=(
            f"{char_display_name(p1_char)} vs {char_display_name(p2_char)}"
            if (p1_char or p2_char)
            and len({(m.p1_char, m.p2_char) for m in matches}) <= 1 else ""
        ),
    )
    out_path.write_text(html, encoding="utf-8")
    print(f"  レポート出力: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Soku Advisor — 東方非想天則 リプレイ分析ツール")
    parser.add_argument("video", nargs="?", help="入力 .mp4 ファイル（--live のみの場合は省略可）")
    parser.add_argument("rep", nargs="?", help="入力 .rep ファイル（省略可）")
    parser.add_argument("-o", "--output", help="出力 HTML パス（省略時は動画と同ディレクトリ）")
    parser.add_argument("--live", help="soku_live_reader.py で記録した JSON ファイル")
    parser.add_argument(
        "--viewpoint", type=int, choices=[1, 2], default=1,
        help="解説視点: 1=P1（左）, 2=P2（右）。自分がどちら側か指定",
    )
    parser.add_argument("--p1-char", help="P1キャラID（例: aya）。省略時はファイル名から推定")
    parser.add_argument("--p2-char", help="P2キャラID（例: yuyuko）")
    parser.add_argument("--ai", action="store_true", help="AI自然文コーチングを生成（要 SOKU_AI_API_KEY）")
    args = parser.parse_args()

    if not args.video and not args.live:
        parser.print_help()
        sys.exit(1)

    video_path: Optional[Path] = None
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"ERROR: {video_path} が見つかりません", file=sys.stderr)
            sys.exit(1)

    live_path: Optional[Path] = None
    if args.live:
        live_path = Path(args.live)
        if not live_path.exists():
            print(f"ERROR: ライブ記録ファイル {live_path} が見つかりません", file=sys.stderr)
            sys.exit(1)

    # .rep 自動探索
    rep_path: Optional[Path] = None
    if args.rep:
        rep_path = Path(args.rep)
    elif video_path:
        candidate = video_path.with_suffix(".rep")
        if candidate.exists():
            rep_path = candidate
            print(f"  .rep 自動検出: {rep_path.name}")

    # 出力先の決定
    if args.output:
        out_path = Path(args.output)
    elif video_path:
        out_path = video_path.with_suffix(".html")
    elif live_path:
        out_path = live_path.with_suffix(".html")
    else:
        out_path = Path("soku_report.html")

    try:
        analyze(
            video_path, rep_path, out_path, live_path,
            viewpoint=args.viewpoint,
            p1_char=normalize_char_id(args.p1_char) if args.p1_char else None,
            p2_char=normalize_char_id(args.p2_char) if args.p2_char else None,
            use_ai=args.ai,
        )
    except AnalyzeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
