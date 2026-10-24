"""
coaching.py — 改善課題・練習メニュー・次の対戦で意識すること・前回の課題の振り返り

analyzer.py が検出した被弾（DamageEvent）とガードの集計（LiveStats）から、
「何を、どう練習するか」を組み立てる。外部のサービスは使わない。

考え方:
  - 課題の候補は、すでに集計している切り口（自分の状況・相手の始動技・負けた自分の技・
    誤ガード・画面端・当てた時のダメージ）から作る。新しく推測で原因を作らない。
  - 数字（回数・ダメージ・何試合で起きたか）は記録から直接分かる「確認済み」。
    原因は、記録で確かめられた範囲だけ「推定」として書き、分からなければ「特定できない」と書く。
  - 練習メニューは、練習モードが無くても通常の対戦とリプレイの見返しだけでできる手順にする。
    コンボルートや「正解の行動」は記録から確かめられないので書かない。
  - 前回の課題との比較は、試合あたりの回数を Poisson 分布とみなして、差が標準誤差の 2 倍を
    超える時だけ「改善の可能性あり / 悪化の可能性あり」とする。試合数や回数が足りない時、
    キャラの組み合わせが違う時は言い切らない。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

# ── 課題の候補にする条件
MIN_TAKEN_EVENTS = 3        # 被弾がこれ未満なら、課題を出さずに「判定不能」にする
CANDIDATE_MIN_COUNT = 2     # 2 回以上起きて、
CANDIDATE_MIN_DAMAGE = 0.15  # 合計 15% 以上取られたものを候補にする
SINGLE_BIG_DAMAGE = 0.25    # 1 回だけでも、これ以上取られたものは候補にする（確信度は低）
MAX_ISSUES = 3              # レポートに出す課題の数
MAX_GOALS = 3               # 次の対戦で意識することの数
MAX_SCENES = 3              # 課題ごとに出す場面の数
OVERLAP_MERGE = 0.6         # 2 つの課題の被弾がこの割合以上重なっていたら、同じ課題としてまとめる
DOMINANT_SHARE = 0.5        # 課題の中で、ある状況・始動技がこの割合以上なら「推定」の原因として書く

# 練習で変えやすいか（自分で選んでいる行動ほど高い）。優先度の重みに使う
TRAINABILITY = {
    "situation": 1.0, "own": 1.0, "guard": 1.0, "starter": 0.8, "corner": 0.8, "offense": 0.7,
}
# 自分で選んでいない状態（食らっている最中など）は、練習で変えにくい
LOW_CONTROL_SITUATIONS = ("やられ中（追撃）", "不明")


@dataclass
class Scene:
    """課題の根拠になる場面（1 回の被弾）"""
    match: int
    match_sec: float              # 試合の頭からの秒数（記録の時刻）
    damage: float
    video_sec: Optional[float]    # 動画の再生位置（時刻合わせできた時だけ）
    video_index: int = 0


@dataclass
class PracticeMenu:
    name: str
    purpose: str
    target: str
    steps: list[str]
    focus: list[str]
    success: str
    in_match: str
    replay_check: str


@dataclass
class Issue:
    key: tuple                    # 課題の種類と中身。前回との比較にも使う
    title: str                    # 課題（何が起きているか）
    evidence: list[str]           # 根拠（確認済みの数字）
    cause: str                    # 考えられる原因
    cause_level: str              # 「推定」「確認済み」「特定できない」
    suggestion: str               # 改善案（次に試す行動）
    menu: PracticeMenu
    check: str                    # 確認方法（改善を見る指標）
    confidence: str               # 「高」「中」「低」
    confidence_reason: str
    score: float
    count: int
    damage: float
    matches_with: int
    n_matches: int
    goal: str                     # 次の対戦で意識すること（短い一文）
    scenes: list[Scene] = field(default_factory=list)
    events: list = field(default_factory=list)      # 根拠の被弾（重複の判定用。保存しない）
    notes: list[str] = field(default_factory=list)  # まとめた別の課題など

    @property
    def persistence(self) -> str:
        """複数試合で繰り返しているか（1 試合だけの解析では分からない）"""
        if self.n_matches < 2:
            return "1試合だけの記録のため、繰り返す課題かは判定できません"
        if self.matches_with >= 2:
            return f"継続課題（{self.n_matches}試合中{self.matches_with}試合で起きています）"
        return f"この記録では1試合だけで起きています（{self.n_matches}試合中）。一時的なミスの可能性もあります"


@dataclass
class Review:
    """前回の課題と今回の比較"""
    title: str
    status: str                   # 「改善の可能性あり」「横ばい」「悪化の可能性あり」「判定不能」
    detail: str
    next_step: str


@dataclass
class Coaching:
    issues: list[Issue]
    goals: list[str]
    notes: list[str]              # データ不足などの注意
    reviews: list[Review] = field(default_factory=list)


@dataclass
class MatchInput:
    """1 試合ぶんの入力"""
    match: int
    start_sec: float
    events: list                  # この試合の DamageEvent（両者の被弾）
    stats: object = None          # 自分の LiveStats（ガードの集計に使う）
    video_index: Optional[int] = None


# ────────────────────────────────────────────
# 被弾の切り口
# ────────────────────────────────────────────
def situation_of(e) -> str:
    """自分の状況（技名を出さない粗い分け方）。起き上がり直後は、その中の行動を問わずまとめる"""
    s = e.what_was_doing()
    return "起き上がり直後" if s.startswith("起き上がり直後") else s


def _own_loss(e, act_name: Callable) -> Optional[tuple[str, str]]:
    context = e.attack_context()
    if not context:
        return None
    kind, act = context
    return kind, act_name(act, e.own_names.get(act))


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _scenes(events: list, match_of: dict, starts: dict, videos: dict) -> list[Scene]:
    out = []
    for e in sorted(events, key=lambda e: -e.damage_pct)[:MAX_SCENES]:
        m = match_of[id(e)]
        out.append(Scene(m, e.time_sec - starts[m], e.damage_pct, e.video_sec, videos.get(m) or 0))
    return sorted(out, key=lambda s: (s.match, s.match_sec))


# ────────────────────────────────────────────
# 練習メニュー（課題の種類ごと）
# ────────────────────────────────────────────
REPLAY_FIRST = "該当する場面をリプレイ（またはレポートの動画）で見返す"


def _menu_situation(situation: str, starter: Optional[str]) -> tuple[PracticeMenu, str, str]:
    by = f"相手の {starter}" if starter else "相手の攻撃"
    if situation in ("飛翔中", "ハイジャンプ中", "ジャンプ中", "空中前ダッシュ中", "空中後ろダッシュ中"):
        return PracticeMenu(
            name=f"{situation}の被弾を減らす",
            purpose=f"{situation}に{by}を食らう場面を減らす",
            target=f"{situation}の被弾",
            steps=[REPLAY_FIRST,
                   "その時の相手との距離・高さ・動き始めたタイミングを書き出す",
                   "次の対戦では、動き始めるタイミングか、近づく道筋のどちらか 1 つだけを変えてみる",
                   "対戦後に同じような場面を探して、食らったかどうかを比べる"],
            focus=["相手が技を出せる距離で動き始めていないか", "同じ高さ・同じ道筋を繰り返していないか"],
            success=f"{situation}に食らった回数とダメージ（試合あたり）が減ること",
            in_match=f"{situation.replace('中', '')}を始める前に、相手との距離を一度確かめる",
            replay_check=f"次のレポートの「{situation}」の被弾回数と、{by}の被ダメージ",
        ), f"{situation.replace('中', '')}を始めるタイミングを変えてみる", "動き始めるタイミングか道筋を 1 つだけ変える"
    if situation in ("前ダッシュ中", "前歩き中", "ダッシュ停止中"):
        return PracticeMenu(
            name="近づき方を見直す",
            purpose=f"{situation}に{by}を食らう場面を減らす",
            target=f"{situation}の被弾",
            steps=[REPLAY_FIRST,
                   "どの距離から近づき始めたか、相手がその時何をしていたかを書き出す",
                   "次の対戦では、近づく前に射撃を挟む・一度止まるなど、近づき方を 1 つ変えてみる",
                   "対戦後に、近づいた回数と食らった回数を比べる"],
            focus=["相手の技が届く距離にまっすぐ入っていないか", "近づくタイミングが毎回同じになっていないか"],
            success=f"{situation}に食らった回数とダメージ（試合あたり）が減ること",
            in_match="まっすぐ近づく前に、一度相手の動きを見る",
            replay_check=f"次のレポートの「{situation}」の被弾回数",
        ), "同じ距離からまっすぐ近づかない", "近づき方を 1 つ変える"
    if situation in ("バックステップ中", "後ろ歩き中"):
        return PracticeMenu(
            name="下がり方を見直す",
            purpose=f"{situation}に{by}を食らう場面を減らす",
            target=f"{situation}の被弾",
            steps=[REPLAY_FIRST,
                   "下がり始めた時の相手との距離と、画面端までの余裕を書き出す",
                   "次の対戦では、下がる代わりにガードする・横に動くなど、別の選び方を 1 つ試す",
                   "対戦後に同じような場面を探して比べる"],
            focus=["下がった先で相手の技が届いていないか", "下がり続けて画面端を背負っていないか"],
            success=f"{situation}に食らった回数とダメージ（試合あたり）が減ること",
            in_match="下がる前に、ガードで様子を見る選択肢を思い出す",
            replay_check=f"次のレポートの「{situation}」の被弾回数",
        ), "下がる前にガードの選択肢を思い出す", "下がる代わりの選び方を 1 つ試す"
    if situation == "起き上がり直後":
        return PracticeMenu(
            name="起き上がりの対応を見直す",
            purpose="起き上がった直後に重ねられた攻撃での被弾を減らす",
            target="起き上がり直後の被弾",
            steps=[REPLAY_FIRST,
                   "起き上がった時に自分が何をしていたか（ガード・技・移動）と、受け身の方向を書き出す",
                   "次の対戦では、起き上がりはまずガードする、受け身の方向を変える、のどちらかを試す",
                   "対戦後に、起き上がり直後の被弾回数を比べる"],
            focus=["起き上がってすぐ技を出していないか", "受け身の方向が毎回同じになっていないか"],
            success="起き上がり直後の被弾回数とダメージ（試合あたり）が減ること",
            in_match="起き上がりはまずガードを意識する",
            replay_check="次のレポートの「起き上がり直後」の被弾回数",
        ), "起き上がりはまずガードを意識する", "起き上がりにまずガードする"
    if situation == "ガードクラッシュ中":
        return PracticeMenu(
            name="ガードクラッシュを減らす",
            purpose="ガードを割られてからの被弾を減らす",
            target="ガードクラッシュからの被弾",
            steps=[REPLAY_FIRST,
                   "割られる前にどのくらいガードし続けていたか、霊力がどれだけ残っていたかを確かめる",
                   "次の対戦では、霊力が少ない時は長くガードし続けず、抜け出す行動を早めに試す",
                   "対戦後に、ガードクラッシュの回数と誤ガードの回数を比べる"],
            focus=["霊力が少ない時にガードを続けていないか", "誤ガードで霊力を削られていないか"],
            success="ガードクラッシュの回数と、その後の被ダメージが減ること",
            in_match="霊力が少ない時は、長くガードし続けない",
            replay_check="次のレポートのガードクラッシュ回数（🛡️ ガードの行）",
        ), "霊力が少ない時は、長くガードし続けない", "霊力が少ない時に早めに抜け出す"
    return PracticeMenu(
        name=f"「{situation}」の被弾を見直す",
        purpose=f"「{situation}」の時の被弾を減らす",
        target=f"「{situation}」の時の被弾",
        steps=[REPLAY_FIRST,
               "その直前に自分が何をしようとしていたか、相手との距離を書き出す",
               "次の対戦で、その場面での選び方を 1 つ変えてみる",
               "対戦後に同じような場面を探して比べる"],
        focus=["同じ場面で同じ選び方を繰り返していないか"],
        success=f"「{situation}」の時の被弾回数とダメージ（試合あたり）が減ること",
        in_match=f"「{situation}」になりやすい場面で、選び方を 1 つ変える",
        replay_check=f"次のレポートの「{situation}」の被弾回数",
    ), f"「{situation}」になりやすい場面で、選び方を変える", "選び方を 1 つ変える"


OWN_LOSS_TEXT = {
    # 種類: (課題の言い方, 確認済みか, 原因, 改善案, 意識すること)
    "blocked": ("ガードされた後に反撃を受けている", "確認済み",
                "その技（連係）を相手にガードされ、硬直中に相手の技が始まっています",
                "ガードされた後の行動（止める・ガードに戻る）を見直す",
                "をガードされたら、無理に続けずに一度ガードに戻る"),
    "whiff": ("空振りの硬直に食らっている", "確認済み",
              "攻撃判定が当たらずに終わり、その後の硬直に相手の技が当たっています",
              "届く距離かどうかを確かめてから出す",
              "を、届くか分からない距離では出さない"),
    "clash": ("打ち合いで負けている", "確認済み",
              "攻撃判定が出ている最中に、相手の技に負けています（技の強さ・判定の差の可能性）",
              "その場面で別の技を出すか、出さずに様子を見る",
              "で打ち合いになりそうな場面では、別の技を試す"),
    "startup": ("出がかりを潰されている", "確認済み",
                "攻撃判定が出る前に、相手の技が当たっています",
                "相手の技が出ている・出そうな時に出さない",
                "を、相手が技を出していそうな時に出さない"),
    "first": ("先に出して、後から出た相手の技に負けている", "推定",
              "技の始まった順から見ると、空振りや硬直を差し返されている可能性があります",
              "届く距離かどうかを確かめてから出す",
              "を出す距離を見直す"),
    "late": ("相手の技が出ているところに出して負けている", "推定",
             "技の始まった順から見ると、相手の技が先に出ていた可能性があります",
             "相手の技が出ている時に重ねて出さない",
             "を、相手が技を出していそうな時に出さない"),
    "hit": ("当てた後に反撃を受けている", "確認済み",
            "自分の技が当たった後に、相手の技を食らっています（コンボが途切れた可能性）",
            "当てた後のつなぎを確かめる",
            "を当てた後のつなぎを確かめる"),
}


def _menu_own(kind: str, move: str) -> PracticeMenu:
    title, _, _, suggestion, _ = OWN_LOSS_TEXT[kind]
    steps = [f"{move} を出して食らった場面をリプレイで見返す",
             "その時の相手との距離と、相手がその直前に何をしていたかを書き出す"]
    if kind == "hit":
        steps += ["当たった後に、どこでつなぎが途切れたかを確かめる",
                  "途切れた所が毎回同じなら、そのつなぎを対戦の前に何度か試しておく"]
    else:
        steps += [f"次の対戦では、同じような場面で {move} を出す・出さないのどちらかを意識して選ぶ",
                  f"対戦後に {move} が絡んだ被弾が減ったかを比べる"]
    return PracticeMenu(
        name=f"{move} の使い方を見直す",
        purpose=f"{move} が{title.replace('ている', 'る')}場面を減らす",
        target=f"{move} が絡んだ被弾",
        steps=steps,
        focus=[suggestion, "同じ距離・同じタイミングで繰り返していないか"],
        success=f"{move} が絡んだ被弾の回数とダメージ（試合あたり）が減ること",
        in_match=f"{move}{OWN_LOSS_TEXT[kind][4]}",
        replay_check=f"次のレポートのアドバイス欄の「{move}」の行（負けた技の一覧）",
    )


def _menu_starter(move: str, situation: Optional[str]) -> PracticeMenu:
    where = f"（多いのは自分が「{situation}」の時）" if situation else ""
    return PracticeMenu(
        name=f"相手の {move} への対応を探す",
        purpose=f"相手の {move} から取られるダメージを減らす{where}",
        target=f"相手の {move} 始動の被弾",
        steps=[f"相手の {move} を食らった場面をリプレイで見返す",
               "その時の距離・高さ・位置（画面端かどうか）と、自分が何をしていたかを書き出す",
               "共通している点を 1 つ選び、次の対戦ではその状況を避ける・ガードで待つなど、試す行動を 1 つ決める",
               "対戦後に、相手の同じ技から取られたダメージを比べる"],
        focus=[f"相手が {move} を出しやすい距離・位置に入っていないか"],
        success=f"相手の {move} 始動の被ダメージ（試合あたり）が減ること",
        in_match=f"相手の {move} が届く距離・位置を意識する",
        replay_check=f"次のレポートの「被弾の内訳」の {move} の行",
    )


def _menu_guard(kind: str, moves: str) -> PracticeMenu:
    if kind == "wrong_low":
        what, change, want = "中段", "立ちガード", "しゃがみガードで中段を受ける"
    else:
        what, change, want = "下段", "しゃがみガード", "立ちガードで下段を受ける"
    return PracticeMenu(
        name=f"{what}のガードを見直す",
        purpose=f"{want}誤ガードを減らし、霊力を削られないようにする",
        target=f"{want}誤ガード",
        steps=[f"誤ガードになった相手の技（{moves}）が出た場面をリプレイで見返す",
               f"その技が来る前の相手の動き（構え・距離）を確かめる",
               f"次の対戦では、その技が来そうな時は{change}を意識する",
               "対戦後に、誤ガードの回数を比べる"],
        focus=[f"相手の {moves} に対してガードの向き（立ち・しゃがみ）が合っているか"],
        success=f"{want}誤ガードの回数（試合あたり）が減ること",
        in_match=f"相手の {moves} には{change}",
        replay_check="次のレポートの「🛡️ ガード」の行の誤ガード回数",
    )


def _menu_corner() -> PracticeMenu:
    return PracticeMenu(
        name="画面端での被弾を減らす",
        purpose="画面端に追い込まれてからの被弾を減らす",
        target="画面端での被弾",
        steps=["画面端で食らった場面をリプレイで見返す",
               "端に追い込まれる前に、どこで下がり始めていたかを確かめる",
               "次の対戦では、端に近づく前に前に出る・入れ替わる行動を 1 つ試す",
               "対戦後に、画面端での被弾と、端に追い込まれていた時間を比べる"],
        focus=["下がり続けて端を背負っていないか"],
        success="画面端での被ダメージ（試合あたり）と、追い込まれていた時間が減ること",
        in_match="端に近づいたら、下がり続けずに前に出る選択肢を思い出す",
        replay_check="次のレポートの「📍 位置」の行と、画面端での被弾回数",
    )


def _menu_offense() -> PracticeMenu:
    return PracticeMenu(
        name="当てた時のダメージを確かめる",
        purpose="攻撃が当たった時に取れるダメージを増やす",
        target="当てた時のダメージ",
        steps=["ダメージが大きかった場面と、小さかった場面をリプレイで見比べる",
               "小さかった場面で、どこでつなぎが途切れたか（届かなかった・入力が遅れたなど）を確かめる",
               "途切れた所が毎回同じなら、そのつなぎを対戦の前に何度か試しておく",
               "別の対戦のレポートで、当てた時の平均ダメージを比べる"],
        focus=["当てた後に、つなぎを最後まで出せているか"],
        success="当てた1回あたりの平均ダメージが増えること",
        in_match="当てた後のつなぎを最後まで出すことを意識する",
        replay_check="次のレポートの「被弾の内訳」の「当てた技」の表の合計ダメージ",
    )


# ────────────────────────────────────────────
# 課題を作る
# ────────────────────────────────────────────
def _confidence(count: int, matches_with: int, n_matches: int, trust: float) -> tuple[str, str]:
    reasons = [f"{count}回"]
    if n_matches >= 2:
        reasons.append(f"{n_matches}試合中{matches_with}試合")
    if trust < 1.0:
        reasons.append("推定を含む")
    if count >= 5 and (n_matches < 2 or matches_with >= 2) and trust >= 0.8:
        return "高", "・".join(reasons)
    if count >= 3 and trust >= 0.6:
        return "中", "・".join(reasons)
    return "低", "・".join(reasons) + "（回数が少ないため、偶然の可能性があります）"


def _score(damage: float, total: float, count: int, matches_with: int, n_matches: int,
           trust: float, train: float) -> float:
    """優先度。影響（被ダメージに占める割合）× 再現性 × 回数 × 確かさ × 練習で変えやすいか"""
    impact = damage / total if total > 0 else 0.0
    repro = 0.75 if n_matches < 2 else matches_with / n_matches
    frequency = 0.8 + 0.2 * min(count, 5) / 5
    return impact * (0.6 + 0.4 * repro) * frequency * trust * train


def build_coaching(
    matches: list[MatchInput],
    self_side: str,
    act_name: Callable,
    session_stats=None,
) -> Coaching:
    """試合ごとの被弾から、改善課題・練習メニュー・次の対戦で意識することを作る"""
    match_of, starts, videos = {}, {}, {}
    taken, dealt = [], []
    for m in matches:
        starts[m.match] = m.start_sec
        videos[m.match] = m.video_index
        for e in m.events:
            if not e.prev_inputs or e.estimated:
                continue
            match_of[id(e)] = m.match
            (taken if e.target == self_side else dealt).append(e)
    n_matches = len(matches)
    notes: list[str] = []
    if len(taken) < MIN_TAKEN_EVENTS:
        notes.append(f"判定不能: 被弾が{len(taken)}回しか記録されていないため、課題を決められません"
                     f"（{MIN_TAKEN_EVENTS}回以上必要です。連戦の記録でもう一度見てください）")
        return Coaching([], [], notes)
    total = sum(e.damage_pct for e in taken)

    def matches_with(events) -> int:
        return len({match_of[id(e)] for e in events})

    def eligible(events) -> bool:
        damage = sum(e.damage_pct for e in events)
        return ((len(events) >= CANDIDATE_MIN_COUNT and damage >= CANDIDATE_MIN_DAMAGE)
                or any(e.damage_pct >= SINGLE_BIG_DAMAGE for e in events))

    def dominant(events, key) -> Optional[tuple[str, int]]:
        counts: dict[str, int] = {}
        for e in events:
            k = key(e)
            if k:
                counts[k] = counts.get(k, 0) + 1
        if not counts:
            return None
        name, n = max(counts.items(), key=lambda x: x[1])
        return (name, n) if n >= 2 and n / len(events) >= DOMINANT_SHARE else None

    def common(events, kind, key, trust, title, evidence, cause, cause_level, suggestion, menu, check, goal):
        count, damage, mw = len(events), sum(e.damage_pct for e in events), matches_with(events)
        conf, why = _confidence(count, mw, n_matches, trust)
        return Issue(
            key=key, title=title, evidence=evidence, cause=cause, cause_level=cause_level,
            suggestion=suggestion, menu=menu, check=check, confidence=conf, confidence_reason=why,
            score=_score(damage, total, count, mw, n_matches, trust, TRAINABILITY[kind]),
            count=count, damage=damage, matches_with=mw, n_matches=n_matches, goal=goal,
            scenes=_scenes(events, match_of, starts, videos), events=events,
        )

    def base_evidence(events) -> list[str]:
        damage = sum(e.damage_pct for e in events)
        lines = [f"{len(events)}回・合計{_pct(damage)}（被ダメージ全体の{_pct(damage / total)}）"]
        if n_matches >= 2:
            lines.append(f"{n_matches}試合中{matches_with(events)}試合で起きています")
        return lines

    issues: list[Issue] = []

    # 1. 自分の状況
    # 自分の技が絡んだ被弾（「打撃の出がかり」など）は、技ごとの課題（3.）で扱う。
    # ただし起き上がり直後とガードクラッシュ中は、技より状況の方が先に効くので、ここで扱う
    by_situation: dict[str, list] = {}
    for e in taken:
        situation = situation_of(e)
        if e.attack_context() and situation not in ("起き上がり直後", "ガードクラッシュ中"):
            continue
        by_situation.setdefault(situation, []).append(e)
    for situation, events in by_situation.items():
        if situation in LOW_CONTROL_SITUATIONS or not eligible(events):
            continue
        top = dominant(events, lambda e: e.opp_move)
        if top:
            cause = f"推定: このうち{top[1]}回は相手の {top[0]} 始動です"
            level = "推定"
        else:
            cause = "原因は特定できません（食らった相手の技はばらばらです）。この場面を減らすことを試す価値があります"
            level = "特定できない"
        menu, goal, suggestion = _menu_situation(situation, top[0] if top else None)
        issues.append(common(
            events, "situation", ("situation", situation), 1.0,
            f"「{situation}」に食らうことが多い", base_evidence(events), cause, level,
            suggestion, menu, menu.success, goal,
        ))

    # 2. 相手の始動技
    by_starter: dict[str, list] = {}
    for e in taken:
        if e.opp_move:
            by_starter.setdefault(e.opp_move, []).append(e)
    for move, events in by_starter.items():
        if not eligible(events):
            continue
        delayed = sum(1 for e in events if e.opp_delayed)
        trust = 1.0 - 0.5 * delayed / len(events)
        evidence = base_evidence(events)
        if delayed:
            evidence.append(f"うち{delayed}回は、相手が技を出し終えた後に当たった被弾で、始動技は推定です")
        top = dominant(events, situation_of)
        if top:
            cause = f"推定: このうち{top[1]}回は自分が「{top[0]}」の時です"
            level = "推定"
        else:
            cause = "原因は特定できません（食らった時の自分の状況はばらばらです）"
            level = "特定できない"
        menu = _menu_starter(move, top[0] if top else None)
        issues.append(common(
            events, "starter", ("starter", move), trust,
            f"相手の {move} から多く取られている", evidence, cause, level,
            f"相手の {move} が届く距離・位置を避けるか、ガードで待つ", menu, menu.success,
            f"相手の {move} が届く距離を意識する",
        ))

    # 3. 自分の技が絡んだ被弾
    by_own: dict[tuple[str, str], list] = {}
    for e in taken:
        loss = _own_loss(e, act_name)
        if loss:
            by_own.setdefault(loss, []).append(e)
    for (kind, move), events in by_own.items():
        if not eligible(events):
            continue
        title, level, cause, suggestion, _ = OWN_LOSS_TEXT[kind]
        trust = 1.0 if level == "確認済み" else 0.7
        evidence = base_evidence(events)
        if level == "確認済み":
            cause = f"確認済み: {cause}。なぜそうなったか（距離・読み合い）は記録からは分かりません"
        else:
            cause = f"推定: {cause}"
        menu = _menu_own(kind, move)
        issues.append(common(
            events, "own", ("own", kind, move), trust,
            f"{move} が{title}", evidence, cause, level if level != "確認済み" else "確認済み",
            suggestion, menu, menu.success, menu.in_match,
        ))

    # 4. 画面端
    with_pos = [e for e in taken if e.distance is not None]
    cornered = [e for e in with_pos if e.cornered]
    if with_pos and eligible(cornered) and sum(e.damage_pct for e in cornered) / total >= 0.3:
        evidence = base_evidence(cornered)
        if session_stats is not None and getattr(session_stats, "pos_frames", 0):
            share = session_stats.cornered_frames / session_stats.pos_frames
            evidence.append(f"画面端に追い込まれていた時間は全体の{_pct(share)}")
        menu = _menu_corner()
        issues.append(common(
            cornered, "corner", ("corner",), 1.0,
            "画面端で食らうことが多い", evidence,
            "原因は特定できません（端に追い込まれた経緯は場面ごとに確認が必要です）", "特定できない",
            "端に近づく前に前に出る・入れ替わる", menu, menu.success, menu.in_match,
        ))

    # 5. 当てた時のダメージ（相手より明らかに低い時だけ）
    if len(dealt) >= 5 and len(taken) >= 5:
        md, mt = _mean_sd([e.damage_pct for e in dealt]), _mean_sd([e.damage_pct for e in taken])
        se = math.sqrt(md[1] ** 2 / len(dealt) + mt[1] ** 2 / len(taken))
        if mt[0] - md[0] > 2 * se and se > 0:
            menu = _menu_offense()
            issue = common(
                dealt, "offense", ("offense",), 1.0,
                "当てた時のダメージが、食らった時より低い",
                [f"当てた {len(dealt)}回の平均 {_pct(md[0])} / 食らった {len(taken)}回の平均 {_pct(mt[0])}",
                 "差は回数のばらつきを考えても大きいです（標準誤差の2倍以上）"],
                "原因は特定できません（つなぎの途切れ・始動の違い・キャラの差などが考えられます）",
                "特定できない", "当てた後のつなぎを確かめる", menu, menu.success, menu.in_match,
            )
            # 優先度は、食らった時との平均ダメージの差の割合で決める
            issue.score = (mt[0] - md[0]) / mt[0] * 0.5 * TRAINABILITY["offense"]
            issue.scenes = []
            issues.append(issue)

    # 6. 誤ガード。誤ガードは HP ではなく霊力を削られ、霊力が尽きるとガードクラッシュになる。
    # 影響は「ガードクラッシュ中に取られたダメージ」で測る（正ガードでも霊力は減るので、
    # すべてが誤ガードのせいとは言えない。そのぶん確かさを下げる）
    if session_stats is not None and getattr(session_stats, "guard_counts", None):
        g = session_stats.guard_counts
        guards = sum(g.values())
        crush_damage = sum(e.damage_pct for e in taken if situation_of(e) == "ガードクラッシュ中")
        for kind in ("wrong_low", "wrong_high"):
            n = g.get(kind, 0)
            if n < 3:
                continue
            per_match = [getattr(m.stats, "guard_counts", {}).get(kind, 0) if m.stats else 0 for m in matches]
            mw = sum(1 for c in per_match if c > 0)
            top = session_stats.wrong_guard_moves[kind].most_common(2)
            moves = "、".join(name for name, _ in top) or "相手の技"
            label = "しゃがみガードで中段を受けている" if kind == "wrong_low" else "立ちガードで下段を受けている"
            menu = _menu_guard(kind, moves)
            conf, why = _confidence(n, mw, n_matches, 1.0)
            evidence = [f"誤ガード {n}回（ガード全体 {guards}回の{_pct(n / guards)}）"]
            if crush_damage:
                evidence.append(f"ガードクラッシュ中に取られたダメージは合計{_pct(crush_damage)}")
            if top:
                evidence.append("多い技: " + "、".join(f"{name} {c}回" for name, c in top))
            if n_matches >= 2:
                evidence.append(f"{n_matches}試合中{mw}試合で起きています")
            crush = g.get("crush", 0)
            issues.append(Issue(
                key=("guard", kind), title=f"{label}（誤ガード）", evidence=evidence,
                cause=f"確認済み: 誤ガードは霊力を削られます（この記録ではガードクラッシュ{crush}回）。"
                      "なぜガードの向きを間違えたか（見えていない・読み違い）は記録からは分かりません",
                cause_level="確認済み", suggestion=f"相手の {moves} にガードの向きを合わせる",
                menu=menu, check=menu.success, confidence=conf, confidence_reason=why,
                score=_score(crush_damage, total, n, mw, n_matches, 0.7, TRAINABILITY["guard"]),
                count=n, damage=0.0, matches_with=mw, n_matches=n_matches, goal=menu.in_match,
            ))

    issues = _merge_overlaps(sorted(issues, key=lambda i: -i.score))
    top = issues[:MAX_ISSUES]
    goals: list[str] = []
    for issue in top:
        if issue.goal not in goals:
            goals.append(issue.goal)
    if len(taken) < 8:
        notes.append(f"被弾が{len(taken)}回と少ないため、課題は偶然の偏りかもしれません。連戦の記録で確かめてください")
    return Coaching(top, goals[:MAX_GOALS], notes)


def _merge_overlaps(issues: list[Issue]) -> list[Issue]:
    """根拠の被弾が大きく重なる課題は、優先度の高い方にまとめる（同じ課題を言い換えて何度も出さない）"""
    kept: list[Issue] = []
    for issue in issues:
        target = None
        for k in kept:
            if not issue.events or not k.events:
                continue
            shared = len({id(e) for e in issue.events} & {id(e) for e in k.events})
            if shared >= OVERLAP_MERGE * min(len(issue.events), len(k.events)):
                target = k
                break
        if target is None:
            kept.append(issue)
        else:
            target.notes.append(f"関連: {issue.title}（{issue.count}回・合計{_pct(issue.damage)}）。"
                                "根拠の被弾が大きく重なるため、この課題にまとめています")
    return kept


def _mean_sd(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1) if len(values) > 1 else 0.0
    return mean, math.sqrt(var)


# ────────────────────────────────────────────
# 前回の課題との比較（成長記録）
# ────────────────────────────────────────────
def measure(key: list | tuple, matches: list[MatchInput], self_side: str, act_name: Callable,
            session_stats=None) -> Optional[dict]:
    """課題の指標を今回の記録で測る。{"count": 回数, "damage": 合計, "matches": 試合数}。測れなければ None"""
    key = tuple(key)
    events = [e for m in matches for e in m.events
              if e.target == self_side and e.prev_inputs and not e.estimated]
    kind = key[0]
    if kind == "situation":
        hit = [e for e in events if situation_of(e) == key[1]]
    elif kind == "starter":
        hit = [e for e in events if e.opp_move == key[1]]
    elif kind == "own":
        hit = [e for e in events if _own_loss(e, act_name) == (key[1], key[2])]
    elif kind == "corner":
        if not any(e.distance is not None for e in events):
            return None
        hit = [e for e in events if e.cornered]
    elif kind == "guard":
        if session_stats is None or not getattr(session_stats, "guard_counts", None):
            return None
        return {"count": session_stats.guard_counts.get(key[1], 0), "damage": 0.0, "matches": len(matches)}
    else:
        return None     # 当てた時のダメージは平均の比較になるので、今は前回との比較をしない
    return {"count": len(hit), "damage": sum(e.damage_pct for e in hit), "matches": len(matches)}


def snapshot(issue: Issue, value: Optional[dict]) -> dict:
    """履歴に残す形"""
    return {"key": list(issue.key), "title": issue.title, "goal": issue.goal, "value": value}


def compare(previous: dict, now: Optional[dict], same_matchup: bool) -> Review:
    """前回と今回の指標を比べる。試合あたりの回数を Poisson 分布とみなし、差が標準誤差の 2 倍を超えた時だけ言い切る"""
    title = previous["title"]
    before = previous.get("value")
    keep = previous.get("goal", "")
    if not before or not now:
        return Review(title, "判定不能", "前回か今回の記録に、この指標を測るデータがありません", keep)
    if not same_matchup:
        return Review(title, "判定不能", "前回とキャラの組み合わせが違うため、比べられません", keep)
    n1, n2 = before["matches"], now["matches"]
    c1, c2 = before["count"], now["count"]
    r1, r2 = c1 / n1, c2 / n2
    detail = (f"前回 {c1}回／{n1}試合（試合あたり {r1:.1f}回・{_pct(before['damage'] / n1)}） → "
              f"今回 {c2}回／{n2}試合（試合あたり {r2:.1f}回・{_pct(now['damage'] / n2)}）")
    if n1 < 2 or n2 < 2 or c1 + c2 < 4:
        return Review(title, "判定不能", detail + "。評価に必要な試合数・回数が足りません", keep)
    se = math.sqrt(c1 / n1 ** 2 + c2 / n2 ** 2) or 1e-9
    if r2 < r1 and (r1 - r2) > 2 * se:
        return Review(title, "改善の可能性あり", detail, "続けて同じことを意識し、次の記録でも減っているか確かめる")
    if r2 > r1 and (r2 - r1) > 2 * se:
        return Review(title, "悪化の可能性あり", detail, "該当する場面をもう一度見返し、意識することを 1 つに絞り直す")
    return Review(title, "横ばい", detail + "。差は試合ごとのばらつきの範囲です", keep)
