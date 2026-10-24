"""
coaching.py（改善課題・練習メニュー・前回の課題の振り返り）の確認用テスト。

実行:
    python -m unittest discover -s tests -v

被弾は analyzer.event_from_record で作る（履歴に残す形と同じ。直前 1.5 秒の両者の動きを
[アクションID, フレーム数] の並びで書く）。自分は p1 側で、食らった被弾は target="p1" になる。
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import analyzer as an
import coaching as coach
import player_history as ph

FLYING = [[0, 50], [214, 10]]           # 飛翔中
DASHING = [[0, 50], [200, 10]]          # 前ダッシュ中
STANDING = [[0, 60]]                    # 立ち
OPP_JA = [[0, 45], [306, 15]]           # 相手の JA が当たる


def hit(mine=FLYING, starter="JA", damage=0.1, t=10.0, theirs=OPP_JA, cornered=False, kind="taken"):
    return an.event_from_record({
        "kind": kind, "t": t, "hp": 1.0, "dmg": damage, "act": 306, "move": starter,
        "v": mine, "a": theirs, "dist": 100, "corner": cornered,
    })


def matches_of(*per_match):
    """試合ごとの被弾の並びから MatchInput を作る"""
    return [coach.MatchInput(n, 0.0, events) for n, events in enumerate(per_match, start=1)]


def build(*per_match, stats=None):
    return coach.build_coaching(matches_of(*per_match), "p1", an._act_name, stats)


class IssueTests(unittest.TestCase):
    def test_issue_with_the_most_damage_is_found_with_its_evidence(self):
        result = build([hit(starter=f"s{i}", damage=0.15) for i in range(4)] + [hit(DASHING, starter="x")])
        issue = result.issues[0]
        self.assertEqual(issue.key, ("situation", "飛翔中"))
        self.assertEqual((issue.count, round(issue.damage, 2)), (4, 0.6))
        self.assertIn("4回・合計60%（被ダメージ全体の86%）", issue.evidence)

    def test_issue_repeated_across_matches_ranks_above_one_with_the_same_damage(self):
        # 飛翔中は 3 試合すべて、前ダッシュ中は 1 試合目だけ。ダメージの合計は同じ
        flying = [[hit(starter=f"f{m}{i}") for i in range(2)] for m in range(3)]
        dashing = [hit(DASHING, starter=f"d{i}") for i in range(6)]
        result = build(flying[0] + dashing, flying[1], flying[2])
        keys = [i.key for i in result.issues]
        self.assertLess(keys.index(("situation", "飛翔中")), keys.index(("situation", "前ダッシュ中")))
        top = result.issues[0]
        self.assertEqual(top.persistence, "継続課題（3試合中3試合で起きています）")
        dash = next(i for i in result.issues if i.key == ("situation", "前ダッシュ中"))
        self.assertIn("一時的なミスの可能性", dash.persistence)

    def test_cause_is_not_asserted_without_a_dominant_pattern(self):
        scattered = build([hit(starter=f"s{i}") for i in range(4)])
        self.assertEqual(scattered.issues[0].cause_level, "特定できない")
        self.assertIn("特定できません", scattered.issues[0].cause)
        mostly_ja = build([hit(starter="JA") for _ in range(3)] + [hit(starter="6A")])
        flying = next(i for i in mostly_ja.issues if i.key == ("situation", "飛翔中"))
        self.assertEqual(flying.cause_level, "推定")
        self.assertIn("3回は相手の JA 始動", flying.cause)

    def test_too_few_hits_gives_no_issues_and_says_why(self):
        result = build([hit(), hit()])
        self.assertEqual((result.issues, result.goals), ([], []))
        self.assertTrue(result.notes[0].startswith("判定不能"))

    def test_rare_single_big_hit_is_shown_with_low_confidence(self):
        result = build([hit(DASHING, starter="a", damage=0.3)] + [hit(STANDING, starter=f"s{i}", damage=0.02)
                                                                   for i in range(3)])
        issue = next(i for i in result.issues if i.key == ("situation", "前ダッシュ中"))
        self.assertEqual(issue.confidence, "低")
        self.assertIn("偶然の可能性", issue.confidence_reason)

    def test_practice_menu_has_every_field(self):
        menus = [coach._menu_situation(sit, starter)[0]
                 for sit in ("飛翔中", "前ダッシュ中", "バックステップ中", "起き上がり直後", "ガードクラッシュ中", "しゃがみ")
                 for starter in ("JA", None)]
        menus += [coach._menu_own(kind, "JA") for kind in coach.OWN_LOSS_TEXT]
        menus += [coach._menu_starter("JA", "飛翔中"), coach._menu_starter("JA", None),
                  coach._menu_guard("wrong_low", "J2A"), coach._menu_guard("wrong_high", "2A"),
                  coach._menu_corner(), coach._menu_offense()]
        for m in menus:
            for name in ("name", "purpose", "target", "success", "in_match", "replay_check"):
                self.assertTrue(getattr(m, name), (m.name, name))
            self.assertGreaterEqual(len(m.steps), 3, m.name)
            self.assertTrue(m.focus, m.name)
        # 課題にも、改善案・確認方法・次の対戦で意識することが付く
        stats = an.LiveStats()
        stats.guard_counts = {"right": 20, "wrong_low": 5}
        stats.wrong_guard_moves["wrong_low"].update({"J2A": 4})
        for issue in build([hit(starter="JA") for _ in range(4)], stats=stats).issues:
            self.assertTrue(issue.check and issue.suggestion and issue.goal and issue.evidence, issue.title)

    def test_issues_resting_on_the_same_hits_are_merged(self):
        # 相手の JA 始動の被弾 4 回は、すべて飛翔中。同じ課題を 2 回出さない
        result = build([hit(starter="JA") for _ in range(4)] + [hit(STANDING, starter="x", damage=0.02)])
        keys = [i.key for i in result.issues]
        self.assertEqual(len([k for k in keys if k in (("situation", "飛翔中"), ("starter", "JA"))]), 1)
        self.assertTrue(any("まとめています" in n for n in result.issues[0].notes))

    def test_goals_are_one_to_three_and_come_from_the_issues(self):
        events = ([hit(starter=f"f{i}") for i in range(4)] + [hit(DASHING, starter=f"d{i}") for i in range(4)]
                  + [hit(STANDING, starter=f"c{i}", cornered=True) for i in range(4)]
                  + [hit([[0, 50], [201, 10]], starter=f"b{i}") for i in range(4)])
        result = build(events)
        self.assertTrue(1 <= len(result.goals) <= coach.MAX_GOALS)
        self.assertEqual(result.goals, list(dict.fromkeys(i.goal for i in result.issues))[:coach.MAX_GOALS])

    def test_lower_damage_when_landing_hits_is_found_only_when_the_gap_is_clear(self):
        taken = [hit(STANDING, starter=f"s{i}", damage=0.2) for i in range(6)]
        weak = [hit(STANDING, kind="dealt", damage=0.05 + 0.01 * (i % 2)) for i in range(6)]
        self.assertIn(("offense",), [i.key for i in build(taken + weak).issues])
        close = [hit(STANDING, kind="dealt", damage=0.19 + 0.02 * (i % 2)) for i in range(6)]
        self.assertNotIn(("offense",), [i.key for i in build(taken + close).issues])


class ReviewTests(unittest.TestCase):
    def _previous(self, count, matches):
        return {"key": ["situation", "飛翔中"], "title": "「飛翔中」に食らうことが多い", "goal": "g",
                "value": {"count": count, "damage": count * 0.1, "matches": matches}}

    def test_clear_drop_is_a_possible_improvement(self):
        review = coach.compare(self._previous(12, 3), {"count": 1, "damage": 0.1, "matches": 3}, True)
        self.assertEqual(review.status, "改善の可能性あり")
        self.assertIn("前回 12回／3試合", review.detail)

    def test_clear_rise_is_a_possible_decline(self):
        review = coach.compare(self._previous(1, 3), {"count": 12, "damage": 1.2, "matches": 3}, True)
        self.assertEqual(review.status, "悪化の可能性あり")

    def test_small_change_is_flat(self):
        review = coach.compare(self._previous(6, 3), {"count": 5, "damage": 0.5, "matches": 3}, True)
        self.assertEqual(review.status, "横ばい")

    def test_no_claim_with_too_few_matches_or_a_different_matchup(self):
        self.assertEqual(coach.compare(self._previous(12, 1), {"count": 0, "damage": 0, "matches": 3}, True).status,
                         "判定不能")
        review = coach.compare(self._previous(12, 3), {"count": 0, "damage": 0, "matches": 3}, False)
        self.assertEqual(review.status, "判定不能")
        self.assertIn("キャラの組み合わせが違う", review.detail)
        self.assertEqual(coach.compare(self._previous(12, 3), None, True).status, "判定不能")

    def test_measure_counts_the_same_hits_as_the_issue(self):
        ms = matches_of([hit(starter="JA") for _ in range(3)], [hit(starter="JA"), hit(DASHING, starter="x")])
        result = coach.build_coaching(ms, "p1", an._act_name)
        for issue in result.issues:
            value = coach.measure(issue.key, ms, "p1", an._act_name)
            self.assertEqual((value["count"], value["matches"]), (issue.count, 2))


class ReportTests(unittest.TestCase):
    """レポートと履歴に組み込んだ時の動き"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._dir = ph.HISTORY_DIR
        ph.HISTORY_DIR = self.tmp / "history"

    def tearDown(self):
        ph.HISTORY_DIR = self._dir

    def _session(self, name, hits_per_match):
        """3 試合。どの試合でも P1 が相手の JA(306) で hits_per_match 回、10% ずつ食らう"""
        frames = []
        for n in range(3):
            base = n * 3000
            for i in range(1500):
                hp = 10000 - 1000 * sum(1 for k in range(hits_per_match) if i >= 200 + k * 200)
                p2_act = 306 if any(185 + k * 200 <= i < 205 + k * 200 for k in range(hits_per_match)) else 0
                frames.append({"t": round((base + i) / 60, 4), "f": base + i, "match": n + 1,
                               "p1": {"hp": hp, "sp": 1000, "act": 0, "a": 0, "b": 0, "c": 0, "d": 0, "x": 0, "y": 0},
                               "p2": {"hp": 10000, "sp": 1000, "act": p2_act, "a": 0, "b": 0, "c": 0, "d": 0,
                                      "x": 0, "y": 0}})
        path = self.tmp / name
        path.write_text(json.dumps({"meta": {"version": 7, "recorded_at": name, "matches": [
            {"match": n + 1, "p1_char": "aya", "p2_char": "sakuya"} for n in range(3)]}, "frames": frames}),
            encoding="utf-8")
        return path

    def _report(self, live, **kw):
        out = self.tmp / (live.stem + ".html")
        an.analyze(None, None, out, live, **kw)
        return out.read_text(encoding="utf-8")

    def test_report_has_issues_menus_and_goals(self):
        report = self._report(self._session("a.json", 3))
        self.assertIn("🧭 改善課題と練習メニュー", report)
        # JA の被弾はすべて立っている時なので、「立ち」の課題 1 つにまとまり、原因に JA が出る
        self.assertIn("練習メニュー: 「立ち」の被弾を見直す", report)
        self.assertIn("推定: このうち9回は相手の JA 始動です", report)
        self.assertIn("🎯 次の対戦で意識すること", report)
        # 動画が無いので、場面は記録の時刻として出し、動画へのリンクは作らない
        section = report[report.index("🧭"):report.index("📋 セッション結果")]
        self.assertIn("（記録の時刻）", section)
        self.assertNotIn('class="seek"', section)
        self.assertNotIn("前回の課題の振り返り", section)    # 履歴を使わなければ振り返りは無い

    def test_next_session_is_compared_with_the_previous_issues(self):
        self._report(self._session("before.json", 4), player_name="tester", use_history=True)
        report = self._report(self._session("after.json", 1), player_name="tester", use_history=True)
        section = report[report.index("📈 前回の課題の振り返り"):]
        self.assertIn("改善の可能性あり", section)
        self.assertIn("前回 12回／3試合", section)
        # 同じ記録から作り直しても、その記録自身とは比べない
        again = self._report(self._session("after.json", 1), player_name="tester", use_history=True)
        self.assertIn("前回 12回／3試合", again[again.index("📈"):])
        self.assertEqual(len(ph.load_coaching("tester")), 2)
        ph.clear_history("tester")
        self.assertEqual(ph.load_coaching("tester"), [])

    def test_history_failure_does_not_stop_the_report(self):
        real = ph.save_coaching
        ph.save_coaching = lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
        try:
            report = self._report(self._session("c.json", 3), player_name="tester", use_history=True)
        finally:
            ph.save_coaching = real
        self.assertIn("🧭 改善課題と練習メニュー", report)
        self.assertIn("振り返りは出していません", report)

    def test_coaching_failure_does_not_stop_the_report(self):
        real = coach.build_coaching
        coach.build_coaching = lambda *a, **k: 1 / 0
        try:
            report = self._report(self._session("d.json", 3))
        finally:
            coach.build_coaching = real
        self.assertNotIn("🧭", report)
        self.assertIn("📋 セッション結果", report)


if __name__ == "__main__":
    unittest.main()
