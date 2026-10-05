"""
analyzer.py / player_history.py の確認用テスト。

実行:
    python -m unittest discover -s tests -v

合成データだけで動く。実ファイルと突き合わせるテスト（RealFileTests）は、
SOKU_TEST_LIVE / SOKU_TEST_REP にパスを入れた時だけ走る。
"""

import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import analyzer as an
import player_history as ph


def side(hp=10000, sp=1000, act=0, a=0, b=0, c=0, d=0, x=0, y=0, combo=0):
    return {"hp": hp, "sp": sp, "act": act, "dir": 0, "a": a, "b": b, "c": c, "d": d,
            "x": x, "y": y, "hx": 0, "hy": 0, "combo": combo}


def make_round(start_f, t0, loser="p2", frames=900, match=1, winner_hp=6000):
    """loser の HP が frames かけて 0 まで減り、その後 60 フレーム 0 のまま続くラウンド"""
    out = []
    for i in range(frames + 60):
        lose_hp = max(0, 10000 - i * (10000 // frames + 1))
        win_hp = 10000 - min(i, frames) * (10000 - winner_hp) // frames
        p1, p2 = (win_hp, lose_hp) if loser == "p2" else (lose_hp, win_hp)
        out.append({"t": round(t0 + i / 60, 4), "f": start_f + i, "match": match,
                    "p1": side(hp=p1), "p2": side(hp=p2)})
    return out


def make_rep(path, mode, stream, c1=11, c2=15, deck=20):
    """実ファイルと同じ並びの .rep を作る。mode: 1=対CPU, 3=対人"""
    head = bytearray(0x0E)
    head[0] = 0xD2
    head[an.REP_MODE_OFS] = mode
    body = bytes(head)
    for n, char in enumerate((c1, c2)):
        body += bytes([char, 0]) + struct.pack("<I", deck) + struct.pack(f"<{deck}H", *range(deck))
        body += b"\x00\x01\x01" if n == 0 else b"\x01\x01\x00"
    body += bytes(7) + struct.pack("<I", len(stream)) + struct.pack(f"<{len(stream)}H", *stream)
    path.write_bytes(body)


class RoundTests(unittest.TestCase):
    def test_winner_is_taken_at_the_ko_not_at_the_last_sample(self):
        # 決着後に両方のバーが消える（動画の結果画面）
        hp = [(i * 0.1, 1.0, max(0.0, 1 - i / 200)) for i in range(250)]
        hp += [(25 + i * 0.1, 0.0, 0.0) for i in range(100)]
        rounds = an.split_rounds(hp)
        self.assertEqual([r.p1_won for r in rounds], [True])
        self.assertAlmostEqual(rounds[0].end_sec, 20.0, places=1)
        self.assertEqual(rounds[0].final_hps(), (1.0, 0.0))
        self.assertEqual(rounds[0].min_hp()[0], 1.0)

    def test_comeback_from_a_sliver_is_not_counted_as_a_loss(self):
        hp = [(i * 0.1, 0.02 if i > 50 else 0.5, max(0.0, 1 - i / 200)) for i in range(260)]
        self.assertEqual([r.p1_won for r in an.split_rounds(hp)], [True])

    def test_unfinished_round_has_no_winner(self):
        # 2ラウンド目は HP が 4% 残ったところで記録が終わる
        frames = make_round(0, 0.0) + make_round(960, 16.0)[:800]
        rounds = an.split_rounds(an.frames_to_hp_samples(frames))
        self.assertEqual([(r.p1_won, r.incomplete) for r in rounds], [(True, False), (None, True)])

    def test_smoothing_removes_short_dips(self):
        hp = [(i * 0.1, 0.7, 0.9) for i in range(100)]
        hp[40] = (4.0, 0.0, 0.2)
        hp[41] = (4.1, 0.1, 0.3)
        smooth = an.smooth_hp(hp)
        self.assertTrue(all(abs(s[1] - 0.7) < 1e-9 and abs(s[2] - 0.9) < 1e-9 for s in smooth))


class DamageTests(unittest.TestCase):
    def _frames(self):
        frames = []
        hp = 10000
        for i in range(600):
            if 100 <= i < 105:       # 5ヒットで 20%
                hp -= 400
            if 300 <= i < 302:       # 2ヒットで 4%（しきい値未満）
                hp -= 200
            frames.append({"t": round(i / 60, 4), "f": i, "match": 1,
                           "p1": side(hp=hp, sp=100 if i < 200 else 1000, act=300 if 90 <= i < 100 else 0),
                           "p2": side()})
        return frames

    def test_combo_is_one_event_and_small_damage_is_ignored(self):
        events = an.detect_damage_events(self._frames())
        self.assertEqual(len(events), 1)
        e = events[0]
        self.assertEqual((e.target, e.hits, e.spirit_before), ("p1", 5, 100))
        self.assertAlmostEqual(e.damage_pct, 0.20)
        self.assertEqual(e.what_was_doing(), "打撃中")

    def test_summary_lists_big_hits_in_time_order(self):
        lines = an.analyze_damage_patterns(an.detect_damage_events(self._frames()), "p1", "P1")
        self.assertTrue(any("霊力1玉未満の時に食らったのは1回" in ln for ln in lines))
        self.assertTrue(any(ln.startswith("  大ダメージ: 1.7秒") for ln in lines))

    def test_video_only_detection_matches_frame_accurate_detection(self):
        frames = self._frames()
        hp = an.frames_to_hp_samples(frames)[::6]
        rounds = [an.RoundData(round_num=1, start_sec=0.0, end_sec=hp[-1][0], hp_timeline=hp)]
        events = an.detect_damage_events_from_hp(rounds)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].damage_pct, 0.20)
        self.assertEqual(events[0].prev_inputs, [])
        lines = an.analyze_damage_patterns(events, "p1", "P1")
        self.assertFalse(any("直前" in ln for ln in lines))

    def _frames_with_opponent(self):
        """P1 が 2 回食らう: 相手の打撃(306)で 10%、ダウン明けに相手の射撃(411)で 20%"""
        frames = []
        hp = 10000
        for i in range(600):
            if 100 <= i < 105:
                hp -= 200
            if 300 <= i < 305:
                hp -= 400
            if 105 <= i < 150:
                act = 98
            elif 240 <= i < 290:
                act = 197            # 290 で起き上がり、10 フレーム後に被弾
            elif 300 <= i < 330:
                act = 71
            else:
                act = 0
            if 80 <= i < 105:
                opp = 306
            elif 250 <= i < 270:
                opp = 411            # 射撃は出し終わってから当たる
            else:
                opp = 0
            frames.append({"t": round(i / 60, 4), "f": i, "match": 1,
                           "p1": side(hp=hp, act=act), "p2": side(act=opp)})
        return frames

    def test_opponent_starter_and_wakeup_are_reported(self):
        events = an.detect_damage_events(self._frames_with_opponent())
        self.assertEqual([(e.opp_act, e.after_down) for e in events], [(306, False), (411, True)])
        self.assertEqual(events[1].what_was_doing(), "起き上がり直後")
        lines = an.analyze_damage_patterns(events, "p1", "P1")
        self.assertIn("  相手の始動技ごとの被ダメージ: 6C 20%・1回、JA 10%・1回", lines)
        self.assertTrue(any(ln.endswith("直前→起き上がり直後 / 相手→6C") for ln in lines))
        # 同じ被弾を攻撃側から見ると、P2 の与ダメージになる
        self.assertEqual(an.analyze_offense(events, "p1", "P2"),
                         ["🗡️ P2 の始動技ごとの与ダメージ: 6C 20%・1回、JA 10%・1回"])
        self.assertEqual(an.analyze_offense(events, "p2", "P1"), [])

    def test_guard_crush_before_a_hit_is_named(self):
        frames = self._frames_with_opponent()
        for f in frames[80:100]:
            f["p1"]["act"] = an.ACT_GUARD_CRUSH
        events = an.detect_damage_events(frames)
        self.assertEqual(events[0].what_was_doing(), "ガードクラッシュ中")

    def test_skills_are_named_by_the_command_that_started_them(self):
        frames = [{"t": i / 60, "f": i, "match": 1, "p1": side(), "p2": side()} for i in range(900)]
        for start in (100, 300, 500):          # 236B で 500 番が 3 回
            for i in range(start - 3, start):
                frames[i]["p1"].update(b=1, combo=2)
            for i in range(start, start + 30):
                frames[i]["p1"]["act"] = 500
        for n, start in enumerate((150, 350, 550)):   # 505 番はコマンドが揃わない（派生など）
            if n == 0:
                frames[start - 1]["p1"].update(c=1, combo=32)
            for i in range(start, start + 30):
                frames[i]["p1"]["act"] = 505
        names = an.skill_command_names(frames)
        self.assertEqual(names, {("p1", 500): "236B"})
        self.assertEqual(an._act_name(500, names.get(("p1", 500))), "236B")
        self.assertEqual(an._act_name(505, names.get(("p1", 505))), "スキル(505)")
        self.assertEqual(an._act_name(310), "打撃(310)")
        # 位置のある記録で、空中で出していれば J が付く
        for f in frames:
            f["p1"]["py"] = 120
        self.assertEqual(an.skill_command_names(frames), {("p1", 500): "J236B"})

    def test_no_opponent_attack_means_no_starter_line(self):
        lines = an.analyze_damage_patterns(an.detect_damage_events(self._frames()), "p1", "P1")
        self.assertFalse(any("始動技" in ln or "相手→" in ln for ln in lines))

    def test_time_format(self):
        self.assertEqual(an._fmt_time(46.94), "46.9秒")
        self.assertEqual(an._fmt_time(214.1), "214.1秒（3:34）")


class LiveStatsTests(unittest.TestCase):
    def test_button_rates_are_reported_as_numbers_only(self):
        frames = [{"t": i / 60, "f": i, "match": 1, "p1": side(a=1 if i % 2 else 0), "p2": side()}
                  for i in range(600)]
        p1, p2 = an.parse_live_stats_from_frames(frames)
        advice: list[str] = []
        an._advice_from_live(advice, p1)
        self.assertTrue(advice[0].startswith("🎮 入力発生率 50.0% / ボタン押下率: A 50.0%"))
        self.assertFalse(any("混ぜて" in ln or "少なめ" in ln for ln in advice))

    def test_spirit_summary(self):
        frames = []
        for i in range(100):
            sp = 1000 if i < 50 else (800 if i < 75 else 100)
            frames.append({"t": i / 60, "f": i, "match": 1,
                           "p1": side(sp=sp, act=150 if i == 50 else 0), "p2": side()})
        p1, _ = an.parse_live_stats_from_frames(frames)
        self.assertEqual((p1.sp_low_frames, p1.sp_guard_drops), (25, 1))
        self.assertIn("1玉未満の時間 25.0%", p1.spirit_summary())

    def _positioned_frames(self):
        """前半は P1 が左端で P2 が近く、後半は中央で離れている。P1 は 100 フレーム目に 20% 食らう"""
        frames = []
        for i in range(600):
            p1x, p2x = (100, 250) if i < 300 else (500, 1000)
            p1 = side(hp=10000 - 2000 * (i >= 100))
            p1.update(px=p1x, py=0, face=1)
            p2 = side()
            p2.update(px=p2x, py=0, face=-1)
            frames.append({"t": round(i / 60, 4), "f": i, "match": 1, "p1": p1, "p2": p2})
        return frames

    def test_position_summary_and_cornered_hits(self):
        frames = self._positioned_frames()
        p1, p2 = an.parse_live_stats_from_frames(frames)
        self.assertEqual((p1.cornered_frames, p1.cornering_frames, p1.dist_frames), (300, 0, [0, 300, 0, 300]))
        self.assertEqual((p2.cornered_frames, p2.cornering_frames), (0, 300))
        self.assertIn("画面端に追い込まれていた時間 50.0%", p1.position_summary())
        events = an.detect_damage_events(frames)
        self.assertEqual((events[0].distance, events[0].cornered), (150, True))
        lines = an.analyze_damage_patterns(events, "p1", "P1")
        self.assertIn("  被弾した時の相手との距離: 密着 0回・0% / 近距離 1回・20% / 中距離 0回・0% / 遠距離 0回・0%", lines)
        self.assertEqual([an.distance_band(d) for d in (39, 60, 219, 220, 449, 450)], [0, 1, 1, 2, 2, 3])
        self.assertIn("  被弾1回のうち、画面端に追い込まれていた時が1回（合計 20%）", lines)
        self.assertTrue(any(ln.startswith("  大ダメージ") and ln.endswith(" / 近距離・画面端") for ln in lines))

    def test_recordings_without_position_have_no_position_lines(self):
        frames = [{"t": i / 60, "f": i, "match": 1, "p1": side(hp=10000 - 2000 * (i >= 100)), "p2": side()}
                  for i in range(300)]
        p1, _ = an.parse_live_stats_from_frames(frames)
        self.assertEqual(p1.position_summary(), "")
        lines = an.analyze_damage_patterns(an.detect_damage_events(frames), "p1", "P1")
        self.assertFalse(any("距離" in ln or "画面端" in ln for ln in lines))

    def test_combo_key_groups_button_variants(self):
        self.assertEqual(an._combo_key(1024), 512)        # 623 の C
        self.assertEqual(an._combo_key(514), 2)           # 複数立っていたら下位
        self.assertEqual(an._combo_key(524288), 524288)   # 未対応のコマンドは生値


class RepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_vs_human_inputs_alternate_p1_p2(self):
        rep = self.tmp / "a.rep"
        make_rep(rep, 3, [0x10, 0x80] * 100 + [0x10])
        p1, p2 = an.parse_rep(rep)
        self.assertEqual((p1.active_player, p1.total_frames, p1.button_pct(0x10)), (1, 101, 100.0))
        self.assertEqual((p2.active_player, p2.total_frames, p2.button_pct(0x80)), (2, 100, 100.0))
        self.assertEqual((p1.p1_char, p1.p2_char), ("aya", "sanae"))

    def test_vs_cpu_has_only_p1(self):
        rep = self.tmp / "b.rep"
        make_rep(rep, 1, [0x10, 0, 0, 0])
        stats = an.parse_rep(rep)
        self.assertEqual(len(stats), 1)
        self.assertEqual((stats[0].total_frames, stats[0].button_pct(0x10)), (4, 25.0))

    def test_broken_file_is_rejected(self):
        rep = self.tmp / "c.rep"
        rep.write_bytes(b"\xd2" + bytes(300))
        with self.assertRaises(an.AnalyzeError):
            an.parse_rep(rep)


class RepSyncTests(unittest.TestCase):
    """.rep と動画の時刻合わせ"""

    def _match(self, rate=1.012, off=150, delay=12):
        import random
        rng = random.Random(7)
        presses = sorted(rng.sample(range(200, 4800, 40), 45))
        stream = [0] * 5000
        for f in presses:
            for k in range(4):
                stream[f + k] = 0x10
        # P1 が A を押した delay フレーム後に P2 の HP が 2% 減る
        hits = sorted(round((f + delay) * rate + off) for f in presses)
        hp, cur = [], 1.0
        for v in range(0, 5400, 6):
            while hits and hits[0] <= v:
                hits.pop(0)
                cur = max(0.0, cur - 0.02)
            hp.append((v / 60, 1.0, cur))
        return stream, hp, presses

    def test_offset_is_recovered(self):
        stream, hp, presses = self._match()
        rounds = an.split_rounds(hp)
        self.assertEqual(len(rounds), 1)
        sync = an.align_rep_to_video(hp, rounds, [tuple(stream)])
        self.assertIsNotNone(sync)
        # 押したフレーム → 動画の時刻 → .rep のフレーム、と戻して 0.3 秒以内に収まる
        for f in presses[::5]:
            video_sec = (f * 1.012 + 150) / 60
            self.assertLess(abs(sync.rep_frame(0, video_sec) - f), 18)

    def test_inputs_before_a_hit_come_from_the_rep(self):
        stream, hp, _ = self._match()
        rounds = an.split_rounds(hp)
        sync = an.align_rep_to_video(hp, rounds, [tuple(stream)])
        events = an.detect_damage_events_from_hp(rounds, sync)
        self.assertTrue(events)
        # 対CPU戦の .rep には P2 の入力が無いので、P2 の被弾には直前の入力が付かない
        self.assertTrue(all(e.target == "p2" and not e.prev_inputs for e in events))
        self.assertEqual(len(sync.inputs_before(0, "p1", 30.0)), 90)

    def test_no_replay_means_no_sync(self):
        _, hp, _ = self._match()
        self.assertIsNone(an.align_rep_to_video(hp, an.split_rounds(hp), []))


class LiveVideoSyncTests(unittest.TestCase):
    """ライブ記録と動画の時刻合わせ"""

    def _live(self, seed, rounds=2):
        """ラウンドごとに両者がばらばらの間隔・大きさで削り合うライブ記録の HP"""
        import random
        rng = random.Random(seed)
        hp, t = [], 0
        for _ in range(rounds):
            p1 = p2 = 1.0
            hits = {f: (rng.choice((1, 2)), rng.uniform(0.02, 0.12))
                    for f in rng.sample(range(120, 2400, 30), 30)}
            for f in range(2700):
                if f in hits:
                    who, dmg = hits[f]
                    if who == 1:
                        p1 = max(0.05, p1 - dmg)
                    else:
                        p2 = max(0.05, p2 - dmg)
                if f == 2500:
                    p2 = 0.0
                hp.append((t / 60, p1, p2))
                t += 1
            for _ in range(300):               # ラウンド間
                hp.append((t / 60, 1.0, 1.0))
                t += 1
        return hp

    def _video(self, live_hp, lead_sec, rate=1.0):
        """live_hp を lead_sec 遅らせて 0.1 秒ごとに読んだ動画の HP（手前はメニュー画面）"""
        end = lead_sec + live_hp[-1][0] * rate
        out = []
        for i in range(int(end * 10)):
            v = i / 10
            k = int((v - lead_sec) / rate * 60)
            out.append((v, 1.0, 1.0) if k < 0 else (v, live_hp[k][1], live_hp[k][2]))
        return out

    def test_video_position_is_recovered(self):
        live = self._live(3)
        rounds = an.split_rounds(live)
        self.assertEqual(len(rounds), 2)
        sync = an.align_live_to_video(live, self._video(live, 23.4, rate=1.004), rounds)
        self.assertIsNotNone(sync)
        for sec in (10.0, 40.0, 60.0, 90.0):
            self.assertAlmostEqual(sync.video_sec(sec), 23.4 + sec * 1.004, delta=0.2)

    def test_round_missing_from_the_video_is_left_unsynced(self):
        live = self._live(3)
        rounds = an.split_rounds(live)
        video = [s for s in self._video(live, 5.0) if s[0] < 5.0 + rounds[0].end_sec + 2]
        sync = an.align_live_to_video(live, video, rounds)
        self.assertIsNotNone(sync.offsets[0])
        self.assertIsNone(sync.offsets[1])
        self.assertIsNone(sync.video_sec(rounds[1].start_sec + 10))

    def test_video_of_another_match_is_not_synced(self):
        live = self._live(3)
        other = self._video(self._live(8), 5.0)
        self.assertIsNone(an.align_live_to_video(live, other, an.split_rounds(live)))


class AnalyzeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._history_dir = ph.HISTORY_DIR
        ph.HISTORY_DIR = self.tmp / "history"

    def tearDown(self):
        ph.HISTORY_DIR = self._history_dir

    def _write(self, name, frames, matches):
        path = self.tmp / name
        path.write_text(json.dumps({"meta": {"version": 3, "matches": matches}, "frames": frames}),
                        encoding="utf-8")
        return path

    def _two_matches(self):
        return make_round(0, 0.0, match=1) + make_round(2000, 100.0, match=2)

    def test_multi_match_report_shows_characters_per_match(self):
        live = self._write("multi.json", self._two_matches(), [
            {"match": 1, "p1_char": "youmu", "p2_char": "patchouli"},
            {"match": 2, "p1_char": "aya", "p2_char": "patchouli"},
        ])
        out = self.tmp / "multi.html"
        an.analyze(None, None, out, live)
        report = out.read_text(encoding="utf-8")
        self.assertIn("魂魄妖夢 vs パチュリー", report)
        self.assertIn("射命丸文 vs パチュリー", report)

    def _session(self, chars):
        """試合ごとに、P1 が相手の JA(306) から 20% 食らい、試合 1 だけ 6C(411) からも 10% 食らう連戦"""
        frames = []
        for n, _ in enumerate(chars):
            group = make_round(n * 2000, n * 100.0, match=n + 1, winner_hp=10000)
            for i, f in enumerate(group):
                f["p2"]["act"] = 306 if 190 <= i < 205 else (411 if n == 0 and 490 <= i < 505 else 0)
                f["p1"]["hp"] -= 2000 * (i >= 200) + 1000 * (n == 0 and i >= 500)
            frames += group
        return self._write("s.json", frames, [
            {"match": n + 1, "p1_char": c1, "p2_char": c2} for n, (c1, c2) in enumerate(chars)
        ])

    def _advice_lines(self, live):
        out = self.tmp / "s.html"
        an.analyze(None, None, out, live)
        import html as html_mod
        import re
        return [html_mod.unescape(m) for m in re.findall(r"<li>(.*?)</li>", out.read_text(encoding="utf-8"))]

    def test_session_totals_and_trend_of_opponent_starters(self):
        lines = self._advice_lines(self._session([("aya", "yuyuko")] * 3))
        self.assertIn("⚠️ 全3試合での相手の始動技ごとの被ダメージ: JA 60%・3回、6C 10%・1回", lines)
        self.assertIn("  JA 始動の被ダメージの推移（試合順）: 20% → 20% → 20%", lines)
        self.assertIn("  6C 始動の被ダメージの推移（試合順）: 10% → 0% → 0%", lines)

    def test_session_totals_skip_matches_with_other_characters(self):
        lines = self._advice_lines(self._session([("aya", "yuyuko"), ("youmu", "yuyuko"), ("aya", "yuyuko")]))
        self.assertTrue(any(
            ln.startswith("⚠️ 射命丸文 vs 西行寺幽々子 の2試合（試合1・3）での相手の始動技ごとの被ダメージ: JA 40%・2回")
            for ln in lines))

    def test_history_is_not_duplicated_and_keeps_notable_advice(self):
        frames = self._two_matches()
        for f in frames:                      # P1 が 1 ラウンドに 1 回大きく食らう
            if f["f"] % 2000 > 300:
                f["p1"]["hp"] -= 3000
        live = self._write("h.json", frames, [])
        for _ in range(2):
            an.analyze(None, None, self.tmp / "h.html", live, player_name="tester", use_history=True)
        sessions = ph.load_history("tester")["sessions"]
        self.assertEqual(len(sessions), 1)
        self.assertTrue(sessions[0]["notable_advice"])   # 「【試合N】」付きの行も拾えている

        other = self._write("h2.json", frames, [])
        an.analyze(None, None, self.tmp / "h2.html", other, player_name="tester", use_history=True)
        trend = ph.build_trend_context("tester")
        self.assertEqual(trend["total_sessions_recorded"], 2)
        self.assertTrue(trend["repeatedly_flagged_issues"])

    def _hp_video(self, name, hp, lead_frames=0):
        """HP バーだけを描いた 640×480 の動画。hp は 1/60 秒ごとの (P1, P2)"""
        import cv2
        import numpy as np
        video = self.tmp / name
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 60, (640, 480))
        for p1, p2 in [(0.0, 0.0)] * lead_frames + hp:
            frame = np.zeros((480, 640, 3), np.uint8)
            frame[30:50, an.P1_X_START:an.P1_X_START + int(an.HP_BAR_FULL * p1)] = (40, 200, 255)
            frame[30:50, an.P2_X_END + 1 - int(an.HP_BAR_FULL * p2):an.P2_X_END + 1] = (40, 200, 255)
            writer.write(frame)
        writer.release()
        return video

    def _big_hit_frames(self):
        """P1 が記録の頭から削られ続け（被弾の開始は 0.0 秒）、P2 が削り切られるラウンド"""
        return make_round(0, 0.0)

    def test_report_links_to_the_synced_video_position(self):
        frames = self._big_hit_frames()
        live = self._write("v.json", frames, [])
        video = self._hp_video("v.mp4", [(f["p1"]["hp"] / 10000, f["p2"]["hp"] / 10000) for f in frames],
                               lead_frames=600)
        out = self.tmp / "v.html"
        an.analyze(video, None, out, live)
        report = out.read_text(encoding="utf-8")
        self.assertIn('<video id="matchVideo" src="v.mp4"', report)
        # 記録の 0.0 秒は、頭に 10 秒足した動画では 10.0 秒
        self.assertRegex(report, r'大ダメージ: 動画 <a class="seek" data-t="(9\.[89]|10\.[012])">[\d.]+秒</a>')
        self.assertRegex(report, r'<a class="seek" data-t="1[12]\.\d">動画 0:(09|10)〜</a>')
        self.assertIn("横軸は動画の再生位置", report)

    def test_video_only_report_links_without_sync(self):
        frames = self._big_hit_frames()
        video = self._hp_video("only.mp4", [(f["p1"]["hp"] / 10000, f["p2"]["hp"] / 10000) for f in frames])
        out = self.tmp / "only.html"
        an.analyze(video, None, out)
        report = out.read_text(encoding="utf-8")
        self.assertRegex(report, r'大ダメージ: <a class="seek" data-t="[01]\.\d">[01]\.\d秒</a>')

    def test_live_only_report_has_no_video_or_links(self):
        out = self.tmp / "live.html"
        an.analyze(None, None, out, self._write("l.json", self._big_hit_frames(), []))
        report = out.read_text(encoding="utf-8")
        # グラフ用の Chart.js はレポートに埋め込む（ネットが無くてもグラフが出る）
        self.assertIn("Chart.js v4", report)
        self.assertNotIn("cdn.jsdelivr.net", report)
        self.assertNotIn("<video", report)
        self.assertNotIn('class="seek" data-t', report)

    def test_wrong_aspect_ratio_video_is_rejected(self):
        import cv2
        import numpy as np
        video = self.tmp / "wide.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 60, (1280, 720))
        for _ in range(30):
            writer.write(np.zeros((720, 1280, 3), np.uint8))
        writer.release()
        with self.assertRaises(an.AnalyzeError):
            an.analyze(video, None, self.tmp / "wide.html")


@unittest.skipUnless(os.environ.get("SOKU_TEST_LIVE") and os.environ.get("SOKU_TEST_REP"),
                     "SOKU_TEST_LIVE / SOKU_TEST_REP が未設定")
class RealFileTests(unittest.TestCase):
    """同じ試合のライブ記録と .rep で、P1 のボタン押下率が合うことを確かめる"""

    def test_rep_matches_live_recording(self):
        raw = json.loads(Path(os.environ["SOKU_TEST_LIVE"]).read_text(encoding="utf-8-sig"))
        live_p1, _ = an.parse_live_stats_from_frames(raw["frames"])
        rep_p1 = an.parse_rep(Path(os.environ["SOKU_TEST_REP"]))[0]
        for key, bit in (("a", 0x10), ("b", 0x20), ("c", 0x40), ("d", 0x80)):
            self.assertAlmostEqual(live_p1.hold_pct(key), rep_p1.button_pct(bit), delta=1.5)


if __name__ == "__main__":
    unittest.main()
