"""
soku_live_reader.py - 天則（th123）リアルタイム入力記録ツール

実行中の天則プロセスからP1・P2の入力とHPを毎フレーム読み取り、
JSONファイルに保存します。保存されたJSONはanalyzer.pyで解析できます。

使い方:
  python soku_live_reader.py [output.json]
  
  ゲームを起動した状態で実行してください。
  Ctrl+C または試合終了後に自動停止します。

依存: ctypes (標準ライブラリのみ、外部パッケージ不要)
"""

import ctypes
import ctypes.wintypes as wintypes
import json
import struct
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# =====================================================================
# メモリアドレス（天則 v1.10a 固定アドレス）
# =====================================================================
PBATTLEMGR       = 0x008985E4  # BattleManager ポインタ
SCENEID          = 0x008A0044  # 現在のシーン番号 (1バイト)
TIMECOUNT        = 0x008985D8  # ゲームフレームカウンタ
WEATHER_CURRENT  = 0x008971C0  # 現在天候
WEATHER_DISPLAY  = 0x008971C4  # 表示天候
WEATHER_COUNTER  = 0x008971CC  # 天候カウンタ

LCHARID          = 0x00899D10  # P1 (左) の選択キャラ番号
RCHARID          = 0x00899D30  # P2 (右) の選択キャラ番号

# キャラ番号 → char_data.json のID（実機で 11=文, 15=早苗 を確認）
CHAR_IDS = [
    "reimu", "marisa", "sakuya", "alice", "patchouli", "youmu", "remilia",
    "yuyuko", "yukari", "suika", "reisen", "aya", "komachi", "iku", "tenshi",
    "sanae", "cirno", "meiling", "utsuho", "suwako",
]

# BattleManager内のキャラオブジェクトオフセット
LCHAR_OFS = 0x0C  # P1 (左) キャラポインタ
RCHAR_OFS = 0x10  # P2 (右) キャラポインタ

# キャラオブジェクト内のフィールドオフセット
CF_CURRENT_HEALTH    = 0x184  # short: 現在HP
CF_MAX_HEALTH        = 0x182  # short: 最大HP (= 10000)
CF_CURRENT_SPIRIT    = 0x49E  # short: 霊力
CF_ACTION_ID         = 0x13C  # short: 現在のアクションID
CF_FRAME_COUNT       = 0x144  # int: アクション経過フレーム数
# 位置と向き（実機で確認: 開幕は P1 が x=480・向き 1、P2 が x=800・向き -1。地上は y=0）
CF_POS_X             = 0xEC   # float: 横位置
CF_POS_Y             = 0xF0   # float: 高さ
CF_DIRECTION         = 0x104  # char: 向き（1=右向き, -1=左向き）

# 入力オフセット（各値はint型）
# 注意: 定数名は歴史的経緯で PRESSED / HELD だが、実機の記録で確認した意味は以下。
#   0x754〜: 押し続けているフレーム数のカウンタ（離すと 0）。軸は符号が向き（左/上が負）。
#   0x774〜: 先行入力バッファ。押した直後の数フレームだけ値が入り、押しっぱなしでも 0 に戻る。
CF_PRESSED_X_AXIS    = 0x754  # 水平軸の押下フレーム数（左が負, 右が正）
CF_PRESSED_Y_AXIS    = 0x758  # 垂直軸の押下フレーム数（上が負, 下が正）
CF_PRESSED_A         = 0x75C  # A の押下フレーム数
CF_PRESSED_B         = 0x760  # B の押下フレーム数
CF_PRESSED_C         = 0x764  # C の押下フレーム数
CF_PRESSED_D         = 0x768  # D の押下フレーム数
CF_HELD_X_AXIS       = 0x774  # 水平軸（先行入力バッファ）
CF_HELD_Y_AXIS       = 0x778  # 垂直軸（先行入力バッファ）
CF_HELD_A            = 0x77C  # A（先行入力バッファ）
CF_HELD_B            = 0x780  # B（先行入力バッファ）
CF_HELD_C            = 0x784  # C（先行入力バッファ）
CF_HELD_D            = 0x788  # D（先行入力バッファ）
CF_PRESSED_COMBO     = 0x7C8  # コマンド入力のビットフラグ (236=bit1 など)

# シーンID
SCENE_BATTLE       = 0x05  # ローカル対戦/リプレイ再生
SCENE_BATTLE_SV    = 0x0D  # ネット対戦（ホスト）
SCENE_BATTLE_CL    = 0x0E  # ネット対戦（クライアント）
SCENE_BATTLE_WATCH = 0x0F  # 観戦
BATTLE_SCENES = (SCENE_BATTLE, SCENE_BATTLE_SV, SCENE_BATTLE_CL, SCENE_BATTLE_WATCH)

# =====================================================================
# Win32 API セットアップ
# =====================================================================
PROCESS_VM_READ = 0x0010
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
PROCESS_ACCESS = PROCESS_VM_READ | PROCESS_QUERY_LIMITED_INFORMATION

ERROR_ACCESS_DENIED = 5

TOKEN_QUERY = 0x0008
TokenElevation = 20

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
user32   = ctypes.WinDLL("user32")
advapi32 = ctypes.WinDLL("advapi32")
shell32  = ctypes.windll.shell32

ReadProcessMemory = kernel32.ReadProcessMemory
OpenProcess       = kernel32.OpenProcess
CloseHandle       = kernel32.CloseHandle

FindWindowW                = user32.FindWindowW
GetWindowThreadProcessId   = user32.GetWindowThreadProcessId

# 64bit の Python ではハンドルやサイズが 8 バイトなので、型を明示しておく
# （指定しないと 4 バイトの int として扱われる）
ReadProcessMemory.argtypes = [
    wintypes.HANDLE, wintypes.LPCVOID, wintypes.LPVOID,
    ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t),
]
ReadProcessMemory.restype = wintypes.BOOL
OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
OpenProcess.restype = wintypes.HANDLE
CloseHandle.argtypes = [wintypes.HANDLE]
CloseHandle.restype = wintypes.BOOL
FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
FindWindowW.restype = wintypes.HWND
GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
GetWindowThreadProcessId.restype = wintypes.DWORD
advapi32.OpenProcessToken.argtypes = [
    wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE),
]
advapi32.OpenProcessToken.restype = wintypes.BOOL
advapi32.GetTokenInformation.argtypes = [
    wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID,
    wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
]
advapi32.GetTokenInformation.restype = wintypes.BOOL

SOKU_WINDOW_CLASS = "th123_110a"


def is_current_process_admin() -> bool:
    """現在のプロセスが管理者権限で動作しているか。"""
    try:
        return bool(shell32.IsUserAnAdmin())
    except Exception:
        return False


def is_process_elevated(pid: int) -> Optional[bool]:
    """対象プロセスが管理者（昇格）で動作しているか。"""
    h_process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h_process:
        return None

    h_token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(h_process, TOKEN_QUERY, ctypes.byref(h_token)):
        CloseHandle(h_process)
        return None

    elevation = ctypes.c_ulong()
    size = ctypes.c_ulong(ctypes.sizeof(elevation))
    ok = advapi32.GetTokenInformation(
        h_token,
        TokenElevation,
        ctypes.byref(elevation),
        size,
        ctypes.byref(size),
    )
    CloseHandle(h_token)
    CloseHandle(h_process)
    if not ok:
        return None
    return bool(elevation.value)


@dataclass
class AttachResult:
    handle: Optional[int] = None
    status: str = "not_running"  # connected | not_running | need_admin | access_denied | read_failed
    pid: int = 0
    game_elevated: Optional[bool] = None
    self_elevated: bool = False
    error_code: int = 0

    @property
    def ok(self) -> bool:
        return self.handle is not None

    @property
    def message(self) -> str:
        if self.status == "connected":
            return "天則プロセスに接続しました"
        if self.status == "not_running":
            return "天則が起動していません"
        if self.status == "need_admin":
            return (
                "天則は管理者権限で起動中です。"
                "SokuAdvisor も管理者として再起動してください。"
            )
        if self.status == "access_denied":
            return "天則プロセスへのアクセスが拒否されました（権限またはセキュリティソフト）"
        if self.status == "read_failed":
            return "天則プロセスには接続できましたが、メモリ読み取りに失敗しました"
        return "天則への接続に失敗しました"


def _read_mem(proc_handle, address: int, size: int) -> Optional[bytes]:
    buf = ctypes.create_string_buffer(size)
    n_read = ctypes.c_size_t(0)
    ok = ReadProcessMemory(proc_handle, address, buf, size, ctypes.byref(n_read))
    if ok and n_read.value == size:
        return bytes(buf)
    return None


def read_int(proc, addr: int) -> Optional[int]:
    b = _read_mem(proc, addr, 4)
    if b is None:
        return None
    return int.from_bytes(b, "little", signed=True)


def read_uint(proc, addr: int) -> Optional[int]:
    b = _read_mem(proc, addr, 4)
    if b is None:
        return None
    return int.from_bytes(b, "little", signed=False)


def read_short(proc, addr: int) -> Optional[int]:
    b = _read_mem(proc, addr, 2)
    if b is None:
        return None
    return int.from_bytes(b, "little", signed=True)


def read_ubyte(proc, addr: int) -> Optional[int]:
    b = _read_mem(proc, addr, 1)
    if b is None:
        return None
    return b[0]


def read_float(proc, addr: int) -> Optional[float]:
    b = _read_mem(proc, addr, 4)
    if b is None:
        return None
    return struct.unpack("<f", b)[0]


# =====================================================================
# プロセス接続
# =====================================================================

def attach_to_game_detailed() -> AttachResult:
    """天則プロセスへの接続結果を詳細に返す。"""
    result = AttachResult(self_elevated=is_current_process_admin())

    hwnd = FindWindowW(SOKU_WINDOW_CLASS, None)
    if not hwnd:
        return result

    pid = ctypes.c_ulong(0)
    GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if pid.value == 0:
        result.status = "not_running"
        return result

    result.pid = pid.value
    result.game_elevated = is_process_elevated(pid.value)

    handle = OpenProcess(PROCESS_ACCESS, False, pid.value)
    if not handle:
        result.error_code = ctypes.get_last_error()
        if result.error_code == ERROR_ACCESS_DENIED:
            if not result.self_elevated and result.game_elevated is not False:
                result.status = "need_admin"
            else:
                result.status = "access_denied"
        else:
            result.status = "access_denied"
        return result

    if read_ubyte(handle, SCENEID) is None:
        CloseHandle(handle)
        result.status = "read_failed"
        return result

    result.handle = handle
    result.status = "connected"
    return result


def close_process_handle(handle: int | None) -> None:
    """OpenProcess で取得したハンドルを解放する。"""
    if handle:
        CloseHandle(handle)


def attach_to_game() -> Optional[int]:
    """天則プロセスに接続してハンドルを返す。失敗時は None。"""
    return attach_to_game_detailed().handle


def restart_current_process_as_admin() -> bool:
    """現在のプロセスを管理者権限で再起動する。UAC ダイアログが表示される。"""
    if is_current_process_admin():
        return True

    exe = sys.executable
    # exe 化していない時は python.exe を起動するので、スクリプトのパス（argv[0]）も渡す
    args = sys.argv[1:] if getattr(sys, "frozen", False) else sys.argv
    params = " ".join(f'"{arg}"' for arg in args)
    ret = shell32.ShellExecuteW(None, "runas", exe, params or None, None, 1)
    return ret > 32


# =====================================================================
# ゲーム状態読み取り
# =====================================================================

@dataclass
class CharState:
    hp: int          # 現在HP (0-10000)
    spirit: int      # 霊力 (0-1000)
    action_id: int   # アクションID
    # 押した瞬間（単発検出に使う）
    pressed_x: int   # -1,0,1
    pressed_y: int   # -1,0,1
    pressed_a: int
    pressed_b: int
    pressed_c: int
    pressed_d: int
    # ホールド中（持続検出に使う）
    held_x: int
    held_y: int
    held_a: int
    held_b: int
    held_c: int
    held_d: int
    # 特殊コマンド入力
    combo_code: int  # ビットフラグ。COMBO_NAMES を参照
    # 位置と向き
    pos_x: float = 0.0
    pos_y: float = 0.0
    facing: int = 0  # 1=右向き, -1=左向き


def read_char_state(proc, char_ptr: int) -> Optional[CharState]:
    """キャラポインタからCharStateを読み取る。"""
    if char_ptr == 0:
        return None

    def ri(offset):
        return read_int(proc, char_ptr + offset) or 0

    def rs(offset):
        return read_short(proc, char_ptr + offset) or 0

    return CharState(
        hp       = max(0, rs(CF_CURRENT_HEALTH)),
        spirit   = max(0, rs(CF_CURRENT_SPIRIT)),
        action_id = rs(CF_ACTION_ID),
        pressed_x = ri(CF_PRESSED_X_AXIS),
        pressed_y = ri(CF_PRESSED_Y_AXIS),
        pressed_a = ri(CF_PRESSED_A),
        pressed_b = ri(CF_PRESSED_B),
        pressed_c = ri(CF_PRESSED_C),
        pressed_d = ri(CF_PRESSED_D),
        held_x   = ri(CF_HELD_X_AXIS),
        held_y   = ri(CF_HELD_Y_AXIS),
        held_a   = ri(CF_HELD_A),
        held_b   = ri(CF_HELD_B),
        held_c   = ri(CF_HELD_C),
        held_d   = ri(CF_HELD_D),
        combo_code = ri(CF_PRESSED_COMBO),
        pos_x    = read_float(proc, char_ptr + CF_POS_X) or 0.0,
        pos_y    = read_float(proc, char_ptr + CF_POS_Y) or 0.0,
        facing   = -1 if (read_ubyte(proc, char_ptr + CF_DIRECTION) or 0) > 127 else 1,
    )


def encode_input(cs: CharState) -> dict:
    """CharStateから入力情報を辞書化する。"""
    # 方向をビットマップで表す（repファイルと同じ形式）
    direction = 0
    if cs.held_y < 0:  direction |= 0x01  # 上
    if cs.held_y > 0:  direction |= 0x02  # 下
    if cs.held_x < 0:  direction |= 0x04  # 左
    if cs.held_x > 0:  direction |= 0x08  # 右
    if cs.held_a:      direction |= 0x10  # A
    if cs.held_b:      direction |= 0x20  # B
    if cs.held_c:      direction |= 0x40  # C
    if cs.held_d:      direction |= 0x80  # D(ダッシュ)

    return {
        "dir": direction,
        "a": cs.pressed_a,
        "b": cs.pressed_b,
        "c": cs.pressed_c,
        "d": cs.pressed_d,
        "x": cs.pressed_x,
        "y": cs.pressed_y,
        "hx": cs.held_x,
        "hy": cs.held_y,
        "combo": cs.combo_code,
        # x / y は入力に使っているので、位置は px / py
        "px": round(cs.pos_x),
        "py": round(cs.pos_y),
        "face": cs.facing,
    }


# =====================================================================
# コンボコードのデコード
# =====================================================================
# 4ビットごとに1コマンドで、その中の並びが A/B/C/D。キーは各コマンドの B のビット
# （analyzer.py の COMBO_NAMES と同じ対応）。
COMBO_NAMES = {
    2:         "236 (波動)",
    32:        "214 (逆波動)",
    512:       "623 (昇龍)",
    8192:      "421 (逆昇龍)",
    536870912: "22 (下下)",
}


def decode_combo(code: int) -> str:
    for bit, name in COMBO_NAMES.items():
        if code & ((bit >> 1) * 0xF):  # そのコマンドの A/B/C/D どれか
            return name
    return f"combo({code})"


# =====================================================================
# メインレコーダー
# =====================================================================

class LiveRecorder:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.proc = None
        self.frames: list[dict] = []
        self.start_time = None
        self.frame_count = 0
        self.in_battle = False
        self.match_id = 0
        self.match_chars: list[dict] = []  # 試合ごとの {"match", "p1_char", "p2_char"}

    def _read_char(self, addr: int) -> Optional[str]:
        """選択キャラ番号を char_data.json のIDにして返す。範囲外・読めない時は None。"""
        n = read_int(self.proc, addr)
        if n is None or not 0 <= n < len(CHAR_IDS):
            return None
        return CHAR_IDS[n]

    def connect(self) -> bool:
        result = attach_to_game_detailed()
        if result.ok:
            self.proc = result.handle
            print(f"[INFO] {result.message}")
            return True

        print(f"[WARN] {result.message}")
        if result.status == "need_admin":
            print("[WARN] SokuAdvisor を管理者として実行してください。")
        return False

    def record_frame(self) -> bool:
        """1フレーム分のデータを記録。バトル中のみ記録する。True=継続, False=停止。"""
        if self.proc is None:
            return False

        scene = read_ubyte(self.proc, SCENEID)
        if scene is None:
            return False

        is_battle = scene in BATTLE_SCENES

        if is_battle and not self.in_battle:
            self.match_id += 1
            chars = {
                "match": self.match_id,
                "p1_char": self._read_char(LCHARID),
                "p2_char": self._read_char(RCHARID),
            }
            self.match_chars.append(chars)
            print(f"[INFO] 試合開始 #{self.match_id} (scene={scene:#04x}) "
                  f"{chars['p1_char'] or '?'} vs {chars['p2_char'] or '?'}")
            self.in_battle = True

        if not is_battle and self.in_battle:
            print(f"[INFO] 試合終了 #{self.match_id} (scene={scene:#04x})")
            self.in_battle = False

        if not is_battle:
            return True  # バトル外はスキップ（継続）

        # バトルマネージャーからキャラポインタを取得
        battle_ptr = read_uint(self.proc, PBATTLEMGR)
        if not battle_ptr:
            return True

        p1_char_ptr = read_uint(self.proc, battle_ptr + LCHAR_OFS)
        p2_char_ptr = read_uint(self.proc, battle_ptr + RCHAR_OFS)

        if not p1_char_ptr or not p2_char_ptr:
            return True

        p1 = read_char_state(self.proc, p1_char_ptr)
        p2 = read_char_state(self.proc, p2_char_ptr)

        if p1 is None or p2 is None:
            return True

        now = time.time()
        elapsed = now - self.start_time if self.start_time else 0.0
        if self.start_time is None:
            self.start_time = now
            elapsed = 0.0

        frame_data = {
            "t": round(elapsed, 4),
            "f": self.frame_count,
            "match": self.match_id,
            "p1": {
                "hp": p1.hp,
                "sp": p1.spirit,
                "act": p1.action_id,
                **encode_input(p1),
            },
            "p2": {
                "hp": p2.hp,
                "sp": p2.spirit,
                "act": p2.action_id,
                **encode_input(p2),
            },
        }
        self.frames.append(frame_data)
        self.frame_count += 1
        return True

    def save(self):
        """記録データをJSONに保存する。"""
        meta = {
            # 3: p1/p2 に x, y（方向の押下フレーム数）、meta に matches（キャラ）を追加
            # 4: p1/p2 に px, py（位置）、face（向き）を追加
            "version": 4,
            "recorded_at": datetime.now().isoformat(),
            "total_frames": self.frame_count,
            "match_count": self.match_id,
            "duration_sec": self.frames[-1]["t"] if self.frames else 0.0,
            "fps": 60,
            "matches": self.match_chars,
        }
        output = {"meta": meta, "frames": self.frames}
        self.output_path.write_text(
            json.dumps(output, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        print(f"[INFO] {self.frame_count}フレーム → {self.output_path}")


# =====================================================================
# エントリーポイント
# =====================================================================

def main():
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    print(f"[INFO] 出力先: {out_path}")
    print("[INFO] Ctrl+C で停止・保存")

    recorder = LiveRecorder(out_path)

    # プロセスを探すループ
    while not recorder.connect():
        try:
            time.sleep(2.0)
        except KeyboardInterrupt:
            print("\n[INFO] 中断されました")
            return

    target_interval = 1.0 / 60.0
    print("[INFO] 記録中... (天則が対戦画面になると記録開始)")

    try:
        while True:
            t0 = time.perf_counter()
            ok = recorder.record_frame()
            if not ok:
                print("[WARN] 読み取りエラー。プロセスが終了した可能性があります。")
                break
            elapsed = time.perf_counter() - t0
            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\n[INFO] 停止しました")

    if recorder.frames:
        recorder.save()
        print(f"[INFO] 記録完了: {recorder.frame_count}フレーム ({recorder.frame_count/60:.1f}秒)")
        # 簡易サマリーを表示
        _print_summary(recorder.frames)
    else:
        print("[INFO] 記録データがありません（バトルは行われませんでした）")


def _print_summary(frames: list[dict]):
    """記録データの簡易サマリーを表示。"""
    total = len(frames)
    if total == 0:
        return

    p1_a = sum(1 for f in frames if f["p1"]["a"])
    p1_b = sum(1 for f in frames if f["p1"]["b"])
    p1_c = sum(1 for f in frames if f["p1"]["c"])
    p2_a = sum(1 for f in frames if f["p2"]["a"])
    p2_b = sum(1 for f in frames if f["p2"]["b"])
    p2_c = sum(1 for f in frames if f["p2"]["c"])

    p1_combos: dict[str, int] = {}
    p2_combos: dict[str, int] = {}
    for f in frames:
        c1 = decode_combo(f["p1"]["combo"]) if f["p1"]["combo"] else ""
        c2 = decode_combo(f["p2"]["combo"]) if f["p2"]["combo"] else ""
        if c1:
            p1_combos[c1] = p1_combos.get(c1, 0) + 1
        if c2:
            p2_combos[c2] = p2_combos.get(c2, 0) + 1

    print("\n=== 記録サマリー ===")
    print(f"総フレーム数: {total} ({total/60:.1f}秒)")
    print(f"P1 ボタン: A={p1_a}回, B={p1_b}回, C={p1_c}回")
    print(f"P2 ボタン: A={p2_a}回, B={p2_b}回, C={p2_c}回")
    if p1_combos:
        print("P1 コマンド入力:")
        for code, count in sorted(p1_combos.items(), key=lambda x: -x[1])[:5]:
            print(f"  {code}: {count}フレーム")
    if p2_combos:
        print("P2 コマンド入力:")
        for code, count in sorted(p2_combos.items(), key=lambda x: -x[1])[:5]:
            print(f"  {code}: {count}フレーム")


if __name__ == "__main__":
    main()
