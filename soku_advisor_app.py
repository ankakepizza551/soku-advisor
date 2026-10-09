"""
soku_advisor_app.py — Soku Advisor GUI アプリケーション

天則プロセスへの接続・ライブ入力記録・HTML レポート生成を
ひとつのウィンドウで操作できます。

依存: tkinter (標準), ctypes (標準), cv2 (opencv-python), numpy
"""

import ctypes
import json
import os
import queue
import sys
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path


def _enable_windows_dpi_awareness() -> None:
    """tkinter  import 前に呼ぶ。Per-Monitor V2 でネイティブ DPI 描画。"""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except Exception:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


_enable_windows_dpi_awareness()

import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, simpledialog, ttk

# ──────────────────────────────────────────────
# 起動時にスクリプトのディレクトリを sys.path に追加
# (exe 化された場合は sys._MEIPASS を使う)
# ──────────────────────────────────────────────
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys._MEIPASS)
else:
    BASE_DIR = Path(__file__).parent

sys.path.insert(0, str(BASE_DIR))

from soku_live_reader import (
    attach_to_game_detailed, close_process_handle, read_ubyte, SCENEID,
    LiveRecorder,
    is_current_process_admin, restart_current_process_as_admin,
)
from analyzer import analyze, AnalyzeError

# 動画・ライブ記録の欄に複数のファイルを入れる時の区切り（| はファイル名に使えない文字）
PATH_SEP = " | "
from char_advisor import char_id_choices, parse_char_choice
from ai_advisor import ai_available

# ──────────────────────────────────────────────
# テーマ
# ──────────────────────────────────────────────
BG          = "#0c0e14"
SURFACE     = "#151922"
SURFACE_ALT = "#1c2230"
BORDER      = "#2d3548"
ACCENT      = "#7eb6ff"
ACCENT_DIM  = "#4a7eb8"
TEXT        = "#eef2f8"
TEXT_DIM    = "#93a0b8"
TEXT_MUTED  = "#667085"
INPUT_BG    = "#0f131c"
P1_COLOR    = "#6eb5ff"
P2_COLOR    = "#ff8a8a"
GREEN       = "#5ecf8f"
RED         = "#ff7070"
ORANGE      = "#f0b45a"

BASE_WIDTH   = 660
BASE_HEIGHT  = 920
BASE_MINSIZE = (560, 640)

_FONT_FAMILY: str | None = None


def _pick_font_family(root: tk.Tk) -> str:
    global _FONT_FAMILY
    if _FONT_FAMILY:
        return _FONT_FAMILY
    families = set(tkfont.families(root))
    for name in ("Yu Gothic UI", "Meiryo UI", "Segoe UI Variable Text", "Segoe UI", "MS UI Gothic"):
        if name in families:
            _FONT_FAMILY = name
            return name
    _FONT_FAMILY = "TkDefaultFont"
    return _FONT_FAMILY


def _init_fonts(root: tk.Tk) -> None:
    family = _pick_font_family(root)
    root._ui_fonts = {
        "title":    tkfont.Font(root=root, family=family, size=20, weight="bold"),
        "subtitle": tkfont.Font(root=root, family=family, size=10),
        "section":  tkfont.Font(root=root, family=family, size=12, weight="bold"),
        "body":     tkfont.Font(root=root, family=family, size=11),
        "small":    tkfont.Font(root=root, family=family, size=10),
        "caption":  tkfont.Font(root=root, family=family, size=9),
        "mono":     tkfont.Font(root=root, family=family, size=10),
        "status":   tkfont.Font(root=root, family=family, size=11, weight="bold"),
        "btn":      tkfont.Font(root=root, family=family, size=11, weight="bold"),
    }


def _font(root: tk.Tk, key: str) -> tkfont.Font:
    return root._ui_fonts[key]


def _configure_style(root: tk.Tk) -> None:
    _init_fonts(root)
    style = ttk.Style(root)
    style.theme_use("clam")

    style.configure(".", background=SURFACE, foreground=TEXT, font=_font(root, "body"))
    style.configure("App.TFrame", background=BG)
    style.configure("Card.TFrame", background=SURFACE)
    style.configure("CardInner.TFrame", background=SURFACE)
    style.configure("Header.TFrame", background=BG)
    style.configure("StatusBar.TFrame", background=SURFACE_ALT)

    style.configure("Title.TLabel", background=BG, foreground=TEXT, font=_font(root, "title"))
    style.configure("Subtitle.TLabel", background=BG, foreground=TEXT_DIM, font=_font(root, "subtitle"))
    style.configure("Section.TLabel", background=SURFACE, foreground=ACCENT, font=_font(root, "section"))
    style.configure("Body.TLabel", background=SURFACE, foreground=TEXT, font=_font(root, "body"))
    style.configure("Dim.TLabel", background=SURFACE, foreground=TEXT_DIM, font=_font(root, "small"))
    style.configure("Caption.TLabel", background=SURFACE, foreground=TEXT_MUTED, font=_font(root, "caption"))
    style.configure("Rec.TLabel", background=SURFACE, foreground=TEXT_DIM, font=_font(root, "body"))
    style.configure("RecActive.TLabel", background=SURFACE, foreground=RED, font=_font(root, "status"))
    style.configure("Field.TLabel", background=SURFACE, foreground=TEXT_DIM,
                    font=_font(root, "small"), width=10, anchor="w")

    style.configure("TButton", background=SURFACE_ALT, foreground=TEXT,
                    bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
                    relief="flat", padding=(18, 10), font=_font(root, "body"))
    style.map("TButton",
              background=[("active", BORDER), ("pressed", ACCENT_DIM)],
              foreground=[("pressed", TEXT)])

    style.configure("Accent.TButton", background=ACCENT, foreground=BG,
                    font=_font(root, "btn"), padding=(20, 11))
    style.map("Accent.TButton",
              background=[("active", "#9ccaff"), ("disabled", BORDER)],
              foreground=[("disabled", TEXT_MUTED)])

    style.configure("Danger.TButton", background=RED, foreground=BG,
                    font=_font(root, "btn"), padding=(20, 11))
    style.map("Danger.TButton",
              background=[("active", "#ff9999"), ("disabled", BORDER)])

    style.configure("Ghost.TButton", background=SURFACE, foreground=TEXT_DIM,
                    padding=(10, 6), font=_font(root, "small"))
    style.map("Ghost.TButton", background=[("active", SURFACE_ALT)])

    style.configure("Browse.TButton", background=SURFACE_ALT, foreground=ACCENT,
                    bordercolor=ACCENT_DIM, lightcolor=ACCENT_DIM, darkcolor=ACCENT_DIM,
                    padding=(16, 8), font=_font(root, "small"), width=9)
    style.map("Browse.TButton",
              background=[("active", "#263248"), ("pressed", ACCENT_DIM)],
              foreground=[("active", "#9ccaff"), ("pressed", BG)])

    style.configure("Secondary.TButton", background=SURFACE_ALT, foreground=TEXT,
                    bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
                    padding=(14, 8), font=_font(root, "body"))
    style.map("Secondary.TButton", background=[("active", BORDER)])

    style.configure("TEntry", fieldbackground=INPUT_BG, foreground=TEXT,
                    insertcolor=ACCENT, bordercolor=BORDER,
                    lightcolor=BORDER, darkcolor=BORDER, padding=(8, 6),
                    font=_font(root, "body"))
    style.configure("TCombobox", fieldbackground=INPUT_BG, foreground=TEXT,
                    background=INPUT_BG, arrowsize=16, padding=(8, 6),
                    font=_font(root, "body"))
    style.map("TCombobox",
              fieldbackground=[("readonly", INPUT_BG)],
              foreground=[("readonly", TEXT)])

    style.configure("TRadiobutton", background=SURFACE, foreground=TEXT,
                    font=_font(root, "body"), padding=(0, 6))
    style.configure("TCheckbutton", background=SURFACE, foreground=TEXT_DIM,
                    font=_font(root, "small"), padding=(0, 6))

    style.configure("TProgressbar", troughcolor=INPUT_BG, background=ACCENT,
                    bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER, thickness=6)
    style.configure("TSeparator", background=BORDER)

    for name, color in (("Ok", GREEN), ("Warn", ORANGE), ("Err", RED), ("Idle", TEXT_DIM)):
        style.configure(f"Status{name}.TLabel", background=SURFACE_ALT,
                        foreground=color, font=_font(root, "status"))

    style.configure("AdminOk.TLabel", background=SURFACE, foreground=GREEN, font=_font(root, "caption"))


# ──────────────────────────────────────────────
# メインウィンドウ
# ──────────────────────────────────────────────
class SokuAdvisorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Soku Advisor")
        self.geometry(f"{BASE_WIDTH}x{BASE_HEIGHT}")
        self.minsize(*BASE_MINSIZE)
        self.configure(bg=BG)
        self.resizable(True, True)
        self._set_window_icon()

        _configure_style(self)

        # 状態変数
        self._recorder: LiveRecorder | None = None
        self._record_thread: threading.Thread | None = None
        self._recording = threading.Event()
        self._elapsed_sec = 0
        self._proc = None
        self._game_connected = False
        self._connection_status = "not_running"
        self._need_admin = False
        # 別スレッドから画面を更新したい時に積む。Tk を触るのはメインスレッドだけにする
        self._ui_queue: queue.Queue = queue.Queue()

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._log_admin_hint()
        self._poll_game()
        self._drain_ui_queue()
        self.after(100, self._fit_scroll_width)

    def _post(self, func, *args) -> None:
        """別スレッドから呼ぶ用。func(*args) をメインスレッドで実行させる"""
        self._ui_queue.put((func, args))

    def _drain_ui_queue(self) -> None:
        try:
            while True:
                func, args = self._ui_queue.get_nowait()
                func(*args)
        except queue.Empty:
            pass
        self.after(30, self._drain_ui_queue)

    def _on_close(self) -> None:
        # 記録中に閉じても、そこまでの記録は保存する
        if self._recording.is_set():
            self._stop_record(ask_rename=False)
        self.destroy()

    def _set_window_icon(self) -> None:
        if getattr(sys, "frozen", False):
            icon = Path(sys._MEIPASS) / "soku_advisor.ico"
        else:
            icon = Path(__file__).parent / "soku_advisor.ico"
        if icon.exists():
            try:
                self.iconbitmap(default=str(icon))
            except Exception:
                pass

    # ── UI 構築 ──────────────────────────────────

    def _fit_scroll_width(self, _event=None):
        if not hasattr(self, "_canvas"):
            return
        self._canvas.itemconfigure(self._scroll_window, width=self._canvas.winfo_width())

    def _bind_mousewheel(self, widget):
        def _on_wheel(event):
            if not hasattr(self, "_canvas"):
                return
            self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        widget.bind_all("<MouseWheel>", _on_wheel, add="+")

    def _card(self, parent, title: str) -> ttk.Frame:
        outer = ttk.Frame(parent, style="App.TFrame")
        outer.pack(fill="x", padx=20, pady=(0, 10))

        shell = tk.Frame(outer, bg=BORDER, padx=1, pady=1)
        shell.pack(fill="x")

        inner = ttk.Frame(shell, style="Card.TFrame", padding=(16, 14))
        inner.pack(fill="x")

        head = ttk.Frame(inner, style="Card.TFrame")
        head.pack(fill="x", pady=(0, 10))
        ttk.Label(head, text=title, style="Section.TLabel").pack(side="left")
        ttk.Separator(inner, orient="horizontal").pack(fill="x", pady=(0, 12))
        return inner

    def _file_row(self, parent, label: str, var: tk.StringVar, browse_cmd, readonly=False):
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=(0, 8))
        row.columnconfigure(1, weight=1)

        ttk.Label(row, text=label, style="Field.TLabel").grid(row=0, column=0, sticky="w")
        entry = ttk.Entry(row, textvariable=var)
        if readonly:
            entry.configure(state="readonly")
        entry.grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Button(row, text="参照...", style="Browse.TButton", command=browse_cmd).grid(
            row=0, column=2, sticky="e", ipadx=4)

    def _build_ui(self):
        outer = ttk.Frame(self, style="App.TFrame")
        outer.pack(fill="both", expand=True)

        self._canvas = tk.Canvas(outer, bg=BG, highlightthickness=0, borderwidth=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        root = ttk.Frame(self._canvas, style="App.TFrame", padding=(0, 16, 8, 16))
        self._scroll_window = self._canvas.create_window((0, 0), window=root, anchor="nw")
        root.bind("<Configure>", lambda _e: self._canvas.configure(
            scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>", self._fit_scroll_width)
        self._bind_mousewheel(self._canvas)

        # ── ヘッダー
        header = ttk.Frame(root, style="Header.TFrame")
        header.pack(fill="x", padx=20, pady=(0, 12))

        title_block = ttk.Frame(header, style="Header.TFrame")
        title_block.pack(side="left", fill="x", expand=True)
        ttk.Label(title_block, text="Soku Advisor", style="Title.TLabel").pack(anchor="w")
        ttk.Label(title_block, text="東方非想天則 — 対戦解析ツール", style="Subtitle.TLabel").pack(
            anchor="w", pady=(2, 0))

        # ── 接続ステータス
        status_card = self._card(root, "接続状態")
        status_row = ttk.Frame(status_card, style="StatusBar.TFrame", padding=(12, 10))
        status_row.pack(fill="x")

        self._status_dot = ttk.Label(status_row, text="●", style="StatusErr.TLabel")
        self._status_dot.pack(side="left", padx=(0, 8))
        self._status_text = ttk.Label(status_row, text="天則未起動", style="StatusErr.TLabel")
        self._status_text.pack(side="left")

        self._admin_label = ttk.Label(status_card, text="", style="Caption.TLabel")
        self._admin_label.pack(anchor="w", pady=(10, 0))

        self._admin_btn = ttk.Button(
            status_card, text="管理者として再起動",
            style="Secondary.TButton", command=self._restart_as_admin,
        )
        self._admin_btn.pack(anchor="w", pady=(8, 0))
        self._admin_btn.pack_forget()
        self._update_admin_label()

        self._build_record_section(root)
        self._build_report_section(root)
        self._build_log_section(root)

    def _build_record_section(self, root):
        frame = self._card(root, "ライブ記録")

        row1 = ttk.Frame(frame, style="Card.TFrame")
        row1.pack(fill="x", pady=(0, 10))

        self._rec_btn = ttk.Button(row1, text="記録開始", style="Accent.TButton",
                                   command=self._toggle_record, width=16)
        self._rec_btn.pack(side="left")

        self._rec_time_label = ttk.Label(row1, text="", style="Rec.TLabel")
        self._rec_time_label.pack(side="left", padx=(14, 0))

        path_row = ttk.Frame(frame, style="Card.TFrame")
        path_row.pack(fill="x")
        path_row.columnconfigure(1, weight=1)
        ttk.Label(path_row, text="保存先", style="Field.TLabel").grid(row=0, column=0, sticky="w")
        self._live_path_var = tk.StringVar(value="（自動：日時名で保存）")
        ttk.Entry(path_row, textvariable=self._live_path_var, state="readonly").grid(
            row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Button(path_row, text="参照...", style="Browse.TButton",
                   command=self._browse_live_out).grid(row=0, column=2, sticky="e", ipadx=4)

    def _build_report_section(self, root):
        frame = self._card(root, "レポート生成")

        self._video_var = tk.StringVar()
        self._json_var = tk.StringVar()
        self._rep_var = tk.StringVar()
        self._file_row(frame, "動画", self._video_var, self._browse_video)
        self._file_row(frame, "ライブ記録", self._json_var, self._browse_json)
        self._file_row(frame, ".rep", self._rep_var, self._browse_rep)

        opts = ttk.Frame(frame, style="Card.TFrame")
        opts.pack(fill="x", pady=(4, 0))
        opts.columnconfigure(1, weight=1)

        ttk.Label(opts, text="解説視点", style="Field.TLabel").grid(row=0, column=0, sticky="nw", pady=4)
        vp = ttk.Frame(opts, style="Card.TFrame")
        vp.grid(row=0, column=1, sticky="w", padx=(8, 0), pady=4)
        self._viewpoint_var = tk.IntVar(value=1)
        ttk.Radiobutton(vp, text="P1（左）", variable=self._viewpoint_var, value=1).pack(
            side="left", padx=(0, 16))
        ttk.Radiobutton(vp, text="P2（右）", variable=self._viewpoint_var, value=2).pack(side="left")

        char_opts = ["（自動）"] + char_id_choices()
        ttk.Label(opts, text="P1キャラ", style="Field.TLabel").grid(row=1, column=0, sticky="w", pady=4)
        self._p1_char_var = tk.StringVar(value="（自動）")
        p1_box = ttk.Combobox(opts, textvariable=self._p1_char_var, values=char_opts, state="readonly")
        p1_box.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=4)

        ttk.Label(opts, text="P2キャラ", style="Field.TLabel").grid(row=2, column=0, sticky="w", pady=4)
        self._p2_char_var = tk.StringVar(value="（自動）")
        p2_box = ttk.Combobox(opts, textvariable=self._p2_char_var, values=char_opts, state="readonly")
        p2_box.grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=4)
        # 欄の上でホイールを回しただけでキャラが変わらないようにする（開いた一覧の中では効く）
        for box in (p1_box, p2_box):
            box.bind("<MouseWheel>", lambda e: "break")

        ttk.Label(opts, text="プレイヤー名", style="Field.TLabel").grid(row=3, column=0, sticky="w", pady=4)
        self._player_name_var = tk.StringVar()
        ttk.Entry(opts, textvariable=self._player_name_var).grid(
            row=3, column=1, sticky="ew", padx=(8, 0), pady=4)

        self._hide_opp_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            opts, text="レポートで相手の名前を伏せる（「相手」と表示）",
            variable=self._hide_opp_var,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 2))

        self._collect_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            opts, text="使った動画・記録・.rep を1つのフォルダにまとめる（コピー）",
            variable=self._collect_var,
        ).grid(row=8, column=0, columnspan=2, sticky="w", pady=(0, 2))

        self._use_ai_var = tk.BooleanVar(value=False)
        ai_state = "normal" if ai_available() else "disabled"
        ttk.Checkbutton(
            opts, text="AI自然文コーチング（要 SOKU_AI_API_KEY）",
            variable=self._use_ai_var, state=ai_state,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(0, 2))

        self._use_history_var = tk.BooleanVar(value=True)
        self._history_cb = ttk.Checkbutton(
            opts, text="過去履歴を蓄積・参照する（プレイヤー名が必要）",
            variable=self._use_history_var,
        )
        self._history_cb.grid(row=6, column=0, columnspan=2, sticky="w", pady=(0, 4))

        self._history_label = ttk.Label(opts, text="", style="Caption.TLabel")
        self._history_label.grid(row=7, column=0, columnspan=2, sticky="w")
        self._player_name_var.trace_add("write", self._update_history_label)
        self._update_history_label()

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=(12, 12))

        self._progress = ttk.Progressbar(frame, mode="indeterminate")
        self._progress.pack(fill="x", pady=(0, 10))

        self._report_btn = ttk.Button(frame, text="レポート生成", style="Accent.TButton",
                                      command=self._generate_report)
        self._report_btn.pack(fill="x", ipady=4)

    def _build_log_section(self, root):
        frame = self._card(root, "ログ")

        self._log = tk.Text(
            frame, bg=INPUT_BG, fg=TEXT_DIM,
            font=_font(self, "mono"),
            height=5, relief="flat", wrap="word",
            borderwidth=0, highlightthickness=1,
            highlightbackground=BORDER, highlightcolor=BORDER,
            padx=10, pady=8, state="disabled",
        )
        self._log.pack(fill="x")

    # ── ゲーム接続ポーリング ──────────────────────

    def _log_admin_hint(self):
        if is_current_process_admin():
            self._log_msg("SokuAdvisor: 管理者権限で起動中")
        else:
            self._log_msg(
                "SokuAdvisor: 通常権限で起動中。"
                "天則を管理者起動している場合は「管理者として再起動」が必要です。"
            )

    def _update_admin_label(self):
        if is_current_process_admin():
            self._admin_label.config(text="本ツール: 管理者権限", style="AdminOk.TLabel")
        else:
            self._admin_label.config(text="本ツール: 通常権限", style="Caption.TLabel")

    def _restart_as_admin(self):
        if is_current_process_admin():
            messagebox.showinfo("再起動不要", "すでに管理者権限で起動しています。")
            return
        if not restart_current_process_as_admin():
            messagebox.showerror(
                "再起動失敗",
                "管理者権限での再起動がキャンセルまたは失敗しました。",
            )
            return
        self._on_close()

    def _disconnect_game(self, log_message: str | None = None):
        close_process_handle(self._proc)
        self._proc = None
        self._game_connected = False
        self._connection_status = "not_running"
        self._need_admin = False
        self._update_status("not_running")
        if log_message:
            self._log_msg(log_message)

    def _poll_game(self):
        """1.5秒ごとに天則プロセスの状態を確認する。"""
        try:
            if self._game_connected and self._proc:
                if read_ubyte(self._proc, SCENEID) is not None:
                    self.after(1500, self._poll_game)
                    return
                self._disconnect_game("天則プロセスが終了しました")

            result = attach_to_game_detailed()
            self._connection_status = result.status
            self._need_admin = result.status == "need_admin"

            if result.ok:
                if self._proc and self._proc != result.handle:
                    close_process_handle(self._proc)
                self._proc = result.handle
                if not self._game_connected:
                    self._game_connected = True
                    self._log_msg(result.message)
                self._update_status("connected")
            else:
                if self._game_connected:
                    self._disconnect_game(result.message)
                else:
                    self._update_status(result.status)
                    if result.status in ("need_admin", "access_denied", "read_failed"):
                        if getattr(self, "_last_warn_status", None) != result.status:
                            self._last_warn_status = result.status
                            self._log_msg(result.message)
                    elif result.status == "not_running":
                        self._last_warn_status = None
        except Exception as exc:
            if self._game_connected:
                self._disconnect_game(f"接続エラー: {exc}")
        self.after(1500, self._poll_game)

    def _update_status(self, status: str):
        if status == "connected":
            style, text = "StatusOk.TLabel", "天則接続中"
            self._admin_btn.pack_forget()
        elif status == "need_admin":
            style, text = "StatusWarn.TLabel", "権限不足（管理者再起動が必要）"
            self._admin_btn.pack(anchor="w", pady=(8, 0))
        elif status == "access_denied":
            style, text = "StatusWarn.TLabel", "接続拒否（権限/セキュリティソフト）"
            self._admin_btn.pack(anchor="w", pady=(8, 0))
        elif status == "read_failed":
            style, text = "StatusWarn.TLabel", "接続できましたが読み取り失敗"
            self._admin_btn.pack_forget()
        else:
            style, text = "StatusErr.TLabel", "天則未起動"
            self._admin_btn.pack_forget()

        self._status_dot.config(style=style)
        self._status_text.config(style=style, text=text)

    # ── 記録 ────────────────────────────────────

    def _toggle_record(self):
        if self._recording.is_set():
            self._stop_record()
        else:
            self._start_record()

    def _start_record(self):
        if self._need_admin:
            messagebox.showwarning(
                "管理者権限が必要",
                "天則が管理者権限で起動しています。\n"
                "SokuAdvisor も「管理者として再起動」してください。",
            )
            return
        if not self._game_connected:
            messagebox.showwarning("未接続", "天則を起動してから記録を開始してください。")
            return

        # 出力先の決定
        raw = self._live_path_var.get().strip()
        if not raw or raw.startswith("（"):
            out = Path.home() / "Videos" / f"soku_live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            out = Path(raw)
        out.parent.mkdir(parents=True, exist_ok=True)

        self._recorder = LiveRecorder(out)
        self._recorder.proc = self._proc
        self._recording.set()
        self._elapsed_sec = 0

        self._rec_btn.config(text="記録停止", style="Danger.TButton")
        self._log_msg(f"記録開始 → {out}")
        self._live_path_var.set(str(out))

        self._record_thread = threading.Thread(
            target=self._record_loop, daemon=True)
        self._record_thread.start()
        self._update_rec_time()

    def _record_loop(self):
        target = 1.0 / 60.0
        abort_msg = None
        next_t = time.perf_counter()
        while self._recording.is_set():
            try:
                ok = self._recorder.record_frame()
                if not ok:
                    abort_msg = "読み取りエラー — 記録停止"
                    break
            except Exception as e:
                abort_msg = f"エラー: {e}"
                break
            # 次の予定時刻まで待つ。sleep が寝過ごした分は次の周回で取り戻す
            next_t += target
            st = next_t - time.perf_counter()
            if st > 0:
                time.sleep(st)
            else:
                next_t = time.perf_counter()
        if abort_msg:
            # 異常終了時もそれまでの記録を保存する
            self._post(self._on_record_aborted, abort_msg)

    def _on_record_aborted(self, msg: str):
        self._log_msg(msg)
        if self._recording.is_set():
            self._stop_record()

    def _stop_record(self, ask_rename: bool = True):
        self._recording.clear()
        if self._record_thread:
            self._record_thread.join(timeout=2.0)

        self._rec_btn.config(text="記録開始", style="Accent.TButton")
        self._rec_time_label.config(text="", style="Rec.TLabel")
        self._live_path_var.set("（自動：日時名で保存）")

        if self._recorder and self._recorder.frames:
            try:
                self._recorder.save()
            except OSError as e:
                self._log_msg(f"記録の保存に失敗: {e}")
                messagebox.showerror("エラー", f"記録の保存に失敗しました:\n{e}")
                return
            n = self._recorder.frame_count
            match_count = self._recorder.match_id
            if match_count > 1:
                self._log_msg(
                    f"記録完了: {n}フレーム ({n/60:.1f}秒) / {match_count}試合"
                )
            else:
                self._log_msg(f"記録完了: {n}フレーム ({n/60:.1f}秒)")
            saved_path = self._recorder.output_path
            # 記録したJSONをレポート欄に自動セット
            self._json_var.set(str(saved_path))
            # リネームダイアログ
            if ask_rename:
                self._ask_rename(saved_path)

    def _ask_rename(self, current_path: Path):
        new_stem = simpledialog.askstring(
            "ファイル名の変更",
            f"記録ファイルを別名で保存しますか？\n"
            f"（空欄のままOKを押すと変更しません）\n\n"
            f"現在のファイル名: {current_path.name}",
            initialvalue=current_path.stem,
            parent=self,
        )
        if not new_stem or new_stem.strip() == current_path.stem:
            return
        new_name = new_stem.strip() + current_path.suffix
        new_path = current_path.parent / new_name
        try:
            current_path.rename(new_path)
            self._json_var.set(str(new_path))
            self._log_msg(f"ファイル名変更: {current_path.name} → {new_name}")
        except Exception as e:
            messagebox.showerror("エラー", f"ファイル名の変更に失敗しました:\n{e}")

    def _update_rec_time(self):
        if self._recording.is_set():
            if self._recorder:
                n = self._recorder.frame_count
                sec = n // 60
                self._rec_time_label.config(
                    text=f"記録中  {sec//60:02d}:{sec%60:02d}  ({n}フレーム)",
                    style="RecActive.TLabel")
            self.after(500, self._update_rec_time)
        else:
            self._rec_time_label.config(text="", style="Rec.TLabel")

    # ── ファイル参照 ──────────────────────────────

    def _browse_live_out(self):
        p = filedialog.asksaveasfilename(
            title="ライブ記録の保存先",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("すべて", "*.*")],
            initialdir=str(Path.home() / "Videos"),
        )
        if p:
            self._live_path_var.set(p)

    def _browse_video(self):
        picked = filedialog.askopenfilenames(
            title="動画ファイルを選択（試合ごとに分かれている時は複数選べます）",
            filetypes=[("動画", "*.mp4 *.avi *.mkv"), ("すべて", "*.*")],
            initialdir=str(Path.home() / "Videos"),
        )
        if picked:
            self._video_var.set(PATH_SEP.join(picked))
            # .rep を同ディレクトリで自動検索
            rep = Path(picked[0]).with_suffix(".rep")
            if len(picked) == 1 and rep.exists() and not self._rep_var.get():
                self._rep_var.set(str(rep))
                self._log_msg(f".rep 自動検出: {rep.name}")

    def _browse_json(self):
        picked = filedialog.askopenfilenames(
            title="ライブ記録JSONを選択（複数選ぶと連戦として1つのレポートになります）",
            filetypes=[("JSON", "*.json"), ("すべて", "*.*")],
            initialdir=str(Path.home() / "Videos"),
        )
        if picked:
            self._json_var.set(PATH_SEP.join(sorted(picked)))

    def _browse_rep(self):
        p = filedialog.askopenfilename(
            title=".rep ファイルを選択",
            filetypes=[("rep", "*.rep"), ("すべて", "*.*")],
        )
        if p:
            self._rep_var.set(p)

    # ── レポート生成 ──────────────────────────────

    def _update_history_label(self, *_):
        name = self._player_name_var.get().strip()
        if not name:
            self._history_label.config(text="※ プレイヤー名を入力すると履歴が有効になります")
            return
        try:
            from player_history import get_session_count
            count = get_session_count(name)
            if count == 0:
                self._history_label.config(text=f"「{name}」の履歴: まだありません（初回記録後から有効）")
            else:
                self._history_label.config(text=f"「{name}」の履歴: {count} セッション蓄積済み")
        except Exception:
            self._history_label.config(text="")

    def _generate_report(self):
        video = self._video_var.get().strip()
        live  = self._json_var.get().strip()
        rep   = self._rep_var.get().strip()

        if not video and not live:
            messagebox.showwarning(
                "入力不足",
                "動画ファイルまたはライブ記録JSONを指定してください。")
            return

        # 動画とライブ記録は複数入っていることがある（PATH_SEP 区切り）
        video_path = [Path(p.strip()) for p in video.split(PATH_SEP.strip()) if p.strip()]
        live_path  = [Path(p.strip()) for p in live.split(PATH_SEP.strip()) if p.strip()]
        rep_path   = Path(rep)   if rep   else None

        # 出力先の決定（複数ある時は最初のファイルの名前）
        out = (video_path or live_path)[0].with_suffix(".html")

        self._log_msg("レポート生成中...")
        self._progress.start(12)
        # 生成中の二重実行を防ぐ（完了・エラー時に戻す）
        self._report_btn.config(state="disabled")

        # Tk の変数は別スレッドから読まない。ここ（メインスレッド）で値にしておく
        _pname = self._player_name_var.get().strip() or None
        _use_hist = self._use_history_var.get() and bool(_pname)
        _viewpoint = self._viewpoint_var.get()
        _p1_char = parse_char_choice(self._p1_char_var.get())
        _p2_char = parse_char_choice(self._p2_char_var.get())
        _use_ai = self._use_ai_var.get()
        _hide_opp = self._hide_opp_var.get()
        _collect = self._collect_var.get()

        def run():
            try:
                # analyzer.py の analyze() を直接呼び出す
                # （別スレッドから Tk を触らないよう after 経由でログに流す）
                _redirect_print(lambda m: self._post(self._log_msg, m))
                done = analyze(
                    video_path, rep_path, out,
                    live_path, viewpoint=_viewpoint,
                    p1_char=_p1_char,
                    p2_char=_p2_char,
                    use_ai=_use_ai,
                    player_name=_pname,
                    use_history=_use_hist,
                    hide_opp_name=_hide_opp,
                    collect=_collect,
                )
                self._post(self._on_report_done, done)
            except AnalyzeError as e:
                self._post(self._on_report_error, str(e))
            except Exception as e:
                import traceback
                msg = traceback.format_exc()
                self._post(self._on_report_error, msg)
            finally:
                _restore_print()

        threading.Thread(target=run, daemon=True).start()

    def _on_report_done(self, out: Path):
        self._progress.stop()
        self._report_btn.config(state="normal")
        self._log_msg(f"レポート完成 → {out}")
        self._update_history_label()
        if messagebox.askyesno("完成", f"レポートを開きますか？\n{out}"):
            webbrowser.open(out.as_uri())

    def _on_report_error(self, msg: str):
        self._progress.stop()
        self._report_btn.config(state="normal")
        short = msg[-400:] if len(msg) > 400 else msg
        self._log_msg(f"エラー:\n{short}")
        messagebox.showerror("エラー", f"レポート生成に失敗しました。\n\n{short[:300]}")

    # ── ログ ─────────────────────────────────────

    def _log_msg(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        self._log.config(state="normal")
        self._log.insert("end", line)
        self._log.see("end")
        self._log.config(state="disabled")


# ──────────────────────────────────────────────
# print → ログリダイレクト (analyzer.py の print を GUI に流す)
# ──────────────────────────────────────────────
_orig_stdout = sys.stdout
_orig_stderr = sys.stderr

class _PrintCapture:
    def __init__(self, callback):
        self._cb = callback
        self._buf = ""

    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._cb(line)

    def flush(self):
        pass


def _redirect_print(callback):
    sys.stdout = _PrintCapture(callback)
    sys.stderr = _PrintCapture(callback)


def _restore_print():
    sys.stdout = _orig_stdout
    sys.stderr = _orig_stderr


# ──────────────────────────────────────────────
# エントリーポイント
# ──────────────────────────────────────────────
def main():
    app = SokuAdvisorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
