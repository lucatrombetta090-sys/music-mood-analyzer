"""
Music Mood Analyzer — Kivy Android version
Riscrittura completa da ttkbootstrap+pygame → Kivy per generazione APK Android.

Dipendenze:
  kivy, plyer, numpy, librosa, mutagen, sqlite3 (built-in)

Audio: kivy.core.audio.SoundLoader (Android: MediaPlayer backend)
File picker: plyer.filechooser
"""

import os
import threading
import random
from pathlib import Path
from typing import Optional

# ── Kivy config (prima di qualsiasi import kivy) ──────────────────────────────
import kivy
kivy.require("2.3.0")

from kivy.config import Config
Config.set("graphics", "width",  "400")
Config.set("graphics", "height", "750")

from kivy.app           import App
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.uix.boxlayout     import BoxLayout
from kivy.uix.gridlayout    import GridLayout
from kivy.uix.scrollview    import ScrollView
from kivy.uix.recycleview   import RecycleView
from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.label         import Label
from kivy.uix.button        import Button
from kivy.uix.textinput     import TextInput
from kivy.uix.slider        import Slider
from kivy.uix.progressbar   import ProgressBar
from kivy.uix.spinner       import Spinner
from kivy.uix.popup         import Popup
from kivy.uix.widget        import Widget
from kivy.clock             import Clock
from kivy.core.audio        import SoundLoader
from kivy.metrics           import dp, sp
from kivy.properties        import (StringProperty, NumericProperty,
                                    BooleanProperty, ListProperty,
                                    ObjectProperty)
from kivy.graphics          import Color, RoundedRectangle, Rectangle
from kivy.utils             import get_color_from_hex

# Backend (invariato dal progetto originale)
from analyze_mp3      import scan_folders
import listening_history as lh

# ── Palette colori ────────────────────────────────────────────────────────────
C = {
    "bg":        "#121212",
    "surface":   "#1E1E2E",
    "card":      "#2A2A3E",
    "primary":   "#7C3AED",
    "accent":    "#A855F7",
    "text":      "#E2E8F0",
    "subtext":   "#94A3B8",
    "energetic": "#F59E0B",
    "positive":  "#10B981",
    "aggressive":"#EF4444",
    "melancholic":"#8B5CF6",
    "success":   "#22C55E",
    "danger":    "#EF4444",
}

MOOD_COLORS = {
    "Energetic":   C["energetic"],
    "Positive":    C["positive"],
    "Aggressive":  C["aggressive"],
    "Melancholic": C["melancholic"],
}

MOOD_ICONS = {
    "Energetic":   "⚡",
    "Positive":    "😊",
    "Aggressive":  "🔥",
    "Melancholic": "🌧",
}

MOODS  = ["Energetic", "Positive", "Aggressive", "Melancholic"]
GENRES = ["Pop", "Rock", "Electronic", "HipHop", "Acoustic", "Jazz", "Classical"]


def hex_color(h: str, alpha: float = 1.0):
    """Converte colore hex → rgba tuple per Kivy."""
    c = get_color_from_hex(h)
    return (c[0], c[1], c[2], alpha)


# =============================================================================
# AppState — stato globale condiviso tra le schermate
# =============================================================================
class AppState:
    def __init__(self):
        self.all_songs:        list[dict] = []
        self.current_playlist: list[dict] = []
        self.playlist_index:   int        = 0
        self.current_song:     Optional[dict] = None
        self._sound:           Optional[object] = None   # SoundLoader instance
        self._loop:            bool       = False
        self._shuffle:         bool       = False
        self._scanning:        bool       = False
        self.scan_progress:    float      = 0.0
        self.scan_status:      str        = ""
        self.callbacks:        dict       = {}  # "on_song_change", "on_list_change"

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def register(self, event: str, fn):
        self.callbacks.setdefault(event, []).append(fn)

    def fire(self, event: str, *args):
        for fn in self.callbacks.get(event, []):
            try:
                fn(*args)
            except Exception as e:
                print(f"[callback error] {event}: {e}")

    # ── Audio ──────────────────────────────────────────────────────────────────
    def play_song(self, song: dict):
        self._stop_sound()
        try:
            sound = SoundLoader.load(song["path"])
            if sound is None:
                print(f"[audio] impossibile caricare: {song['path']}")
                return
            self._sound = sound
            self._sound.volume = getattr(self, "_volume", 1.0)
            self._sound.bind(on_stop=self._on_sound_stop)
            self._sound.play()
            self.current_song = song
            lh.log_play(song)
            self.fire("on_song_change", song)
        except Exception as e:
            print(f"[audio] errore riproduzione: {e}")

    def _stop_sound(self):
        if self._sound:
            try:
                self._sound.stop()
                self._sound.unload()
            except Exception:
                pass
            self._sound = None

    def _on_sound_stop(self, *args):
        if self._loop and self.current_song:
            self.play_song(self.current_song)
        elif self.current_playlist:
            self.next_song()

    def toggle_play_pause(self):
        if not self._sound:
            return
        if self._sound.state == "play":
            self._sound.stop()
        else:
            self._sound.play()
        self.fire("on_song_change", self.current_song)

    def next_song(self):
        if not self.current_playlist:
            return
        self.playlist_index = (self.playlist_index + 1) % len(self.current_playlist)
        self.play_song(self.current_playlist[self.playlist_index])

    def prev_song(self):
        if not self.current_playlist:
            return
        self.playlist_index = (self.playlist_index - 1) % len(self.current_playlist)
        self.play_song(self.current_playlist[self.playlist_index])

    def stop(self):
        self._stop_sound()
        self.current_song = None
        self.fire("on_song_change", None)

    def set_volume(self, v: float):
        self._volume = v
        if self._sound:
            self._sound.volume = v

    def toggle_loop(self):
        self._loop = not self._loop

    def toggle_shuffle(self):
        self._shuffle = not self._shuffle
        if self._shuffle and self.current_playlist:
            random.shuffle(self.current_playlist)
            self.playlist_index = 0

    def get_position(self) -> tuple[float, float]:
        """(pos_seconds, duration_seconds)"""
        if self._sound:
            pos = self._sound.get_pos() or 0.0
            dur = self._sound.length or 0.0
            return pos, dur
        return 0.0, 0.0

    def is_playing(self) -> bool:
        return self._sound is not None and self._sound.state == "play"

    # ── Playlist per mood ──────────────────────────────────────────────────────
    def play_mood_playlist(self, mood: str, genre_filter: str = "All"):
        playlist = [
            s for s in self.all_songs
            if s.get("mood") == mood and
               (genre_filter == "All" or s.get("genre") == genre_filter)
        ]
        if not playlist:
            return False
        if self._shuffle:
            random.shuffle(playlist)
        self.current_playlist = playlist
        self.playlist_index   = 0
        self.play_song(playlist[0])
        return True

    def play_filtered(self, songs: list[dict], index: int = 0):
        if not songs:
            return
        self.current_playlist = songs[:]
        if self._shuffle:
            random.shuffle(self.current_playlist)
            index = 0
        self.playlist_index = index
        self.play_song(self.current_playlist[index])


# Singleton
state = AppState()
lh.init_database()


# =============================================================================
# Componenti UI riusabili
# =============================================================================

class DarkButton(Button):
    """Pulsante scuro con bordi arrotondati."""
    bg_color = ListProperty([0.42, 0.20, 0.78, 1])  # viola primario

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_normal   = ""
        self.background_color    = [0, 0, 0, 0]
        self.color               = hex_color(C["text"])
        self.font_size           = sp(13)
        self.bind(pos=self._redraw, size=self._redraw)

    def _redraw(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*self.bg_color)
            RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(8)])

    def on_bg_color(self, *args):
        self._redraw()


class MoodButton(DarkButton):
    mood = StringProperty("")

    def __init__(self, mood: str, **kwargs):
        color = hex_color(MOOD_COLORS.get(mood, C["primary"]))
        super().__init__(bg_color=color,
                         text=f"{MOOD_ICONS.get(mood, '')} {mood}",
                         **kwargs)
        self.mood = mood


class SongRow(RecycleDataViewBehavior, BoxLayout):
    """Riga della song list con RecycleView."""
    index      = NumericProperty(0)
    song_data  = ObjectProperty(None, allownone=True)
    selected   = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.padding     = [dp(12), dp(6)]
        self.spacing     = dp(2)
        self.size_hint_y = None
        self.height      = dp(72)

        self.title_lbl   = Label(text="", font_size=sp(13), bold=True,
                                 color=hex_color(C["text"]),
                                 halign="left", valign="middle",
                                 size_hint_y=None, height=dp(22),
                                 text_size=(None, None))
        self.meta_lbl    = Label(text="", font_size=sp(11),
                                 color=hex_color(C["subtext"]),
                                 halign="left", valign="middle",
                                 size_hint_y=None, height=dp(18),
                                 text_size=(None, None))
        self.add_widget(self.title_lbl)
        self.add_widget(self.meta_lbl)

        self.bind(pos=self._redraw, size=self._redraw)

    def _redraw(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            if self.selected:
                Color(*hex_color(C["primary"], 0.3))
            else:
                Color(*hex_color(C["card"]))
            RoundedRectangle(pos=(self.x + dp(4), self.y + dp(2)),
                             size=(self.width - dp(8), self.height - dp(4)),
                             radius=[dp(8)])

    def refresh_view_attrs(self, rv, index, data):
        self.index     = index
        self.song_data = data.get("song")
        self.selected  = data.get("selected", False)

        song = data.get("song", {})
        mood = song.get("mood", "")
        icon = MOOD_ICONS.get(mood, "•")
        title = song.get("title", "?")
        # Rimuovi estensione
        if title.lower().endswith(".mp3"):
            title = title[:-4]
        # Tronca
        if len(title) > 38:
            title = title[:38] + "…"

        genre = song.get("genre", "?")
        alt_g = song.get("genre_alt")
        if alt_g:
            genre += f"/{alt_g}"
        bpm = song.get("tempo", 0)

        mood_color = MOOD_COLORS.get(mood, C["subtext"])

        self.title_lbl.text       = f"{icon} {title}"
        self.title_lbl.color      = hex_color(mood_color)
        self.meta_lbl.text        = f"{genre}  ·  {bpm:.0f} BPM  ·  {mood}"
        self._redraw()
        return super().refresh_view_attrs(rv, index, data)

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            if self.song_data:
                App.get_running_app().on_song_tap(self.song_data, self.index)
            return True
        return super().on_touch_down(touch)


class SongRecycleView(RecycleView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.viewclass = "SongRow"
        self.data      = []
        with self.canvas.before:
            Color(*hex_color(C["bg"]))
            self._bg = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._upd_bg, size=self._upd_bg)

    def _upd_bg(self, *args):
        self._bg.pos  = self.pos
        self._bg.size = self.size


# =============================================================================
# Libreria Screen
# =============================================================================
class LibraryScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._filtered: list[dict] = []
        self._build()
        state.register("on_list_change", self._refresh)
        state.register("on_song_change", self._highlight_current)

    def _build(self):
        root = BoxLayout(orientation="vertical", spacing=dp(6),
                         padding=[dp(10), dp(8), dp(10), dp(8)])
        root.canvas.before.clear()
        with root.canvas.before:
            Color(*hex_color(C["bg"]))
            Rectangle(pos=root.pos, size=root.size)

        # ── Header ────────────────────────────────────────────────────────────
        header = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(6))

        self.search_box = TextInput(
            hint_text="🔍 Cerca brano…",
            background_color=hex_color(C["card"]),
            foreground_color=hex_color(C["text"]),
            cursor_color=hex_color(C["accent"]),
            hint_text_color=hex_color(C["subtext"]),
            font_size=sp(13),
            multiline=False,
            size_hint_x=0.6,
            padding=[dp(10), dp(10)],
        )
        self.search_box.bind(text=lambda *_: self._refresh())

        scan_btn = DarkButton(
            text="📂 Scansiona",
            size_hint_x=0.4,
            bg_color=hex_color(C["primary"]),
        )
        scan_btn.bind(on_press=self._start_scan)
        header.add_widget(self.search_box)
        header.add_widget(scan_btn)
        root.add_widget(header)

        # ── Filtri ────────────────────────────────────────────────────────────
        filter_row = BoxLayout(size_hint_y=None, height=dp(36), spacing=dp(6))
        mood_vals  = ["Mood: All"] + MOODS
        genre_vals = ["Genere: All"] + GENRES

        self.mood_spinner  = Spinner(
            text="Mood: All", values=mood_vals,
            background_normal="", background_color=hex_color(C["card"]),
            color=hex_color(C["text"]), font_size=sp(11),
            option_cls="SpinnerOption",
        )
        self.genre_spinner = Spinner(
            text="Genere: All", values=genre_vals,
            background_normal="", background_color=hex_color(C["card"]),
            color=hex_color(C["text"]), font_size=sp(11),
        )
        self.mood_spinner.bind(text=lambda *_: self._refresh())
        self.genre_spinner.bind(text=lambda *_: self._refresh())

        self.count_lbl = Label(
            text="0 brani", font_size=sp(11),
            color=hex_color(C["subtext"]),
            size_hint_x=0.25,
        )

        filter_row.add_widget(self.mood_spinner)
        filter_row.add_widget(self.genre_spinner)
        filter_row.add_widget(self.count_lbl)
        root.add_widget(filter_row)

        # ── Progress scansione ────────────────────────────────────────────────
        self.scan_bar = ProgressBar(
            max=100, value=0,
            size_hint_y=None, height=dp(4),
        )
        self.scan_status_lbl = Label(
            text="", font_size=sp(10),
            color=hex_color(C["subtext"]),
            size_hint_y=None, height=dp(14),
        )
        root.add_widget(self.scan_bar)
        root.add_widget(self.scan_status_lbl)

        # ── Song list ─────────────────────────────────────────────────────────
        self.rv = SongRecycleView(size_hint=(1, 1))
        root.add_widget(self.rv)

        # ── Mood Playlists ────────────────────────────────────────────────────
        mood_row = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(6))
        for mood in MOODS:
            btn = MoodButton(mood=mood, size_hint_x=0.25, font_size=sp(10))
            btn.bind(on_press=lambda b, m=mood: self._play_mood(m))
            mood_row.add_widget(btn)
        root.add_widget(mood_row)

        self.add_widget(root)

    # ── Filtro & refresh ──────────────────────────────────────────────────────
    def _get_filtered(self) -> list[dict]:
        q      = self.search_box.text.lower().strip()
        mood   = self.mood_spinner.text.replace("Mood: ", "")
        genre  = self.genre_spinner.text.replace("Genere: ", "")

        return [
            s for s in state.all_songs
            if (not q      or q     in s.get("title", "").lower())
            and (mood  == "All" or s.get("mood")  == mood)
            and (genre == "All" or s.get("genre") == genre)
        ]

    def _refresh(self, *args):
        self._filtered = self._get_filtered()
        current = state.current_song

        self.rv.data = [
            {
                "song":     s,
                "selected": (current is not None and
                             s.get("path") == current.get("path")),
            }
            for s in self._filtered
        ]
        self.count_lbl.text = f"{len(self._filtered)} brani"

    def _highlight_current(self, song, *args):
        self._refresh()

    # ── Scan ──────────────────────────────────────────────────────────────────
    def _start_scan(self, *args):
        if state._scanning:
            return
        try:
            from plyer import filechooser
            filechooser.open_file(
                title="Seleziona cartella (scegli un file dentro di essa)",
                filters=["*.mp3"],
                on_selection=self._on_folder_selected,
            )
        except Exception:
            # Fallback: scansiona cartella Music Android
            self._scan_path(_default_music_path())

    def _on_folder_selected(self, selection):
        if not selection:
            return
        folder = os.path.dirname(selection[0])
        self._scan_path(folder)

    def _scan_path(self, folder: str):
        if not folder or not os.path.isdir(folder):
            self._show_popup("Errore", f"Cartella non trovata:\n{folder}")
            return

        state._scanning   = True
        self.scan_bar.value = 0
        self.scan_status_lbl.text = "Analisi in corso…"

        def _progress(current, total, fname):
            pct = (current / total * 100) if total else 0
            Clock.schedule_once(
                lambda dt: self._update_progress(pct, f"[{current}/{total}] {fname[:30]}"),
                0,
            )

        def _worker():
            songs = scan_folders([folder], progress_callback=_progress)
            Clock.schedule_once(lambda dt: self._on_scan_done(songs), 0)

        threading.Thread(target=_worker, daemon=True).start()

    def _update_progress(self, pct: float, txt: str):
        self.scan_bar.value       = pct
        self.scan_status_lbl.text = txt

    def _on_scan_done(self, songs: list[dict]):
        state._scanning    = False
        state.all_songs    = songs
        self.scan_bar.value = 100
        self.scan_status_lbl.text = f"✓ {len(songs)} brani trovati"
        state.fire("on_list_change")
        self._refresh()

    def _play_mood(self, mood: str):
        genre = self.genre_spinner.text.replace("Genere: ", "")
        if not state.play_mood_playlist(mood, genre):
            self._show_popup("Playlist vuota",
                             f"Nessun brano con mood: {mood}")
        else:
            App.get_running_app().go_player()

    def _show_popup(self, title: str, msg: str):
        content = BoxLayout(orientation="vertical", padding=dp(16), spacing=dp(10))
        content.add_widget(Label(text=msg, color=hex_color(C["text"]),
                                 font_size=sp(13)))
        btn = DarkButton(text="OK", size_hint_y=None, height=dp(40))
        p = Popup(title=title, content=content, size_hint=(0.8, 0.4),
                  background_color=hex_color(C["surface"]),
                  title_color=hex_color(C["text"]))
        btn.bind(on_press=p.dismiss)
        content.add_widget(btn)
        p.open()


# =============================================================================
# Player Screen
# =============================================================================
class PlayerScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._build()
        state.register("on_song_change", self._update_ui)
        Clock.schedule_interval(self._tick_progress, 0.5)

    def _build(self):
        root = BoxLayout(orientation="vertical", spacing=dp(12),
                         padding=[dp(16), dp(20), dp(16), dp(16)])
        with root.canvas.before:
            Color(*hex_color(C["bg"]))
            self._bg_rect = Rectangle(pos=root.pos, size=root.size)
        root.bind(pos=self._upd_bg, size=self._upd_bg)

        # ── Artwork placeholder ────────────────────────────────────────────────
        art_box = BoxLayout(size_hint_y=None, height=dp(180))
        self.art_lbl = Label(
            text="🎵",
            font_size=sp(80),
            size_hint=(1, 1),
        )
        with self.art_lbl.canvas.before:
            Color(*hex_color(C["card"]))
            self._art_rect = RoundedRectangle(
                pos=self.art_lbl.pos,
                size=self.art_lbl.size,
                radius=[dp(20)],
            )
        self.art_lbl.bind(pos=self._upd_art, size=self._upd_art)
        art_box.add_widget(self.art_lbl)
        root.add_widget(art_box)

        # ── Mood badge ────────────────────────────────────────────────────────
        self.mood_badge = Label(
            text="", font_size=sp(12),
            size_hint_y=None, height=dp(24),
            color=hex_color(C["accent"]),
        )
        root.add_widget(self.mood_badge)

        # ── Titolo ────────────────────────────────────────────────────────────
        self.title_lbl = Label(
            text="Nessun brano",
            font_size=sp(16), bold=True,
            color=hex_color(C["text"]),
            size_hint_y=None, height=dp(40),
            text_size=(None, None),
            halign="center",
        )
        root.add_widget(self.title_lbl)

        # ── Meta (genere | BPM) ───────────────────────────────────────────────
        self.meta_lbl = Label(
            text="",
            font_size=sp(12),
            color=hex_color(C["subtext"]),
            size_hint_y=None, height=dp(22),
        )
        root.add_widget(self.meta_lbl)

        # ── Progress ──────────────────────────────────────────────────────────
        self.progress = ProgressBar(max=100, value=0,
                                    size_hint_y=None, height=dp(6))
        root.add_widget(self.progress)

        self.time_lbl = Label(
            text="0:00 / 0:00",
            font_size=sp(11),
            color=hex_color(C["subtext"]),
            size_hint_y=None, height=dp(20),
        )
        root.add_widget(self.time_lbl)

        # ── Controlli principali ──────────────────────────────────────────────
        ctrl = BoxLayout(size_hint_y=None, height=dp(64), spacing=dp(10))

        prev_btn = DarkButton(text="⏮", font_size=sp(22),
                               bg_color=hex_color(C["card"]))
        prev_btn.bind(on_press=lambda *_: state.prev_song())

        self.play_btn = DarkButton(text="▶", font_size=sp(26),
                                    bg_color=hex_color(C["primary"]))
        self.play_btn.bind(on_press=lambda *_: state.toggle_play_pause())

        next_btn = DarkButton(text="⏭", font_size=sp(22),
                               bg_color=hex_color(C["card"]))
        next_btn.bind(on_press=lambda *_: state.next_song())

        ctrl.add_widget(prev_btn)
        ctrl.add_widget(self.play_btn)
        ctrl.add_widget(next_btn)
        root.add_widget(ctrl)

        # ── Controlli secondari ───────────────────────────────────────────────
        ctrl2 = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(10))

        self.loop_btn  = DarkButton(text="🔁 Loop",  font_size=sp(13),
                                     bg_color=hex_color(C["card"]))
        self.shuf_btn  = DarkButton(text="🔀 Shuffle", font_size=sp(13),
                                     bg_color=hex_color(C["card"]))
        stop_btn       = DarkButton(text="⏹ Stop",   font_size=sp(13),
                                     bg_color=hex_color(C["danger"]))

        self.loop_btn.bind(on_press=self._toggle_loop)
        self.shuf_btn.bind(on_press=self._toggle_shuffle)
        stop_btn.bind(on_press=lambda *_: state.stop())

        ctrl2.add_widget(self.loop_btn)
        ctrl2.add_widget(self.shuf_btn)
        ctrl2.add_widget(stop_btn)
        root.add_widget(ctrl2)

        # ── Volume ────────────────────────────────────────────────────────────
        vol_row = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(8))
        vol_row.add_widget(Label(text="🔈", font_size=sp(16),
                                 size_hint_x=None, width=dp(28),
                                 color=hex_color(C["subtext"])))
        self.vol_slider = Slider(min=0, max=1, value=1,
                                  cursor_size=(dp(20), dp(20)))
        self.vol_slider.bind(value=lambda _, v: state.set_volume(v))
        vol_row.add_widget(self.vol_slider)
        vol_row.add_widget(Label(text="🔊", font_size=sp(16),
                                 size_hint_x=None, width=dp(28),
                                 color=hex_color(C["subtext"])))
        root.add_widget(vol_row)

        self.add_widget(root)

    def _upd_bg(self, instance, *args):
        self._bg_rect.pos  = instance.pos
        self._bg_rect.size = instance.size

    def _upd_art(self, instance, *args):
        self._art_rect.pos  = instance.pos
        self._art_rect.size = instance.size

    def _update_ui(self, song, *args):
        if not song:
            self.title_lbl.text = "Nessun brano"
            self.meta_lbl.text  = ""
            self.mood_badge.text = ""
            self.play_btn.text  = "▶"
            return

        title = song.get("title", "")
        if title.lower().endswith(".mp3"):
            title = title[:-4]
        if len(title) > 35:
            title = title[:35] + "…"

        mood  = song.get("mood", "")
        genre = song.get("genre", "?")
        alt_g = song.get("genre_alt")
        if alt_g:
            genre += f"/{alt_g}"
        bpm   = song.get("tempo", 0)
        key   = "Maggiore" if song.get("mode_major") else "Minore"

        mood_icon  = MOOD_ICONS.get(mood, "")
        mood_color = MOOD_COLORS.get(mood, C["accent"])

        self.title_lbl.text  = title
        self.mood_badge.text = f"{mood_icon} {mood}"
        self.mood_badge.color = hex_color(mood_color)
        self.meta_lbl.text   = f"{genre}  ·  {bpm:.0f} BPM  ·  {key}"

        # Emoji artwork in base al mood
        art_emoji = {"Energetic": "⚡", "Positive": "🌞",
                     "Aggressive": "🔥", "Melancholic": "🌧"}.get(mood, "🎵")
        self.art_lbl.text = art_emoji

        self.play_btn.text = "⏸" if state.is_playing() else "▶"

    def _tick_progress(self, dt):
        pos, dur = state.get_position()
        if dur > 0:
            self.progress.value = min(100, pos / dur * 100)
            self.time_lbl.text  = f"{_fmt(pos)} / {_fmt(dur)}"
        if state.is_playing():
            self.play_btn.text = "⏸"
        elif state.current_song:
            self.play_btn.text = "▶"

    def _toggle_loop(self, *args):
        state.toggle_loop()
        self.loop_btn.bg_color = (
            hex_color(C["success"]) if state._loop else hex_color(C["card"])
        )

    def _toggle_shuffle(self, *args):
        state.toggle_shuffle()
        self.shuf_btn.bg_color = (
            hex_color(C["success"]) if state._shuffle else hex_color(C["card"])
        )


# =============================================================================
# Stats Screen
# =============================================================================
class StatsScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._build()

    def _build(self):
        root = BoxLayout(orientation="vertical", spacing=dp(8),
                         padding=[dp(12), dp(10), dp(12), dp(10)])
        with root.canvas.before:
            Color(*hex_color(C["bg"]))
            Rectangle(pos=root.pos, size=root.size)

        header = Label(
            text="📊 Statistiche Ascolti",
            font_size=sp(18), bold=True,
            color=hex_color(C["text"]),
            size_hint_y=None, height=dp(40),
        )
        root.add_widget(header)

        refresh_btn = DarkButton(
            text="🔄 Aggiorna",
            size_hint_y=None, height=dp(40),
            bg_color=hex_color(C["primary"]),
        )
        refresh_btn.bind(on_press=lambda *_: self._load_stats())
        root.add_widget(refresh_btn)

        scroll = ScrollView()
        self.stats_grid = GridLayout(
            cols=1, spacing=dp(8),
            size_hint_y=None, padding=[dp(4), dp(4)],
        )
        self.stats_grid.bind(minimum_height=self.stats_grid.setter("height"))
        scroll.add_widget(self.stats_grid)
        root.add_widget(scroll)

        self.add_widget(root)
        self._load_stats()

    def on_enter(self):
        self._load_stats()

    def _load_stats(self, *args):
        self.stats_grid.clear_widgets()

        try:
            total    = lh.get_total_plays()
            top_mood = lh.get_top_mood_last_n_days(7) or "N/D"
            mood_dist = lh.get_mood_distribution_last_n_days(7)
            genre_dist = lh.get_genre_distribution_last_n_days(7)
            top_songs  = lh.get_most_played_songs_last_n_days(7, 10)
            slots      = lh.get_time_slot_mood_preference()
        except Exception as e:
            self._add_card(f"Errore caricamento statistiche:\n{e}")
            return

        self._add_card(f"🎧  Totale riproduzioni: {total}")
        self._add_card(f"⭐  Mood preferito (7gg): {top_mood}")

        # Libreria
        self._add_card(
            f"📚  Brani in libreria: {len(state.all_songs)}\n"
            f"    Generi: {', '.join(set(s.get('genre','?') for s in state.all_songs[:50]))}"
            if state.all_songs else "📚  Libreria vuota — scansiona prima una cartella"
        )

        # Mood distribution
        if mood_dist:
            lines = "\n".join(
                f"  {MOOD_ICONS.get(m,'•')} {m}: {c}" for m, c in mood_dist.items()
            )
            self._add_card(f"🎭  Distribuzione mood (7gg):\n{lines}")

        # Genre distribution
        if genre_dist:
            lines = "\n".join(f"  {g}: {c}" for g, c in genre_dist.items())
            self._add_card(f"🎸  Generi ascoltati (7gg):\n{lines}")

        # Fasce orarie
        if slots:
            slot_icons = {"morning": "🌅", "afternoon": "☀️",
                          "evening": "🌆", "night": "🌙"}
            lines = "\n".join(
                f"  {slot_icons.get(k,'•')} {k.capitalize()}: {v}"
                for k, v in slots.items()
            )
            self._add_card(f"🕐  Preferenze orarie:\n{lines}")

        # Top songs
        if top_songs:
            lines = "\n".join(
                f"  {i+1}. {s['song_title'][:30]} ({s['play_count']}x)"
                for i, s in enumerate(top_songs[:5])
            )
            self._add_card(f"🏆  Top 5 brani (7gg):\n{lines}")

    def _add_card(self, text: str):
        lbl = Label(
            text=text,
            font_size=sp(12),
            color=hex_color(C["text"]),
            size_hint_y=None,
            halign="left",
            valign="top",
            text_size=(None, None),
        )
        lbl.bind(texture_size=lambda l, s: setattr(l, "height", s[1] + dp(20)))

        card = BoxLayout(
            size_hint_y=None,
            padding=[dp(14), dp(10)],
        )
        card.bind(minimum_height=card.setter("height"))
        with card.canvas.before:
            Color(*hex_color(C["card"]))
            RoundedRectangle(pos=card.pos, size=card.size, radius=[dp(10)])
        card.bind(pos=lambda c, *_: self._upd_card(c),
                  size=lambda c, *_: self._upd_card(c))
        card._rect = None
        card.add_widget(lbl)
        self.stats_grid.add_widget(card)

    def _upd_card(self, card):
        card.canvas.before.clear()
        with card.canvas.before:
            Color(*hex_color(C["card"]))
            RoundedRectangle(pos=card.pos, size=card.size, radius=[dp(10)])


# =============================================================================
# Bottom Navigation Bar
# =============================================================================
class NavBar(BoxLayout):
    def __init__(self, sm: ScreenManager, **kwargs):
        super().__init__(**kwargs)
        self.sm          = sm
        self.orientation = "horizontal"
        self.size_hint_y = None
        self.height      = dp(56)
        self.spacing     = 0

        with self.canvas.before:
            Color(*hex_color(C["surface"]))
            self._bg = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._upd, size=self._upd)

        tabs = [
            ("library",  "🎵 Libreria"),
            ("player",   "▶ Player"),
            ("stats",    "📊 Stats"),
        ]
        self._btns = {}
        for name, label in tabs:
            btn = Button(
                text=label, font_size=sp(12),
                background_normal="", background_color=[0, 0, 0, 0],
                color=hex_color(C["subtext"]),
            )
            btn.bind(on_press=lambda b, n=name: self._nav(n))
            self._btns[name] = btn
            self.add_widget(btn)

        self._set_active("library")

    def _upd(self, *args):
        self._bg.pos  = self.pos
        self._bg.size = self.size

    def _nav(self, name: str):
        self.sm.current = name
        self._set_active(name)

    def _set_active(self, name: str):
        for n, b in self._btns.items():
            b.color = hex_color(C["accent"]) if n == name else hex_color(C["subtext"])


# =============================================================================
# App principale
# =============================================================================
class MusicMoodApp(App):
    def build(self):
        self.title = "Music Mood Analyzer"

        root = BoxLayout(orientation="vertical")
        with root.canvas.before:
            Color(*hex_color(C["bg"]))
            Rectangle(pos=root.pos, size=root.size)

        self.sm = ScreenManager(transition=FadeTransition(duration=0.15))
        self.sm.add_widget(LibraryScreen(name="library"))
        self.sm.add_widget(PlayerScreen(name="player"))
        self.sm.add_widget(StatsScreen(name="stats"))

        self.navbar = NavBar(self.sm)

        root.add_widget(self.sm)
        root.add_widget(self.navbar)

        return root

    def go_player(self):
        self.sm.current = "player"
        self.navbar._set_active("player")

    def on_song_tap(self, song: dict, index: int):
        """Chiamato da SongRow al tap."""
        # Trova i brani filtrati dalla LibraryScreen corrente
        lib = self.sm.get_screen("library")
        filtered = lib._filtered
        state.play_filtered(filtered, filtered.index(song) if song in filtered else 0)
        self.go_player()


# =============================================================================
# Utility
# =============================================================================
def _fmt(sec: float) -> str:
    s = int(sec)
    return f"{s // 60}:{s % 60:02d}"


def _default_music_path() -> str:
    """Percorso Music predefinito per Android."""
    try:
        from android.storage import primary_external_storage_path
        return os.path.join(primary_external_storage_path(), "Music")
    except ImportError:
        # Desktop fallback
        return os.path.join(Path.home(), "Music")


if __name__ == "__main__":
    MusicMoodApp().run()
