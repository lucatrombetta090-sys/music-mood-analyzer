"""
Music Mood Analyzer v18
Fixes:
  - Richiesta permessi runtime (Android 6+)
  - Gestione MANAGE_EXTERNAL_STORAGE (Android 11+)
  - Rimossi emoji → testo plain (compatibile con tutti i font Android)
  - UI migliorata con etichette chiare
"""

import os
import threading
import random

import kivy
kivy.require("2.3.0")

from kivy.config import Config
Config.set("graphics", "width",  "400")
Config.set("graphics", "height", "750")

from kivy.app               import App
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
from kivy.clock             import Clock
from kivy.core.audio        import SoundLoader
from kivy.metrics           import dp, sp
from kivy.properties        import NumericProperty, BooleanProperty
from kivy.graphics          import Color, RoundedRectangle, Rectangle
from kivy.utils             import get_color_from_hex

import listening_history as lh
from analyze_mp3 import scan_folders

lh.init_database()

# ── Rilevamento Android ───────────────────────────────────────────────────────
try:
    from android.permissions import (
        request_permissions, check_permission, Permission
    )
    from jnius import autoclass
    ANDROID = True
except ImportError:
    ANDROID = False

# ── Palette ───────────────────────────────────────────────────────────────────
C = {
    "bg":         "#0F0F1A",
    "surface":    "#1A1A2E",
    "card":       "#252540",
    "card2":      "#2E2E50",
    "primary":    "#6C3FC5",
    "primary2":   "#8B5CF6",
    "accent":     "#A78BFA",
    "text":       "#F1F0FF",
    "subtext":    "#9CA3AF",
    "border":     "#3D3D6B",
    "energetic":  "#F59E0B",
    "positive":   "#10B981",
    "aggressive": "#EF4444",
    "melancholic":"#7C3AED",
    "success":    "#22C55E",
    "danger":     "#DC2626",
    "warn":       "#D97706",
}

MOOD_COLORS = {
    "Energetic":  C["energetic"],
    "Positive":   C["positive"],
    "Aggressive": C["aggressive"],
    "Melancholic":C["melancholic"],
}
MOOD_TAG = {
    "Energetic": "ENERG",
    "Positive":  "POS",
    "Aggressive":"AGGR",
    "Melancholic":"MEL",
}
MOODS  = ["Energetic","Positive","Aggressive","Melancholic"]
GENRES = ["Pop","Rock","Electronic","HipHop","Acoustic","Jazz","Classical"]

def hc(h, a=1.0):
    c = get_color_from_hex(h)
    return (c[0], c[1], c[2], a)

def _fmt(s):
    s = int(s)
    return f"{s//60}:{s%60:02d}"


# ── Permessi runtime ──────────────────────────────────────────────────────────
_perm_granted = False

def _request_storage_permissions(callback=None):
    """Richiede READ/WRITE_EXTERNAL_STORAGE a runtime (Android 6+)."""
    global _perm_granted
    if not ANDROID:
        _perm_granted = True
        if callback: callback(True)
        return

    # Android 11+ (API 30): serve MANAGE_EXTERNAL_STORAGE via Settings
    try:
        Build = autoclass("android.os.Build$VERSION")
        if Build.SDK_INT >= 30:
            Environment = autoclass("android.os.Environment")
            if not Environment.isExternalStorageManager():
                _open_all_files_settings()
                # Non possiamo sapere se l'utente ha accettato, continuiamo con i permessi base
    except Exception as e:
        print(f"[perm] Android 11 check error: {e}")

    perms = [
        Permission.READ_EXTERNAL_STORAGE,
        Permission.WRITE_EXTERNAL_STORAGE,
    ]
    # Aggiunge READ_MEDIA_AUDIO per Android 13+
    try:
        perm_media = autoclass("android.Manifest$permission")
        if hasattr(perm_media, "READ_MEDIA_AUDIO"):
            perms.append("android.permission.READ_MEDIA_AUDIO")
    except Exception:
        pass

    def _on_result(permissions, results):
        global _perm_granted
        _perm_granted = all(results)
        print(f"[perm] granted={_perm_granted} {list(zip(permissions,results))}")
        if callback:
            Clock.schedule_once(lambda dt: callback(_perm_granted), 0)

    request_permissions(perms, _on_result)


def _open_all_files_settings():
    """Apre la pagina Impostazioni per MANAGE_EXTERNAL_STORAGE (Android 11+)."""
    try:
        Intent  = autoclass("android.content.Intent")
        Settings= autoclass("android.provider.Settings")
        Uri     = autoclass("android.net.Uri")
        PythonActivity = autoclass("org.kivy.android.PythonActivity")
        ctx = PythonActivity.mActivity
        pkg = ctx.getPackageName()
        intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION)
        intent.setData(Uri.parse(f"package:{pkg}"))
        ctx.startActivity(intent)
    except Exception as e:
        print(f"[perm] open settings error: {e}")


# ── Cerca cartelle Music ──────────────────────────────────────────────────────
def _find_music_dirs() -> list:
    roots = [
        "/storage/emulated/0/Music",
        "/sdcard/Music",
        "/storage/emulated/0/Download",
        "/sdcard/Download",
        "/storage/emulated/0/Downloads",
        "/storage/emulated/0",
        "/sdcard",
    ]
    try:
        for s in os.listdir("/storage"):
            p = f"/storage/{s}"
            if s not in ("emulated", "self") and os.path.isdir(p):
                roots += [p, f"{p}/Music"]
    except Exception:
        pass

    seen, results = set(), []
    for r in roots:
        if not os.path.isdir(r) or r in seen:
            continue
        seen.add(r)
        count = 0
        try:
            for e in os.scandir(r):
                if e.name.lower().endswith(".mp3"):
                    count += 1
            for sub in os.scandir(r):
                if sub.is_dir(follow_symlinks=False):
                    try:
                        for e in os.scandir(sub.path):
                            if e.name.lower().endswith(".mp3"):
                                count += 1
                    except PermissionError:
                        pass
        except PermissionError:
            pass
        if count > 0:
            results.append((r, count))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def _count_mp3(folder: str) -> int:
    n = 0
    try:
        for _, _, files in os.walk(folder):
            n += sum(1 for f in files if f.lower().endswith(".mp3"))
    except Exception:
        pass
    return n


# =============================================================================
# AppState
# =============================================================================
class AppState:
    def __init__(self):
        self.all_songs=[]; self.current_playlist=[]; self.playlist_index=0
        self.current_song=None; self._sound=None
        self._loop=False; self._shuffle=False; self._volume=1.0; self._cbs={}

    def on(self,ev,fn): self._cbs.setdefault(ev,[]).append(fn)
    def fire(self,ev,*a):
        for fn in self._cbs.get(ev,[]):
            try: fn(*a)
            except Exception as e: print(f"[cb]{ev}:{e}")

    def _stop(self):
        if self._sound:
            try: self._sound.stop(); self._sound.unload()
            except: pass
            self._sound = None

    def play(self, song):
        self._stop()
        s = SoundLoader.load(song["path"])
        if not s: return
        self._sound=s; self._sound.volume=self._volume
        self._sound.bind(on_stop=self._ended)
        self._sound.play(); self.current_song=song
        lh.log_play(song); self.fire("song",song)

    def _ended(self,*_):
        if self._loop and self.current_song:
            self.play(self.current_song); return
        if self.current_playlist:
            nxt=(self.playlist_index+1)%len(self.current_playlist)
            if nxt==0 and not self._loop:
                self.current_song=None; self.fire("song",None)
            else:
                self.playlist_index=nxt; self.play(self.current_playlist[nxt])

    def toggle_pause(self):
        if not self._sound: return
        if self._sound.state=="play": self._sound.stop()
        else: self._sound.play()
        self.fire("song",self.current_song)

    def next(self):
        if not self.current_playlist: return
        self.playlist_index=(self.playlist_index+1)%len(self.current_playlist)
        self.play(self.current_playlist[self.playlist_index])

    def prev(self):
        if not self.current_playlist: return
        self.playlist_index=(self.playlist_index-1)%len(self.current_playlist)
        self.play(self.current_playlist[self.playlist_index])

    def stop(self): self._stop(); self.current_song=None; self.fire("song",None)
    def set_vol(self,v):
        self._volume=v
        if self._sound: self._sound.volume=v
    def toggle_loop(self): self._loop=not self._loop
    def toggle_shuffle(self): self._shuffle=not self._shuffle

    def play_list(self,songs,idx=0):
        if not songs: return
        lst=songs[:]
        if self._shuffle: random.shuffle(lst); idx=0
        self.current_playlist=lst; self.playlist_index=idx; self.play(lst[idx])

    def play_mood(self,mood,genre="All"):
        pl=[s for s in self.all_songs
            if s.get("mood")==mood and (genre=="All" or s.get("genre")==genre)]
        if not pl: return False
        self.play_list(pl); return True

    def is_playing(self):
        return self._sound is not None and self._sound.state=="play"

    def position(self):
        if self._sound: return self._sound.get_pos() or 0.0, self._sound.length or 0.0
        return 0.0, 0.0


state = AppState()


# =============================================================================
# UI helpers
# =============================================================================
def mk_label(text, size=13, bold=False, color=None, halign="left",
             sh_y=None, height=None):
    kw = dict(text=text, font_size=sp(size), bold=bold,
              color=hc(color or C["text"]), halign=halign)
    if sh_y is not None: kw["size_hint_y"] = sh_y
    if height is not None: kw["height"] = dp(height)
    return Label(**kw)


class RndRect(BoxLayout):
    """BoxLayout con sfondo arrotondato."""
    def __init__(self, bg=None, radius=8, **kw):
        super().__init__(**kw)
        self._bg_col = hc(bg or C["card"])
        self._r = radius
        self.bind(pos=self._draw, size=self._draw)
    def _draw(self, *_):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*self._bg_col)
            RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(self._r)])


class DkBtn(Button):
    def __init__(self, bg=None, radius=8, **kw):
        super().__init__(**kw)
        self._bg  = bg or hc(C["primary"])
        self._r   = radius
        self.background_normal = ""
        self.background_color  = [0,0,0,0]
        self.color = hc(C["text"])
        self.font_size = sp(13)
        self._draw()
        self.bind(pos=lambda *_: self._draw(), size=lambda *_: self._draw())

    def _draw(self):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*self._bg)
            RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(self._r)])

    def set_bg(self, c):
        self._bg = c; self._draw()


# =============================================================================
# SongRow / SongRV
# =============================================================================
class SongRow(RecycleDataViewBehavior, BoxLayout):
    index    = NumericProperty(0)
    selected = BooleanProperty(False)

    def __init__(self, **kw):
        super().__init__(**kw)
        self.orientation = "vertical"
        self.padding = [dp(14), dp(6)]
        self.spacing = dp(2)
        self.size_hint_y = None
        self.height = dp(70)
        self._song = None
        self.t = Label(font_size=sp(13), bold=True, color=hc(C["text"]),
                       halign="left", valign="middle",
                       size_hint_y=None, height=dp(22))
        self.m = Label(font_size=sp(11), color=hc(C["subtext"]),
                       halign="left", valign="middle",
                       size_hint_y=None, height=dp(18))
        self.add_widget(self.t); self.add_widget(self.m)
        self.bind(pos=self._rd, size=self._rd)

    def _rd(self, *_):
        self.canvas.before.clear()
        with self.canvas.before:
            if self.selected:
                Color(*hc(C["primary"], 0.35))
            else:
                Color(*hc(C["card"]))
            RoundedRectangle(
                pos=(self.x+dp(4), self.y+dp(2)),
                size=(self.width-dp(8), self.height-dp(4)),
                radius=[dp(10)]
            )

    def refresh_view_attrs(self, rv, index, data):
        self.index = index
        self._song = data.get("song", {})
        self.selected = data.get("selected", False)
        s = self._song
        mood = s.get("mood", "")
        title = s.get("title", "?")
        if title.lower().endswith(".mp3"): title = title[:-4]
        if len(title) > 38: title = title[:38] + "..."

        genre = s.get("genre", "?")
        if s.get("genre_alt"): genre += "/" + s["genre_alt"]
        tag = MOOD_TAG.get(mood, mood[:5].upper() if mood else "")
        mood_col = MOOD_COLORS.get(mood, C["subtext"])

        self.t.text = f"[{tag}] {title}"
        self.t.color = hc(mood_col)
        self.m.text = f"{genre}  |  {s.get('tempo',0):.0f} BPM  |  {mood or '---'}"
        self._rd()
        return super().refresh_view_attrs(rv, index, data)

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos) and self._song:
            App.get_running_app().on_song_tap(self._song)
            return True
        return super().on_touch_down(touch)


class SongRV(RecycleView):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.viewclass = "SongRow"
        self.data = []
        with self.canvas.before:
            Color(*hc(C["bg"]))
            self._bg = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=lambda *_: setattr(self._bg,"pos",self.pos),
                  size=lambda *_: setattr(self._bg,"size",self.size))


# =============================================================================
# Banner permessi
# =============================================================================
class PermBanner(RndRect):
    def __init__(self, on_grant, **kw):
        super().__init__(bg=C["warn"], radius=10,
                         size_hint_y=None, height=dp(68),
                         spacing=dp(8), padding=[dp(12),dp(6)], **kw)
        self._cb = on_grant
        info = BoxLayout(orientation="vertical", size_hint_x=0.72)
        info.add_widget(Label(
            text="Accesso storage non concesso",
            font_size=sp(12), bold=True, color=hc("#FFF"),
            halign="left", size_hint_y=None, height=dp(22)))
        info.add_widget(Label(
            text="Tocca per abilitare lettura file",
            font_size=sp(10), color=hc("#FFE"), halign="left",
            size_hint_y=None, height=dp(18)))
        btn = DkBtn(text="Abilita", size_hint_x=0.28,
                    bg=hc(C["success"]), font_size=sp(12))
        btn.bind(on_press=lambda *_: _request_storage_permissions(self._on_result))
        self.add_widget(info); self.add_widget(btn)

    def _on_result(self, ok):
        self._cb(ok)


# =============================================================================
# Popup selettore cartelle
# =============================================================================
class FolderPickerPopup(Popup):
    def __init__(self, on_folder_selected, **kw):
        self._cb = on_folder_selected
        content = BoxLayout(orientation="vertical", spacing=dp(8),
                            padding=[dp(10), dp(8)])

        # Istruzioni
        info_box = RndRect(bg=C["card2"], radius=8,
                           size_hint_y=None, height=dp(52),
                           padding=[dp(10), dp(6)])
        info_box.add_widget(Label(
            text="Seleziona la cartella con i tuoi MP3.\n"
                 "Le cartelle trovate sul dispositivo appaiono qui sotto.",
            font_size=sp(11), color=hc(C["subtext"]),
            halign="center", text_size=(dp(300), None)))
        content.add_widget(info_box)

        # Lista cartelle
        scroll = ScrollView(size_hint=(1, 1))
        self._grid = GridLayout(cols=1, spacing=dp(6),
                                size_hint_y=None, padding=[0, dp(4)])
        self._grid.bind(minimum_height=self._grid.setter("height"))
        scroll.add_widget(self._grid)
        content.add_widget(scroll)

        # Percorso manuale
        sep = Label(text="--- oppure inserisci percorso manuale ---",
                    font_size=sp(10), color=hc(C["subtext"]),
                    size_hint_y=None, height=dp(22))
        content.add_widget(sep)

        path_row = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(6))
        self._inp = TextInput(
            hint_text="/storage/emulated/0/Music",
            background_color=hc(C["card"]),
            foreground_color=hc(C["text"]),
            hint_text_color=hc(C["subtext"]),
            font_size=sp(11), multiline=False,
            size_hint_x=0.70, padding=[dp(8), dp(10)])
        go = DkBtn(text="Vai", size_hint_x=0.30, bg=hc(C["primary"]))
        go.bind(on_press=self._custom)
        path_row.add_widget(self._inp); path_row.add_widget(go)
        content.add_widget(path_row)

        cancel = DkBtn(text="Annulla", size_hint_y=None, height=dp(40),
                       bg=hc(C["card"]))
        cancel.bind(on_press=self.dismiss)
        content.add_widget(cancel)

        super().__init__(
            title="Scegli cartella Music",
            content=content,
            size_hint=(0.93, 0.80),
            background_color=hc(C["surface"]),
            title_color=hc(C["text"]),
            **kw
        )

        self._loading = Label(
            text="Cerco cartelle con MP3...",
            font_size=sp(12), color=hc(C["subtext"]),
            size_hint_y=None, height=dp(40))
        self._grid.add_widget(self._loading)
        threading.Thread(target=self._find, daemon=True).start()

    def _find(self):
        dirs = _find_music_dirs()
        Clock.schedule_once(lambda dt: self._show(dirs), 0)

    def _show(self, dirs):
        self._grid.clear_widgets()
        if not dirs:
            msg = RndRect(bg=C["card2"], radius=8,
                          size_hint_y=None, height=dp(80),
                          padding=[dp(12), dp(10)])
            msg.add_widget(Label(
                text="Nessuna cartella MP3 trovata.\n\n"
                     "Verifica i permessi storage nelle\n"
                     "Impostazioni > App > Music Mood Analyzer\n"
                     "oppure inserisci il percorso manualmente.",
                font_size=sp(11), color=hc(C["subtext"]),
                halign="center", text_size=(dp(280), None)))
            self._grid.add_widget(msg)
            return

        for path, count in dirs:
            row = RndRect(bg=C["card"], radius=10,
                          size_hint_y=None, height=dp(60),
                          spacing=dp(8), padding=[dp(10), dp(6)])
            info = BoxLayout(orientation="vertical", size_hint_x=0.70)
            # Abbrevia percorso
            short = (path
                     .replace("/storage/emulated/0", "[INT]")
                     .replace("/sdcard", "[INT]"))
            info.add_widget(Label(
                text=short, font_size=sp(11), bold=True,
                color=hc(C["text"]), halign="left",
                size_hint_y=None, height=dp(24)))
            info.add_widget(Label(
                text=f"{count} file MP3", font_size=sp(10),
                color=hc(C["accent"]), halign="left",
                size_hint_y=None, height=dp(20)))
            btn = DkBtn(text="Seleziona", size_hint_x=0.30,
                        bg=hc(C["primary"]), font_size=sp(11))
            btn.bind(on_press=lambda _, p=path: self._pick(p))
            row.add_widget(info); row.add_widget(btn)
            self._grid.add_widget(row)

    def _pick(self, path):
        self.dismiss(); self._cb(path)

    def _custom(self, *_):
        p = self._inp.text.strip()
        if p: self.dismiss(); self._cb(p)


# =============================================================================
# LibraryScreen
# =============================================================================
class LibraryScreen(Screen):
    def __init__(self, **kw):
        super().__init__(**kw)
        self._filtered = []
        self._perm_ok = not ANDROID  # su desktop non serve
        self._build()
        state.on("song", self._highlight)
        # Richiedi permessi all'avvio
        if ANDROID:
            Clock.schedule_once(self._check_perms, 1.0)

    def _check_perms(self, *_):
        _request_storage_permissions(self._on_perm_result)

    def _on_perm_result(self, ok):
        self._perm_ok = ok
        if ok:
            # Rimuovi banner se presente
            if hasattr(self, "_perm_banner") and self._perm_banner.parent:
                self._main_col.remove_widget(self._perm_banner)
            self.status.text = "Permessi OK - tocca SCANSIONA per analizzare"
        else:
            self.status.text = "!! Permessi storage negati !!"

    def _build(self):
        root = BoxLayout(orientation="vertical", spacing=dp(0),
                         padding=[dp(10), dp(8), dp(10), dp(6)])
        with root.canvas.before:
            Color(*hc(C["bg"]))
            self._root_bg = Rectangle(pos=root.pos, size=root.size)
        root.bind(pos=lambda *_: setattr(self._root_bg,"pos",root.pos),
                  size=lambda *_: setattr(self._root_bg,"size",root.size))

        # ── Banner permessi (visibile se non granted) ──
        self._perm_banner = PermBanner(
            on_grant=self._on_perm_result,
            size_hint_y=None, height=dp(68))
        if ANDROID:
            root.add_widget(self._perm_banner)

        # ── Titolo ──
        title_row = BoxLayout(size_hint_y=None, height=dp(40))
        title_row.add_widget(Label(
            text="MUSIC MOOD ANALYZER",
            font_size=sp(16), bold=True,
            color=hc(C["accent"]), halign="center"))
        root.add_widget(title_row)

        # ── Barra ricerca + pulsante ──
        hdr = BoxLayout(size_hint_y=None, height=dp(46), spacing=dp(8))
        self.search = TextInput(
            hint_text="Cerca brano...",
            background_color=hc(C["card"]),
            foreground_color=hc(C["text"]),
            hint_text_color=hc(C["subtext"]),
            font_size=sp(13), multiline=False,
            size_hint_x=0.55, padding=[dp(10), dp(12)])
        self.search.bind(text=lambda *_: self._refresh())

        scan_btn = DkBtn(text="SCANSIONA", size_hint_x=0.45,
                         bg=hc(C["primary"]))
        scan_btn.bind(on_press=self._open_picker)
        hdr.add_widget(self.search); hdr.add_widget(scan_btn)
        root.add_widget(hdr)

        # ── Filtri ──
        frow = BoxLayout(size_hint_y=None, height=dp(38), spacing=dp(6))
        self.mood_sp = Spinner(
            text="Mood: Tutti",
            values=["Mood: Tutti"] + MOODS,
            background_normal="", background_color=hc(C["card2"]),
            color=hc(C["text"]), font_size=sp(11))
        self.genre_sp = Spinner(
            text="Genere: Tutti",
            values=["Genere: Tutti"] + GENRES,
            background_normal="", background_color=hc(C["card2"]),
            color=hc(C["text"]), font_size=sp(11))
        self.cnt = Label(text="0 brani", font_size=sp(11),
                         color=hc(C["accent"]), size_hint_x=0.28)
        self.mood_sp.bind(text=lambda *_: self._refresh())
        self.genre_sp.bind(text=lambda *_: self._refresh())
        frow.add_widget(self.mood_sp); frow.add_widget(self.genre_sp)
        frow.add_widget(self.cnt)
        root.add_widget(frow)

        # ── Status + progress ──
        self.status = Label(
            text="Tocca SCANSIONA per analizzare la tua musica",
            font_size=sp(11), color=hc(C["subtext"]),
            size_hint_y=None, height=dp(18))
        self.pbar = ProgressBar(max=100, value=0, size_hint_y=None, height=dp(5))
        root.add_widget(self.status); root.add_widget(self.pbar)

        # Separatore
        root.add_widget(BoxLayout(size_hint_y=None, height=dp(4)))

        # ── Lista brani ──
        self.rv = SongRV(size_hint=(1, 1))
        root.add_widget(self.rv)

        # ── Pulsanti mood rapidi ──
        root.add_widget(BoxLayout(size_hint_y=None, height=dp(6)))
        mood_lbl = Label(text="Riproduci per mood:",
                         font_size=sp(10), color=hc(C["subtext"]),
                         size_hint_y=None, height=dp(16))
        root.add_widget(mood_lbl)
        mrow = BoxLayout(size_hint_y=None, height=dp(42), spacing=dp(6))
        mood_labels = {
            "Energetic":"ENERGETIC","Positive":"POSITIVE",
            "Aggressive":"AGGRESS.","Melancholic":"MELANC."
        }
        for mood in MOODS:
            b = DkBtn(text=mood_labels[mood],
                      bg=hc(MOOD_COLORS[mood]),
                      size_hint_x=0.25, font_size=sp(10))
            b.bind(on_press=lambda _, m=mood: self._play_mood(m))
            mrow.add_widget(b)
        root.add_widget(mrow)
        self.add_widget(root)
        self._main_col = root

    def _get_filtered(self):
        q = self.search.text.lower().strip()
        mood = self.mood_sp.text.replace("Mood: Tutti", "").replace("Mood: ","")
        genre = self.genre_sp.text.replace("Genere: Tutti","").replace("Genere: ","")
        return [s for s in state.all_songs
                if (not q or q in s.get("title","").lower())
                and (not mood or s.get("mood")==mood)
                and (not genre or s.get("genre")==genre)]

    def _refresh(self, *_):
        self._filtered = self._get_filtered(); cur = state.current_song
        self.rv.data = [{"song": s, "selected": bool(cur and s.get("path")==cur.get("path"))}
                        for s in self._filtered]
        self.cnt.text = f"{len(self._filtered)} brani"

    def _highlight(self, *_): self._refresh()

    def _open_picker(self, *_):
        if ANDROID and not self._perm_ok:
            self._toast(
                "Permessi storage non concessi.\n\n"
                "Tocca il pulsante 'Abilita' in alto\noppure vai in:\n"
                "Impostazioni > Applicazioni >\nMusic Mood Analyzer > Autorizzazioni\n"
                "e abilita 'File e file multimediali'.")
            return
        FolderPickerPopup(on_folder_selected=self._do_scan).open()

    def _do_scan(self, folder):
        folder = folder.strip()
        if not folder:
            self._toast("Percorso vuoto."); return
        if not os.path.isdir(folder):
            self._toast(
                f"Cartella non trovata:\n{folder}\n\n"
                "Verifica il percorso.\n"
                "Nota: su Android i file sono in\n"
                "/storage/emulated/0/Music"); return
        n = _count_mp3(folder)
        if n == 0:
            self._toast(
                f"Nessun file .mp3 trovato in:\n{folder}\n\n"
                "Controlla che la cartella contenga\nfile MP3 e che i permessi siano OK."); return

        self.status.text = f"Analisi in corso... {n} brani trovati"
        self.pbar.value = 2

        def _prog(cur, tot, fname):
            pct = cur/tot*100 if tot else 0
            short = fname[:28] if len(fname) > 28 else fname
            Clock.schedule_once(
                lambda dt: self._upd(pct, f"[{cur}/{tot}] {short}"), 0)

        def _worker():
            try:
                songs = scan_folders([folder], progress_callback=_prog)
                Clock.schedule_once(lambda dt: self._done(songs, folder), 0)
            except Exception as e:
                import traceback as tb
                Clock.schedule_once(
                    lambda dt: self._err(str(e), tb.format_exc()), 0)

        threading.Thread(target=_worker, daemon=True).start()

    def _upd(self, pct, txt):
        self.pbar.value = pct; self.status.text = txt

    def _done(self, songs, folder):
        self.pbar.value = 100
        if songs:
            state.all_songs = songs
            self.status.text = f"Completato: {len(songs)} brani analizzati"
            self._refresh()
        else:
            self.status.text = "Attenzione: 0 brani elaborati"
            self._toast(
                "Analisi completata ma 0 brani.\n\n"
                "Possibili cause:\n"
                "- Permessi storage non completi\n"
                "- File MP3 non leggibili\n\n"
                "Su Android 11+: vai in Impostazioni >\n"
                "Applicazioni > Music Mood Analyzer >\n"
                "Autorizzazioni > File e file multimediali\n"
                "e imposta 'Consenti accesso a tutti i file'.")

    def _err(self, short, full):
        self.pbar.value = 0
        self.status.text = f"Errore: {short[:50]}"
        self._toast(f"Errore scansione:\n\n{short[:250]}")
        print("[SCAN ERROR]", full)

    def _play_mood(self, mood):
        genre = self.genre_sp.text.replace("Genere: Tutti","").replace("Genere: ","")
        if not state.play_mood(mood, genre or "All"):
            self._toast(f"Nessun brano con mood: {mood}\n\nEsegui prima una scansione.")
        else:
            App.get_running_app().go("player")

    def _toast(self, msg):
        c = BoxLayout(orientation="vertical", padding=dp(16), spacing=dp(10))
        lbl = Label(text=msg, color=hc(C["text"]), font_size=sp(12),
                    halign="center", text_size=(dp(270), None))
        lbl.bind(texture_size=lambda l,s: setattr(l,"height",s[1]+dp(4)))
        c.add_widget(lbl)
        b = DkBtn(text="OK", size_hint_y=None, height=dp(44))
        p = Popup(title="Info", content=c,
                  size_hint=(0.88, None), height=dp(360),
                  background_color=hc(C["surface"]),
                  title_color=hc(C["text"]))
        b.bind(on_press=p.dismiss); c.add_widget(b); p.open()


# =============================================================================
# PlayerScreen
# =============================================================================
class PlayerScreen(Screen):
    def __init__(self, **kw):
        super().__init__(**kw); self._build()
        state.on("song", self._upd)
        Clock.schedule_interval(self._tick, 0.5)

    def _build(self):
        root = BoxLayout(orientation="vertical", spacing=dp(12),
                         padding=[dp(16), dp(16), dp(16), dp(12)])
        with root.canvas.before:
            Color(*hc(C["bg"]))
            self._bg = Rectangle(pos=root.pos, size=root.size)
        root.bind(pos=lambda *_: setattr(self._bg,"pos",root.pos),
                  size=lambda *_: setattr(self._bg,"size",root.size))

        # ── Artwork box ──
        art_box = RndRect(bg=C["card"], radius=16,
                          size_hint_y=None, height=dp(150))
        self.art = Label(text="[PLAY]", font_size=sp(28), bold=True,
                         color=hc(C["accent"]))
        art_box.add_widget(self.art)
        root.add_widget(art_box)

        # ── Mood badge ──
        self.badge = Label(text="", font_size=sp(13), bold=True,
                           color=hc(C["accent"]),
                           size_hint_y=None, height=dp(24))
        root.add_widget(self.badge)

        # ── Titolo ──
        self.title_lbl = Label(
            text="Nessun brano in riproduzione",
            font_size=sp(15), bold=True,
            color=hc(C["text"]),
            size_hint_y=None, height=dp(46),
            halign="center", text_size=(None, None))
        root.add_widget(self.title_lbl)

        # ── Meta ──
        self.meta = Label(text="", font_size=sp(12),
                          color=hc(C["subtext"]),
                          size_hint_y=None, height=dp(20))
        root.add_widget(self.meta)

        # ── Progress ──
        self.prog = ProgressBar(max=100, value=0, size_hint_y=None, height=dp(6))
        root.add_widget(self.prog)
        self.time_lbl = Label(text="0:00 / 0:00", font_size=sp(11),
                              color=hc(C["subtext"]),
                              size_hint_y=None, height=dp(18))
        root.add_widget(self.time_lbl)

        # ── Controlli principali ──
        ctrl = BoxLayout(size_hint_y=None, height=dp(62), spacing=dp(10))
        prev_b = DkBtn(text="<< Prec", font_size=sp(13), bg=hc(C["card2"]))
        prev_b.bind(on_press=lambda *_: state.prev())
        self.play_b = DkBtn(text="  PLAY  ", font_size=sp(15), bold=True,
                            bg=hc(C["primary"]))
        self.play_b.font_size = sp(16)
        self.play_b.bind(on_press=lambda *_: state.toggle_pause())
        next_b = DkBtn(text="Succ >>", font_size=sp(13), bg=hc(C["card2"]))
        next_b.bind(on_press=lambda *_: state.next())
        ctrl.add_widget(prev_b); ctrl.add_widget(self.play_b); ctrl.add_widget(next_b)
        root.add_widget(ctrl)

        # ── Controlli secondari ──
        ctrl2 = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(8))
        self.loop_b = DkBtn(text="Loop: OFF", font_size=sp(12), bg=hc(C["card2"]))
        self.shuf_b = DkBtn(text="Shuffle: OFF", font_size=sp(12), bg=hc(C["card2"]))
        stop_b = DkBtn(text="STOP", font_size=sp(12), bg=hc(C["danger"]))
        self.loop_b.bind(on_press=self._loop)
        self.shuf_b.bind(on_press=self._shuf)
        stop_b.bind(on_press=lambda *_: state.stop())
        ctrl2.add_widget(self.loop_b); ctrl2.add_widget(self.shuf_b)
        ctrl2.add_widget(stop_b)
        root.add_widget(ctrl2)

        # ── Volume ──
        vrow = BoxLayout(size_hint_y=None, height=dp(42), spacing=dp(8))
        vrow.add_widget(Label(text="VOL", font_size=sp(11),
                              color=hc(C["subtext"]),
                              size_hint_x=None, width=dp(32)))
        self.vol = Slider(min=0, max=1, value=1)
        self.vol.bind(value=lambda _, v: state.set_vol(v))
        vrow.add_widget(self.vol)
        vrow.add_widget(Label(text="MAX", font_size=sp(11),
                              color=hc(C["subtext"]),
                              size_hint_x=None, width=dp(32)))
        root.add_widget(vrow)
        self.add_widget(root)

    def _upd(self, song, *_):
        if not song:
            self.title_lbl.text = "Nessun brano in riproduzione"
            self.badge.text = ""
            self.meta.text = ""
            self.art.text = "[PLAY]"
            self.play_b.text = "  PLAY  "
            return
        t = song.get("title","")
        if t.lower().endswith(".mp3"): t = t[:-4]
        if len(t) > 30: t = t[:30] + "..."
        mood = song.get("mood","")
        genre = song.get("genre","?")
        if song.get("genre_alt"): genre += "/" + song["genre_alt"]
        self.title_lbl.text = t
        tag = MOOD_TAG.get(mood, mood.upper()[:6] if mood else "")
        self.badge.text = f"-- {tag} --"
        self.badge.color = hc(MOOD_COLORS.get(mood, C["accent"]))
        mode = "Maggiore" if song.get("mode_major") else "Minore"
        self.meta.text = f"{genre}  |  {song.get('tempo',0):.0f} BPM  |  {mode}"
        self.art.text = f">> {tag} <<"
        self.art.color = hc(MOOD_COLORS.get(mood, C["accent"]))
        self.play_b.text = " PAUSA  " if state.is_playing() else "  PLAY  "

    def _tick(self, dt):
        pos, dur = state.position()
        if dur > 0:
            self.prog.value = min(100, pos/dur*100)
            self.time_lbl.text = f"{_fmt(pos)} / {_fmt(dur)}"
        self.play_b.text = " PAUSA  " if state.is_playing() else "  PLAY  "

    def _loop(self, *_):
        state.toggle_loop()
        self.loop_b.text = "Loop: ON" if state._loop else "Loop: OFF"
        self.loop_b.set_bg(hc(C["success"] if state._loop else C["card2"]))

    def _shuf(self, *_):
        state.toggle_shuffle()
        self.shuf_b.text = "Shuffle: ON" if state._shuffle else "Shuffle: OFF"
        self.shuf_b.set_bg(hc(C["success"] if state._shuffle else C["card2"]))


# =============================================================================
# StatsScreen
# =============================================================================
class StatsScreen(Screen):
    def __init__(self, **kw):
        super().__init__(**kw); self._build()

    def _build(self):
        root = BoxLayout(orientation="vertical", spacing=dp(8),
                         padding=[dp(12), dp(10), dp(12), dp(10)])
        with root.canvas.before:
            Color(*hc(C["bg"]))
            self._bg = Rectangle(pos=root.pos, size=root.size)
        root.bind(pos=lambda *_: setattr(self._bg,"pos",root.pos),
                  size=lambda *_: setattr(self._bg,"size",root.size))

        root.add_widget(Label(text="STATISTICHE ASCOLTI",
                              font_size=sp(16), bold=True,
                              color=hc(C["accent"]),
                              size_hint_y=None, height=dp(42)))
        rb = DkBtn(text="Aggiorna statistiche",
                   size_hint_y=None, height=dp(40),
                   bg=hc(C["primary"]))
        rb.bind(on_press=lambda *_: self._load())
        root.add_widget(rb)

        sc = ScrollView()
        self.grid = GridLayout(cols=1, spacing=dp(8),
                               size_hint_y=None, padding=[dp(4), dp(4)])
        self.grid.bind(minimum_height=self.grid.setter("height"))
        sc.add_widget(self.grid); root.add_widget(sc)
        self.add_widget(root)
        self._load()

    def on_enter(self): self._load()

    def _load(self, *_):
        self.grid.clear_widgets()
        total  = lh.get_total_plays()
        mood_d = lh.get_mood_distribution_last_n_days(7)
        genre_d= lh.get_genre_distribution_last_n_days(7)
        top    = lh.get_most_played_songs_last_n_days(7, 5)
        slots  = lh.get_time_slot_mood_preference()

        self._card(f"Totale riproduzioni: {total}")
        self._card(f"Brani in libreria: {len(state.all_songs)}")
        if mood_d:
            self._card("Mood (ultimi 7 giorni):\n" + "\n".join(
                f"  {MOOD_TAG.get(m,m[:4].upper())}: {c}" for m,c in mood_d.items()))
        if genre_d:
            self._card("Generi (ultimi 7 giorni):\n" + "\n".join(
                f"  {g}: {c}" for g,c in genre_d.items()))
        if slots:
            slot_lbl = {"morning":"Mattina","afternoon":"Pomeriggio",
                        "evening":"Sera","night":"Notte"}
            self._card("Preferenze orarie:\n" + "\n".join(
                f"  {slot_lbl.get(k,k)}: {v}" for k,v in slots.items()))
        if top:
            self._card("Top 5 brani (7 giorni):\n" + "\n".join(
                f"  {i+1}. {s['song_title'][:32]} ({s['play_count']}x)"
                for i,s in enumerate(top)))

    def _card(self, txt):
        lbl = Label(text=txt, font_size=sp(12), color=hc(C["text"]),
                    size_hint_y=None, halign="left", valign="top",
                    text_size=(None, None))
        lbl.bind(texture_size=lambda l,s: setattr(l,"height",s[1]+dp(20)))
        card = RndRect(bg=C["card"], radius=10, size_hint_y=None,
                       padding=[dp(14), dp(10)])
        card.bind(minimum_height=card.setter("height"))
        card.add_widget(lbl); self.grid.add_widget(card)


# =============================================================================
# NavBar + App
# =============================================================================
class NavBar(BoxLayout):
    def __init__(self, sm, **kw):
        super().__init__(**kw)
        self.sm = sm
        self.orientation = "horizontal"
        self.size_hint_y = None
        self.height = dp(54)
        with self.canvas.before:
            Color(*hc(C["surface"]))
            self._bg = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=lambda *_: setattr(self._bg,"pos",self.pos),
                  size=lambda *_: setattr(self._bg,"size",self.size))
        self._btns = {}
        tabs = [("library","Libreria"), ("player","Player"), ("stats","Stats")]
        for name, lbl in tabs:
            b = Button(text=lbl, font_size=sp(13), bold=False,
                       background_normal="", background_color=[0,0,0,0],
                       color=hc(C["subtext"]))
            b.bind(on_press=lambda _, n=name: self._go(n))
            self._btns[name] = b; self.add_widget(b)
        self._active("library")

    def _go(self, name): self.sm.current = name; self._active(name)
    def _active(self, name):
        for n, b in self._btns.items():
            b.color = hc(C["accent"]) if n == name else hc(C["subtext"])
            b.bold  = (n == name)


class MusicMoodApp(App):
    def build(self):
        self.title = "Music Mood Analyzer"
        root = BoxLayout(orientation="vertical")
        with root.canvas.before:
            Color(*hc(C["bg"]))
            Rectangle(pos=root.pos, size=root.size)
        self.sm = ScreenManager(transition=FadeTransition(duration=0.12))
        self.sm.add_widget(LibraryScreen(name="library"))
        self.sm.add_widget(PlayerScreen(name="player"))
        self.sm.add_widget(StatsScreen(name="stats"))
        self.nav = NavBar(self.sm)
        root.add_widget(self.sm); root.add_widget(self.nav)
        return root

    def go(self, name):
        self.sm.current = name; self.nav._active(name)

    def on_song_tap(self, song):
        lib = self.sm.get_screen("library")
        fl = lib._filtered
        idx = fl.index(song) if song in fl else 0
        state.play_list(fl, idx); self.go("player")


if __name__ == "__main__":
    MusicMoodApp().run()
