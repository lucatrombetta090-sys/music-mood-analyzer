"""
Music Mood Analyzer — Kivy Android Edition (100% on-device)
============================================================
Analisi audio completa sul dispositivo con scipy/numpy.
Nessuna dipendenza da librosa o da un PC.

Dipendenze (tutte con recipe p4a):
  kivy, numpy, scipy, pyjnius, mutagen, plyer, pillow
"""

import os
import threading
import random
from typing import Optional

import kivy
kivy.require("2.3.0")

from kivy.config import Config
Config.set("graphics", "width",  "400")
Config.set("graphics", "height", "750")

from kivy.app                    import App
from kivy.uix.screenmanager      import ScreenManager, Screen, FadeTransition
from kivy.uix.boxlayout          import BoxLayout
from kivy.uix.gridlayout         import GridLayout
from kivy.uix.scrollview         import ScrollView
from kivy.uix.recycleview        import RecycleView
from kivy.uix.recycleview.views  import RecycleDataViewBehavior
from kivy.uix.label              import Label
from kivy.uix.button             import Button
from kivy.uix.textinput          import TextInput
from kivy.uix.slider             import Slider
from kivy.uix.progressbar        import ProgressBar
from kivy.uix.spinner            import Spinner
from kivy.uix.popup              import Popup
from kivy.clock                  import Clock
from kivy.core.audio             import SoundLoader
from kivy.metrics                import dp, sp
from kivy.properties             import NumericProperty, BooleanProperty, ObjectProperty
from kivy.graphics               import Color, RoundedRectangle, Rectangle
from kivy.utils                  import get_color_from_hex

import listening_history as lh
from analyze_mp3 import scan_folders

lh.init_database()

# ── Palette ───────────────────────────────────────────────────────────────────
C = {
    "bg":         "#121212", "surface":    "#1E1E2E", "card":       "#2A2A3E",
    "primary":    "#7C3AED", "accent":     "#A855F7",
    "text":       "#E2E8F0", "subtext":    "#94A3B8",
    "energetic":  "#F59E0B", "positive":   "#10B981",
    "aggressive": "#EF4444", "melancholic":"#8B5CF6",
    "success":    "#22C55E", "danger":     "#EF4444",
}
MOOD_COLORS = {
    "Energetic": C["energetic"], "Positive":    C["positive"],
    "Aggressive":C["aggressive"],"Melancholic": C["melancholic"],
}
MOOD_ICONS  = {"Energetic":"⚡","Positive":"😊","Aggressive":"🔥","Melancholic":"🌧"}
MOOD_ART    = {"Energetic":"⚡","Positive":"🌞","Aggressive":"🔥","Melancholic":"🌧"}
MOODS  = ["Energetic","Positive","Aggressive","Melancholic"]
GENRES = ["Pop","Rock","Electronic","HipHop","Acoustic","Jazz","Classical"]

def hc(h, a=1.0):
    c = get_color_from_hex(h); return (c[0],c[1],c[2],a)

def _fmt(s):
    s=int(s); return f"{s//60}:{s%60:02d}"

def _music_dir():
    try:
        from android.storage import primary_external_storage_path
        return os.path.join(primary_external_storage_path(), "Music")
    except ImportError:
        from pathlib import Path; return str(Path.home()/"Music")


# =============================================================================
# AppState
# =============================================================================
class AppState:
    def __init__(self):
        self.all_songs        = []
        self.current_playlist = []
        self.playlist_index   = 0
        self.current_song     = None
        self._sound           = None
        self._loop            = False
        self._shuffle         = False
        self._volume          = 1.0
        self._cbs             = {}

    def on(self, ev, fn):  self._cbs.setdefault(ev,[]).append(fn)
    def fire(self, ev, *a):
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
        if not s: print(f"[audio] Cannot load {song['path']}"); return
        self._sound = s
        self._sound.volume = self._volume
        self._sound.bind(on_stop=self._ended)
        self._sound.play()
        self.current_song = song
        lh.log_play(song)
        self.fire("song", song)

    def _ended(self, *_):
        if self._loop and self.current_song:
            self.play(self.current_song); return
        if self.current_playlist:
            nxt = (self.playlist_index+1) % len(self.current_playlist)
            if nxt == 0 and not self._loop:
                self.current_song = None; self.fire("song", None)
            else:
                self.playlist_index = nxt
                self.play(self.current_playlist[nxt])

    def toggle_pause(self):
        if not self._sound: return
        if self._sound.state == "play": self._sound.stop()
        else: self._sound.play()
        self.fire("song", self.current_song)

    def next(self):
        if not self.current_playlist: return
        self.playlist_index = (self.playlist_index+1)%len(self.current_playlist)
        self.play(self.current_playlist[self.playlist_index])

    def prev(self):
        if not self.current_playlist: return
        self.playlist_index = (self.playlist_index-1)%len(self.current_playlist)
        self.play(self.current_playlist[self.playlist_index])

    def stop(self):
        self._stop(); self.current_song=None; self.fire("song",None)

    def set_vol(self, v):
        self._volume=v
        if self._sound: self._sound.volume=v

    def toggle_loop(self):    self._loop=not self._loop
    def toggle_shuffle(self): self._shuffle=not self._shuffle

    def play_list(self, songs, idx=0):
        if not songs: return
        lst=songs[:]
        if self._shuffle: random.shuffle(lst); idx=0
        self.current_playlist=lst; self.playlist_index=idx
        self.play(lst[idx])

    def play_mood(self, mood, genre="All"):
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
# UI Components
# =============================================================================
class DkBtn(Button):
    def __init__(self, bg=None, **kw):
        super().__init__(**kw)
        self._bg = bg or hc(C["primary"])
        self.background_normal=""; self.background_color=[0,0,0,0]
        self.color=hc(C["text"]); self.font_size=sp(13)
        self._draw()
        self.bind(pos=lambda *_:self._draw(), size=lambda *_:self._draw())
    def _draw(self):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*self._bg)
            RoundedRectangle(pos=self.pos,size=self.size,radius=[dp(8)])
    def set_bg(self,c): self._bg=c; self._draw()


class SongRow(RecycleDataViewBehavior, BoxLayout):
    index    = NumericProperty(0)
    selected = BooleanProperty(False)

    def __init__(self, **kw):
        super().__init__(**kw)
        self.orientation="vertical"; self.padding=[dp(12),dp(6)]
        self.spacing=dp(2); self.size_hint_y=None; self.height=dp(72)
        self._song=None
        self.t=Label(font_size=sp(13),bold=True,color=hc(C["text"]),
                     halign="left",valign="middle",size_hint_y=None,height=dp(22))
        self.m=Label(font_size=sp(11),color=hc(C["subtext"]),
                     halign="left",valign="middle",size_hint_y=None,height=dp(18))
        self.add_widget(self.t); self.add_widget(self.m)
        self.bind(pos=self._rd,size=self._rd)

    def _rd(self,*_):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*hc(C["primary"],0.3)) if self.selected else Color(*hc(C["card"]))
            RoundedRectangle(pos=(self.x+dp(4),self.y+dp(2)),
                             size=(self.width-dp(8),self.height-dp(4)),radius=[dp(8)])

    def refresh_view_attrs(self,rv,index,data):
        self.index=index; self._song=data.get("song",{}); self.selected=data.get("selected",False)
        s=self._song; mood=s.get("mood","")
        title=s.get("title","?")
        if title.lower().endswith(".mp3"): title=title[:-4]
        if len(title)>38: title=title[:38]+"…"
        genre=s.get("genre","?")
        if s.get("genre_alt"): genre+="/"+s["genre_alt"]
        self.t.text=f"{MOOD_ICONS.get(mood,'•')} {title}"
        self.t.color=hc(MOOD_COLORS.get(mood,C["text"]))
        self.m.text=f"{genre}  ·  {s.get('tempo',0):.0f} BPM  ·  {mood or '—'}"
        self._rd()
        return super().refresh_view_attrs(rv,index,data)

    def on_touch_down(self,touch):
        if self.collide_point(*touch.pos) and self._song:
            App.get_running_app().on_song_tap(self._song); return True
        return super().on_touch_down(touch)


class SongRV(RecycleView):
    def __init__(self,**kw):
        super().__init__(**kw); self.viewclass="SongRow"; self.data=[]
        with self.canvas.before:
            Color(*hc(C["bg"])); self._bg=Rectangle(pos=self.pos,size=self.size)
        self.bind(pos=lambda *_:setattr(self._bg,"pos",self.pos),
                  size=lambda *_:setattr(self._bg,"size",self.size))


# =============================================================================
# LibraryScreen
# =============================================================================
class LibraryScreen(Screen):
    def __init__(self,**kw):
        super().__init__(**kw); self._filtered=[]; self._build()
        state.on("song",self._highlight)

    def _build(self):
        root=BoxLayout(orientation="vertical",spacing=dp(6),
                       padding=[dp(10),dp(8),dp(10),dp(8)])
        with root.canvas.before:
            Color(*hc(C["bg"])); Rectangle(pos=root.pos,size=root.size)

        # Header
        hdr=BoxLayout(size_hint_y=None,height=dp(44),spacing=dp(6))
        self.search=TextInput(hint_text="🔍 Cerca brano…",
                              background_color=hc(C["card"]),foreground_color=hc(C["text"]),
                              hint_text_color=hc(C["subtext"]),font_size=sp(13),
                              multiline=False,size_hint_x=0.58,padding=[dp(10),dp(10)])
        self.search.bind(text=lambda *_:self._refresh())
        scan_btn=DkBtn(text="📂 Scansiona",size_hint_x=0.42,bg=hc(C["primary"]))
        scan_btn.bind(on_press=self._scan)
        hdr.add_widget(self.search); hdr.add_widget(scan_btn)
        root.add_widget(hdr)

        # Filtri
        frow=BoxLayout(size_hint_y=None,height=dp(36),spacing=dp(6))
        self.mood_sp=Spinner(text="Mood: All",values=["Mood: All"]+MOODS,
                             background_normal="",background_color=hc(C["card"]),
                             color=hc(C["text"]),font_size=sp(11))
        self.genre_sp=Spinner(text="Genere: All",values=["Genere: All"]+GENRES,
                              background_normal="",background_color=hc(C["card"]),
                              color=hc(C["text"]),font_size=sp(11))
        self.cnt=Label(text="0 brani",font_size=sp(11),color=hc(C["subtext"]),size_hint_x=0.26)
        self.mood_sp.bind(text=lambda *_:self._refresh())
        self.genre_sp.bind(text=lambda *_:self._refresh())
        frow.add_widget(self.mood_sp); frow.add_widget(self.genre_sp); frow.add_widget(self.cnt)
        root.add_widget(frow)

        # Status + progress
        self.status=Label(text="Tocca 📂 per scansionare la tua cartella Music",
                          font_size=sp(11),color=hc(C["subtext"]),
                          size_hint_y=None,height=dp(18))
        self.pbar=ProgressBar(max=100,value=0,size_hint_y=None,height=dp(4))
        root.add_widget(self.status); root.add_widget(self.pbar)

        # Lista
        self.rv=SongRV(size_hint=(1,1)); root.add_widget(self.rv)

        # Mood rapidi
        mrow=BoxLayout(size_hint_y=None,height=dp(44),spacing=dp(6))
        for mood in MOODS:
            b=DkBtn(text=f"{MOOD_ICONS[mood]} {mood}",
                    bg=hc(MOOD_COLORS[mood]),size_hint_x=0.25,font_size=sp(10))
            b.bind(on_press=lambda _,m=mood:self._play_mood(m))
            mrow.add_widget(b)
        root.add_widget(mrow)
        self.add_widget(root)

    def _get_filtered(self):
        q=self.search.text.lower().strip()
        mood=self.mood_sp.text.replace("Mood: ","")
        genre=self.genre_sp.text.replace("Genere: ","")
        return [s for s in state.all_songs
                if (not q or q in s.get("title","").lower())
                and (mood=="All" or s.get("mood")==mood)
                and (genre=="All" or s.get("genre")==genre)]

    def _refresh(self,*_):
        self._filtered=self._get_filtered()
        cur=state.current_song
        self.rv.data=[{"song":s,"selected":bool(cur and s.get("path")==cur.get("path"))}
                      for s in self._filtered]
        self.cnt.text=f"{len(self._filtered)} brani"

    def _highlight(self,*_): self._refresh()

    def _scan(self,*_):
        try:
            from plyer import filechooser
            filechooser.open_file(title="Seleziona un MP3 nella cartella da scansionare",
                                   filters=["*.mp3"],on_selection=self._on_sel)
        except Exception:
            self._do_scan(_music_dir())

    def _on_sel(self,sel):
        if sel: self._do_scan(os.path.dirname(sel[0]))

    def _do_scan(self,folder):
        if not os.path.isdir(folder):
            self._toast(f"Cartella non trovata:\n{folder}"); return
        self.status.text="⏳ Analisi in corso (può richiedere alcuni minuti)…"
        self.pbar.value=5

        def _prog(cur,tot,fname):
            pct=cur/tot*100 if tot else 0
            Clock.schedule_once(
                lambda dt:self._upd_prog(pct,f"[{cur}/{tot}] {fname[:28]}"),0)

        def _worker():
            songs=scan_folders([folder],progress_callback=_prog)
            Clock.schedule_once(lambda dt:self._done(songs),0)

        threading.Thread(target=_worker,daemon=True).start()

    def _upd_prog(self,pct,txt):
        self.pbar.value=pct; self.status.text=txt

    def _done(self,songs):
        state.all_songs=songs
        self.pbar.value=100
        self.status.text=f"✓ {len(songs)} brani analizzati"
        self._refresh()

    def _play_mood(self,mood):
        genre=self.genre_sp.text.replace("Genere: ","")
        if not state.play_mood(mood,genre):
            self._toast(f"Nessun brano con mood: {mood}\nEsegui prima una scansione.")
        else:
            App.get_running_app().go("player")

    def _toast(self,msg):
        c=BoxLayout(orientation="vertical",padding=dp(16),spacing=dp(10))
        c.add_widget(Label(text=msg,color=hc(C["text"]),font_size=sp(12),
                           halign="center",text_size=(dp(260),None)))
        b=DkBtn(text="OK",size_hint_y=None,height=dp(40))
        p=Popup(title="Info",content=c,size_hint=(0.85,0.42),
                background_color=hc(C["surface"]),title_color=hc(C["text"]))
        b.bind(on_press=p.dismiss); c.add_widget(b); p.open()


# =============================================================================
# PlayerScreen
# =============================================================================
class PlayerScreen(Screen):
    def __init__(self,**kw):
        super().__init__(**kw); self._build()
        state.on("song",self._upd)
        Clock.schedule_interval(self._tick,0.5)

    def _build(self):
        root=BoxLayout(orientation="vertical",spacing=dp(10),
                       padding=[dp(16),dp(20),dp(16),dp(16)])
        with root.canvas.before:
            Color(*hc(C["bg"]))
            self._bg=Rectangle(pos=root.pos,size=root.size)
        root.bind(pos=lambda *_:setattr(self._bg,"pos",root.pos),
                  size=lambda *_:setattr(self._bg,"size",root.size))

        # Artwork
        art_box=BoxLayout(size_hint_y=None,height=dp(160))
        self.art=Label(text="🎵",font_size=sp(72))
        with self.art.canvas.before:
            Color(*hc(C["card"]))
            self._art_r=RoundedRectangle(pos=self.art.pos,size=self.art.size,radius=[dp(20)])
        self.art.bind(pos=lambda *_:setattr(self._art_r,"pos",self.art.pos),
                      size=lambda *_:setattr(self._art_r,"size",self.art.size))
        art_box.add_widget(self.art); root.add_widget(art_box)

        # Badge mood
        self.badge=Label(text="",font_size=sp(13),color=hc(C["accent"]),
                         size_hint_y=None,height=dp(24))
        root.add_widget(self.badge)

        # Titolo
        self.title_lbl=Label(text="Nessun brano in riproduzione",font_size=sp(15),bold=True,
                             color=hc(C["text"]),size_hint_y=None,height=dp(44),
                             halign="center",text_size=(None,None))
        root.add_widget(self.title_lbl)

        # Meta
        self.meta=Label(text="",font_size=sp(12),color=hc(C["subtext"]),
                        size_hint_y=None,height=dp(20))
        root.add_widget(self.meta)

        # Progress
        self.prog=ProgressBar(max=100,value=0,size_hint_y=None,height=dp(6))
        root.add_widget(self.prog)
        self.time_lbl=Label(text="0:00 / 0:00",font_size=sp(11),
                            color=hc(C["subtext"]),size_hint_y=None,height=dp(18))
        root.add_widget(self.time_lbl)

        # Controlli principali
        ctrl=BoxLayout(size_hint_y=None,height=dp(64),spacing=dp(10))
        prev_b=DkBtn(text="⏮",font_size=sp(22),bg=hc(C["card"]))
        prev_b.bind(on_press=lambda *_:state.prev())
        self.play_b=DkBtn(text="▶",font_size=sp(26),bg=hc(C["primary"]))
        self.play_b.bind(on_press=lambda *_:state.toggle_pause())
        next_b=DkBtn(text="⏭",font_size=sp(22),bg=hc(C["card"]))
        next_b.bind(on_press=lambda *_:state.next())
        ctrl.add_widget(prev_b); ctrl.add_widget(self.play_b); ctrl.add_widget(next_b)
        root.add_widget(ctrl)

        # Controlli secondari
        ctrl2=BoxLayout(size_hint_y=None,height=dp(44),spacing=dp(10))
        self.loop_b=DkBtn(text="🔁 Loop",font_size=sp(13),bg=hc(C["card"]))
        self.shuf_b=DkBtn(text="🔀 Shuffle",font_size=sp(13),bg=hc(C["card"]))
        stop_b=DkBtn(text="⏹ Stop",font_size=sp(13),bg=hc(C["danger"]))
        self.loop_b.bind(on_press=self._loop)
        self.shuf_b.bind(on_press=self._shuf)
        stop_b.bind(on_press=lambda *_:state.stop())
        ctrl2.add_widget(self.loop_b); ctrl2.add_widget(self.shuf_b); ctrl2.add_widget(stop_b)
        root.add_widget(ctrl2)

        # Volume
        vrow=BoxLayout(size_hint_y=None,height=dp(40),spacing=dp(8))
        vrow.add_widget(Label(text="🔈",font_size=sp(16),size_hint_x=None,
                              width=dp(28),color=hc(C["subtext"])))
        self.vol=Slider(min=0,max=1,value=1)
        self.vol.bind(value=lambda _,v:state.set_vol(v))
        vrow.add_widget(self.vol)
        vrow.add_widget(Label(text="🔊",font_size=sp(16),size_hint_x=None,
                              width=dp(28),color=hc(C["subtext"])))
        root.add_widget(vrow)
        self.add_widget(root)

    def _upd(self,song,*_):
        if not song:
            self.title_lbl.text="Nessun brano in riproduzione"
            self.badge.text=""; self.meta.text=""; self.art.text="🎵"
            self.play_b.text="▶"; return
        t=song.get("title","")
        if t.lower().endswith(".mp3"): t=t[:-4]
        if len(t)>32: t=t[:32]+"…"
        mood=song.get("mood","")
        genre=song.get("genre","?")
        if song.get("genre_alt"): genre+="/"+song["genre_alt"]
        self.title_lbl.text=t
        self.badge.text=f"{MOOD_ICONS.get(mood,'')} {mood}"
        self.badge.color=hc(MOOD_COLORS.get(mood,C["accent"]))
        self.meta.text=f"{genre}  ·  {song.get('tempo',0):.0f} BPM  ·  {'Magg.' if song.get('mode_major') else 'Min.'}"
        self.art.text=MOOD_ART.get(mood,"🎵")
        self.play_b.text="⏸" if state.is_playing() else "▶"

    def _tick(self,dt):
        pos,dur=state.position()
        if dur>0: self.prog.value=min(100,pos/dur*100); self.time_lbl.text=f"{_fmt(pos)} / {_fmt(dur)}"
        self.play_b.text="⏸" if state.is_playing() else "▶"

    def _loop(self,*_):
        state.toggle_loop()
        self.loop_b.set_bg(hc(C["success"] if state._loop else C["card"]))

    def _shuf(self,*_):
        state.toggle_shuffle()
        self.shuf_b.set_bg(hc(C["success"] if state._shuffle else C["card"]))


# =============================================================================
# StatsScreen
# =============================================================================
class StatsScreen(Screen):
    def __init__(self,**kw):
        super().__init__(**kw); self._build()

    def _build(self):
        root=BoxLayout(orientation="vertical",spacing=dp(8),
                       padding=[dp(12),dp(10),dp(12),dp(10)])
        with root.canvas.before:
            Color(*hc(C["bg"])); Rectangle(pos=root.pos,size=root.size)
        root.add_widget(Label(text="📊 Statistiche Ascolti",font_size=sp(17),bold=True,
                              color=hc(C["text"]),size_hint_y=None,height=dp(40)))
        rb=DkBtn(text="🔄 Aggiorna",size_hint_y=None,height=dp(38),bg=hc(C["primary"]))
        rb.bind(on_press=lambda *_:self._load())
        root.add_widget(rb)
        sc=ScrollView()
        self.grid=GridLayout(cols=1,spacing=dp(8),size_hint_y=None,padding=[dp(4),dp(4)])
        self.grid.bind(minimum_height=self.grid.setter("height"))
        sc.add_widget(self.grid); root.add_widget(sc)
        self.add_widget(root)
        self._load()

    def on_enter(self): self._load()

    def _load(self,*_):
        self.grid.clear_widgets()
        total=lh.get_total_plays()
        mood_d=lh.get_mood_distribution_last_n_days(7)
        genre_d=lh.get_genre_distribution_last_n_days(7)
        top=lh.get_most_played_songs_last_n_days(7,5)
        slots=lh.get_time_slot_mood_preference()
        self._card(f"🎧  Totale riproduzioni: {total}")
        self._card(f"📚  Brani in libreria: {len(state.all_songs)}")
        if mood_d:
            self._card("🎭  Mood (7 giorni):\n"+"\n".join(
                f"  {MOOD_ICONS.get(m,'•')} {m}: {c}" for m,c in mood_d.items()))
        if genre_d:
            self._card("🎸  Generi (7 giorni):\n"+"\n".join(
                f"  {g}: {c}" for g,c in genre_d.items()))
        if slots:
            icons={"morning":"🌅","afternoon":"☀️","evening":"🌆","night":"🌙"}
            self._card("🕐  Preferenze orarie:\n"+"\n".join(
                f"  {icons.get(k,'•')} {k.capitalize()}: {v}" for k,v in slots.items()))
        if top:
            self._card("🏆  Top 5 brani (7gg):\n"+"\n".join(
                f"  {i+1}. {s['song_title'][:30]} ({s['play_count']}x)"
                for i,s in enumerate(top)))

    def _card(self,txt):
        lbl=Label(text=txt,font_size=sp(12),color=hc(C["text"]),
                  size_hint_y=None,halign="left",valign="top",text_size=(None,None))
        lbl.bind(texture_size=lambda l,s:setattr(l,"height",s[1]+dp(20)))
        card=BoxLayout(size_hint_y=None,padding=[dp(14),dp(10)])
        card.bind(minimum_height=card.setter("height"))
        def _draw(c,*_):
            c.canvas.before.clear()
            with c.canvas.before:
                Color(*hc(C["card"])); RoundedRectangle(pos=c.pos,size=c.size,radius=[dp(10)])
        card.bind(pos=_draw,size=_draw)
        card.add_widget(lbl); self.grid.add_widget(card)


# =============================================================================
# NavBar
# =============================================================================
class NavBar(BoxLayout):
    def __init__(self,sm,**kw):
        super().__init__(**kw); self.sm=sm
        self.orientation="horizontal"; self.size_hint_y=None; self.height=dp(56)
        with self.canvas.before:
            Color(*hc(C["surface"])); self._bg=Rectangle(pos=self.pos,size=self.size)
        self.bind(pos=lambda *_:setattr(self._bg,"pos",self.pos),
                  size=lambda *_:setattr(self._bg,"size",self.size))
        self._btns={}
        for name,lbl in [("library","🎵 Libreria"),("player","▶ Player"),("stats","📊 Stats")]:
            b=Button(text=lbl,font_size=sp(12),background_normal="",
                     background_color=[0,0,0,0],color=hc(C["subtext"]))
            b.bind(on_press=lambda _,n=name:self._go(n))
            self._btns[name]=b; self.add_widget(b)
        self._active("library")

    def _go(self,name): self.sm.current=name; self._active(name)
    def _active(self,name):
        for n,b in self._btns.items():
            b.color=hc(C["accent"]) if n==name else hc(C["subtext"])


# =============================================================================
# App
# =============================================================================
class MusicMoodApp(App):
    def build(self):
        self.title="Music Mood Analyzer"
        root=BoxLayout(orientation="vertical")
        with root.canvas.before:
            Color(*hc(C["bg"])); Rectangle(pos=root.pos,size=root.size)
        self.sm=ScreenManager(transition=FadeTransition(duration=0.15))
        self.sm.add_widget(LibraryScreen(name="library"))
        self.sm.add_widget(PlayerScreen(name="player"))
        self.sm.add_widget(StatsScreen(name="stats"))
        self.nav=NavBar(self.sm)
        root.add_widget(self.sm); root.add_widget(self.nav)
        return root

    def go(self,name): self.sm.current=name; self.nav._active(name)

    def on_song_tap(self,song):
        lib=self.sm.get_screen("library")
        fl=lib._filtered
        idx=fl.index(song) if song in fl else 0
        state.play_list(fl,idx); self.go("player")


if __name__=="__main__":
    MusicMoodApp().run()
