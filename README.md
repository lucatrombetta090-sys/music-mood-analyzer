[README.md](https://github.com/user-attachments/files/25603761/README.md)
# 🎵 Music Mood Analyzer — Versione Kivy (Android)

Riscrittura completa per Android dell'app originale, usando **Kivy** al posto di
tkinter/ttkbootstrap. Tutta la logica di analisi audio è invariata.

---

## 📱 Struttura del progetto

```
kivy_app/
├── main.py              ← App Kivy (UI + player + navigazione)
├── analyze_mp3.py       ← Analisi audio (INVARIATO dall'originale)
├── listening_history.py ← Database SQLite (INVARIATO dall'originale)
├── buildozer.spec       ← Configurazione build APK
├── requirements.txt     ← Dipendenze Python
└── README.md            ← Questo file
```

---

## 🚀 Come compilare l'APK

### Prerequisiti (su Linux/macOS o WSL2 su Windows)

```bash
# 1. Installa dipendenze sistema (Ubuntu/Debian)
sudo apt update
sudo apt install -y \
    python3-pip \
    python3-venv \
    git \
    zip \
    unzip \
    openjdk-17-jdk \
    ccache \
    libffi-dev \
    libssl-dev \
    autoconf \
    automake \
    libtool \
    pkg-config \
    zlib1g-dev \
    libncurses5-dev \
    cmake

# 2. Installa buildozer
pip3 install --user buildozer cython

# 3. Clona la repo e naviga nella cartella
cd kivy_app/

# 4. Prima build (scarica Android SDK, NDK — richiede ~10 GB e 30-60 minuti)
buildozer android debug

# L'APK sarà in: bin/musicmoodanalyzer-1.0-arm64-v8a-debug.apk
```

### Installare l'APK sul telefono

```bash
# Con ADB (telefono collegato via USB con debug abilitato)
adb install bin/musicmoodanalyzer-1.0-*.apk

# Oppure copia il file APK e aprilo dal gestore file
```

---

## 🛠 Build con Docker (più semplice)

```bash
docker run --rm \
    -v "$(pwd)":/home/user/hostcwd \
    kivy/buildozer \
    android debug
```

---

## 📱 Funzionalità nell'app Android

| Feature              | Desktop (originale) | Android (Kivy) |
|---------------------|--------------------|--------------------|
| Analisi MP3         | ✅ librosa          | ✅ librosa          |
| Classificazione mood | ✅ V-A model        | ✅ invariato        |
| Classificazione genere | ✅ multi-score   | ✅ invariato        |
| Cache analisi       | ✅ JSON             | ✅ invariato        |
| Database ascolti    | ✅ SQLite           | ✅ invariato        |
| Player audio        | ✅ pygame           | ✅ Kivy SoundLoader |
| Playlist per mood   | ✅                  | ✅                  |
| Filtri (mood/genere)| ✅                  | ✅                  |
| Ricerca brani       | ✅                  | ✅                  |
| Loop / Shuffle      | ✅                  | ✅                  |
| Volume slider       | ✅                  | ✅                  |
| Progress bar        | ✅                  | ✅                  |
| Statistiche         | ✅ grafici matplotlib| ✅ testo (no grafici)|
| Scatter V/A plot    | ✅ matplotlib       | ⛔ rimosso (troppo pesante) |

---

## 🎨 UI — Schermate

```
┌─────────────────────────┐
│      🎵 Libreria        │  ← Ricerca + Filtri mood/genere
│  [Cerca…]  [Mood▼][Gen▼]│
│  ████████████████  95%  │  ← Barra scansione
│ ⚡ Artist - Song.mp3    │
│   Rock · 128 BPM · ...  │  ← Lista brani (RecycleView)
│ 😊 Artist2 - Song2.mp3  │
│  ⚡Energetic 😊Positive  │  ← Playlist rapide per mood
│  🔥Aggressive 🌧Melanch │
├─────────────────────────┤
│   🎵 Lib  ▶ Player 📊  │  ← Bottom navigation
└─────────────────────────┘

┌─────────────────────────┐
│          ⚡             │  ← Artwork emoji animata per mood
│    Artist - Song        │
│  ⚡ Energetic           │  ← Badge mood colorato
│  Rock · 128 BPM · Magg │
│  ████████████░░░  2:13  │  ← Progress bar
│   ⏮    ⏸    ⏭         │
│  🔁Loop  🔀Shuf  ⏹Stop │
│  🔈 ──────────── 🔊    │  ← Volume
├─────────────────────────┤
│   🎵 Lib  ▶ Player 📊  │
└─────────────────────────┘
```

---

## ⚠️ Note importanti su Android

### Accesso ai file (Android 10+)
A causa dello **scoped storage** di Android 10+, l'app usa `plyer.filechooser`
per permettere all'utente di selezionare un file MP3 nella cartella Music.
L'intera cartella verrà poi scansionata.

### Dimensione APK
Libreria librosa + scipy + numpy = APK di circa **80-120 MB**.
Normale per un'app di analisi audio avanzata.

### Prima scansione
La prima analisi di una cartella richiede tempo (librosa analizza ogni brano).
I risultati vengono cachati in `music_cache.json` per le scansioni successive.

### Permessi richiesti
- `READ_EXTERNAL_STORAGE` / `READ_MEDIA_AUDIO` — lettura file MP3
- `WRITE_EXTERNAL_STORAGE` — salvataggio cache e database

---

## 🐛 Troubleshooting

**Errore compilazione librosa:** Assicurati di usare p4a dal branch `develop`
(già configurato in `buildozer.spec`).

**APK si chiude all'avvio:** Controlla i log con `adb logcat | grep python`.

**File MP3 non trovati:** Su Android 13+, l'app chiede il permesso
`READ_MEDIA_AUDIO` — assicurati di concederlo.

---

## 📦 Versione minima Android

Android 7.0 (API 24) o superiore.
