"""
listening_history.py — Tracking e analisi cronologia ascolti

Gestisce un database SQLite locale che registra ogni riproduzione:
- timestamp di quando un brano è stato ascoltato
- metadati del brano (titolo, mood, genere, BPM, etc.)

Fornisce funzionalità di analisi:
- Statistiche settimanali aggregate
- Pattern di ascolto per fascia oraria
- Mood prevalente per periodo
- Grafici temporali di evoluzione mood
"""

import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

DB_PATH = Path("listening_history.db")


# ===========================================================================
# Database setup
# ===========================================================================

def _get_connection() -> sqlite3.Connection:
    """Connessione al database con row_factory per dict-like access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Crea le tabelle se non esistono."""
    conn = _get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS listening_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            song_path TEXT NOT NULL,
            song_title TEXT NOT NULL,
            mood TEXT,
            genre TEXT,
            tempo REAL,
            valence REAL,
            arousal REAL,
            duration REAL,
            hour INTEGER,
            weekday INTEGER,
            date TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_timestamp 
        ON listening_history(timestamp)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_date 
        ON listening_history(date)
    """)
    conn.commit()
    conn.close()
    log.info("Database listening_history inizializzato")


# ===========================================================================
# Tracking ascolti
# ===========================================================================

def log_play(song: dict):
    """
    Registra un ascolto nel database.
    
    Args:
        song: dict con almeno {path, title, mood, genre, tempo, valence, arousal, duration}
    """
    now = datetime.now()
    
    conn = _get_connection()
    conn.execute("""
        INSERT INTO listening_history 
        (timestamp, song_path, song_title, mood, genre, tempo, 
         valence, arousal, duration, hour, weekday, date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now.isoformat(),
        song.get("path", ""),
        song.get("title", "Unknown"),
        song.get("mood", "Unknown"),
        song.get("genre", "Unknown"),
        song.get("tempo", 0.0),
        song.get("valence", 0.5),
        song.get("arousal", 0.5),
        song.get("duration", 0.0),
        now.hour,
        now.weekday(),  # 0=Monday, 6=Sunday
        now.date().isoformat(),
    ))
    conn.commit()
    conn.close()


# ===========================================================================
# Query statistiche
# ===========================================================================

def get_total_plays() -> int:
    """Numero totale di riproduzioni registrate."""
    conn = _get_connection()
    row = conn.execute("SELECT COUNT(*) as cnt FROM listening_history").fetchone()
    conn.close()
    return row["cnt"] if row else 0


def get_plays_last_n_days(n: int = 7) -> list[dict]:
    """
    Restituisce tutte le riproduzioni degli ultimi N giorni.
    
    Returns:
        Lista di dict (uno per ogni riproduzione) con tutti i campi.
    """
    cutoff = (datetime.now() - timedelta(days=n)).isoformat()
    conn = _get_connection()
    rows = conn.execute("""
        SELECT * FROM listening_history 
        WHERE timestamp >= ?
        ORDER BY timestamp DESC
    """, (cutoff,)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_mood_distribution_last_n_days(n: int = 7) -> dict[str, int]:
    """Conteggio mood negli ultimi N giorni."""
    cutoff = (datetime.now() - timedelta(days=n)).isoformat()
    conn = _get_connection()
    rows = conn.execute("""
        SELECT mood, COUNT(*) as cnt 
        FROM listening_history 
        WHERE timestamp >= ? AND mood IS NOT NULL
        GROUP BY mood
        ORDER BY cnt DESC
    """, (cutoff,)).fetchall()
    conn.close()
    return {row["mood"]: row["cnt"] for row in rows}


def get_genre_distribution_last_n_days(n: int = 7) -> dict[str, int]:
    """Conteggio generi negli ultimi N giorni."""
    cutoff = (datetime.now() - timedelta(days=n)).isoformat()
    conn = _get_connection()
    rows = conn.execute("""
        SELECT genre, COUNT(*) as cnt 
        FROM listening_history 
        WHERE timestamp >= ? AND genre IS NOT NULL
        GROUP BY genre
        ORDER BY cnt DESC
    """, (cutoff,)).fetchall()
    conn.close()
    return {row["genre"]: row["cnt"] for row in rows}


def get_top_mood_last_n_days(n: int = 7) -> Optional[str]:
    """Mood più ascoltato negli ultimi N giorni."""
    dist = get_mood_distribution_last_n_days(n)
    if not dist:
        return None
    return max(dist.items(), key=lambda x: x[1])[0]


def get_hourly_mood_distribution() -> dict[int, dict[str, int]]:
    """
    Distribuzione mood per ora del giorno (ultimi 30 giorni).
    
    Returns:
        {hour: {mood: count}} dove hour ∈ [0, 23]
    """
    cutoff = (datetime.now() - timedelta(days=30)).isoformat()
    conn = _get_connection()
    rows = conn.execute("""
        SELECT hour, mood, COUNT(*) as cnt 
        FROM listening_history 
        WHERE timestamp >= ? AND mood IS NOT NULL
        GROUP BY hour, mood
    """, (cutoff,)).fetchall()
    conn.close()
    
    result: dict[int, dict[str, int]] = {}
    for row in rows:
        h = row["hour"]
        m = row["mood"]
        if h not in result:
            result[h] = {}
        result[h][m] = row["cnt"]
    
    return result


def get_daily_mood_evolution(days: int = 14) -> list[dict]:
    """
    Evoluzione mood giorno per giorno negli ultimi N giorni.
    
    Returns:
        Lista di dict {date: str, mood_counts: {mood: count}, avg_valence: float, avg_arousal: float}
    """
    cutoff = (datetime.now() - timedelta(days=days)).date().isoformat()
    conn = _get_connection()
    
    # Aggregazione per data
    rows = conn.execute("""
        SELECT 
            date, 
            mood, 
            COUNT(*) as cnt,
            AVG(valence) as avg_v,
            AVG(arousal) as avg_a
        FROM listening_history 
        WHERE date >= ?
        GROUP BY date, mood
        ORDER BY date ASC
    """, (cutoff,)).fetchall()
    conn.close()
    
    # Riorganizza per data
    by_date: dict[str, dict] = {}
    for row in rows:
        d = row["date"]
        if d not in by_date:
            by_date[d] = {"date": d, "mood_counts": {}, "valence": [], "arousal": []}
        by_date[d]["mood_counts"][row["mood"]] = row["cnt"]
        if row["avg_v"] is not None:
            by_date[d]["valence"].append(row["avg_v"])
        if row["avg_a"] is not None:
            by_date[d]["arousal"].append(row["avg_a"])
    
    # Media valence/arousal per data
    result = []
    for d, data in sorted(by_date.items()):
        result.append({
            "date": d,
            "mood_counts": data["mood_counts"],
            "avg_valence": float(np.mean(data["valence"])) if data["valence"] else 0.5,
            "avg_arousal": float(np.mean(data["arousal"])) if data["arousal"] else 0.5,
        })
    
    return result


def get_most_played_songs_last_n_days(n: int = 7, limit: int = 10) -> list[dict]:
    """Top N brani più ascoltati negli ultimi N giorni."""
    cutoff = (datetime.now() - timedelta(days=n)).isoformat()
    conn = _get_connection()
    rows = conn.execute("""
        SELECT 
            song_title, song_path, mood, genre, 
            COUNT(*) as play_count
        FROM listening_history 
        WHERE timestamp >= ?
        GROUP BY song_path
        ORDER BY play_count DESC
        LIMIT ?
    """, (cutoff, limit)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


# ===========================================================================
# Analisi fasce orarie
# ===========================================================================

def get_time_slot_mood_preference() -> dict[str, str]:
    """
    Mood prevalente per fascia oraria (ultimi 30 giorni).
    
    Returns:
        {"morning": "Energetic", "afternoon": "Positive", "evening": "Melancholic", "night": "Aggressive"}
    """
    hourly = get_hourly_mood_distribution()
    
    slots = {
        "morning":   list(range(6, 12)),    # 06:00 - 11:59
        "afternoon": list(range(12, 18)),   # 12:00 - 17:59
        "evening":   list(range(18, 23)),   # 18:00 - 22:59
        "night":     list(range(0, 6)) + [23],  # 23:00 - 05:59
    }
    
    result = {}
    for slot_name, hours in slots.items():
        mood_total: dict[str, int] = {}
        for h in hours:
            if h in hourly:
                for mood, cnt in hourly[h].items():
                    mood_total[mood] = mood_total.get(mood, 0) + cnt
        
        if mood_total:
            result[slot_name] = max(mood_total.items(), key=lambda x: x[1])[0]
        else:
            result[slot_name] = "Unknown"
    
    return result


# ===========================================================================
# Playlist intelligenti
# ===========================================================================

def get_recommended_songs_for_current_time(all_songs: list[dict], limit: int = 20) -> list[dict]:
    """
    Playlist consigliata per l'ora corrente basata su pattern di ascolto storici.
    
    Args:
        all_songs: Lista completa dei brani disponibili
        limit: Numero massimo di brani da restituire
    
    Returns:
        Lista di brani ordinati per rilevanza (max `limit` elementi)
    """
    now = datetime.now()
    current_hour = now.hour
    
    # Trova il mood prevalente per l'ora corrente (ultimi 30 giorni)
    cutoff = (now - timedelta(days=30)).isoformat()
    conn = _get_connection()
    rows = conn.execute("""
        SELECT mood, COUNT(*) as cnt 
        FROM listening_history 
        WHERE timestamp >= ? AND hour = ? AND mood IS NOT NULL
        GROUP BY mood
        ORDER BY cnt DESC
        LIMIT 1
    """, (cutoff, current_hour)).fetchall()
    conn.close()
    
    if not rows:
        # Nessun dato storico per quest'ora: usa fascia oraria generica
        slots = get_time_slot_mood_preference()
        if 6 <= current_hour < 12:
            target_mood = slots.get("morning", "Energetic")
        elif 12 <= current_hour < 18:
            target_mood = slots.get("afternoon", "Positive")
        elif 18 <= current_hour < 23:
            target_mood = slots.get("evening", "Melancholic")
        else:
            target_mood = slots.get("night", "Aggressive")
    else:
        target_mood = rows[0]["mood"]
    
    # Filtra brani per mood target e mescola
    candidates = [s for s in all_songs if s.get("mood") == target_mood]
    
    if not candidates:
        return []
    
    # Shuffle e limita
    np.random.shuffle(candidates)
    return candidates[:limit]
