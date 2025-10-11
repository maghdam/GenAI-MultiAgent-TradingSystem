import sqlite3
import threading
from pathlib import Path
from typing import Optional

# Database file path
DB_FILE = Path(__file__).parent.parent / "data" / "journal.db"

# Thread-local storage for database connections to ensure thread safety
local = threading.local()

def _get_db():
    """Establishes a new database connection or returns the existing one for the current thread."""
    if not hasattr(local, 'connection'):
        DB_FILE.parent.mkdir(parents=True, exist_ok=True)
        local.connection = sqlite3.connect(DB_FILE, check_same_thread=False)
        local.connection.row_factory = sqlite3.Row
    return local.connection

def _init_db():
    """Initializes the database schema if it doesn't already exist."""
    db = _get_db()
    cursor = db.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL,
            volume REAL NOT NULL,
            entry_price REAL,
            stop_loss REAL,
            take_profit REAL,
            rationale TEXT,
            exit_price REAL,
            pnl REAL
        );
    """)
    db.commit()

def add_trade_entry(
    symbol: str,
    direction: str,
    volume: float,
    entry_price: Optional[float] = None,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    rationale: Optional[str] = None
):
    """Adds a new trade record to the journal."""
    db = _get_db()
    cursor = db.cursor()
    cursor.execute("""
        INSERT INTO trades (symbol, direction, volume, entry_price, stop_loss, take_profit, rationale)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (symbol, direction, volume, entry_price, stop_loss, take_profit, rationale))
    db.commit()
    return cursor.lastrowid

def get_all_trades(limit: int = 100) -> list:
    """Retrieves all trade records from the journal, most recent first."""
    db = _get_db()
    cursor = db.cursor()
    cursor.execute("""
        SELECT id, timestamp, symbol, direction, volume, entry_price, stop_loss, take_profit, rationale, exit_price, pnl
        FROM trades
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))
    return [dict(row) for row in cursor.fetchall()]

# Initialize the database on module load
_init_db()
