"""
============================================
Paper Trading SQLite Database Persistence
============================================
Handles persistent storage of virtual accounts and positions for paper trading.
Prevents state loss upon bot or dashboard restarts.
"""

import os
import sqlite3
from typing import Dict, Any, List, Optional
from config.settings import config
from utils.logger import get_logger

logger = get_logger()


class PaperDBManager:
    """
    Manages persistence of paper trading accounts and open positions using SQLite.
    Each symbol and timeframe will have its own isolated database inside its log directory.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Fallback path if log_dir is empty
            db_dir = config.paths.logs_dir or os.path.join(config.paths.base_dir, "logs")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "paper_trading.db")

        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        """Create a new database connection."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialise database schema if not exists."""
        logger.info(f"Initialising Paper Trading database at: {self.db_path}")
        try:
            with self._connect() as conn:
                # Account table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS account (
                        account_key TEXT PRIMARY KEY,
                        balance REAL,
                        equity REAL,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                # Positions table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS positions (
                        ticket INTEGER PRIMARY KEY,
                        symbol TEXT,
                        type TEXT,
                        volume REAL,
                        open_price REAL,
                        sl REAL,
                        tp REAL,
                        comment TEXT,
                        time REAL
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Error initialising Paper Trading SQLite DB: {e}", exc_info=True)

    def get_account_state(self, default_balance: float) -> Dict[str, float]:
        """Fetch balance and equity from DB, initializing if not present."""
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT balance, equity FROM account WHERE account_key = 'main'"
                ).fetchone()
                if row:
                    return {"balance": row["balance"], "equity": row["equity"]}
                
                # If not present, initialize with default
                conn.execute(
                    "INSERT INTO account (account_key, balance, equity) VALUES ('main', ?, ?)",
                    (default_balance, default_balance)
                )
                conn.commit()
                return {"balance": default_balance, "equity": default_balance}
        except Exception as e:
            logger.error(f"Error getting account state from DB: {e}", exc_info=True)
            return {"balance": default_balance, "equity": default_balance}

    def save_account_state(self, balance: float, equity: float):
        """Upsert balance and equity into DB."""
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO account (account_key, balance, equity, last_updated)
                    VALUES ('main', ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(account_key) DO UPDATE SET
                        balance = excluded.balance,
                        equity = excluded.equity,
                        last_updated = CURRENT_TIMESTAMP
                    """,
                    (balance, equity)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving account state to DB: {e}", exc_info=True)

    def get_positions(self) -> List[Dict[str, Any]]:
        """Fetch all open virtual positions."""
        try:
            with self._connect() as conn:
                rows = conn.execute("SELECT * FROM positions").fetchall()
                positions = []
                for row in rows:
                    positions.append({
                        "ticket": row["ticket"],
                        "symbol": row["symbol"],
                        "type": row["type"],
                        "volume": row["volume"],
                        "open_price": row["open_price"],
                        "sl": row["sl"],
                        "tp": row["tp"],
                        "comment": row["comment"],
                        "time": row["time"]
                    })
                return positions
        except Exception as e:
            logger.error(f"Error getting positions from DB: {e}", exc_info=True)
            return []

    def save_position(self, pos: Dict[str, Any]):
        """Save a new or modified position."""
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO positions (ticket, symbol, type, volume, open_price, sl, tp, comment, time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pos["ticket"], pos["symbol"], pos["type"], pos["volume"],
                        pos["open_price"], pos["sl"], pos["tp"], pos["comment"], pos["time"]
                    )
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving position to DB: {e}", exc_info=True)

    def delete_position(self, ticket: int):
        """Delete a position by ticket."""
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM positions WHERE ticket = ?", (ticket,))
                conn.commit()
        except Exception as e:
            logger.error(f"Error deleting position from DB: {e}", exc_info=True)

    def get_next_ticket(self, start_ticket: int = 100000) -> int:
        """Find max ticket number and return the next ticket."""
        try:
            with self._connect() as conn:
                row = conn.execute("SELECT MAX(ticket) as max_t FROM positions").fetchone()
                if row and row["max_t"]:
                    return max(start_ticket, row["max_t"]) + 1
                return start_ticket
        except Exception as e:
            logger.error(f"Error getting max ticket: {e}", exc_info=True)
            return start_ticket
