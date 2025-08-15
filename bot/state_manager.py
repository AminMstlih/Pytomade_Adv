# bot/state_manager.py
"""Simple state management for the bot."""

import time

# --- Global State Variables ---
# Using module-level variables for simplicity.
# For persistence across restarts, consider saving/loading to/from a file or database.

_last_signal = None
_last_trade_time = 0
# Add more state variables as needed, e.g., daily PnL, error counts

def get_last_signal():
    """Gets the last signal that was acted upon."""
    global _last_signal
    return _last_signal

def set_last_signal(signal):
    """Sets the last signal that was acted upon."""
    global _last_signal
    _last_signal = signal

def get_last_trade_time():
    """Gets the timestamp of the last trade attempt."""
    global _last_trade_time
    return _last_trade_time

def set_last_trade_time(timestamp):
    """Sets the timestamp of the last trade attempt."""
    global _last_trade_time
    _last_trade_time = timestamp

def is_signal_cooldown_active(cooldown_period_seconds):
    """
    Checks if the signal cooldown period is still active.
    """
    current_time = time.time()
    last_time = get_last_trade_time()
    return (current_time - last_time) < cooldown_period_seconds

# --- Optional: Functions for more complex state ---
# def update_daily_pnl(...):
# def get_daily_pnl():
# def reset_daily_state(): # If tracking daily stats
# def load_state_from_file(filepath):
# def save_state_to_file(filepath):