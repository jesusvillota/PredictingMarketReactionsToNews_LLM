"""Data utility functions."""

import bisect
from datetime import datetime, date
from typing import List
import pandas as pd

from src.config import get_logger, config_settings

logger = get_logger("data.utils")


class TradingCalendarAdjustments:
    """Handle trading calendar adjustments for article publication dates."""
    
    def __init__(self, trading_days: List[date], closing_time: str = "17:30:00"):
        """
        Initialize trading calendar adjustments.
        
        Args:
            trading_days: List of trading days
            closing_time: Market closing time in format 'HH:MM:SS'
        """
        self.trading_days = sorted(trading_days)
        self.closing_time = datetime.strptime(closing_time, '%H:%M:%S').time()
    
    def is_trading_day(self, date: date) -> bool:
        """Check if a date is a trading day."""
        return date in self.trading_days
    
    def closest_trading_day_at_or_before(self, day_x: date) -> date:
        """Find the closest trading day at or before the given date."""
        index = bisect.bisect_right(self.trading_days, day_x) - 1
        if index < 0:
            return None
        return self.trading_days[index]
    
    def closest_trading_day_at_or_after(self, day_x: date) -> date:
        """Find the closest trading day at or after the given date."""
        index = bisect.bisect_left(self.trading_days, day_x)
        if index == len(self.trading_days):
            return None
        return self.trading_days[index]
    
    def next_trading_day(self, date: date) -> date:
        """Get the next trading day after the given date."""
        date_corrected = self.closest_trading_day_at_or_before(date)
        if date_corrected is None:
            return None
        index = self.trading_days.index(date_corrected) + 1
        if index >= len(self.trading_days):
            return None
        return self.trading_days[index]
    
    def impute_date_affect(self, publ_datetime: datetime) -> date:
        """
        Impute the effective treatment date based on publication datetime.
        
        If published on a trading day before closing time, use that day.
        Otherwise, use the next trading day.
        
        Args:
            publ_datetime: Publication datetime
        
        Returns:
            Effective treatment date
        """
        publ_date = publ_datetime.date()
        publ_time = publ_datetime.time()
        
        if self.is_trading_day(publ_date) and publ_time < self.closing_time:
            return publ_date
        else:
            return self.next_trading_day(publ_date)


def create_trading_calendar_adjustments(r_data: pd.DataFrame) -> TradingCalendarAdjustments:
    """
    Create TradingCalendarAdjustments instance from return data.
    
    Args:
        r_data: DataFrame with trading days as index
    
    Returns:
        TradingCalendarAdjustments instance
    """
    trading_days = [dt.date() if isinstance(dt, datetime) else dt for dt in r_data.index]
    closing_time = config_settings.data_config.get("closing_time", "17:30:00")
    return TradingCalendarAdjustments(trading_days, closing_time)

