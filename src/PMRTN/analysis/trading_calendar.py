"""Trading calendar utilities for date adjustments.

This module provides functionality to handle trading day calculations and adjustments
for financial market data, ensuring that news publication dates are correctly mapped
to the appropriate trading days.
"""

import bisect
from datetime import datetime, date, time
from typing import List, Optional


class TradingCalendarAdjustments:
    """Manages trading calendar and date adjustments.
    
    This class provides methods to determine trading days, find the closest trading
    days, and impute the effective date when a news article affects the market based
    on publication time and trading calendar.
    
    The imputation logic follows:
    - If article is published on a trading day before closing time → same day
    - Otherwise → next available trading day
    
    Attributes:
        trading_days: Sorted list of trading days (date objects)
        closing_time: Market closing time (time object, default 17:30:00)
    """
    
    def __init__(self, trading_days: List[date], closing_time: str = '17:30:00'):
        """Initialize TradingCalendarAdjustments.
        
        Args:
            trading_days: List of trading days as date objects
            closing_time: Market closing time in 'HH:MM:SS' format (default '17:30:00')
            
        Raises:
            ValueError: If closing_time format is invalid
        """
        self.trading_days = sorted(trading_days)  # Ensure the trading days are sorted
        try:
            self.closing_time = datetime.strptime(closing_time, '%H:%M:%S').time()
        except ValueError as e:
            raise ValueError(f"Invalid closing_time format. Expected 'HH:MM:SS': {e}")
    
    def is_trading_day(self, date_to_check: date) -> bool:
        """Check if a given date is a trading day.
        
        Args:
            date_to_check: Date to check
            
        Returns:
            True if the date is a trading day, False otherwise
        """
        return date_to_check in self.trading_days

    def closest_trading_day_at_or_before(self, day_x: date) -> Optional[date]:
        """Find the closest trading day at or before a given date.
        
        Uses binary search for efficient lookup.
        
        Args:
            day_x: Reference date
            
        Returns:
            The closest trading day at or before day_x, or None if no such day exists
        """
        index = bisect.bisect_right(self.trading_days, day_x) - 1
        if index < 0:
            return None
        return self.trading_days[index]
    
    def closest_trading_day_at_or_after(self, day_x: date) -> Optional[date]:
        """Find the closest trading day at or after a given date.
        
        Uses binary search for efficient lookup.
        
        Args:
            day_x: Reference date
            
        Returns:
            The closest trading day at or after day_x, or None if no such day exists
        """
        index = bisect.bisect_left(self.trading_days, day_x)
        if index == len(self.trading_days):
            return None
        return self.trading_days[index]

    def next_trading_day(self, date_ref: date) -> Optional[date]:
        """Get the next trading day after a given date.
        
        Args:
            date_ref: Reference date
            
        Returns:
            The next trading day after date_ref, or None if no such day exists
        """
        date_corrected = self.closest_trading_day_at_or_before(date_ref)
        if date_corrected is None:
            return None
        index = self.trading_days.index(date_corrected) + 1
        if index >= len(self.trading_days):
            return None
        return self.trading_days[index]

    def impute_date_affect(self, publ_datetime: datetime) -> Optional[date]:
        """Impute the effective date when an article affects the market.
        
        The imputation follows the logic:
        $$\\tilde{d}_0^i:=\\left\\{\\begin{array}{lll}d_0^i & \\text{if} & d_0^i \\in \\tilde{\\mathfrak{d}} \\wedge t_0^i<17:30:00.000 \\\\ \\Lambda\\left(d_0^i\\right) & \\text{if} & d_0^i \\notin \\tilde{\\mathfrak{d}} \\vee t_0^i>17:30:00.000\\end{array}\\right.$$
        
        where $\\Lambda(d):=\\min \\{\\tilde{d} \\in \\tilde{\\mathfrak{d}} \\mid \\tilde{d} \\geq d\\}$
        
        Args:
            publ_datetime: Publication datetime of the article
            
        Returns:
            The effective date when the article affects the market, or None if no valid
            trading day is available
        """
        publ_date = publ_datetime.date()
        publ_time = publ_datetime.time()
        
        if self.is_trading_day(publ_date) and publ_time < self.closing_time:
            return publ_date
        else:
            return self.next_trading_day(publ_date)
