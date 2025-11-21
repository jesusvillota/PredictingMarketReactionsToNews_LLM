"""Tests for trading calendar module."""

from datetime import datetime, date, time

import pytest

from PMRTN.analysis import TradingCalendarAdjustments


@pytest.fixture
def sample_trading_days():
    """Create a sample list of trading days for testing."""
    # Create some trading days (Monday to Friday for two weeks)
    trading_days = [
        date(2024, 1, 1),   # Monday
        date(2024, 1, 2),   # Tuesday
        date(2024, 1, 3),   # Wednesday
        date(2024, 1, 4),   # Thursday
        date(2024, 1, 5),   # Friday
        # Weekend (6-7) skipped
        date(2024, 1, 8),   # Monday
        date(2024, 1, 9),   # Tuesday
        date(2024, 1, 10),  # Wednesday
        date(2024, 1, 11),  # Thursday
        date(2024, 1, 12),  # Friday
    ]
    return trading_days


@pytest.fixture
def trading_calendar(sample_trading_days):
    """Create a TradingCalendarAdjustments instance for testing."""
    return TradingCalendarAdjustments(sample_trading_days)


def test_init_default_closing_time(sample_trading_days):
    """Test initialization with default closing time."""
    adj = TradingCalendarAdjustments(sample_trading_days)
    
    assert adj.trading_days == sorted(sample_trading_days)
    assert adj.closing_time == time(17, 30, 0)


def test_init_custom_closing_time(sample_trading_days):
    """Test initialization with custom closing time."""
    adj = TradingCalendarAdjustments(sample_trading_days, closing_time='16:00:00')
    
    assert adj.closing_time == time(16, 0, 0)


def test_init_invalid_closing_time(sample_trading_days):
    """Test initialization with invalid closing time format."""
    with pytest.raises(ValueError, match="Invalid closing_time format"):
        TradingCalendarAdjustments(sample_trading_days, closing_time='invalid')


def test_init_sorts_trading_days():
    """Test that trading days are sorted during initialization."""
    unsorted_days = [
        date(2024, 1, 5),
        date(2024, 1, 2),
        date(2024, 1, 8),
        date(2024, 1, 1),
    ]
    adj = TradingCalendarAdjustments(unsorted_days)
    
    expected_sorted = [
        date(2024, 1, 1),
        date(2024, 1, 2),
        date(2024, 1, 5),
        date(2024, 1, 8),
    ]
    assert adj.trading_days == expected_sorted


def test_is_trading_day_true(trading_calendar):
    """Test is_trading_day returns True for trading days."""
    assert trading_calendar.is_trading_day(date(2024, 1, 1)) is True
    assert trading_calendar.is_trading_day(date(2024, 1, 5)) is True


def test_is_trading_day_false(trading_calendar):
    """Test is_trading_day returns False for non-trading days."""
    # Weekend day
    assert trading_calendar.is_trading_day(date(2024, 1, 6)) is False
    assert trading_calendar.is_trading_day(date(2024, 1, 7)) is False
    # Day outside range
    assert trading_calendar.is_trading_day(date(2024, 1, 15)) is False


def test_closest_trading_day_at_or_before_exact_match(trading_calendar):
    """Test finding closest trading day when exact match exists."""
    result = trading_calendar.closest_trading_day_at_or_before(date(2024, 1, 3))
    assert result == date(2024, 1, 3)


def test_closest_trading_day_at_or_before_non_trading_day(trading_calendar):
    """Test finding closest trading day before a non-trading day."""
    # Weekend day - should return Friday
    result = trading_calendar.closest_trading_day_at_or_before(date(2024, 1, 6))
    assert result == date(2024, 1, 5)


def test_closest_trading_day_at_or_before_before_all_days(trading_calendar):
    """Test finding closest trading day when date is before all trading days."""
    result = trading_calendar.closest_trading_day_at_or_before(date(2023, 12, 31))
    assert result is None


def test_closest_trading_day_at_or_after_exact_match(trading_calendar):
    """Test finding closest trading day at or after when exact match exists."""
    result = trading_calendar.closest_trading_day_at_or_after(date(2024, 1, 3))
    assert result == date(2024, 1, 3)


def test_closest_trading_day_at_or_after_non_trading_day(trading_calendar):
    """Test finding closest trading day after a non-trading day."""
    # Weekend day - should return Monday
    result = trading_calendar.closest_trading_day_at_or_after(date(2024, 1, 6))
    assert result == date(2024, 1, 8)


def test_closest_trading_day_at_or_after_after_all_days(trading_calendar):
    """Test finding closest trading day when date is after all trading days."""
    result = trading_calendar.closest_trading_day_at_or_after(date(2024, 1, 15))
    assert result is None


def test_next_trading_day_normal(trading_calendar):
    """Test getting next trading day in normal case."""
    result = trading_calendar.next_trading_day(date(2024, 1, 1))
    assert result == date(2024, 1, 2)


def test_next_trading_day_before_weekend(trading_calendar):
    """Test getting next trading day when current day is before weekend."""
    result = trading_calendar.next_trading_day(date(2024, 1, 5))
    assert result == date(2024, 1, 8)


def test_next_trading_day_from_weekend(trading_calendar):
    """Test getting next trading day from a weekend day."""
    # Saturday
    result = trading_calendar.next_trading_day(date(2024, 1, 6))
    assert result == date(2024, 1, 8)


def test_next_trading_day_last_day(trading_calendar):
    """Test getting next trading day when current day is the last trading day."""
    result = trading_calendar.next_trading_day(date(2024, 1, 12))
    assert result is None


def test_next_trading_day_before_all_days(trading_calendar):
    """Test getting next trading day when date is before all trading days."""
    result = trading_calendar.next_trading_day(date(2023, 12, 31))
    assert result is None


def test_impute_date_affect_trading_day_before_closing(trading_calendar):
    """Test impute_date_affect when article published on trading day before closing."""
    # Article published at 10:00 AM on a trading day
    publ_datetime = datetime(2024, 1, 2, 10, 0, 0)
    result = trading_calendar.impute_date_affect(publ_datetime)
    
    # Should return the same day
    assert result == date(2024, 1, 2)


def test_impute_date_affect_trading_day_after_closing(trading_calendar):
    """Test impute_date_affect when article published on trading day after closing."""
    # Article published at 18:00 (after 17:30 closing) on a trading day
    publ_datetime = datetime(2024, 1, 2, 18, 0, 0)
    result = trading_calendar.impute_date_affect(publ_datetime)
    
    # Should return next trading day
    assert result == date(2024, 1, 3)


def test_impute_date_affect_weekend_day(trading_calendar):
    """Test impute_date_affect when article published on weekend."""
    # Article published on Saturday
    publ_datetime = datetime(2024, 1, 6, 10, 0, 0)
    result = trading_calendar.impute_date_affect(publ_datetime)
    
    # Should return next Monday
    assert result == date(2024, 1, 8)


def test_impute_date_affect_at_closing_time(trading_calendar):
    """Test impute_date_affect when article published exactly at closing time."""
    # Article published exactly at 17:30:00
    publ_datetime = datetime(2024, 1, 2, 17, 30, 0)
    result = trading_calendar.impute_date_affect(publ_datetime)
    
    # At closing time is not before closing time, so should be next day
    assert result == date(2024, 1, 3)


def test_impute_date_affect_one_second_before_closing(trading_calendar):
    """Test impute_date_affect when article published one second before closing."""
    # Article published at 17:29:59
    publ_datetime = datetime(2024, 1, 2, 17, 29, 59)
    result = trading_calendar.impute_date_affect(publ_datetime)
    
    # Should return the same day
    assert result == date(2024, 1, 2)


def test_impute_date_affect_friday_after_closing(trading_calendar):
    """Test impute_date_affect when article published on Friday after closing."""
    # Article published on Friday after closing
    publ_datetime = datetime(2024, 1, 5, 18, 0, 0)
    result = trading_calendar.impute_date_affect(publ_datetime)
    
    # Should return next Monday
    assert result == date(2024, 1, 8)


def test_impute_date_affect_custom_closing_time(sample_trading_days):
    """Test impute_date_affect with custom closing time."""
    # Create calendar with 16:00 closing time
    adj = TradingCalendarAdjustments(sample_trading_days, closing_time='16:00:00')
    
    # Article published at 16:30 (after custom closing)
    publ_datetime = datetime(2024, 1, 2, 16, 30, 0)
    result = adj.impute_date_affect(publ_datetime)
    
    # Should return next trading day
    assert result == date(2024, 1, 3)


def test_impute_date_affect_after_last_trading_day(trading_calendar):
    """Test impute_date_affect when article published after all trading days."""
    # Article published after the last trading day
    publ_datetime = datetime(2024, 1, 15, 10, 0, 0)
    result = trading_calendar.impute_date_affect(publ_datetime)
    
    # Should return None (no valid trading day)
    assert result is None
