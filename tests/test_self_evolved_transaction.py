from MAS.self_evolved.transaction import augment_with_transaction_tools


def test_calendar_gap_verifier_accounts_for_duration_and_all_rows() -> None:
    events = [
        {"event_id": "1", "event_start": "2023-12-01 09:00:00", "duration": "60"},
        {"event_id": "2", "event_start": "2023-12-01 10:00:00", "duration": "120"},
        {"event_id": "3", "event_start": "2023-12-01 12:00:00", "duration": "60"},
        {"event_id": "4", "event_start": "2023-12-01 13:30:00", "duration": "30"},
    ]
    tools = [
        {
            "name": "calendar.search_events",
            "handler": lambda arguments: events,
        }
    ]

    augmented = augment_with_transaction_tools(tools)
    verifier = next(
        tool for tool in augmented if tool["name"] == "calendar.find_first_available_slot"
    )
    result = verifier["handler"](
        {
            "time_min": "2023-12-01 09:00:00",
            "time_max": "2023-12-01 18:00:00",
            "duration": "30",
        }
    )

    assert result["available"] is True
    assert result["event_start"] == "2023-12-01 13:00:00"
    assert [item["event_id"] for item in result["considered_intervals"]] == ["1", "2", "3", "4"]


def test_calendar_gap_verifier_is_not_added_without_search_capability() -> None:
    assert augment_with_transaction_tools([{"name": "calendar.create_event"}]) == [
        {"name": "calendar.create_event"}
    ]


def test_parse_duration_minutes_accepts_model_spellings() -> None:
    from MAS.self_evolved.transaction import parse_duration_minutes

    assert parse_duration_minutes("30") == 30
    assert parse_duration_minutes(30) == 30
    assert parse_duration_minutes("30m") == 30
    assert parse_duration_minutes("30 minutes") == 30
    assert parse_duration_minutes("2h") == 120
    assert parse_duration_minutes("2 hours") == 120
    assert parse_duration_minutes("1.5 hours") == 90
    assert parse_duration_minutes("1:30") == 90
    assert parse_duration_minutes("01:30:00") == 90
    assert parse_duration_minutes("00:30:00") == 30
    assert parse_duration_minutes("PT30M") == 30
    assert parse_duration_minutes("PT1H30M") == 90
    assert parse_duration_minutes(None) == 30
    assert parse_duration_minutes("") == 30


def test_parse_duration_minutes_rejects_gibberish() -> None:
    import pytest

    from MAS.self_evolved.transaction import parse_duration_minutes

    with pytest.raises(ValueError, match="unrecognized duration"):
        parse_duration_minutes("soonish")


def test_calendar_gap_verifier_accepts_unit_suffixed_duration() -> None:
    events = [
        {"event_id": "1", "event_start": "2023-12-01 09:00:00", "duration": "60"},
    ]
    tools = [{"name": "calendar.search_events", "handler": lambda arguments: events}]

    verifier = next(
        tool
        for tool in augment_with_transaction_tools(tools)
        if tool["name"] == "calendar.find_first_available_slot"
    )
    result = verifier["handler"](
        {
            "time_min": "2023-12-01 09:00:00",
            "time_max": "2023-12-01 18:00:00",
            "duration": "30m",
        }
    )

    assert result["available"] is True
    assert result["event_start"] == "2023-12-01 10:00:00"
