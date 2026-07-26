from MAS.self_evolved.executor import TurnExecutor
from MAS.self_evolved.transaction import (
    augment_with_transaction_tools,
    calendar_scheduling_mode,
    successful_mutation_record,
)


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


def test_fixed_time_calendar_request_does_not_get_gap_verifier() -> None:
    tools = [
        {"name": "calendar.search_events", "handler": lambda arguments: []},
        {"name": "calendar.create_event", "handler": lambda arguments: "00000001"},
    ]
    prompt = [
        {"role": "system", "content": "Use the provided workplace tools."},
        {
            "role": "user",
            "content": (
                "Schedule a 30 minute meeting called 'Budget Review' at 10:30 on December 8."
            ),
        },
    ]

    augmented = augment_with_transaction_tools(tools, task_prompt=prompt)

    assert calendar_scheduling_mode(prompt) == "fixed"
    assert all(tool["name"] != "calendar.find_first_available_slot" for tool in augmented)


def test_first_free_calendar_request_keeps_gap_verifier() -> None:
    tools = [
        {"name": "calendar.search_events", "handler": lambda arguments: []},
        {"name": "calendar.create_event", "handler": lambda arguments: "00000001"},
    ]
    prompt = [
        {
            "role": "user",
            "content": "Book a 30-minute meeting at the first time I'm free tomorrow.",
        }
    ]

    augmented = augment_with_transaction_tools(tools, task_prompt=prompt)

    assert calendar_scheduling_mode(prompt) == "flexible"
    assert any(tool["name"] == "calendar.find_first_available_slot" for tool in augmented)


def test_non_calendar_task_is_unchanged() -> None:
    tools = [
        {"name": "calendar.search_events", "handler": lambda arguments: []},
        {"name": "calendar.create_event", "handler": lambda arguments: "00000001"},
    ]

    augmented = augment_with_transaction_tools(
        tools,
        task_prompt="Find the company matching these information-retrieval clues.",
    )

    assert calendar_scheduling_mode("Find the company matching these clues.") == ""
    assert augmented == tools


def test_mutation_commit_requires_successful_tool_result() -> None:
    mutation_names = {"calendar.create_event"}

    assert not successful_mutation_record(
        {
            "tool_name": "calendar.create_event",
            "arguments": {"duration": "30"},
            "status": "completed",
            "output": "Event name not provided.",
        },
        mutation_names,
    )
    assert not successful_mutation_record(
        {
            "tool_name": "calendar.create_event",
            "arguments": {},
            "status": "completed",
            "output": {"success": False, "error": "missing required arguments"},
        },
        mutation_names,
    )
    assert successful_mutation_record(
        {
            "tool_name": "calendar.create_event",
            "arguments": {
                "event_name": "Budget Review",
                "participant_email": "alex@example.com",
                "event_start": "2023-12-08 10:30:00",
                "duration": "30",
            },
            "status": "completed",
            "output": "00000042",
        },
        mutation_names,
    )


def test_transaction_directive_preserves_fixed_time_without_availability_gate() -> None:
    class CapturingStage:
        directive = ""

        def _execute_agent_stage(self, state, **kwargs):
            self.directive = kwargs["directive"]
            return {"artifacts": []}

    class Context:
        def record_evidence(self, state, artifact, *, agent_id):
            return None

    stage = CapturingStage()
    executor = TurnExecutor(stage, Context())
    state = {
        "tools": [
            {"name": "calendar.search_events"},
            {"name": "calendar.create_event"},
        ],
        "self_evolved_mutation_tool_names": ["calendar.create_event"],
        "self_evolved_committer_id": "agent_0",
        "self_evolved_calendar_scheduling_mode": "fixed",
        "round_index": 0,
    }

    executor._run_stage(
        state,
        agent_id="agent_0",
        node_name="transaction_commit",
        stage_role="aggregator",
        directive="Aggregate.",
        visible_messages=[],
    )

    assert "Do not add an availability precondition" in stage.directive
    assert "call calendar.find_first_available_slot" not in stage.directive
    assert "exact event name, participant email, start time, and duration" in stage.directive


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
