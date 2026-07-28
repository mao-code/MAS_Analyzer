from __future__ import annotations

import json

from scripts.stabletoolbench_virtual_server import StableToolBenchCacheServer


def test_cache_lookup_falls_back_to_more_specific_subset_key(tmp_path):
    cache_file = (
        tmp_path
        / "Location"
        / "egypt_api_for_Location"
        / "facilities_lookup.json"
    )
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text(
        json.dumps(
            {
                "{'gov': 'Cairo', 'type': 'college', 'limit': '5'}": {
                    "error": "",
                    "response": {"college": [{"address": "Cairo"}]},
                }
            }
        ),
        encoding="utf-8",
    )

    server = StableToolBenchCacheServer(cache_root=tmp_path)
    response = server.lookup(
        category="Location",
        tool_name="Egypt API",
        api_name="Facilities Lookup",
        tool_input={
            "gov": "Cairo",
            "type": "college",
            "city": "Cairo",
            "limit": "5",
        },
    )

    assert response["error"] == ""
    assert response["response"] == {"college": [{"address": "Cairo"}]}


def test_cache_lookup_matches_parameter_names_case_insensitively(tmp_path):
    cache_file = (
        tmp_path
        / "SMS"
        / "virtual_number_for_SMS"
        / "get_number_by_country_id.json"
    )
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text(
        json.dumps(
            {
                "{'countryid': '380'}": {
                    "error": "",
                    "response": [{"number": "380123456789"}],
                }
            }
        ),
        encoding="utf-8",
    )

    server = StableToolBenchCacheServer(cache_root=tmp_path)
    response = server.lookup(
        category="SMS",
        tool_name="Virtual Number",
        api_name="Get Number By Country Id",
        tool_input={"countryId": "380"},
    )

    assert response["error"] == ""
    assert response["response"] == [{"number": "380123456789"}]


def test_cache_lookup_matches_toolbench_reserved_parameter_aliases(tmp_path):
    cache_file = (
        tmp_path
        / "Sports"
        / "rugbyapi2_for_Sports"
        / "categorytournaments.json"
    )
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text(
        json.dumps(
            {
                "{'is_id': 82}": {
                    "error": "",
                    "response": {"groups": [{"uniqueTournaments": []}]},
                }
            }
        ),
        encoding="utf-8",
    )

    server = StableToolBenchCacheServer(cache_root=tmp_path)
    response = server.lookup(
        category="Sports",
        tool_name="RugbyAPI2",
        api_name="CategoryTournaments",
        tool_input={"id": 82},
    )

    assert response["error"] == ""
    assert response["response"] == {"groups": [{"uniqueTournaments": []}]}
