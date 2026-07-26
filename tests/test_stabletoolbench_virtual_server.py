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
