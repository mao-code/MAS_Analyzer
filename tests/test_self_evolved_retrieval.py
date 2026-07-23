from MAS.self_evolved.retrieval import _bounded_numeric_variants, augment_with_constraint_search


def test_bounded_numeric_variants_split_ranges_for_literal_search() -> None:
    assert _bounded_numeric_variants("formed 2001..2004 rock band") == [
        "formed 2001 rock band",
        "formed 2002 rock band",
        "formed 2003 rock band",
        "formed 2004 rock band",
    ]
    assert _bounded_numeric_variants("born May 1981 1982 1983 1984 composer") == [
        "born May 1981 composer",
        "born May 1982 composer",
        "born May 1983 composer",
        "born May 1984 composer",
    ]


def test_bounded_numeric_variants_keep_wide_ranges_bounded_and_searchable() -> None:
    assert _bounded_numeric_variants("attendance 61700..61906 football") == [
        "attendance football",
        "attendance 61700 football",
        "attendance 61803 football",
        "attendance 61906 football",
    ]
    assert len(_bounded_numeric_variants("died 1880..1889 glassmaker")) == 10


def test_constraint_search_fuses_distinct_clause_support() -> None:
    results_by_query = {
        "formed 2001 band": [{"docid": "noise", "score": 9, "snippet": "noise"}],
        "formed 2002 band": [{"docid": "noise", "score": 9, "snippet": "noise"}],
        "formed 2003 band": [
            {"docid": "candidate", "score": 8, "snippet": "formed in 2003"},
            {"docid": "noise", "score": 7, "snippet": "noise"},
        ],
        "formed 2004 band": [{"docid": "noise", "score": 9, "snippet": "noise"}],
        "debut album August": [
            {"docid": "candidate", "score": 5, "snippet": "August debut"},
            {"docid": "album-only", "score": 6, "snippet": "album"},
        ],
        "first single cover": [
            {"docid": "candidate", "score": 4, "snippet": "cover single"},
            {"docid": "single-only", "score": 7, "snippet": "single"},
        ],
    }

    def search(args):
        return results_by_query.get(args["query"], [])

    tools = [
        {"name": "search", "handler": search},
        {"name": "get_document", "handler": lambda args: args},
    ]
    augmented = augment_with_constraint_search(tools)
    constraint_tool = next(tool for tool in augmented if tool["name"] == "constraint_search")

    results = constraint_tool["handler"](
        {
            "queries": "["
            '"formed 2001 2002 2003 2004 band", '
            '"debut album August", '
            '"first single cover"]'
        }
    )

    candidate = next(item for item in results if item["docid"] == "candidate")
    assert candidate["constraint_support"] == 3
    assert results[0]["docid"] == "candidate"
    assert any(item["docid"] == "album-only" for item in results)
    assert len(tools) == 2


def test_constraint_search_is_only_added_to_readable_retrieval_runs() -> None:
    tools = [{"name": "search", "handler": lambda args: []}]

    assert augment_with_constraint_search(tools) == tools


def test_constraint_search_bridges_entity_title_into_another_clue() -> None:
    def search(args):
        query = args["query"]
        if query == "hospital renamed later":
            return [
                {
                    "docid": "rename",
                    "score": 10,
                    "snippet": "---\ntitle: Coney Island Hospital renamed\n---\nRenamed later.",
                }
            ]
        if "Coney Island Hospital renamed" in query and "second child same birthday" in query:
            return [
                {
                    "docid": "birth",
                    "score": 20,
                    "snippet": "A couple welcomed a second child on the same birthday.",
                }
            ]
        return []

    augmented = augment_with_constraint_search(
        [
            {"name": "search", "handler": search},
            {"name": "get_document", "handler": lambda args: args},
        ]
    )
    constraint_tool = next(tool for tool in augmented if tool["name"] == "constraint_search")

    results = constraint_tool["handler"](
        {"queries": ["hospital renamed later", "second child same birthday"]}
    )

    birth = next(item for item in results if item["docid"] == "birth")
    assert birth["constraint_support"] == 1
    assert birth["direct_constraint_support"] == 0
    assert birth["bridge_constraint_support"] == 1


def test_entity_bridges_do_not_displace_direct_anchor_recall() -> None:
    def search(args):
        query = args["query"]
        if query == "anchor founded 2009 coordinator":
            return [
                {"docid": f"anchor-{rank}", "score": 20 - rank, "snippet": "noise"}
                for rank in range(1, 5)
            ] + [{"docid": "gold", "score": 10, "snippet": "the direct anchor match"}]
        if query == "secondary clue":
            return [
                {
                    "docid": "secondary",
                    "score": 20,
                    "snippet": "---\ntitle: Candidate entity\n---\nsecondary match",
                }
            ]
        if "Candidate entity" in query and "anchor founded 2009 coordinator" in query:
            return [
                {
                    "docid": f"bridge-{rank}",
                    "score": 30 - rank,
                    "snippet": "entity bridge noise",
                }
                for rank in range(1, 8)
            ]
        return []

    augmented = augment_with_constraint_search(
        [
            {"name": "search", "handler": search},
            {"name": "get_document", "handler": lambda args: args},
        ]
    )
    constraint_tool = next(tool for tool in augmented if tool["name"] == "constraint_search")

    results = constraint_tool["handler"](
        {"queries": ["anchor founded 2009 coordinator", "secondary clue"]}
    )

    assert any(item["docid"] == "gold" for item in results)


def test_constraint_search_uses_deeper_internal_recall_without_expanding_output() -> None:
    requested_depths: list[int] = []

    def search(args):
        requested_depths.append(args["k"])
        noise = [
            {"docid": f"{args['query']}-noise-{rank}", "score": 30 - rank, "snippet": "noise"}
            for rank in range(1, 14)
        ]
        return noise + [{"docid": "cross-clause", "score": 5, "snippet": "decisive evidence"}]

    augmented = augment_with_constraint_search(
        [
            {"name": "search", "handler": search},
            {"name": "get_document", "handler": lambda args: args},
        ]
    )
    constraint_tool = next(tool for tool in augmented if tool["name"] == "constraint_search")

    results = constraint_tool["handler"]({"queries": ["rare clue alpha", "rare clue beta"]})

    assert set(requested_depths) == {20}
    assert any(item["docid"] == "cross-clause" for item in results)
    assert len(results) <= 16
