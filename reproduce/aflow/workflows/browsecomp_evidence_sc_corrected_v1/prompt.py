BROWSECOMP_PLAN_PROMPT = """
Break this BrowseComp question into a compact retrieval plan.

Return exactly these sections:
Clues:
- 3 to 6 atomic clues that the final answer must satisfy.

Search queries:
- 3 to 5 targeted queries. Each query should combine rare clue terms, names, dates, titles, places,
  or quoted phrases. Avoid generic web-search wording.

Candidate rules:
- State what kind of entity/name/title/number is being requested.
- State which clues are mandatory for accepting a candidate.

Do not answer the question in this planning step.
"""
