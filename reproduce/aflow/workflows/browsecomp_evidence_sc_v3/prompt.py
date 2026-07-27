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

BROWSECOMP_ANSWER_PROMPT = """
You are solving one BrowseComp question with retrieval tools.

Question:
{question}

Retrieval plan:
{plan}

Procedure:
1. Issue 2 to 3 targeted searches using rare terms from multiple clues at once.
2. Open the most promising document when a snippet contains a plausible candidate.
3. Track whether the candidate satisfies EACH mandatory clue.
4. Stop searching once you have a plausible candidate and answer.

Return exactly this format:
Evidence:
- clue: evidence or missing

Candidate: exact candidate answer
Final answer: exact short answer only

Do not answer with "insufficient evidence" or refusal text. If uncertain, choose the best
candidate supported by the retrieved evidence.
"""

BROWSECOMP_VERIFY_PROMPT = """
You are the evidence verifier for a BrowseComp question.

Question:
{question}

Retrieval plan and mandatory clues:
{plan}

Candidate solutions:
{solutions}

Use search/get_document only if needed to verify a candidate against the mandatory clues.
Pick the candidate that best satisfies ALL clues. Penalize candidates whose evidence only
matches one clue or comes from generic search noise. If none is perfect, choose the most
evidence-supported candidate.

Return exactly:
Verification:
- candidate: supported/missing/conflict notes
Final answer: exact short answer only
"""
