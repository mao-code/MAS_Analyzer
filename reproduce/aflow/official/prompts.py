from __future__ import annotations

WORKFLOW_OPTIMIZE_PROMPT = """You are building a Graph and corresponding Prompt to jointly solve {type} problems.
Referring to the given graph and prompt, which forms a basic example of a {type} solution approach,
please reconstruct and optimize them. You can add, modify, or delete nodes, parameters, or prompts.
Include your single modification in XML tags in your reply. Ensure they are complete and correct to
avoid runtime failures. When optimizing, you can incorporate critical thinking methods like review,
revise, ensemble, selfAsk, loops, and conditional statements. The graph complexity should not exceed
10. Ensure that all prompts required by the current graph from prompt_custom are included.
Only generate prompts used by prompt_custom; remove unused prompts.
The generated prompt must not contain placeholders."""

BROWSECOMP_WORKFLOW_GUIDANCE = """

Benchmark-specific guidance for BrowseComp:
- This is a retrieval-heavy question answering task. Strong workflows should gather evidence before
  committing to an answer.
- Prefer graph changes that make the workflow search/read evidence, compare candidate entities
  against every clue, then format a short final answer.
- Penalize workflows that return refusal text, "insufficient evidence", "no-answer", empty answers,
  long explanations, or answers based only on general memory.
- The final output should be only the exact requested entity/name/title/number.
"""

WORKFLOW_INPUT = """
Here is a graph and the corresponding prompt that performed well in a previous iteration
(maximum score is 1). You must make one focused optimization based on this graph.

<sample>
    <experience>{experience}</experience>
    <modification>(such as:add /delete /modify/ ...)</modification>
    <score>{score}</score>
    <graph>{graph}</graph>
    <prompt>{prompt}</prompt>
    <operator_description>{operator_description}</operator_description>
</sample>

Below are logs from the selected graph. Use them as references for optimization:
{log}

First, provide optimization ideas. Only one detail point can be modified at a time, and no more than
5 lines of code may be changed per modification. The graph must output a non-empty final answer.
The graph should be Python code for a class Workflow that subclasses OfficialWorkflowBase. Do not
import prompt_custom or operator_custom yourself; they are provided as self.prompt_custom and
self.operator_custom by OfficialWorkflowBase.
"""

WORKFLOW_CUSTOM_USE = """
Example usage:
```
analysis = await self.custom(input=problem, instruction=self.prompt_custom.ANALYZE_PROMPT)
solution = await self.answer_generate(input=f"Task:\\n{problem}\\nAnalysis:\\n{analysis['response']}")
```
Operators available on self:
- custom(input: str, instruction: str) -> {'response': str}
- answer_generate(input: str) -> {'thought': str, 'answer': str}
- sc_ensemble(solutions: list[str], problem: str) -> {'response': str}
- review(problem: str, solution: str) -> {'thought': str, 'review_result': str, 'feedback': str}
- revise(problem: str, solution: str, feedback: str) -> {'thought': str, 'solution': str}
- format(problem: str, solution: str) -> {'solution': str}

Return `(final_answer, total_cost)` from `__call__`, matching the official AFlow workflow contract.
"""

XML_RESPONSE_INSTRUCTION = """
Return exactly these XML fields:
<modification>short description of the single change</modification>
<graph>complete Python class Workflow code</graph>
<prompt>complete Python prompt module code, or comments if no custom prompt is needed</prompt>
"""

ANSWER_GENERATION_PROMPT = """
Solve the problem using the available tools when external evidence is needed.
1. In the "thought" field, write at most 2 short sentences. Do not write a long chain of thought,
   numbered derivation, transcript, or repeated self-checks.
2. In the "answer" field, provide only the final answer, concisely and directly. For name/entity
   questions, return only the exact name/entity. If tool evidence is partial or noisy, still return
   the most plausible candidate answer. Do not output refusal text or an insufficient-evidence
   placeholder.
3. For BrowseComp-style questions, search with multiple targeted queries, compare candidates against
   all clues, and prefer evidence from retrieved snippets/documents over memory.
Your task: {input}
"""

FORMAT_PROMPT = """
For the question described as {problem_description}, extract a short final answer from this solution:
{solution}
Return only the answer.
"""

SC_ENSEMBLE_PROMPT = """
Given the question:
{question}

Several solutions have been generated:
{solutions}

Identify the most reliable solution. In "thought", explain briefly. In "solution_letter", output only
the single letter ID.
"""

REVIEW_PROMPT = """
Given a problem and solution, review whether the solution answers the task correctly.
Keep the response short. Do not write a long chain of thought, transcript, numbered derivation, or
repeated self-checks.

problem: {problem}
solution: {solution}

If you are more than 95 percent confident the final answer is incorrect, put False in
"review_result" and give at most 2 short sentences in "feedback". Otherwise put True in
"review_result" and give at most 1 short sentence in "feedback".
"""

REVISE_PROMPT = """
Given a problem, a solution, and feedback, revise the solution to better solve the task.
Keep the response short. Do not include long reasoning or analysis. In "solution", output only the
revised final answer, not an explanation. If evidence is partial or noisy, still output the most
plausible candidate answer. Do not output refusal text or an insufficient-evidence placeholder.

problem: {problem}
solution: {solution}
feedback: {feedback}
"""
