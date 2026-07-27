from reproduce.aflow.official.runtime import OfficialWorkflowBase


class Workflow(OfficialWorkflowBase):
    async def __call__(self, problem: str):
        plan = await self.custom(input=problem, instruction=self.prompt_custom.BROWSECOMP_PLAN_PROMPT)
        prompt = (
            "Question:\n"
            f"{problem}\n\n"
            "Search plan and clues:\n"
            f"{plan['response']}\n\n"
            "Use the plan to search, compare candidates against the clues, and return the "
            "single best final answer."
        )
        solutions = []
        for _ in range(3):
            answer = await self.answer_generate(input=prompt)
            solutions.append(answer["answer"])
        selected = await self.sc_ensemble(solutions=solutions, problem=problem)
        final = await self.format(problem=problem, solution=selected["response"])
        return final["solution"], self.llm.get_usage_summary()["total_cost"]
