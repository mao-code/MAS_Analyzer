from reproduce.aflow.official.runtime import OfficialWorkflowBase


class Workflow(OfficialWorkflowBase):
    async def __call__(self, problem: str):
        plan = await self.custom(input=problem, instruction=self.prompt_custom.BROWSECOMP_PLAN_PROMPT)
        answer = await self.answer_generate(
            input=(
                "Question:\n"
                f"{problem}\n\n"
                "Search plan and clues:\n"
                f"{plan['response']}\n\n"
                "Use the plan to search, compare candidates against the clues, and return the "
                "single best final answer."
            )
        )
        final = await self.format(problem=problem, solution=answer["answer"])
        return final["solution"], self.llm.get_usage_summary()["total_cost"]
