from reproduce.aflow.official.runtime import OfficialWorkflowBase


class Workflow(OfficialWorkflowBase):
    async def __call__(self, problem: str):
        plan = await self.custom(input=problem, instruction=self.prompt_custom.BROWSECOMP_PLAN_PROMPT)
        solutions = []
        for _ in range(3):
            answer = await self.answer_generate(
                input=self.prompt_custom.BROWSECOMP_ANSWER_PROMPT.format(
                    question=problem,
                    plan=plan["response"],
                )
            )
            solutions.append(answer["answer"])
        selected = await self.sc_ensemble(solutions=solutions, problem=problem)
        final = await self.format(problem=problem, solution=selected["response"])
        return final["solution"], self.llm.get_usage_summary()["total_cost"]
