from reproduce.aflow.official.runtime import OfficialWorkflowBase


class Workflow(OfficialWorkflowBase):
    async def __call__(self, problem: str):
        plan = await self.custom(input=problem, instruction=self.prompt_custom.BROWSECOMP_PLAN_PROMPT)
        solutions = []
        for _ in range(4):
            answer = await self.answer_generate(
                input=self.prompt_custom.BROWSECOMP_ANSWER_PROMPT.format(
                    question=problem,
                    plan=plan["response"],
                )
            )
            solutions.append(answer["answer"])
        solution_text = "\n\n".join(
            f"Candidate {index + 1}:\n{solution}" for index, solution in enumerate(solutions)
        )
        verified = await self.custom(
            input=self.prompt_custom.BROWSECOMP_VERIFY_PROMPT.format(
                question=problem,
                plan=plan["response"],
                solutions=solution_text,
            ),
            instruction="",
        )
        final = await self.format(problem=problem, solution=verified["response"])
        return final["solution"], self.llm.get_usage_summary()["total_cost"]
