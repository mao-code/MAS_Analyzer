import unittest

from MAS.llm import OpenRouterLLMClient


class TestLLMToolNames(unittest.TestCase):
    def test_normalize_tools_sanitizes_names_for_openai(self) -> None:
        def noop(args):
            return args

        defs, handlers, original_names = OpenRouterLLMClient._normalize_tools(
            [
                {
                    "name": "calendar.delete_event",
                    "description": "Delete an event.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                    "handler": noop,
                },
                {
                    "name": "customer_relationship_manager.update_customer",
                    "description": "Update a customer.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                    "handler": noop,
                },
            ]
        )

        tool_names = [tool["function"]["name"] for tool in defs]
        self.assertEqual(
            tool_names,
            ["calendar_delete_event", "customer_relationship_manager_update_customer"],
        )
        self.assertEqual(set(handlers.keys()), set(tool_names))
        self.assertEqual(
            original_names,
            {
                "calendar_delete_event": "calendar.delete_event",
                "customer_relationship_manager_update_customer": (
                    "customer_relationship_manager.update_customer"
                ),
            },
        )


if __name__ == "__main__":
    unittest.main()
