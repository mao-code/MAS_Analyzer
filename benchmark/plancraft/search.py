"""Faithful read-only recipe search for the PlanCraft environment.

Upstream's ``GoldSearchActionHandler`` samples one accepted ingredient for each
recipe slot. That is a valid example, but it hides alternatives and can make a
solvable task look impossible. This handler renders the same upstream ``RECIPES``
objects exhaustively; it never reads an example, reward, optimal path, or label.
"""

from __future__ import annotations

import re

from plancraft.environment.actions import ActionHandlerBase, convert_from_slot_index
from plancraft.environment.recipes import (
    RECIPES,
    ShapedRecipe,
    ShapelessRecipe,
    SmeltingRecipe,
    id_to_item,
)


def exhaustive_recipe_search(recipe_name: str) -> str:
    recipes = RECIPES.get(recipe_name)
    if not recipes:
        return f"No supported crafting or smelting recipe was found for {recipe_name}."

    lines = [
        f"Recipes to craft {recipe_name}:",
        "All accepted ingredient alternatives are shown; `|` means either item is valid.",
    ]
    for index, recipe in enumerate(recipes, start=1):
        lines.append(f"recipe {index} ({recipe.recipe_type}):")
        if isinstance(recipe, ShapedRecipe):
            for row_index, row in enumerate(recipe.kernel):
                for col_index, item_ids in enumerate(row):
                    alternatives = sorted(
                        item
                        for item in (id_to_item(item_id) for item_id in item_ids)
                        if item is not None
                    )
                    if not alternatives:
                        continue
                    slot = convert_from_slot_index(row_index * 3 + col_index + 1)
                    lines.append(f"- {'|'.join(alternatives)} at {slot}")
        elif isinstance(recipe, ShapelessRecipe):
            for variant_index, ingredients in enumerate(recipe.ingredients, start=1):
                parts = [f"{count} {item}" for item, count in sorted(ingredients.items())]
                lines.append(
                    f"- variant {variant_index}, place in any grid slots: {', '.join(parts)}"
                )
        elif isinstance(recipe, SmeltingRecipe):
            lines.append(f"- smelt {'|'.join(sorted(recipe.ingredient))}")
        lines.append(f"- produces {recipe.result.count} {recipe.result.item}")
    return "\n".join(lines)


class ExhaustiveRecipeSearchActionHandler(ActionHandlerBase):
    @property
    def prompt_description(self) -> str:
        return "Look up every accepted recipe ingredient alternative for an item"

    @property
    def prompt_format_example(self) -> str:
        return "`search: <recipe name>`"

    @property
    def regex_pattern(self) -> str:
        return r"search:\s*\w+"

    @property
    def action_name(self) -> str:
        return "search"

    def match(self, generated_text: str, **_kwargs) -> str | None:
        match = re.search(r"search:\s*(\w+)", generated_text, flags=re.IGNORECASE)
        if match is None:
            return None
        return exhaustive_recipe_search(match.group(1).lower())
