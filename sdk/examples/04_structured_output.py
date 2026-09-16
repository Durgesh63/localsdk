"""Example 4 -- structured output validated into a pydantic model.

`structured()` sends the model a JSON schema and validates what comes back, so
you get a typed object instead of a string you have to trust.

Run:
    python examples/04_structured_output.py
"""

from pydantic import BaseModel, Field

from localsdk import Client, StructuredOutputError, LocalSDKError


class Ingredient(BaseModel):
    """One line of a recipe."""

    name: str
    quantity: str


class Recipe(BaseModel):
    """A short recipe."""

    title: str
    minutes: int = Field(description="Total time in minutes")
    ingredients: list[Ingredient]
    steps: list[str]


PROMPT = [
    {"role": "system", "content": "You reply only with JSON that fits the given schema."},
    {"role": "user", "content": "A 3-step recipe for masala chai."},
]


def main() -> None:
    with Client() as client:
        try:
            recipe = client.structured(PROMPT, Recipe, temperature=0)
        except StructuredOutputError as exc:
            # Small models sometimes narrate instead of answering. The raw
            # text is kept so you can see what it actually said.
            print("the model did not return usable JSON:")
            print(exc.text)
            return

        print("{0} ({1} min)".format(recipe.title, recipe.minutes))
        for item in recipe.ingredients:
            print("  - {0}: {1}".format(item.quantity, item.name))
        for index, step in enumerate(recipe.steps, start=1):
            print("  {0}. {1}".format(index, step))

        # Embeddings, for completeness: one vector per input string.
        vectors = client.embed(["masala chai", "green tea"])
        print("\nembeddings: {0} vectors, {1} dimensions".format(
            len(vectors), len(vectors[0]) if vectors else 0
        ))


if __name__ == "__main__":
    try:
        main()
    except LocalSDKError as exc:
        raise SystemExit("error: {0}".format(exc))
