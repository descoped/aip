from aip import PromptBuilder
from aip.processors import (
    create_chain_of_thought_processor,
    create_few_shot_processor,
    create_structured_output_processor,
    create_variable_processor
)


def test_prompt_processor():
    # Create a prompt with better IDE navigation
    prompt = (
        PromptBuilder()
        .configure_variables(model="GPT-4", task="summarization")
        .step_by_step(detailed=True)
        .prompt("Create a summary using {model} for {task} task.")
    )

    # Print the final prompt
    print("Final prompt:")
    print(str(prompt))


def test_advanced_test():
    # Example 1: Variable substitution
    prompt = (
        PromptBuilder()
        .add_processor(create_variable_processor({
            "model": "GPT-4",
            "task": "summarization",
            "length": "short"
        }))
        .prompt("Create a {length} summary using {model} for {task} task.")
    )
    print(str(prompt))

    # Example 2: JSON data inclusion
    data = {
        "user": {"name": "Alice", "preferences": ["science", "history"]},
        "settings": {"language": "en", "format": "paragraph"}
    }
    prompt = (
        PromptBuilder()
        .prompt("Generate content based on user preferences")
        .with_json(data)
    )
    print(f"\n---\n{str(prompt)}\n---\n")

    # Example 3: Combining multiple processors
    examples = [
        ("Summarize: The quick brown fox jumps over the lazy dog.",
         "A fox quickly jumps over a dog that is resting."),
        ("Summarize: The study showed significant results with p<0.05.",
         "The research findings were statistically significant.")
    ]
    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "length": {"type": "integer"},
            "topics": {"type": "array", "items": {"type": "string"}}
        }
    }
    prompt = (
        PromptBuilder()
        .prompt("Summarize the following text")
        .add_processor(create_chain_of_thought_processor())
        .add_processor(create_few_shot_processor(examples))
        .add_processor(create_structured_output_processor(schema))
    )
    print(str(prompt))


def test_lambda_processor():
    # Example of using the new process method with a lambda
    prompt = (
        PromptBuilder()
        .prompt("Standard prompt text")
        .process(lambda content, _: f"Custom prefix: {content}")
        .process(lambda content, _: f"{content}\nCustom suffix")
    )
    print()
    print(str(prompt))
