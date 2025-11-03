"""
Example usage of DeepThinkLLM with the electoral college voting method.

Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from deepconf import DeepThinkLLM

def main():
    # Initialize model
    deep_llm = DeepThinkLLM(model="deepseek-ai/deepseek-llm-7b-base")

    # Prepare prompt
    question = "What is the capital of the United States?"

    messages = [
        {"role": "user", "content": question}
    ]

    prompt = deep_llm.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Run offline mode with multiple voting
    result = deep_llm.deepthink(
        prompt=prompt,
        mode="offline",
        budget=100,
        compute_multiple_voting=True
    )

    # Print the electoral college result
    if result.voting_results and 'electoral_college' in result.voting_results:
        electoral_result = result.voting_results['electoral_college']
        if electoral_result and electoral_result.get('answer'):
            print(f"Electoral College Answer: {electoral_result['answer']}")
        else:
            print("Electoral college voting did not produce a result.")
    else:
        print("Electoral college voting results not found.")

if __name__ == "__main__":
    main()
