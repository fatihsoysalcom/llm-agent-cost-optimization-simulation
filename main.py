import time

# Simulate a simple LLM call.
# In a real scenario, this would be an API call to OpenAI, Anthropic, etc.
def simulate_llm_call(prompt_text, task_description="process"):
    """
    Simulates an LLM API call, returning a mock response and a token cost.
    The cost is based on the length of the prompt and a mock response.
    """
    # Simulate network latency or processing time
    time.sleep(0.05)

    # Simple mock response generation
    if "summarize" in task_description.lower():
        response_text = f"Summary of '{prompt_text[:50]}...': Key points on {task_description}."
    elif "combine" in task_description.lower():
        response_text = f"Combined report based on '{prompt_text[:50]}...': All essential information integrated."
    else:
        response_text = f"Processed '{prompt_text[:50]}...': Result of {task_description}."

    # Simulate token cost: A simple heuristic (e.g., 1 token per 4 characters)
    # In reality, tokenization is more complex and includes both input and output tokens.
    input_tokens = len(prompt_text) // 4
    output_tokens = len(response_text) // 4
    total_tokens = input_tokens + output_tokens

    print(f"  [SIMULATED LLM CALL] Task: {task_description}, Prompt length: {len(prompt_text)} chars, Tokens: {total_tokens}")
    return response_text, total_tokens

class AgentWorkflow:
    def __init__(self, name):
        self.name = name
        self.total_cost = 0
        self.llm_calls = 0

    def _make_llm_call(self, prompt, task):
        response, cost = simulate_llm_call(prompt, task)
        self.total_cost += cost
        self.llm_calls += 1
        return response

    def run_unoptimized_workflow(self, documents):
        print(f"\n--- Running {self.name}: Unoptimized Workflow ---")
        individual_summaries = []
        # Unoptimized strategy: Make an LLM call for each document to summarize it.
        # This increases the total number of API calls.
        for i, doc in enumerate(documents):
            print(f"  > Calling LLM to summarize Document {i+1}")
            summary = self._make_llm_call(doc, f"summarize document {i+1}") # First set of LLM calls
            individual_summaries.append(summary)

        # Unoptimized strategy: Make another LLM call to combine the individual summaries.
        # This adds yet another API call.
        combined_prompt = "Combine the following summaries into a concise report:\n" + "\n".join(individual_summaries)
        print("  > Calling LLM to combine individual summaries")
        final_report = self._make_llm_call(combined_prompt, "combine summaries") # Second LLM call for final aggregation

        print(f"--- Unoptimized Workflow Finished ---")
        print(f"  Total LLM Calls: {self.llm_calls}")
        print(f"  Total Simulated Tokens: {self.total_cost}")
        return final_report

    def run_optimized_workflow(self, documents):
        print(f"\n--- Running {self.name}: Optimized Workflow ---")
        # Optimized strategy: Combine all documents into a single prompt.
        # This reduces the number of LLM calls to just one for the entire task.
        combined_documents_prompt = "Please summarize the following documents into a single, concise report. " \
                                    "Ensure all key points from each document are covered:\n\n"
        for i, doc in enumerate(documents):
            combined_documents_prompt += f"--- Document {i+1} ---\n{doc}\n\n"

        print("  > Calling LLM to summarize all documents in one go")
        final_report = self._make_llm_call(combined_documents_prompt, "summarize all documents") # Single LLM call for the entire task

        print(f"--- Optimized Workflow Finished ---")
        print(f"  Total LLM Calls: {self.llm_calls}")
        print(f"  Total Simulated Tokens: {self.total_cost}")
        return final_report

# --- Main execution ---
if __name__ == "__main__":
    sample_documents = [
        "Document 1: The quarterly financial report indicates a 15% growth in revenue, primarily driven by new product launches in Q2. Operating expenses remained stable, leading to a healthy profit margin. Market share increased by 2% in key regions.",
        "Document 2: Employee satisfaction survey results show a significant improvement in work-life balance perception, largely due to the new flexible working policy. Training and development opportunities were also rated higher. Areas for improvement include internal communication.",
        "Document 3: Our latest marketing campaign, 'Innovate & Grow', exceeded expectations, reaching over 5 million unique users. Social media engagement saw a 200% increase. The campaign successfully positioned our brand as a leader in sustainable technology.",
        "Document 4: Research and development efforts are focused on AI-driven solutions for supply chain optimization. Early prototypes show promising results in reducing logistics costs by up to 10%. Patent applications for two new algorithms are underway."
    ]

    print("--- Simulating LLM Agent Cost Optimization ---")

    # Run unoptimized workflow
    unoptimized_agent = AgentWorkflow("Unoptimized Agent")
    unoptimized_agent.run_unoptimized_workflow(sample_documents)

    # Run optimized workflow
    optimized_agent = AgentWorkflow("Optimized Agent")
    optimized_agent.run_optimized_workflow(sample_documents)

    print("\n--- Comparison ---")
    print(f"Unoptimized Agent: {unoptimized_agent.llm_calls} calls, {unoptimized_agent.total_cost} simulated tokens")
    print(f"Optimized Agent:   {optimized_agent.llm_calls} calls, {optimized_agent.total_cost} simulated tokens")

    if optimized_agent.total_cost < unoptimized_agent.total_cost:
        print(f"\nOptimization successful! Saved {unoptimized_agent.total_cost - optimized_agent.total_cost} simulated tokens.")
    elif optimized_agent.total_cost == unoptimized_agent.total_cost:
        print("\nOptimization resulted in similar costs. This can happen if combining prompts makes the single prompt very long.")
    else:
        print(f"\nOptimization resulted in higher costs. This can happen if combining prompts makes the single prompt significantly longer than the sum of individual prompts plus the final combination, or due to specific model pricing tiers.")

    print("\nNote: This is a simplified simulation. Real LLM costs depend on actual tokenization, model pricing, and API usage patterns.")
