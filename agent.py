# agent.py
from planner import Planner
from executor import Executor
from memory.case_memory import CaseMemory

class Agent:
    """Orchestrates the Planner, Executor, and Memory to solve tasks."""
    def __init__(self, api_key=None):
        """Initialize the agent with OpenAI integration."""
        self.planner = Planner(api_key=api_key)
        self.executor = Executor(api_key=api_key)
        self.memory = CaseMemory(api_key=api_key)

    def solve(self, task):
        """Solve a given task (question or objective) using the agent."""
        # Step 1: Memory retrieval
        retrieved_cases = self.memory.retrieve(task, top_k=3)
        # We might extract just summaries of cases for the planner
        case_summaries = [case['summary'] for case in retrieved_cases]

        # Step 2: Planning
        plan_steps = self.planner.plan(task, retrieved_cases=case_summaries)
        print("Plan:", plan_steps)

        # Step 3: Execution
        self.executor.clear_tool_memory()  # Clear previous execution context
        answer, step_logs = self.executor.run_plan(plan_steps)
        print("Answer:", answer)

        # Step 4: (Optional evaluation of answer)
        reward = self._evaluate_answer(task, answer)
        # For a user query, reward might be 1 (success) by default or determined by some criteria.

        # Step 5: Memory update
        # Convert step_logs dict to list for storage
        steps_list = [f"Step {i}: {step_data['step']} -> {step_data['output'][:100]}..." 
                     for i, step_data in step_logs.items()]
        self.memory.store_case(task, answer, reward, steps_log=steps_list)
        return answer

    def _evaluate_answer(self, task, answer):
        # Placeholder: in an interactive setting, we might not have an automatic reward.
        # Could return 1 for now, or use some heuristic or user feedback.
        # Task and answer could be used for more sophisticated evaluation
        _ = task, answer  # Acknowledge parameters to avoid warnings
        return 1.0
