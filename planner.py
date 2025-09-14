# planner.py
import os
from openai import OpenAI

class Planner:
    """The Planner uses OpenAI to decompose tasks into subtasks, possibly using past cases."""
    def __init__(self, api_key=None):
        """
        Initialize the Planner with OpenAI client.
        :param api_key: OpenAI API key, if None will read from environment
        """
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))

    def plan(self, task_description, retrieved_cases=None):
        """
        Generate a plan (list of subtasks) for the given task.
        :param task_description: str describing the high-level task or query.
        :param retrieved_cases: list of past case summaries or info to inform planning.
        :return: plan as a list of step descriptions.
        """
        prompt = self._compose_prompt(task_description, retrieved_cases)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            plan_text = response.choices[0].message.content
            steps = self._parse_plan(plan_text)
            return steps
        except Exception as e:
            print(f"Error generating plan: {e}")
            return [f"Search for information about: {task_description}"]

    def _compose_prompt(self, task, cases):
        # Construct a prompt string for the LLM
        prompt = f"""You are an expert research planner. Break down complex tasks into clear, actionable steps.

Task: {task}
"""
        if cases:
            prompt += "\nRelevant past cases:\n"
            for c in cases:
                prompt += f"- {c}\n"
        
        prompt += """
Plan the steps to solve this task. Each step should be:
1. Specific and actionable
2. Use available tools (search, document processing)
3. Build towards answering the main question

Format your response as a numbered list of steps."""
        return prompt

    def _parse_plan(self, plan_text):
        # Convert the LLM's output into a list of step strings
        steps = []
        for line in plan_text.splitlines():
            line = line.strip()
            # Remove numbering, bullets, or other prefixes
            if line and not line.isspace():
                # Remove common prefixes like "1.", "•", "-", etc.
                import re
                cleaned = re.sub(r'^\d+\.\s*', '', line)  # Remove "1. "
                cleaned = re.sub(r'^[•\-\*]\s*', '', cleaned)  # Remove "• ", "- ", "* "
                if cleaned:
                    steps.append(cleaned)
        return steps
