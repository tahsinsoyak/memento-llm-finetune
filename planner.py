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
        prompt = f"""You are an expert task planner for an AI agent with search and document processing tools.

Task: {task}
"""
        if cases:
            prompt += "\nRelevant past cases:\n"
            for c in cases:
                prompt += f"- {c}\n"
        
        prompt += """
Create a concise, actionable plan with 2-4 steps that uses the available tools:

Available Tools:
- search: Search the web for current information
- search_news: Search for recent news articles
- document: Process documents or web pages

Plan Requirements:
1. Each step must be specific and tool-focused
2. Use "search for [specific query]" for web searches
3. Use "search news for [specific query]" for recent news
4. Keep steps minimal and focused on the main question
5. Don't create generic "analysis" steps - let the search results speak for themselves

Format as a simple numbered list (2-4 steps maximum).

Example for "What is the latest iPhone price?":
1. Search for "iPhone 15 price official Apple store 2024"
2. Search news for "iPhone price announcement latest"
3. Compare prices from official sources

Now create your plan:"""
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
