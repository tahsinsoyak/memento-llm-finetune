# executor.py
import re
import os
from typing import Dict, Any, List, Tuple
from openai import OpenAI
from tools.search_tool import SearchTool
from tools.document_tool import DocumentTool

class Executor:
    """Executes plan steps by invoking tools and using OpenAI for step-level reasoning."""
    
    def __init__(self, api_key=None):
        """
        Initialize the Executor.
        
        :param api_key: OpenAI API key, if None will read from environment
        """
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        # Initialize available tools
        self.search_tool = SearchTool(max_results=3)
        self.doc_tool = DocumentTool(max_content_length=5000)
        self.tool_memory = []  # log of tool usage for the current task

    def execute_step(self, step_instruction: str) -> str:
        """
        Execute a single plan step and return the result/output.
        
        :param step_instruction: The instruction for this step
        :return: Result of executing the step
        """
        step_lower = step_instruction.lower()
        result = None
        
        # Enhanced tool selection based on keywords and patterns
        if self._is_search_step(step_lower):
            query = self._extract_search_query(step_instruction)
            if query:
                if "news" in step_lower or "recent" in step_lower or "current" in step_lower:
                    result = self.search_tool.search_news(query)
                else:
                    result = self.search_tool.search(query)
                self.tool_memory.append(f"SEARCH[{query}] -> Found {len(result.split('Result')) - 1} results")
            else:
                result = "Error: Could not extract search query from instruction"
                
        elif self._is_document_step(step_lower):
            target = self._extract_target(step_instruction)
            if target:
                if target.startswith(('http://', 'https://')):
                    content = self.doc_tool.fetch_content(target)
                    # Summarize if content is very long
                    if len(content) > 2000:
                        summary_prompt = f"Please provide a concise summary of the following content that is relevant to our research:\n\n{content[:2000]}..."
                        result = self._call_openai(summary_prompt)
                    else:
                        result = content
                else:
                    result = self.doc_tool.fetch_content(target)
                self.tool_memory.append(f"DOCUMENT[{target}] -> Processed content")
            else:
                result = "Error: Could not extract target URL or file from instruction"
                
        elif self._is_analysis_step(step_lower):
            # For analysis, reasoning, or synthesis steps, use the LLM
            analysis_prompt = self._create_analysis_prompt(step_instruction)
            result = self._call_openai(analysis_prompt)
            self.tool_memory.append(f"ANALYSIS[{step_instruction[:30]}...] -> Generated analysis")
            
        else:
            # Default: Use LLM for general reasoning
            result = self._call_openai(step_instruction)
            self.tool_memory.append(f"LLM[{step_instruction[:30]}...] -> {result[:50]}...")
        
        return result

    def _call_openai(self, prompt: str) -> str:
        """Helper method to call OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500
            )
            return response.choices[0].message.content or "No response generated"
        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            return f"Error: Could not generate response - {str(e)}"

    def run_plan(self, plan_steps: List[str]) -> Tuple[str, Dict[int, Dict[str, Any]]]:
        """
        Execute a sequence of plan steps and collect the final answer.
        
        :param plan_steps: List of step instructions
        :return: Tuple of (final_answer, intermediate_results)
        """
        intermediate_results = {}
        context = []  # Build context from previous steps
        final_answer = None
        
        for i, step in enumerate(plan_steps, start=1):
            print(f"Executing step {i}: {step}")
            
            # Add context from previous steps to current step if needed
            contextualized_step = self._contextualize_step(step, context)
            output = self.execute_step(contextualized_step)
            
            intermediate_results[i] = {
                "step": step,
                "output": output,
                "contextualized_step": contextualized_step
            }
            
            # Add this step's output to context for future steps
            context.append(f"Step {i}: {step}\nResult: {output[:200]}...")
            
            # Check if this is the final step or produces final answer
            if self._is_concluding_step(step) or i == len(plan_steps):
                if self._is_synthesis_needed(plan_steps, intermediate_results):
                    # Synthesize final answer from all steps
                    final_answer = self._synthesize_final_answer(intermediate_results)
                else:
                    final_answer = output
                break
        
        return final_answer or "No final answer generated", intermediate_results

    def _is_search_step(self, step_lower: str) -> bool:
        """Check if step requires web search."""
        search_keywords = ['search', 'find', 'look up', 'research', 'investigate', 'discover']
        return any(keyword in step_lower for keyword in search_keywords)

    def _is_document_step(self, step_lower: str) -> bool:
        """Check if step requires document/URL processing."""
        doc_keywords = ['read', 'open', 'fetch', 'download', 'access', 'retrieve', 'url', 'website', 'document']
        return any(keyword in step_lower for keyword in doc_keywords)

    def _is_analysis_step(self, step_lower: str) -> bool:
        """Check if step requires analysis or reasoning."""
        analysis_keywords = ['analyze', 'compare', 'evaluate', 'assess', 'examine', 'summarize', 'synthesize', 'conclude']
        return any(keyword in step_lower for keyword in analysis_keywords)

    def _extract_search_query(self, instruction: str) -> str:
        """Extract search query from instruction."""
        # Try different patterns to extract the query
        patterns = [
            r'search\s+for\s+"([^"]+)"',  # search for "query"
            r'search\s+for\s+(.+?)(?:\.|$)',  # search for query
            r'find\s+information\s+about\s+(.+?)(?:\.|$)',  # find information about query
            r'research\s+(.+?)(?:\.|$)',  # research query
            r'look\s+up\s+(.+?)(?:\.|$)',  # look up query
        ]
        
        for pattern in patterns:
            match = re.search(pattern, instruction, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: return everything after common search terms
        search_terms = ['search for', 'find', 'research', 'look up']
        instruction_lower = instruction.lower()
        
        for term in search_terms:
            if term in instruction_lower:
                query = instruction[instruction_lower.find(term) + len(term):].strip()
                # Remove common endings
                query = re.sub(r'\s+and\s+.+', '', query)
                query = re.sub(r'\s+to\s+.+', '', query)  
                query = re.sub(r'\s+in\s+order\s+to\s+.+', '', query)
                return query.strip(' ."')
        
        return ""

    def _extract_target(self, instruction: str) -> str:
        """Extract URL or file path from instruction."""
        # Look for URLs first
        url_pattern = r'https?://[^\s]+'
        url_match = re.search(url_pattern, instruction)
        if url_match:
            return url_match.group()
        
        # Look for quoted file paths
        file_pattern = r'"([^"]+\.[a-zA-Z0-9]+)"'
        file_match = re.search(file_pattern, instruction)
        if file_match:
            return file_match.group(1)
        
        # Look for common file indicators
        file_indicators = ['file:', 'document:', 'read ', 'open ']
        for indicator in file_indicators:
            if indicator in instruction.lower():
                # Extract what comes after the indicator
                start = instruction.lower().find(indicator) + len(indicator)
                target = instruction[start:].split()[0] if instruction[start:].split() else ""
                return target.strip(' "\'')
        
        return ""

    def _is_concluding_step(self, step: str) -> bool:
        """Check if this step should produce the final answer."""
        concluding_keywords = [
            'provide the final answer', 'conclude', 'final answer', 'summarize', 
            'synthesis', 'combine', 'compile the answer', 'answer the question'
        ]
        step_lower = step.lower()
        return any(keyword in step_lower for keyword in concluding_keywords)

    def _contextualize_step(self, step: str, context: List[str]) -> str:
        """Add relevant context to the current step if needed."""
        if not context:
            return step
        
        # If step refers to previous results, add context
        if any(word in step.lower() for word in ['this', 'that', 'it', 'them', 'above', 'previous']):
            context_summary = "\n".join(context[-2:])  # Use last 2 steps for context
            return f"Context from previous steps:\n{context_summary}\n\nCurrent step: {step}"
        
        return step

    def _is_synthesis_needed(self, plan_steps: List[str], _results: Dict[int, Dict[str, Any]]) -> bool:
        """Check if final synthesis is needed based on multiple information gathering steps."""
        # If there are multiple steps and the last step isn't explicitly a synthesis step
        if len(plan_steps) > 2:
            last_step = plan_steps[-1].lower()
            synthesis_indicators = ['synthesize', 'combine', 'compile', 'summarize all', 'bring together']
            return not any(indicator in last_step for indicator in synthesis_indicators)
        return False

    def _synthesize_final_answer(self, results: Dict[int, Dict[str, Any]]) -> str:
        """Synthesize a final answer from all step results."""
        # Compile all the information gathered
        all_info = []
        for step_num, result in results.items():
            step = result['step']
            output = result['output']
            all_info.append(f"From step {step_num} ({step}):\n{output}")
        
        synthesis_prompt = f"""
Based on the following information gathered from multiple research steps, please provide a comprehensive and accurate final answer:

{chr(10).join(all_info)}

Please synthesize this information into a clear, coherent, and complete answer to the original question.
"""
        
        return self._call_openai(synthesis_prompt)

    def _create_analysis_prompt(self, instruction: str) -> str:
        """Create an enhanced prompt for analysis steps."""
        context_info = ""
        if self.tool_memory:
            # Add context from recent tool usage
            recent_tools = self.tool_memory[-3:]  # Last 3 tool calls
            context_info = "\nRecent research context:\n" + "\n".join(recent_tools)
        
        return f"""
{instruction}

{context_info}

Please provide a thorough and well-reasoned response based on the available information.
"""

    def get_execution_summary(self) -> str:
        """Get a summary of tool usage during execution."""
        if not self.tool_memory:
            return "No tools were used during execution."
        
        return "Tool usage summary:\n" + "\n".join(self.tool_memory)

    def clear_tool_memory(self):
        """Clear the tool usage memory for a new task."""
        self.tool_memory = []
