# main.py
import os
from dotenv import load_dotenv
from agent import Agent

# Load environment variables from .env file
load_dotenv()

def main():
    """Main function to initialize and run the agent."""
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_key_here")
        return
    
    print("Initializing Memento LLM Agent...")
    
    # Initialize the agent - it will handle all sub-components internally
    try:
        agent = Agent(api_key=api_key)
        print("✓ Agent initialized successfully!")
        
        # Print memory stats
        stats = agent.memory.get_stats()
        print(f"✓ Memory stats: {stats['total_cases']} cases stored using {stats['storage_type']}")
        
        return agent
        
    except Exception as e:
        print(f"Error initializing agent: {e}")
        return None

def run_example_query(agent):
    """Run an example query to test the agent."""
    if not agent:
        return
    
    print("\n" + "="*60)
    print("Running example query...")
    print("="*60)
    
    # Example query for the agent
    query = " what is the score of latest real madrid match ? "
    print(f"Query: {query}")
    print("\nProcessing...")
    
    try:
        answer = agent.solve(query)
        print("\n" + "="*60)
        print("FINAL ANSWER:")
        print("="*60)
        print(answer)
        print("="*60)
        
        # Show updated memory stats
        stats = agent.memory.get_stats()
        print(f"\nMemory updated: {stats['total_cases']} total cases")
        
    except Exception as e:
        print(f"Error processing query: {e}")

if __name__ == "__main__":
    # Initialize the agent
    agent = main()
    
    # Run example query
    run_example_query(agent)
    
    # Interactive mode (optional)
    if agent:
        print("\n" + "="*60)
        print("Interactive mode (type 'quit' to exit)")
        print("="*60)
        
        while True:
            try:
                user_query = input("\nEnter your query: ").strip()
                if user_query.lower() in ['quit', 'exit', 'q']:
                    break
                if user_query:
                    print("\nProcessing...")
                    answer = agent.solve(user_query)
                    print(f"\nAnswer: {answer}")
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("Goodbye!")
