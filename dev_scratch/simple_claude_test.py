"""
Simple test to verify Claude works with current llm_withtools.py
"""
from llm_withtools import chat_with_agent

def test_claude_simple():
    """Simple Claude test"""
    
    print("=== Simple Claude Test ===")
    
    # Use Claude model
    claude_model = 'claude-3-5-sonnet-20241022'
    msg = "Hello! Please respond with 'Claude is working' and use the bash tool to run 'echo test'"
    
    print(f"Calling Claude with: {msg}")
    
    try:
        history = chat_with_agent(
            msg=msg,
            model=claude_model,
            msg_history=[],
            logging=print,
            convert=False
        )
        
        print(f"\nGot history with {len(history)} messages")
        
        # Print the last message to see Claude's response
        if history:
            last_msg = history[-1]
            print(f"Last message: {last_msg}")
        
        print("✅ Claude test completed successfully!")
        return history
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_claude_simple()