"""
Test the current convert_msg_history function with Claude tool calling.
"""
import os
from llm_withtools import chat_with_agent, convert_msg_history
from utils.eval_utils import msg_history_to_report

def test_current_claude_conversion():
    """Test current Claude conversion using existing llm_withtools.py"""
    
    if 'ANTHROPIC_API_KEY' not in os.environ:
        print("❌ No Claude API access - need ANTHROPIC_API_KEY")
        return
    
    print("=== Testing Current Claude Logic ===")
    
    # Use the CLAUDE_MODEL constant from llm_withtools
    claude_model = 'bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0'
    msg = "Please list the files in the current directory using the bash tool"
    
    print(f"Calling Claude with: {msg}")
    
    def detailed_logging(x):
        print(f"🔍 LOG: {x}")
    
    try:
        # Get raw Claude history (convert=False)
        print("\n=== Getting Raw Claude History ===")
        raw_history = chat_with_agent(
            msg=msg,
            model=claude_model,
            msg_history=[],
            logging=detailed_logging,
            convert=False  # Get raw format
        )
        
        print(f"\n=== Raw History Analysis ({len(raw_history)} messages) ===")
        for i, msg_item in enumerate(raw_history):
            print(f"Message {i}:")
            print(f"  Type: {type(msg_item)}")
            if isinstance(msg_item, dict):
                role = msg_item.get('role', 'unknown')
                content = msg_item.get('content', [])
                print(f"  Role: {role}")
                print(f"  Content type: {type(content)}")
                if isinstance(content, list):
                    for j, block in enumerate(content):
                        if hasattr(block, 'type'):
                            # Pydantic object
                            block_type = block.type
                            print(f"    Block {j}: {block_type} (Pydantic)")
                            if block_type == 'tool_result':
                                tool_content = getattr(block, 'content', '')
                                print(f"      Tool result: {str(tool_content)[:50]}...")
                        elif isinstance(block, dict):
                            # Dict object
                            block_type = block.get('type', 'unknown')
                            print(f"    Block {j}: {block_type} (dict)")
                            if block_type == 'tool_result':
                                tool_content = block.get('content', '')
                                print(f"      Tool result: {str(tool_content)[:50]}...")
        
        print(f"\n=== Testing convert_msg_history Function ===")
        
        # Test the current convert_msg_history function
        converted_history = convert_msg_history(raw_history, model=claude_model)
        
        print(f"Converted history has {len(converted_history)} messages")
        for i, msg_item in enumerate(converted_history):
            role = msg_item.get('role', 'unknown')
            content = msg_item.get('content', [])
            print(f"Converted {i}: role={role}")
            
            if isinstance(content, list):
                for j, block in enumerate(content):
                    if isinstance(block, dict):
                        text = block.get('text', '')
                        print(f"  Block {j}: {text[:60]}...")
                        
                        # Check for tool results
                        if 'Tool Result:' in text:
                            print(f"    ✅ FOUND CONVERTED TOOL RESULT!")
        
        print(f"\n=== Testing msg_history_to_report ===")
        
        # Test the reporting function
        test_report = msg_history_to_report('dgm', converted_history, model=claude_model)
        
        print(f"Generated test report: {test_report}")
        
        if test_report:
            print("✅ SUCCESS: msg_history_to_report worked!")
        else:
            print("❌ FAILED: msg_history_to_report returned empty")
            
        return converted_history
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_current_claude_conversion()