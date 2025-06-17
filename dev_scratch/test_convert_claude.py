"""
Test the current convert_msg_history function with Claude tool usage.
This will help us understand the expected format transformation.
"""
import os
from llm_withtools import chat_with_agent, convert_msg_history
from utils.eval_utils import msg_history_to_report

def test_claude_conversion():
    """Test Claude tool calling and message conversion using chat_with_agent"""
    
    # Check if we have access to Claude
    if 'ANTHROPIC_API_KEY' not in os.environ:
        print("❌ No Claude API access - need ANTHROPIC_API_KEY")
        return
    
    print("=== Testing Claude Tool Calling with chat_with_agent ===")
    
    try:
        # Use chat_with_agent to get real Claude message history with test execution
        # The updated parser now handles both old and new pytest formats
        msg = "Please run the project tests using the bash tool: 'python -m pytest tests/ --tb=no'"
        print(f"Calling Claude with: {msg}")
        
        # Get raw Claude message history from chat_with_agent
        raw_history = chat_with_agent(
            msg=msg,
            model='claude-3-5-sonnet-20241022',
            msg_history=[],
            logging=print,
            convert=False  # Don't convert yet, we want raw format
        )
        
        print(f"\n=== Raw Claude Message History ===")
        print(f"Raw history has {len(raw_history)} messages")
        
        for i, msg_item in enumerate(raw_history):
            role = msg_item.get('role')
            content = msg_item.get('content')
            print(f"Raw {i}: role={role}, content_type={type(content)}")
            
            # Show content structure
            if isinstance(content, list):
                print(f"  Content blocks: {len(content)}")
                for j, block in enumerate(content):
                    if isinstance(block, dict):
                        block_type = block.get('type', 'unknown')
                        print(f"    Block {j}: type={block_type}")
                        if block_type == 'tool_result':
                            content_preview = str(block.get('content', ''))[:100]
                            print(f"      Tool result preview: {content_preview}...")
                        elif block_type == 'tool_use':
                            tool_name = block.get('name', 'unknown')
                            print(f"      Tool use: {tool_name}")
            else:
                print(f"  Content: {str(content)[:100]}...")
        
        print(f"\n=== Testing convert_msg_history ===")
        
        # Test conversion
        converted_history = convert_msg_history(raw_history, model='claude-3-5-sonnet-20241022')
        
        print(f"Converted history has {len(converted_history)} messages")
        for i, msg in enumerate(converted_history):
            role = msg.get('role')
            content = msg.get('content')
            print(f"Converted {i}: role={role}, content_type={type(content)}")
            
            # Look for tool results in converted format
            if isinstance(content, list) and content:
                for j, block in enumerate(content):
                    if isinstance(block, dict):
                        text = block.get('text', '')
                        if 'Tool Result:' in text:
                            print(f"  -> FOUND CONVERTED TOOL RESULT in block {j}!")
                            print(f"     Preview: {text[:150]}...")
                        elif '<tool_use>' in text:
                            print(f"  -> FOUND CONVERTED TOOL USE in block {j}!")
                            print(f"     Preview: {text[:150]}...")
        
        print(f"\n=== Testing msg_history_to_report ===")
        
        # Debug: Let's manually check what the function should find
        print("DEBUG: Looking for Tool Result messages in converted history...")
        for i, msg in enumerate(converted_history):
            if msg['role'] == 'user':
                content = msg['content']
                if isinstance(content, list) and content:
                    text = content[0].get('text', '') if isinstance(content[0], dict) else str(content[0])
                    if 'Tool Result:' in text:
                        print(f"  Found Tool Result in message {i}")
                        print(f"  Content preview: {text[:300]}...")
                        
                        # Try parsing this content directly
                        from utils.eval_utils import parse_eval_output
                        parsed = parse_eval_output('dgm', text)
                        print(f"  Direct parse result: {parsed}")
        
        # Test the reporting function with converted history
        test_report = msg_history_to_report('dgm', converted_history, model='claude-3-5-sonnet-20241022')
        
        print(f"Generated test report length: {len(test_report) if test_report else 0}")
        if test_report:
            print(f"Report preview: {test_report}")
        
        if test_report:
            print("✅ SUCCESS: msg_history_to_report worked with Claude conversion!")
        else:
            print("❌ FAILED: msg_history_to_report returned empty report")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

def test_qwen_format_analysis():
    """Compare with our qwen format to understand the differences"""
    
    print("\n=== Comparing with Qwen Format ===")
    
    try:
        from qwen_chat import chat_with_qwen
        
        # Get qwen history for comparison
        qwen_history = chat_with_qwen('List files in current directory', msg_history=[], logging=lambda x: None)
        
        print(f"Qwen history has {len(qwen_history)} messages")
        for i, msg in enumerate(qwen_history):
            if hasattr(msg, 'role') and not isinstance(msg, dict):
                print(f"Qwen {i}: Pydantic, role={msg.role}")
            else:
                print(f"Qwen {i}: dict, role={msg.get('role')}")
                if msg.get('role') == 'tool':
                    content = msg.get('content', '')
                    print(f"  Tool content preview: {str(content)[:100]}...")
                    
    except Exception as e:
        print(f"Qwen comparison failed: {e}")

if __name__ == "__main__":
    test_claude_conversion()
    test_qwen_format_analysis()