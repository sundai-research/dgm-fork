"""
Ultra-simplified chat with tools for qwen/qwen3-32b via Groq only.
NO backward compatibility - clean, simple API.
"""
import json
import backoff
import groq
from groq import Groq

from prompts.tooluse_prompt import get_tooluse_prompt
from tools import load_all_tools


def process_tool_call(tools_dict, tool_name, tool_input):
    """
    Execute a tool call. Copied exactly from original.
    """
    try:
        if tool_name in tools_dict:
            return tools_dict[tool_name]['function'](**tool_input)
        else:
            return f"Error: Tool '{tool_name}' not found"
    except Exception as e:
        return f"Error executing tool '{tool_name}': {str(e)}"


@backoff.on_exception(
    backoff.expo,
    (groq.RateLimitError, groq.APIConnectionError, groq.APIStatusError, groq.APITimeoutError),
    max_time=600,
    max_value=60,
)
def get_response_withtools(
    messages, tools, tool_choice="auto", logging=None, max_retry=3
):
    """
    Get response from qwen/qwen3-32b with tools via Groq.
    Simplified version that only handles qwen/qwen3-32b.
    """
    client = Groq()  # Always use Groq
    model = "qwen/qwen3-32b"  # Always use qwen3-32b
    
    try:
        # Two-stage token handling (copied from original qwen/ branch)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            parallel_tool_calls=False,
            stream=False,
            max_completion_tokens=10,
        )
        num_prompt_tokens = response.usage.prompt_tokens
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=False,
            stream=False,
            max_completion_tokens=min(128000-num_prompt_tokens, 16384),
        )
        
        content = response.choices[0].message.content
        response.choices[0].message.reasoning = None
        response.choices[0].message.content = content.split('</think>\n\n')[-1] if content else content
        
        return response
    except Exception as e:
        if logging:
            logging(f"Error in get_response_withtools: {str(e)}")
        if max_retry > 0:
            return get_response_withtools(messages, tools, tool_choice, logging, max_retry - 1)
        raise


def check_for_tool_use(response):
    """
    Check if qwen/qwen3-32b response contains tool calls.
    Only handles Groq/qwen format - no other models.
    """
    if hasattr(response, 'choices') and response.choices:
        msg = response.choices[0].message
        tool_calls = getattr(msg, 'tool_calls', None)
        if tool_calls:
            tc = tool_calls[0]
            return {
                'tool_id': tc.id,
                'tool_name': tc.function.name,
                'tool_input': json.loads(tc.function.arguments),
            }
    return None


def chat_with_qwen(msg, msg_history=None, logging=print):
    """
    Chat with qwen/qwen3-32b with tool support via Groq.
    Simple, clean API - no model parameter needed.
    
    Args:
        msg (str): User message
        msg_history (list, optional): Previous message history
        logging (function): Logging function
        
    Returns:
        list: Updated message history including new conversation
    """
    separator = '=' * 10
    logging(f"\n{separator} User Instruction {separator}\n{msg}")
    
    # Initialize conversation
    new_msg_history = [{"role": "user", "content": msg}]
    
    try:
        if msg_history is None:
            msg_history = []
            
        # Load tools and prepare definitions
        all_tools = load_all_tools(logging=logging)
        tools_defs = []
        tools_dict = {}
        for tool in all_tools:
            info = tool['info']
            tools_defs.append({
                "type": "function",
                "function": {
                    "name": info['name'],
                    "description": info['description'],
                    "parameters": info['input_schema'],
                },
                "strict": True,
            })
            tools_dict[info['name']] = tool

        # First API call with tool definitions
        response = get_response_withtools(
            messages=msg_history + new_msg_history,
            tools=tools_defs,
            tool_choice="auto",
            logging=logging,
        )
        logging(f"\n{separator} Agent Response {separator}\n{response}")

        # Loop over tool uses
        new_msg_history.append(response.choices[0].message)
        tool_use = check_for_tool_use(response)
        while tool_use:
            # Execute tool
            result = process_tool_call(
                tools_dict, tool_use['tool_name'], tool_use['tool_input']
            )
            logging(f"Tool Used: {tool_use['tool_name']}")
            logging(f"Tool Input: {tool_use['tool_input']}")
            logging(f"Tool Result: {result}")
            
            # Append tool result
            new_msg_history.append({
                "role": "tool",
                "name": tool_use['tool_name'],
                "tool_call_id": tool_use['tool_id'],
                "content": result,
            })
            
            # Next API call
            response = get_response_withtools(
                messages=msg_history + new_msg_history,
                tools=tools_defs,
                tool_choice="auto",
                logging=logging,
            )
            new_msg_history.append(response.choices[0].message)
            tool_use = check_for_tool_use(response)
            logging(f"Tool Response: {response}")
            
    except Exception:
        import traceback
        logging(f"Error in chat_with_qwen: {traceback.format_exc()}")
        pass

    return msg_history + new_msg_history


def convert_qwen_msg_history(msg_history):
    """
    Convert qwen message history to the format expected by msg_history_to_report.
    Based on convert_msg_history_openai in llm_withtools.py.
    """
    new_msg_history = []

    for msg in msg_history:
        if isinstance(msg, dict):
            # Dict format (tool result messages)
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            if role == 'tool':
                # Convert tool result to "Tool Result:" format
                new_msg = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Tool Result: {content}",
                        }
                    ],
                }
            else:
                new_msg = {
                    "role": role,
                    "content": content,
                }
        else:
            # Pydantic object (ChatCompletionMessage)
            role = getattr(msg, 'role', None)
            content = getattr(msg, 'content', None)
            tool_calls = getattr(msg, 'tool_calls', None)

            if tool_calls:
                # Convert tool calls to <tool_use> format
                tool_call = tool_calls[0]
                function_name = getattr(tool_call.function, 'name', '')
                function_args = getattr(tool_call.function, 'arguments', '')
                new_msg = {
                    "role": role,
                    "content": [
                        {
                            "type": "text",
                            "text": f"<tool_use>\n{{'tool_name': {function_name}, 'tool_input': {function_args}}}\n</tool_use>",
                        }
                    ],
                }
            else:
                # Regular message
                new_msg = {
                    "role": role,
                    "content": [
                        {
                            "type": "text",
                            "text": content or "",
                        }
                    ],
                }

        new_msg_history.append(new_msg)

    return new_msg_history