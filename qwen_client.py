"""
Ultra-simplified LLM client for qwen/qwen3-32b via Groq API.
Replaces the complex multi-provider llm.py with single-purpose functions.
"""
import json
import re
from groq import Groq


def get_response_from_llm(
    msg,
    system_message="You are a helpful assistant.",
    msg_history=None,
    temperature=0.7,
    print_debug=False,
    **kwargs  # Accept but ignore other parameters for compatibility
):
    """
    Get a completion from qwen/qwen3-32b via Groq.
    Simplified version of original get_response_from_llm() - no client/model parameters needed.
    
    Args:
        msg (str): User message
        system_message (str): System message
        msg_history (list, optional): Previous message history  
        temperature (float): Sampling temperature
        print_debug (bool): Whether to print debug output
        **kwargs: Ignored for compatibility (client, model, etc.)
        
    Returns:
        tuple: (response_content, new_msg_history)
    """
    client = Groq()  # Always use Groq
    
    if msg_history is None:
        msg_history = []
    
    # Build message history for this call
    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    
    # Two-stage approach for token limit handling (copied from original qwen/ logic)
    # First call to get prompt token count
    response = client.chat.completions.create(
        model="qwen/qwen3-32b",
        messages=new_msg_history,
        max_completion_tokens=10,
    )
    num_prompt_tokens = response.usage.prompt_tokens
    
    # Second call with calculated max tokens
    response = client.chat.completions.create(
        model="qwen/qwen3-32b", 
        messages=new_msg_history,
        max_completion_tokens=min(128000 - num_prompt_tokens, 16384),
        temperature=temperature,
    )
    
    content = response.choices[0].message.content
    
    # Remove thinking tags if present (copied from original qwen/ logic)
    content = content.split('</think>\n\n')[-1] if content else content
    
    # Add assistant response to history
    new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    
    # Print debug output if requested (copied from original llm.py)
    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        print(f'User: {new_msg_history[-2]["content"]}')
        print(f'Assistant: {new_msg_history[-1]["content"]}')
        print("*" * 21 + " LLM END " + "*" * 21)
        print()
    
    return content, new_msg_history


def extract_json_between_markers(llm_output):
    """
    Extract JSON from LLM output, looking for ```json code blocks first, then fallback.
    Copied exactly from original llm.py - this function is used by self_improve_step.py.
    
    Args:
        llm_output (str): LLM response text
        
    Returns:
        dict or None: Parsed JSON object or None if no valid JSON found
    """
    inside_json_block = False
    json_lines = []
    
    # Split the output into lines and iterate
    for line in llm_output.split('\n'):
        striped_line = line.strip()
        
        # Check for start of JSON code block
        if striped_line.startswith("```json"):
            inside_json_block = True
            continue
        
        # Check for end of code block
        if inside_json_block and striped_line.startswith("```"):
            # We've reached the closing triple backticks.
            inside_json_block = False
            break
        
        # If we're inside the JSON block, collect the lines
        if inside_json_block:
            json_lines.append(line)
    
    # If we never found a JSON code block, fallback to any JSON-like content
    if not json_lines:
        # Fallback: Try a regex that finds any JSON-like object in the text
        fallback_pattern = r"\{.*?\}"
        matches = re.findall(fallback_pattern, llm_output, re.DOTALL)
        for candidate in matches:
            candidate = candidate.strip()
            if candidate:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # Attempt to clean control characters and re-try
                    candidate_clean = re.sub(r"[\x00-\x1F\x7F]", "", candidate)
                    try:
                        return json.loads(candidate_clean)
                    except json.JSONDecodeError:
                        continue
        return None

    # Join all lines in the JSON block into a single string
    json_string = "\n".join(json_lines).strip()
    
    # Try to parse the collected JSON lines
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        # Attempt to remove invalid control characters and re-parse
        json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
        try:
            return json.loads(json_string_clean)
        except json.JSONDecodeError:
            return None