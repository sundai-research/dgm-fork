#!/usr/bin/env python3
"""
Dummy script to call Together AI Qwen model using the same API as llm.py for debugging purposes.
"""

import os
import sys
import json
import openai

# Add support for Groq client
from groq import Groq



def create_client(model: str):
    """
    Create and return an LLM client for the specified model.
    Adapted from llm.py.
    """
    # Support Groq model qwen/qwen3-32b
    if model == "qwen/qwen3-32b":
        print(f"Using Groq API with model {model}.")
        client = Groq()
        return client, model

    if model.startswith("Qwen/"):
        print(f"Using Together OpenAI-compatible API with model {model}.")
        client = openai.OpenAI(
            api_key=os.getenv("TOGETHER_API_KEY"),
            base_url="https://api.together.xyz/v1"
        )
        return client, model
    elif model.startswith("o"):
        print(f"Using OpenAI API with model {model}.")
        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        return client, model
    else:
        raise ValueError(f"Model {model} not supported in this dummy script.")


def get_current_weather(location: str, unit: str = "fahrenheit") -> str:
    """
    Dummy implementation of a weather tool.
    """
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "warm"],
    }
    return json.dumps(weather_info)


def main():
    # model = "Qwen/Qwen3-235B-A22B-fp8-tput"
    model = "qwen/qwen3-32b"
    # Specify the Together AI model
    # model = "o4-mini-2025-04-16"
    client, client_model = create_client(model)


    # messages = [
    #     {
    #         "role": "user",
    #         "content": "I have uploaded a Python code repository in the directory /dgm/. Help solve the following problem.\n\n<problem_description>\n# Coding Agent Summary\n\n- **Main File**: `coding_agent.py`\n  - Primary Class: `AgenticSystem`\n  - The `forward()` function is the central entry point.\n  - Prompts are located either within the `forward()` function or in the `prompts/` directory.\n- **Tools**: `tools/`\n  - The `tools/` directory contains various tools that LLMs can use to perform specific tasks.\n  - Each tool must have a `tool_info()` function that returns a JSON object containing 'name', 'description', and 'input_schema'. The 'input_schema' should be a JSON object containing 'type', 'properties', and 'required'.\n  - Each tool must have a `tool_function()` function that takes the arguments defined in input_schema, performs the tool's task, and returns a string.\n  - See other tools for reference.\n- **Utilities**: `utils/`\n  - The `utils/` directory contains utility functions used across the codebase.\n\n- **Additional Details**:\n  - The agent is very good at automatically utilizing the right available tools at the right time. So do not have an agentic flow that explicitly forces a tool's usage.\n  - Common tools, such as file editing and bash commands, are easy for the agent to recognize and use appropriately. However, more complex and niche tools may require explicit instructions in the prompt.\n  - Tools should be designed to be as general as possible, ensuring they work across any GitHub repository. Avoid hardcoding repository-specific details or behaviors (e.g., paths).\n  - Do not use 'while True' loops in the agent's code. This can cause the agent to get stuck and not respond.\n  - Verify the implementation details of helper functions prior to usage to ensure proper integration and expected behavior.\n  - Do not install additional packages or dependencies directly. Update `requirements.txt` if new dependencies are required and install them using `pip install -r requirements.txt`.\n\n\n# To Implement\n\nModify llm_withtools.py's chat_with_agent function to: 1) Add token counting middleware that tracks input length, 2) Implement a content analyzer that identifies low-relevance code sections using file edit history and test relevance, 3) Create a compression function that replaces low-relevance code blocks with semantic summaries (e.g., \"// 150 lines of utility functions for data validation\"), 4) Add a retry mechanism that automatically compresses context when 400 errors occur. Integrate with the editor tool to selectively expand compressed sections when needed.\n\nThe agent currently fails when handling large codebases due to context window limitations (200k tokens). We need to implement an adaptive context management system that automatically optimizes input content before hitting token limits. The solution should: 1) Monitor token usage in real-time, 2) Prioritize content based on file modification status and test relevance, 3) Dynamically compress non-critical code sections into semantic summaries, and 4) Maintain ability to restore compressed content when needed. The implementation should integrate with existing editor tool functions to access original code when summaries are insufficient.\n</problem_description>\n\n<test_description>\nThe tests in the repository can be run with the bash command `cd /dgm/ && pytest -rA <specific test files>`. If no specific test files are provided, all tests will be run. The given command-line options must be used EXACTLY as specified. Do not use any other command-line options. ONLY test tools and utils. NEVER try to test or run agentic_system.forward().\n</test_description>\n\nYour task is to make changes to the files in the /dgm/ directory to address the <problem_description>. I have already taken care of the required dependencies.\n"
    #     }
    # ]
    messages = [
        {
            "role": "user",
            "content": "what's the current weather in Boston, MA?"
        }
    ]
    # messages = [
    #     {
    #         "role": "user",
    #         "content": "How would you compute the volume of a shape made of the union of two spheres of the same radius where the centers coincide with the surface of the other sphere?"
    #     }
    # ]

    """
        # Original OpenAI-style function definitions (preserved for reference)
        functions = [
            {
                "type": "function",
                "name": "bash",
                "description": "Run commands in a bash shell\n\n* When invoking this tool, the contents of the \"command\" parameter does NOT need to be XML-escaped.\n\n* You don't have access to the internet via this tool.\n\n* You do have access to a mirror of common linux and python packages via apt and pip.\n\n* State is persistent across command calls and discussions with the user.\n\n* To inspect a particular line range of a file, e.g. lines 10-25, try 'sed -n 10,25p /path/to/the/file'.\n\n* Please avoid commands that may produce a very large amount of output.\n\n* Please run long lived commands in the background, e.g. 'sleep 10 &' or start a server in the background.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command to run."
                        }
                    },
                    "required": ["command"],
                    "additionalProperties": 'false'
                },
                "strict": 'true'
            },
            {
                "type": "function",
                "name": "editor",
                "description": "Custom editing tool for viewing, creating, and editing files\n\n* State is persistent across command calls and discussions with the user.\n\n* If `path` is a file, `view` displays the entire file with line numbers. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep.\n\n* The `create` command cannot be used if the specified `path` already exists as a file.\n\n* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`.\n\n* The `edit` command overwrites the entire file with the provided `file_text`.\n\n* No partial/line-range edits or partial viewing are supported.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "enum": ["view", "create", "edit"],
                            "description": "The command to run: `view`, `create`, or `edit`."
                        },
                        "path": {
                            "description": "Absolute path to file or directory, e.g. `/repo/file.py` or `/repo`.",
                            "type": "string"
                        },
                        "file_text": {
                            "description": "Required parameter of `create` or `edit` command, containing the content for the entire file.",
                            "type": ["string", "null"]
                        }
                    },
                    "required": ["command", "path", "file_text"],
                    "additionalProperties": 'false'
                },
                "strict": 'true'
            },
            {
                "type": "function",
                "name": "get_current_weather",
                "description": "Get the current weather for a given location, returning temperature, humidity, and conditions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and optional state/country, e.g. San Francisco, CA or London, UK."
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["fahrenheit", "celsius"],
                            "description": "Temperature unit, either 'fahrenheit' or 'celsius'."
                        }
                    },
                    "required": ["location", "unit"],
                    "additionalProperties": 'false'
                },
                "strict": 'true'
            },
        ]
    """
    '''
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Evaluate a mathematical expression",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to evaluate",
                        }
                    },
                    "required": ["expression"],
                },
            },
        }
    '''
    # New Groq agentic tooling nested function format
    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run commands in a bash shell\n\n* When invoking this tool, the contents of the \"command\" parameter does NOT need to be XML-escaped.\n\n* You don't have access to the internet via this tool.\n\n* You do have access to a mirror of common linux and python packages via apt and pip.\n\n* State is persistent across command calls and discussions with the user.\n\n* To inspect a particular line range of a file, e.g. lines 10-25, try 'sed -n 10,25p /path/to/the/file'.\n\n* Please avoid commands that may produce a very large amount of output.\n\n* Please run long lived commands in the background, e.g. 'sleep 10 &' or start a server in the background.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command to run."
                        }
                    },
                    "required": ["command"],
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "editor",
                "description": "Custom editing tool for viewing, creating, and editing files\n\n* State is persistent across command calls and discussions with the user.\n\n* If `path` is a file, `view` displays the entire file with line numbers. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep.\n\n* The `create` command cannot be used if the specified `path` already exists as a file.\n\n* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`.\n\n* The `edit` command overwrites the entire file with the provided `file_text`.\n\n* No partial/line-range edits or partial viewing are supported.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "enum": ["view", "create", "edit"],
                            "description": "The command to run: `view`, `create`, or `edit`."
                        },
                        "path": {
                            "description": "Absolute path to file or directory, e.g. `/repo/file.py` or `/repo`.",
                            "type": "string"
                        },
                        "file_text": {
                            "description": "Required parameter of `create` or `edit` command, containing the content for the entire file.",
                            "type": ["string", "null"]
                        }
                    },
                    "required": ["command", "path", "file_text"],
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather for a given location, returning temperature, humidity, and conditions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and optional state/country, e.g. San Francisco, CA or London, UK."
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["fahrenheit", "celsius"],
                            "description": "Temperature unit, either 'fahrenheit' or 'celsius'."
                        }
                    },
                    "required": ["location", "unit"],
                },
                "strict": 'true'
            }
        }
    ]
    # "tool_choice": "auto"
    tr = client.chat.completions.create(
        model=client_model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        stream=False,
        parallel_tool_calls=False,
        max_completion_tokens=10,
    )
    num_prompt_tokens = tr.usage.prompt_tokens
    response1 = client.chat.completions.create(
        model=client_model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        stream=False,
        parallel_tool_calls=False,
        max_completion_tokens=min(128000-num_prompt_tokens, 16384),
    )
    msg1 = response1.choices[0].message
    print("Assistant first response:", msg1)
    msg1.content = msg1.content.split('</think>')[-1] if msg1.content else msg1.content

    # Check if model requested a tool call
    if getattr(msg1, 'tool_calls', None):
        tool_call = msg1.tool_calls[0]
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        # Execute the tool
        function_response = get_current_weather(**function_args)

        # Append assistant message and tool result
        messages.append(msg1)
        messages.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": function_response,
        })

        # Second API call: get final response from model
        response2 = client.chat.completions.create(
            model=client_model,
            messages=messages
        )
        content = response2.choices[0].message.content
        response2.choices[0].message.content = content.split('</think>\n\n')[-1] if content else content
        print("Assistant final response:", response2.choices[0].message.content)
    else:
        # No tool use, just print content
        print("Assistant:", msg1.content)


if __name__ == "__main__":
    main()