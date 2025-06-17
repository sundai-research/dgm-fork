"""
Configuration and fixtures for Groq/Qwen functionality tests.
"""
import pytest
import os
import sys

# Add the parent directory to Python path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def groq_api_key():
    """Fixture to provide GROQ_API_KEY for tests."""
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        pytest.skip("GROQ_API_KEY environment variable not set")
    return api_key


@pytest.fixture
def groq_only_env():
    """Fixture that ensures only GROQ_API_KEY is available in environment."""
    # Store original environment
    original_env = os.environ.copy()
    
    # Clear all LLM-related API keys except Groq
    llm_keys_to_remove = [
        'ANTHROPIC_API_KEY',
        'OPENAI_API_KEY', 
        'TOGETHER_API_KEY',
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY',
        'AWS_REGION',
        'AWS_REGION_NAME'
    ]
    
    for key in llm_keys_to_remove:
        if key in os.environ:
            del os.environ[key]
    
    # Ensure GROQ_API_KEY is available if it was in original env
    if 'GROQ_API_KEY' in original_env:
        os.environ['GROQ_API_KEY'] = original_env['GROQ_API_KEY']
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def sample_swe_problem():
    """Fixture providing a simple SWE-bench problem for testing."""
    return {
        "instance_id": "test__test-123",
        "problem_statement": "Fix a simple bug in the add function that returns incorrect results.",
        "test_description": "Run pytest tests/test_math.py to verify the fix works correctly."
    }


@pytest.fixture
def sample_qwen_response():
    """Fixture providing a typical Qwen response structure."""
    return {
        "content": "This is a sample response from Qwen model.",
        "thinking_removed": "This response had thinking tags removed.",
        "with_tools": "I'll help you with that task. Let me use the appropriate tools."
    }


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "real_api: mark test as requiring real API calls"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )