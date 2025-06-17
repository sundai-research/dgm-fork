"""
Test core LLM functionality specifically for Groq/Qwen models.
These tests validate that the current Groq integration works before refactoring.
"""
import pytest
import os
import sys
from unittest.mock import patch, MagicMock

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_client import get_response_from_llm, extract_json_between_markers
from groq import Groq


class TestQwenClientBasics:
    """Test qwen_client.py basic functionality"""
    
    @pytest.mark.skipif(not os.getenv('GROQ_API_KEY'), reason="GROQ_API_KEY not set")
    def test_qwen_client_basic_call(self):
        """Test that qwen_client get_response_from_llm works"""
        
        # Test basic call (no client/model parameters needed)
        content, msg_history = get_response_from_llm(
            msg="Say 'test successful' and nothing else",
            system_message="You are a helpful assistant."
        )
        
        # Verify response structure
        assert isinstance(content, str), f"Expected string content, got {type(content)}"
        assert len(content) > 0, "Response content should not be empty"
        assert isinstance(msg_history, list), f"Expected list msg_history, got {type(msg_history)}"
        assert len(msg_history) == 2, f"Expected 2 messages in history, got {len(msg_history)}"
    
    def test_qwen_client_missing_groq_key(self):
        """Test behavior when GROQ_API_KEY is missing"""
        
        # Remove GROQ_API_KEY if it exists - should raise error
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(Exception):  # GroqError when no API key
                get_response_from_llm(
                    msg="test",
                    system_message="test"
                )


class TestGroqLLMResponses:
    """Test LLM response generation with Groq"""
    
    @pytest.fixture
    def mock_groq_response(self):
        """Mock Groq API response structure"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response from Qwen model"
        mock_response.usage.prompt_tokens = 100
        return mock_response
    
    @pytest.mark.skipif(not os.getenv('GROQ_API_KEY'), reason="GROQ_API_KEY not set")
    def test_get_response_from_llm_qwen_real(self):
        """Test real API call to Qwen model (requires API key)"""
        
        # Simple test message
        msg = "Hello, please respond with just 'OK'"
        system_message = "You are a helpful assistant."
        
        try:
            content, msg_history = get_response_from_llm(
                msg=msg,
                system_message=system_message,
                print_debug=False,
                msg_history=None,
                temperature=0.1
            )
            
            # Verify response structure
            assert isinstance(content, str), f"Expected string content, got {type(content)}"
            assert len(content) > 0, "Response content should not be empty"
            assert isinstance(msg_history, list), f"Expected list msg_history, got {type(msg_history)}"
            assert len(msg_history) == 2, f"Expected 2 messages in history, got {len(msg_history)}"
            
            # Verify message history structure
            user_msg = msg_history[0]
            assistant_msg = msg_history[1]
            
            assert user_msg["role"] == "user"
            assert user_msg["content"] == msg
            assert assistant_msg["role"] == "assistant"
            assert assistant_msg["content"] == content
            
            
        except Exception as e:
            pytest.fail(f"Real API call failed: {str(e)}")
    
    def test_get_response_from_llm_qwen_mocked(self, mock_groq_response):
        """Test Qwen response handling with mocked API"""
        
        with patch('qwen_client.Groq') as mock_groq_class:
            mock_client = MagicMock()
            mock_groq_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_groq_response
            
            msg = "Test message"
            system_message = "Test system"
            
            content, msg_history = get_response_from_llm(
                msg=msg,
                system_message=system_message,
                print_debug=False,
                msg_history=None
            )
            
            # Verify mocked response was processed correctly
            assert content == "Test response from Qwen model"
            assert len(msg_history) == 2
            assert msg_history[0]["role"] == "user"
            assert msg_history[1]["role"] == "assistant"
    
    def test_thinking_tag_removal(self, mock_groq_response):
        """Test that </think> tags are properly removed from Qwen responses"""
        
        # Mock response with thinking tags
        mock_groq_response.choices[0].message.content = "<think>Internal reasoning</think>\n\nActual response"
        
        with patch('qwen_client.Groq') as mock_groq_class:
            mock_client = MagicMock()
            mock_groq_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_groq_response
            
            content, msg_history = get_response_from_llm(
                msg="Test",
                system_message="Test",
                print_debug=False,
                msg_history=None
            )
            
            # Verify thinking tags were removed (this tests line 307 in llm.py)
            assert content == "Actual response"
            assert "<think>" not in content
            assert "</think>" not in content


class TestUtilityFunctions:
    """Test utility functions used with LLM responses"""
    
    def test_extract_json_between_markers_valid(self):
        """Test JSON extraction from LLM output"""
        llm_output = '''
Here is some text.

```json
{
    "key": "value",
    "number": 42,
    "array": [1, 2, 3]
}
```

More text after.
        '''
        
        result = extract_json_between_markers(llm_output)
        
        assert result is not None
        assert result["key"] == "value"
        assert result["number"] == 42
        assert result["array"] == [1, 2, 3]
    
    def test_extract_json_between_markers_fallback(self):
        """Test JSON extraction fallback when no code block"""
        llm_output = 'Here is {"status": "success", "count": 5} in the text.'
        
        result = extract_json_between_markers(llm_output)
        
        assert result is not None
        assert result["status"] == "success"
        assert result["count"] == 5
    
    def test_extract_json_between_markers_invalid(self):
        """Test JSON extraction with invalid JSON"""
        llm_output = '''
```json
{invalid json here
```
        '''
        
        result = extract_json_between_markers(llm_output)
        
        assert result is None


class TestGroqSpecificBehavior:
    """Test Qwen/Groq specific behavior and edge cases"""
    
    def test_qwen_token_limit_handling(self):
        """Test that Qwen token limit logic works (two-stage approach)"""
        
        # Mock the two-stage token checking approach
        first_response = MagicMock()
        first_response.usage.prompt_tokens = 50000  # Mock prompt token count
        
        # Create second response mock
        second_response = MagicMock()
        second_response.choices = [MagicMock()]
        second_response.choices[0].message.content = "Test response from Qwen model"
        second_response.usage.prompt_tokens = 100
        
        with patch('qwen_client.Groq') as mock_groq_class:
            mock_client = MagicMock()
            mock_groq_class.return_value = mock_client
            
            # First call returns token count, second returns actual response
            mock_client.chat.completions.create.side_effect = [first_response, second_response]
            
            content, msg_history = get_response_from_llm(
                msg="Long message",
                system_message="System",
                print_debug=False,
                msg_history=None
            )
            
            # Verify two API calls were made
            assert mock_client.chat.completions.create.call_count == 2
            
            # Verify second call used calculated max_completion_tokens
            second_call_kwargs = mock_client.chat.completions.create.call_args_list[1][1]
            assert 'max_completion_tokens' in second_call_kwargs
            # Should be min(128000-50000, 16384) = 16384
            assert second_call_kwargs['max_completion_tokens'] == 16384


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])