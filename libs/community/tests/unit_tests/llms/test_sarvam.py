from unittest.mock import patch, MagicMock
from langchain_community.llms.sarvam import SarvamAI

def test_sarvam_llm_type():
    llm = SarvamAI(api_key="test-key")
    assert llm._llm_type == "sarvam-ai"

def test_sarvam_default_model():
    llm = SarvamAI(api_key="test-key")
    assert llm.model == "saaras:v1"

@patch("langchain_community.llms.sarvam.requests.post")
def test_sarvam_call(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "नमस्ते!"}}]
    }
    mock_post.return_value = mock_response

    llm = SarvamAI(api_key="test-key")
    result = llm._call("Hello in Hindi")
    assert result == "नमस्ते!"
