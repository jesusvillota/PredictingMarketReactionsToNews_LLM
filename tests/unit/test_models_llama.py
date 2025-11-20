"""Tests for LLAMA news parser model."""

import json
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from news_market_analysis.models.llama import (
    FirmShock,
    LLAMANewsParser,
    LLAMAParserError,
    create_parser,
)


# FirmShock Tests


def test_firm_shock_initialization():
    """Test FirmShock initialization."""
    shock = FirmShock(
        firm="Iberdrola",
        ticker="IBE.MC",
        shock_type="demand",
        shock_magnitude="major",
        shock_direction="positive",
    )

    assert shock.firm == "Iberdrola"
    assert shock.ticker == "IBE.MC"
    assert shock.shock_type == "demand"
    assert shock.shock_magnitude == "major"
    assert shock.shock_direction == "positive"


def test_firm_shock_minimal():
    """Test FirmShock with minimal data."""
    shock = FirmShock(firm="Telefonica")

    assert shock.firm == "Telefonica"
    assert shock.ticker == ""
    assert shock.shock_type == ""
    assert shock.shock_magnitude == ""
    assert shock.shock_direction == ""


def test_firm_shock_invalid_type():
    """Test FirmShock with invalid shock type."""
    with pytest.raises(ValueError, match="Invalid shock_type"):
        FirmShock(firm="Test", shock_type="invalid")


def test_firm_shock_invalid_magnitude():
    """Test FirmShock with invalid magnitude."""
    with pytest.raises(ValueError, match="Invalid shock_magnitude"):
        FirmShock(firm="Test", shock_magnitude="huge")


def test_firm_shock_invalid_direction():
    """Test FirmShock with invalid direction."""
    with pytest.raises(ValueError, match="Invalid shock_direction"):
        FirmShock(firm="Test", shock_direction="neutral")


def test_firm_shock_to_dict():
    """Test FirmShock to_dict conversion."""
    shock = FirmShock(
        firm="Santander",
        ticker="SAN.MC",
        shock_type="financial",
        shock_magnitude="minor",
        shock_direction="negative",
    )

    shock_dict = shock.to_dict()
    assert isinstance(shock_dict, dict)
    assert shock_dict["firm"] == "Santander"
    assert shock_dict["ticker"] == "SAN.MC"
    assert shock_dict["shock_type"] == "financial"
    assert shock_dict["shock_magnitude"] == "minor"
    assert shock_dict["shock_direction"] == "negative"


def test_firm_shock_repr():
    """Test FirmShock string representation."""
    shock = FirmShock(
        firm="BBVA", ticker="BBVA.MC", shock_type="policy", shock_magnitude="major"
    )

    repr_str = repr(shock)
    assert "FirmShock" in repr_str
    assert "BBVA" in repr_str
    assert "BBVA.MC" in repr_str


# LLAMANewsParser Tests


@pytest.fixture
def mock_groq_client():
    """Create a mock Groq client."""
    with patch("news_market_analysis.models.llama.Groq") as mock:
        client = MagicMock()
        mock.return_value = client
        yield client


def test_parser_initialization_no_groq():
    """Test parser initialization without groq installed."""
    with patch("news_market_analysis.models.llama.Groq", None):
        with pytest.raises(LLAMAParserError, match="groq package not installed"):
            LLAMANewsParser(api_key="test_key")


def test_parser_initialization_invalid_model(mock_groq_client):
    """Test parser initialization with invalid model."""
    with pytest.raises(LLAMAParserError, match="Invalid model"):
        LLAMANewsParser(api_key="test_key", model="invalid-model")


def test_parser_initialization_success(mock_groq_client):
    """Test successful parser initialization."""
    parser = LLAMANewsParser(api_key="test_key", model="llama3-70b-8192")

    assert parser.api_key == "test_key"
    assert parser.model == "llama3-70b-8192"
    assert parser.max_retries == 3
    assert parser.retry_delay == 2.0


def test_parser_initialization_custom_params(mock_groq_client):
    """Test parser initialization with custom parameters."""
    parser = LLAMANewsParser(
        api_key="test_key",
        model="llama3-8b-8192",
        max_retries=5,
        retry_delay=1.0,
    )

    assert parser.max_retries == 5
    assert parser.retry_delay == 1.0


def test_get_tools_definition(mock_groq_client):
    """Test tools definition generation."""
    parser = LLAMANewsParser(api_key="test_key")
    tools = parser._get_tools_definition()

    assert isinstance(tools, list)
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "news_parser"
    assert "parameters" in tools[0]["function"]


def test_parse_article_success(mock_groq_client):
    """Test successful article parsing."""
    parser = LLAMANewsParser(api_key="test_key")

    # Mock the API response
    mock_response = Mock()
    mock_message = Mock()
    mock_tool_call = Mock()
    mock_tool_call.function.arguments = json.dumps(
        {
            "firms": [
                {
                    "firm": "Iberdrola",
                    "ticker": "IBE.MC",
                    "shock_type": "demand",
                    "shock_magnitude": "major",
                    "shock_direction": "positive",
                }
            ]
        }
    )
    mock_message.tool_calls = [mock_tool_call]
    mock_response.choices = [Mock(message=mock_message)]

    # Mock second response
    mock_second_response = Mock()
    mock_second_response.choices = [Mock(message=Mock(content="Analysis complete"))]

    mock_groq_client.chat.completions.create.side_effect = [
        mock_response,
        mock_second_response,
    ]

    article = "Iberdrola ha anunciado nuevos proyectos de energía renovable."
    response_text, firm_shocks = parser.parse_article(article)

    assert response_text == "Analysis complete"
    assert len(firm_shocks) == 1
    assert isinstance(firm_shocks[0], FirmShock)
    assert firm_shocks[0].firm == "Iberdrola"
    assert firm_shocks[0].ticker == "IBE.MC"


def test_parse_article_no_function_call(mock_groq_client):
    """Test parsing when no function is called."""
    parser = LLAMANewsParser(api_key="test_key")

    # Mock response without tool calls
    mock_response = Mock()
    mock_message = Mock()
    mock_message.tool_calls = None
    mock_message.content = "No firms identified"
    mock_response.choices = [Mock(message=mock_message)]

    mock_groq_client.chat.completions.create.return_value = mock_response

    article = "This is a general article with no specific firms."
    response_text, firm_shocks = parser.parse_article(article)

    assert response_text == "No firms identified"
    assert len(firm_shocks) == 0


def test_parse_article_invalid_shock_data(mock_groq_client):
    """Test parsing with invalid shock data."""
    parser = LLAMANewsParser(api_key="test_key")

    # Mock response with invalid shock type
    mock_response = Mock()
    mock_message = Mock()
    mock_tool_call = Mock()
    mock_tool_call.function.arguments = json.dumps(
        {
            "firms": [
                {
                    "firm": "Test",
                    "ticker": "TEST.MC",
                    "shock_type": "invalid_type",  # Invalid
                    "shock_magnitude": "major",
                    "shock_direction": "positive",
                }
            ]
        }
    )
    mock_message.tool_calls = [mock_tool_call]
    mock_response.choices = [Mock(message=mock_message)]

    mock_second_response = Mock()
    mock_second_response.choices = [Mock(message=Mock(content="Complete"))]

    mock_groq_client.chat.completions.create.side_effect = [
        mock_response,
        mock_second_response,
    ]

    # Should skip invalid shock but not fail
    article = "Test article"
    response_text, firm_shocks = parser.parse_article(article)

    assert len(firm_shocks) == 0  # Invalid shock skipped


def test_parse_article_retry_on_failure(mock_groq_client):
    """Test retry logic on API failure."""
    parser = LLAMANewsParser(api_key="test_key", max_retries=2, retry_delay=0.1)

    # First call fails, second succeeds
    mock_groq_client.chat.completions.create.side_effect = [
        Exception("API Error"),
        Mock(
            choices=[Mock(message=Mock(tool_calls=None, content="Success after retry"))]
        ),
    ]

    article = "Test article"
    response_text, firm_shocks = parser.parse_article(article)

    assert response_text == "Success after retry"
    assert mock_groq_client.chat.completions.create.call_count == 2


def test_parse_article_max_retries_exceeded(mock_groq_client):
    """Test failure after max retries."""
    parser = LLAMANewsParser(api_key="test_key", max_retries=2, retry_delay=0.1)

    # All calls fail
    mock_groq_client.chat.completions.create.side_effect = Exception("API Error")

    article = "Test article"
    with pytest.raises(LLAMAParserError, match="Failed to parse article"):
        parser.parse_article(article)


def test_parse_articles_batch(mock_groq_client):
    """Test batch parsing of multiple articles."""
    parser = LLAMANewsParser(api_key="test_key")

    # Mock responses for multiple articles
    def create_mock_response(firm_name):
        mock_response = Mock()
        mock_message = Mock()
        mock_tool_call = Mock()
        mock_tool_call.function.arguments = json.dumps(
            {
                "firms": [
                    {
                        "firm": firm_name,
                        "ticker": f"{firm_name}.MC",
                        "shock_type": "demand",
                        "shock_magnitude": "minor",
                        "shock_direction": "positive",
                    }
                ]
            }
        )
        mock_message.tool_calls = [mock_tool_call]
        mock_response.choices = [Mock(message=mock_message)]
        return mock_response

    # Create responses for 3 articles
    mock_groq_client.chat.completions.create.side_effect = [
        create_mock_response("Firm1"),
        Mock(choices=[Mock(message=Mock(content="Response1"))]),
        create_mock_response("Firm2"),
        Mock(choices=[Mock(message=Mock(content="Response2"))]),
        create_mock_response("Firm3"),
        Mock(choices=[Mock(message=Mock(content="Response3"))]),
    ]

    articles = ["Article 1", "Article 2", "Article 3"]
    results = parser.parse_articles_batch(articles, verbose=False)

    assert len(results) == 3
    assert all(len(shocks) == 1 for _, shocks in results)


def test_parse_articles_batch_with_failure(mock_groq_client):
    """Test batch parsing with some failures."""
    parser = LLAMANewsParser(api_key="test_key", max_retries=1, retry_delay=0.1)

    # First article succeeds, second fails
    mock_success = Mock()
    mock_success.choices = [Mock(message=Mock(tool_calls=None, content="Success"))]

    mock_groq_client.chat.completions.create.side_effect = [
        mock_success,
        Exception("API Error"),
        Exception("API Error"),
    ]

    articles = ["Article 1", "Article 2"]
    results = parser.parse_articles_batch(articles, verbose=False)

    assert len(results) == 2
    assert results[0][0] == "Success"  # First succeeded
    assert results[1] == (None, [])  # Second failed


def test_parse_dataframe(mock_groq_client):
    """Test parsing DataFrame with articles."""
    parser = LLAMANewsParser(api_key="test_key")

    # Create sample DataFrame
    df = pd.DataFrame(
        {
            "articles": [
                "Iberdrola anuncia nuevos proyectos.",
                "Telefonica firma acuerdo.",
            ],
            "date": ["2024-01-01", "2024-01-02"],
        }
    )

    # Mock responses
    def create_mock_response(firm_name):
        mock_response = Mock()
        mock_message = Mock()
        mock_tool_call = Mock()
        mock_tool_call.function.arguments = json.dumps(
            {
                "firms": [
                    {
                        "firm": firm_name,
                        "ticker": f"{firm_name[:3].upper()}.MC",
                        "shock_type": "policy",
                        "shock_magnitude": "major",
                        "shock_direction": "negative",
                    }
                ]
            }
        )
        mock_message.tool_calls = [mock_tool_call]
        mock_response.choices = [Mock(message=mock_message)]
        return mock_response

    mock_groq_client.chat.completions.create.side_effect = [
        create_mock_response("Iberdrola"),
        Mock(choices=[Mock(message=Mock(content="Response1"))]),
        create_mock_response("Telefonica"),
        Mock(choices=[Mock(message=Mock(content="Response2"))]),
    ]

    result_df = parser.parse_dataframe(df, verbose=False)

    assert "llama_response" in result_df.columns
    assert "llama_firms" in result_df.columns
    assert "llama_tickers" in result_df.columns
    assert "llama_shock_types" in result_df.columns
    assert "llama_shock_magnitudes" in result_df.columns
    assert "llama_shock_directions" in result_df.columns
    assert "llama_num_firms" in result_df.columns

    assert result_df["llama_num_firms"].iloc[0] == 1
    assert result_df["llama_firms"].iloc[0] == ["Iberdrola"]
    assert result_df["llama_tickers"].iloc[0] == ["IBE.MC"]


def test_parse_dataframe_missing_column(mock_groq_client):
    """Test parsing DataFrame with missing article column."""
    parser = LLAMANewsParser(api_key="test_key")

    df = pd.DataFrame({"text": ["Some text"], "date": ["2024-01-01"]})

    with pytest.raises(LLAMAParserError, match="not found"):
        parser.parse_dataframe(df, article_col="articles")


# create_parser Tests


def test_create_parser_with_api_key(mock_groq_client):
    """Test creating parser with provided API key."""
    parser = create_parser(api_key="my_key", model="llama3-8b-8192")

    assert isinstance(parser, LLAMANewsParser)
    assert parser.api_key == "my_key"
    assert parser.model == "llama3-8b-8192"


def test_create_parser_from_environment(mock_groq_client):
    """Test creating parser from environment variable."""
    with patch.dict("os.environ", {"GROQ_API_KEY": "env_key"}):
        parser = create_parser()

        assert parser.api_key == "env_key"


def test_create_parser_no_api_key(mock_groq_client):
    """Test creating parser without API key."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(LLAMAParserError, match="API key must be provided"):
            create_parser()


# Integration-style Tests


def test_full_parsing_workflow(mock_groq_client):
    """Test a complete parsing workflow."""
    parser = LLAMANewsParser(api_key="test_key")

    # Mock a realistic response
    mock_response = Mock()
    mock_message = Mock()
    mock_tool_call = Mock()
    mock_tool_call.function.arguments = json.dumps(
        {
            "firms": [
                {
                    "firm": "Telefonica",
                    "ticker": "TEF.MC",
                    "shock_type": "financial",
                    "shock_magnitude": "major",
                    "shock_direction": "negative",
                },
                {
                    "firm": "Cellnex",
                    "ticker": "CLNX.MC",
                    "shock_type": "demand",
                    "shock_magnitude": "minor",
                    "shock_direction": "positive",
                },
            ]
        }
    )
    mock_message.tool_calls = [mock_tool_call]
    mock_response.choices = [Mock(message=mock_message)]

    mock_second = Mock()
    mock_second.choices = [Mock(message=Mock(content="Completed analysis"))]

    mock_groq_client.chat.completions.create.side_effect = [
        mock_response,
        mock_second,
    ]

    article = """
    Cellnex tendrá más competencia en Europa. La filial de Telefónica (TEF.MC) 
    Telxius Telecom ha acordado vender su división de torres de telecomunicaciones.
    """

    response_text, firm_shocks = parser.parse_article(article)

    assert len(firm_shocks) == 2
    assert firm_shocks[0].firm == "Telefonica"
    assert firm_shocks[1].firm == "Cellnex"
    assert firm_shocks[0].shock_direction == "negative"
    assert firm_shocks[1].shock_direction == "positive"
