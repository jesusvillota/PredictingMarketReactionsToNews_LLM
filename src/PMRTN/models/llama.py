"""LLAMA-based news parser for extracting firm-specific shocks from articles.

This module provides a wrapper around the Groq API for LLAMA models to parse
Spanish business news articles and extract structured information about
firms, their tickers, and the shocks affecting them.
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from groq import Groq
except ImportError:
    Groq = None


class LLAMAParserError(Exception):
    """Raised when LLAMA parsing operations fail."""

    pass


class FirmShock:
    """Represents a shock affecting a specific firm.

    Attributes:
        firm: Name of the affected firm.
        ticker: Yahoo Finance ticker symbol (e.g., 'IBE.MC' for Spanish firms).
        shock_type: Type of shock ('demand', 'supply', 'financial', 'policy', 'technology').
        shock_magnitude: Magnitude of shock ('minor' or 'major').
        shock_direction: Direction of shock ('positive' or 'negative').
    """

    VALID_SHOCK_TYPES = ["demand", "supply", "financial", "policy", "technology"]
    VALID_MAGNITUDES = ["minor", "major"]
    VALID_DIRECTIONS = ["positive", "negative"]

    def __init__(
        self,
        firm: str,
        ticker: str = "",
        shock_type: str = "",
        shock_magnitude: str = "",
        shock_direction: str = "",
    ) -> None:
        """Initialize a FirmShock.

        Args:
            firm: Name of the affected firm.
            ticker: Yahoo Finance ticker symbol.
            shock_type: Type of shock.
            shock_magnitude: Magnitude of shock.
            shock_direction: Direction of shock.

        Raises:
            ValueError: If shock values are invalid.
        """
        self.firm = firm
        self.ticker = ticker
        self.shock_type = shock_type
        self.shock_magnitude = shock_magnitude
        self.shock_direction = shock_direction

        # Validate shock values if provided
        if shock_type and shock_type not in self.VALID_SHOCK_TYPES:
            raise ValueError(
                f"Invalid shock_type '{shock_type}'. Must be one of {self.VALID_SHOCK_TYPES}"
            )
        if shock_magnitude and shock_magnitude not in self.VALID_MAGNITUDES:
            raise ValueError(
                f"Invalid shock_magnitude '{shock_magnitude}'. Must be one of {self.VALID_MAGNITUDES}"
            )
        if shock_direction and shock_direction not in self.VALID_DIRECTIONS:
            raise ValueError(
                f"Invalid shock_direction '{shock_direction}'. Must be one of {self.VALID_DIRECTIONS}"
            )

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all firm shock attributes.
        """
        return {
            "firm": self.firm,
            "ticker": self.ticker,
            "shock_type": self.shock_type,
            "shock_magnitude": self.shock_magnitude,
            "shock_direction": self.shock_direction,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FirmShock(firm='{self.firm}', ticker='{self.ticker}', "
            f"type='{self.shock_type}', magnitude='{self.shock_magnitude}', "
            f"direction='{self.shock_direction}')"
        )


class LLAMANewsParser:
    """LLAMA-based parser for Spanish business news articles.

    This class uses the Groq API with LLAMA models to parse Spanish business
    news articles and extract structured information about firms and shocks.

    Attributes:
        api_key: Groq API key.
        model: LLAMA model name (default: 'llama3-70b-8192').
        client: Groq client instance.
        max_retries: Maximum number of retries on API failure.
        retry_delay: Delay in seconds between retries.
    """

    DEFAULT_MODEL = "llama3-70b-8192"
    AVAILABLE_MODELS = [
        "llama3-70b-8192",
        "llama3-8b-8192",
        "llama2-70b-4096",
    ]

    SYSTEM_PROMPT = """
You are a function calling LLM that analyses business news in Spanish. 
For every article, you must identify the firms directly affected by the news. Do not include every firm mentioned in the article, only include those that are directly affected by the shocks narrated therein. 
The identified firms must be Spanish and should be publicly listed in the Spanish exchange (their ticker is of the form 'TICKER.MC'). Do not include non-Spanish foreign firms. Do not include Spanish firms that are not publicly traded.
For each identified firm, classify the shocks that affect them (type, magnitude, category). The type of shock can be 'demand', 'supply', 'financial', 'policy', or 'technology'. The magnitude can be 'minor' or 'major'. The direction can be 'positive' or 'negative'.
If a firm is affected neutrally by the news article, don't include it in the analysis.
"""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        """Initialize the LLAMA news parser.

        Args:
            api_key: Groq API key.
            model: LLAMA model name.
            max_retries: Maximum number of retries on API failure.
            retry_delay: Delay in seconds between retries.

        Raises:
            LLAMAParserError: If Groq library not installed or model invalid.
        """
        if Groq is None:
            raise LLAMAParserError(
                "groq package not installed. Install with: pip install groq"
            )

        if model not in self.AVAILABLE_MODELS:
            raise LLAMAParserError(
                f"Invalid model '{model}'. Must be one of {self.AVAILABLE_MODELS}"
            )

        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Initialize Groq client
        try:
            self.client = Groq(api_key=api_key)
        except Exception as e:
            raise LLAMAParserError(f"Failed to initialize Groq client: {str(e)}")

    def _get_tools_definition(self) -> List[Dict[str, Any]]:
        """Get the function calling tools definition for the API.

        Returns:
            List with tool definition dictionary.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "news_parser",
                    "description": """
For every article, you must identify the firms directly affected by the news. Do not include every firm mentioned in the article, only include those that are directly affected by the shocks narrated therein. 
The identified firms must be Spanish and should be publicly listed in the Spanish exchange (their ticker is of the form 'TICKER.MC'). Do not include non-Spanish foreign firms. Do not include Spanish firms that are not publicly traded.
For each identified firm, classify the shocks that affect them (type, magnitude, category). The type of shock can be 'demand', 'supply', 'financial', 'policy', or 'technology'. The magnitude can be 'minor' or 'major'. The direction can be 'positive' or 'negative'.
If a firm is neutral to the article, do NOT include it in the analysis.
""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "firms": {
                                "type": "array",
                                "description": """
List the Spanish firms impacted by the reported news. Such firms must be publicly listed in the Spanish stock exchange and have a stock market ticker of the form TICKER.MC. 
Foreign firms (not listed in the Spanish exchange and whose ticker is not TICKER.MC) are not to be included here. Do not include firms that are mentioned just for contextual comparison but are not directly affected by the events described in the article.
If a firm is neutral to the article, do not include it in the list.
Some times the article mentions explicitly the Spanish ticker of those firms that are directly affected (and hence, the firms to include here). e.g: Iberdrola (IBE.MC).
""",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "firm": {
                                            "type": "string",
                                            "description": "State the Spanish firm (within the list 'firms') in which you will focus the analysis. This firm should be publicly traded in the Spanish exchange with a ticker of the form 'TICKER.MC'.",
                                        },
                                        "ticker": {
                                            "type": "string",
                                            "description": "Specify its stock market ticker of the Spanish firm in Yahoo Finance format (note that Spanish firms' tickers end with '.MC', e.g., ITX.MC for Inditex, ACX.MC for Acerinox, SAN.MC for Banco Santander, NTGY.MC for Naturgy).",
                                        },
                                        "shock_type": {
                                            "type": "string",
                                            "enum": [
                                                "demand",
                                                "supply",
                                                "financial",
                                                "policy",
                                                "technology",
                                            ],
                                            "description": "Classify the type of shock implied by the news article. Choose 'demand' for events impacting consumer demand, 'supply' for events affecting the supply of goods or services, 'financial' for events related to financial markets or conditions, 'policy' for events stemming from changes in government policies or regulations, and 'technology' for events resulting from significant technological advancements or disruptions.",
                                        },
                                        "shock_magnitude": {
                                            "type": "string",
                                            "enum": ["minor", "major"],
                                            "description": "How strong do you expect the shock to be: 'minor' or 'major'?",
                                        },
                                        "shock_direction": {
                                            "type": "string",
                                            "enum": ["positive", "negative"],
                                            "description": """
In what direction do you expect the shock to affect this firm? Choose one of the available options: 'positive' or 'negative'.
Choose 'positive' for beneficial impacts and 'negative' for adverse impacts.
Do not state 'neutral' here. If the firm is neutral to the article, do not include it in the list of firms.
""",
                                        },
                                    },
                                    "required": ["firm"],
                                },
                            },
                        },
                        "required": ["firms"],
                    },
                },
            },
        ]

    def parse_article(
        self,
        article_text: str,
        max_tokens: int = 4096,
    ) -> Tuple[Optional[str], List[FirmShock]]:
        """Parse a single article to extract firm shocks.

        Args:
            article_text: The article text to parse.
            max_tokens: Maximum tokens in response.

        Returns:
            Tuple of (response_text, list of FirmShock objects).

        Raises:
            LLAMAParserError: If parsing fails after all retries.
        """
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": article_text},
        ]

        tools = self._get_tools_definition()

        for attempt in range(self.max_retries):
            try:
                # Call the API
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=max_tokens,
                )

                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls

                # Check if the model called the function
                if tool_calls:
                    # Extract function arguments
                    function_args = json.loads(tool_calls[0].function.arguments)
                    firms_data = function_args.get("firms", [])

                    # Convert to FirmShock objects
                    firm_shocks = []
                    for firm_data in firms_data:
                        try:
                            shock = FirmShock(
                                firm=firm_data.get("firm", ""),
                                ticker=firm_data.get("ticker", ""),
                                shock_type=firm_data.get("shock_type", ""),
                                shock_magnitude=firm_data.get("shock_magnitude", ""),
                                shock_direction=firm_data.get("shock_direction", ""),
                            )
                            firm_shocks.append(shock)
                        except ValueError as e:
                            # Skip invalid shocks but continue processing
                            print(f"Warning: Skipping invalid shock: {e}")

                    # Get the final response (optional)
                    messages.append(response_message)
                    messages.append(
                        {
                            "role": "function",
                            "name": "news_parser",
                            "content": json.dumps([s.to_dict() for s in firm_shocks]),
                        }
                    )

                    try:
                        second_response = self.client.chat.completions.create(
                            model=self.model, messages=messages
                        )
                        response_text = second_response.choices[0].message.content
                    except:
                        response_text = None

                    return response_text, firm_shocks

                else:
                    # No function call made
                    return response_message.content, []

            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(
                        f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {self.retry_delay}s..."
                    )
                    time.sleep(self.retry_delay)
                else:
                    raise LLAMAParserError(
                        f"Failed to parse article after {self.max_retries} attempts: {str(e)}"
                    )

        # Should not reach here
        raise LLAMAParserError("Unexpected error in parse_article")

    def parse_articles_batch(
        self,
        articles: List[str],
        max_tokens: int = 4096,
        verbose: bool = True,
    ) -> List[Tuple[Optional[str], List[FirmShock]]]:
        """Parse multiple articles in batch.

        Args:
            articles: List of article texts to parse.
            max_tokens: Maximum tokens per response.
            verbose: Whether to print progress.

        Returns:
            List of (response_text, firm_shocks) tuples for each article.
        """
        results = []

        for i, article in enumerate(articles):
            if verbose:
                print(f"Processing article {i + 1}/{len(articles)}...")

            try:
                result = self.parse_article(article, max_tokens=max_tokens)
                results.append(result)
            except LLAMAParserError as e:
                print(f"Failed to parse article {i + 1}: {str(e)}")
                results.append((None, []))

            # Add delay between requests to avoid rate limiting
            if i < len(articles) - 1:
                time.sleep(0.5)

        return results

    def parse_dataframe(
        self,
        df: pd.DataFrame,
        article_col: str = "articles",
        max_tokens: int = 4096,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Parse articles from a DataFrame and add parsed data.

        Args:
            df: DataFrame containing articles.
            article_col: Name of column with article text.
            max_tokens: Maximum tokens per response.
            verbose: Whether to print progress.

        Returns:
            DataFrame with added columns:
                - 'llama_response': The LLM response text
                - 'llama_firms': List of extracted firm names
                - 'llama_tickers': List of extracted tickers
                - 'llama_shock_types': List of shock types
                - 'llama_shock_magnitudes': List of shock magnitudes
                - 'llama_shock_directions': List of shock directions
                - 'llama_num_firms': Number of firms identified

        Raises:
            LLAMAParserError: If article_col not found in DataFrame.
        """
        if article_col not in df.columns:
            raise LLAMAParserError(f"Column '{article_col}' not found in DataFrame")

        df = df.copy()

        # Initialize new columns
        df["llama_response"] = None
        df["llama_firms"] = [[] for _ in range(len(df))]
        df["llama_tickers"] = [[] for _ in range(len(df))]
        df["llama_shock_types"] = [[] for _ in range(len(df))]
        df["llama_shock_magnitudes"] = [[] for _ in range(len(df))]
        df["llama_shock_directions"] = [[] for _ in range(len(df))]
        df["llama_num_firms"] = 0

        # Parse each article
        articles = df[article_col].tolist()
        results = self.parse_articles_batch(
            articles, max_tokens=max_tokens, verbose=verbose
        )

        # Populate DataFrame
        for i, (response_text, firm_shocks) in enumerate(results):
            df.at[i, "llama_response"] = response_text
            df.at[i, "llama_firms"] = [s.firm for s in firm_shocks]
            df.at[i, "llama_tickers"] = [s.ticker for s in firm_shocks]
            df.at[i, "llama_shock_types"] = [s.shock_type for s in firm_shocks]
            df.at[i, "llama_shock_magnitudes"] = [
                s.shock_magnitude for s in firm_shocks
            ]
            df.at[i, "llama_shock_directions"] = [
                s.shock_direction for s in firm_shocks
            ]
            df.at[i, "llama_num_firms"] = len(firm_shocks)

        return df


def create_parser(
    api_key: Optional[str] = None,
    model: str = LLAMANewsParser.DEFAULT_MODEL,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> LLAMANewsParser:
    """Create a LLAMA news parser instance.

    Args:
        api_key: Groq API key. If None, will try to read from environment.
        model: LLAMA model name.
        max_retries: Maximum number of retries on API failure.
        retry_delay: Delay in seconds between retries.

    Returns:
        LLAMANewsParser instance.

    Raises:
        LLAMAParserError: If API key not provided and not in environment.
    """
    import os

    if api_key is None:
        api_key = os.getenv("GROQ_API_KEY")
        if api_key is None:
            raise LLAMAParserError(
                "API key must be provided or set in GROQ_API_KEY environment variable"
            )

    return LLAMANewsParser(
        api_key=api_key,
        model=model,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )
