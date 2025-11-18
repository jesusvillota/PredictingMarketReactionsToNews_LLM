"""LLAMA news parser using Groq API for structured news analysis."""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Any

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from src.config import get_paths, get_logger

logger = get_logger("llm.llama_parser")


def get_groq_client(api_key: Optional[str] = None) -> 'Groq':
    """
    Get Groq API client.
    
    Args:
        api_key: Groq API key. If None, reads from environment variable GROQ_API_KEY.
    
    Returns:
        Groq client instance
    
    Raises:
        ImportError: If groq package is not installed
        ValueError: If API key is not provided
    """
    if not GROQ_AVAILABLE:
        raise ImportError(
            "groq package is not installed. Install it with: pip install groq"
        )
    
    if api_key is None:
        api_key = os.environ.get('GROQ_API_KEY')
    
    if not api_key:
        raise ValueError(
            "Groq API key not found. Set GROQ_API_KEY environment variable "
            "or pass api_key parameter."
        )
    
    return Groq(api_key=api_key)


def news_parser(firms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Placeholder function for news parsing.
    
    This function processes the structured output from the LLM.
    
    Args:
        firms: List of firm dictionaries with parsed information
    
    Returns:
        List of processed firm information
    """
    response = []
    for firm in firms:
        response.append({
            "firm": firm.get("firm", ""),
            "ticker": firm.get("ticker", ""),
            "shock_type": firm.get("shock_type", ""),
            "shock_magnitude": firm.get("shock_magnitude", ""),
            "shock_direction": firm.get("shock_direction", ""),
        })
    return response


def run_conversation(
    user_prompt: str,
    client: 'Groq',
    model: str = 'llama3-70b-8192'
) -> tuple:
    """
    Run a conversation with the LLM to parse news articles.
    
    Args:
        user_prompt: The news article text to analyze
        client: Groq client instance
        model: Model name to use
    
    Returns:
        Tuple of (completion_text, structured_output)
    """
    messages = [
        {
            "role": "system",
            "content": """You are a function calling LLM that analyses business news in Spanish. 
            Only identify the firms whose ticker is specified in parenthesis. Do not include firms whose ticker is not specified in parenthesis.
            For example, if an articles mentions 'Firm X (FIRMX.MC) will do Y and Firm Z will do W', you should only include 'Firm X' in the list of firms."""
        },
        {
            "role": "user",
            "content": user_prompt,
        }
    ]
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "news_parser",
                "description": """Analyze the impact of a business news article on the firms affected by it (that is, firms whose ticker is specified in parenthesis).
                Only identify the firms whose ticker is specified in parenthesis. Do not include firms whose ticker is not specified in parenthesis.
                For example, if an articles mentions 'Firm X (FIRMX.MC) will do Y and Firm Z will do W', you should only include 'Firm X' in the list of firms.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "firms": {
                            "type": "array",
                            "description": """Only identify the firms whose ticker is specified in parenthesis. Do not include firms whose ticker is not specified in parenthesis.
                            For example, if an articles mentions 'Firm X (FIRMX.MC) will do Y and Firm Z will do W', you should only include 'Firm X' in the list of firms.""",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "firm": {
                                        "type": "string",
                                        "description": "State the Spanish firm (within the list 'firms') in which you will focus the analysis.",
                                    },
                                    "ticker": {
                                        "type": "string",
                                        "description": "Specify its stock market ticker of the Spanish firm in Yahoo Finance format (note that Spanish firms' tickers end with '.MC', e.g., ITX.MC for Inditex, ACX.MC for Acerinox, SAN.MC for Banco Santander, NTGY.MC for Naturgy).",
                                    },
                                    "shock_type": {
                                        "type": "string",
                                        "enum": ["demand", "supply", "financial", "policy", "technology"],
                                        "description": "Classify the type of shock implied by the news article.",
                                    },
                                    "shock_magnitude": {
                                        "type": "string",
                                        "enum": ["minor", "major"],
                                        "description": "How strong do you expect the shock to be: 'minor' or 'major'?",
                                    },
                                    "shock_direction": {
                                        "type": "string",
                                        "enum": ["positive", "negative"],
                                        "description": "In what direction do you expect the shock to affect this firm? Choose 'positive' for beneficial impacts and 'negative' for adverse impacts. Do not state 'neutral' here. If the firm is neutral to the article, do not include it in the list of firms.",
                                    },
                                },
                                "required": ["firm"],
                            }
                        }
                    },
                    "required": ["firms"],
                }
            }
        }
    ]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=8096
    )
    
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    
    if tool_calls:
        available_functions = {
            "news_parser": news_parser,
        }
        messages.append(response_message)
        
        function_response = None
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                firms=function_args.get("firms")
            )
            messages.append(
                {
                    "role": "function",
                    "name": function_name,
                    "content": json.dumps(function_response),
                }
            )
        
        second_response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        
        return second_response.choices[0].message.content, function_response
    
    return response_message.content, None


def parse_articles_with_llama(
    articles_df: pd.DataFrame,
    api_key: Optional[str] = None,
    model: str = 'llama3-70b-8192',
    save_output: bool = True,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Parse articles using LLAMA via Groq API.
    
    Args:
        articles_df: DataFrame with articles to parse
        api_key: Groq API key. If None, reads from environment variable.
        model: Model name to use
        save_output: Whether to save output to CSV
        output_path: Path to save output. If None, uses config default.
    
    Returns:
        DataFrame with parsed article information
    """
    if output_path is None:
        path_manager = get_paths()
        output_path = path_manager.get_raw_data_path()
    
    logger.info("Initializing Groq client")
    client = get_groq_client(api_key)
    
    logger.info(f"Parsing {len(articles_df)} articles with LLAMA")
    
    results = []
    errors = []
    
    for idx, row in articles_df.iterrows():
        try:
            article_text = row.get('articles', '')
            completion_text, structured_output = run_conversation(
                article_text,
                client,
                model
            )
            
            if structured_output:
                for firm_info in structured_output:
                    results.append({
                        'article_idx': idx,
                        'publ_datetime': row.get('publ_datetime'),
                        **firm_info
                    })
        except Exception as e:
            logger.error(f"Error processing article {idx}: {e}")
            errors.append({'article_idx': idx, 'error': str(e)})
    
    logger.info(f"Successfully parsed {len(results)} firm-article pairs")
    logger.info(f"Encountered {len(errors)} errors")
    
    result_df = pd.DataFrame(results)
    
    if save_output:
        output_file = output_path / 'LLAMA_parsed_news.csv'
        result_df.to_csv(output_file, index=False)
        logger.info(f"Saved parsed news to {output_file}")
    
    return result_df

