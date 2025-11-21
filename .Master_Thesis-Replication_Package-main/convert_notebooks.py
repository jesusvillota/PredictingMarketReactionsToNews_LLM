#!/usr/bin/env python3
"""
Conversion script to convert Jupyter notebooks to Python scripts.
"""

import json
import re
from pathlib import Path


def convert_markdown_to_comment(markdown_text):
    """Convert markdown/HTML content to Python comments."""
    if not markdown_text.strip():
        return ""
    
    lines = markdown_text.split('\n')
    comment_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # Convert markdown headers to comments
        if stripped.startswith('#'):
            # Markdown header (##, ###, etc.)
            header_level = len(stripped) - len(stripped.lstrip('#'))
            header_text = stripped.lstrip('#').strip()
            # Convert to Python comment with appropriate spacing
            if header_text.startswith('**') and header_text.endswith('**'):
                # Bold header
                header_text = header_text.strip('*')
                comment_lines.append(f"# {header_text}")
            else:
                comment_lines.append(f"# {header_text}")
        elif stripped.startswith('<!--'):
            # HTML comment, keep as comment
            comment_lines.append(f"# {line}")
        elif stripped.startswith('<'):
            # HTML tag, convert to comment
            comment_lines.append(f"# {line}")
        elif stripped:
            # Regular markdown/HTML line
            comment_lines.append(f"# {line}")
        else:
            # Empty line
            comment_lines.append("")
    
    return '\n'.join(comment_lines)


def handle_magic_commands(code):
    """Handle Jupyter magic commands by commenting them out."""
    lines = code.split('\n')
    processed_lines = []
    
    for line in lines:
        stripped = line.strip()
        # Check for magic commands (%reset, %matplotlib, etc.)
        if stripped.startswith('%') and not stripped.startswith('%%'):
            # Single-line magic command
            if stripped.startswith('%reset'):
                processed_lines.append(f"# {line}  # Jupyter magic command removed")
            else:
                processed_lines.append(f"# {line}  # Jupyter magic command")
        elif stripped.startswith('%%'):
            # Cell magic command
            processed_lines.append(f"# {line}  # Jupyter cell magic command")
        else:
            processed_lines.append(line)
    
    return '\n'.join(processed_lines)


def convert_notebook_to_script(notebook_path, output_path):
    """Convert a Jupyter notebook to a Python script."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    script_lines = []
    
    # Add header comment
    notebook_name = Path(notebook_path).stem
    script_lines.append(f'"""')
    script_lines.append(f'Script converted from notebook: {Path(notebook_path).name}')
    script_lines.append(f'Original notebook: {notebook_name}')
    script_lines.append(f'"""')
    script_lines.append('')
    
    # Process cells in order
    for cell in notebook.get('cells', []):
        cell_type = cell.get('cell_type', '')
        source = ''.join(cell.get('source', []))
        
        if cell_type == 'markdown':
            # Convert markdown to comments
            comment_text = convert_markdown_to_comment(source)
            if comment_text.strip():
                script_lines.append(comment_text)
                script_lines.append('')
        
        elif cell_type == 'code':
            # Process code cell
            if source.strip():
                # Handle magic commands
                processed_code = handle_magic_commands(source)
                script_lines.append(processed_code)
                script_lines.append('')
    
    # Write to output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(script_lines))
    
    print(f"Converted {notebook_path.name} -> {output_path.name}")


def main():
    """Convert all notebooks to scripts."""
    base_dir = Path(__file__).parent
    notebooks_dir = base_dir / 'notebooks'
    scripts_dir = base_dir / 'scripts'
    
    # List of notebooks to convert
    notebooks = [
        '0_data_articles.ipynb',
        '1_data_description.ipynb',
        '2_data_tickers.ipynb',
        '3_data_embeddings.ipynb',
        '4_kmeans_clustering.ipynb',
        '5_0_llama_news_parser.ipynb',
        '5_llama_clustering.ipynb',
    ]
    
    for notebook_name in notebooks:
        notebook_path = notebooks_dir / notebook_name
        if notebook_path.exists():
            script_name = notebook_name.replace('.ipynb', '.py')
            output_path = scripts_dir / script_name
            convert_notebook_to_script(notebook_path, output_path)
        else:
            print(f"Warning: {notebook_path} not found")


if __name__ == '__main__':
    main()

