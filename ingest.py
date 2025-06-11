import base64
import os
import json
import sys
from google import genai
from google.genai import types
from gitingest import ingest as gitingest_ingest


def get_important_files(repo_path):
    """
    Use Gemini to identify the most important files in a repository
    Returns a list of file patterns to exclude all but the important files
    """
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    # First, get repository data with a very small file size limit for Gemini
    summary, tree, content = gitingest_ingest(
        source=repo_path,
        max_file_size=10 * 1024  # 10KB limit for Gemini to stay under token limits
    )

    # Truncate content to just show file paths and first few lines of each file
    truncated_content = []
    current_file = None
    line_count = 0
    for line in content.split('\n'):
        if line.startswith('=== '):  # New file marker
            if current_file and line_count > 0:
                truncated_content.append(f"... ({line_count} more lines)\n")
            current_file = line
            truncated_content.append(line + '\n')
            line_count = 0
        elif current_file:
            if line_count < 40:  # Only keep first 20 lines of each file
                truncated_content.append(line + '\n')
            line_count += 1
    if current_file and line_count > 0:
        truncated_content.append(f"... ({line_count} more lines)\n")

    # Combine data with truncated content
    repo_data = summary + "\n" + tree + "\n" + "".join(truncated_content)

    model = "gemini-2.5-flash-preview-05-20"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=f"""Analyze this repository and provide ONLY a comma-separated list of file patterns to exclude. Keep only the most important files that contribute to core logic and functionality. Don't include any model files.

Return the response as a simple JSON array of strings, like: ["pattern1", "pattern2", "pattern3"]

Repository data:
{repo_data}"""),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="application/json",
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    
    # Debug: Print raw response before JSON parsing
    print(f"DEBUG: Raw response text: {repr(response.text)}")
    print(f"DEBUG: Raw response length: {len(response.text)}")
    
    # Convert the JSON response to a set of strings
    try:
        exclude_patterns = json.loads(response.text)
        print(f"DEBUG: Gemini response type: {type(exclude_patterns)}")
        print(f"DEBUG: Gemini response content: {exclude_patterns}")
    except json.JSONDecodeError as e:
        print(f"DEBUG: JSON parsing failed: {e}")
        print(f"DEBUG: Attempting to parse as plain text...")
        # If JSON parsing fails, treat as plain text
        if ',' in response.text:
            exclude_patterns = [pattern.strip() for pattern in response.text.split(',')]
        else:
            exclude_patterns = [response.text.strip()]
        print(f"DEBUG: Parsed as text: {exclude_patterns}")
    
    # Handle different response formats from Gemini
    if isinstance(exclude_patterns, list):
        # Response is a list of patterns
        exclude_patterns_set = set(exclude_patterns)
    elif isinstance(exclude_patterns, dict):
        # Response is a dictionary, extract values or keys that look like file patterns
        patterns = []
        # Try to extract patterns from common dictionary structures
        if 'files' in exclude_patterns:
            patterns.extend(exclude_patterns['files'])
        elif 'patterns' in exclude_patterns:
            patterns.extend(exclude_patterns['patterns'])
        elif 'exclude' in exclude_patterns:
            patterns.extend(exclude_patterns['exclude'])
        else:
            # If no common keys, try to extract all string values
            for value in exclude_patterns.values():
                if isinstance(value, list):
                    patterns.extend(value)
                elif isinstance(value, str):
                    patterns.append(value)
        exclude_patterns_set = set(patterns)
    elif isinstance(exclude_patterns, str):
        # Response is a single string, possibly comma-separated
        if ',' in exclude_patterns:
            exclude_patterns_set = set(pattern.strip() for pattern in exclude_patterns.split(','))
        else:
            exclude_patterns_set = {exclude_patterns}
    else:
        # Fallback: convert to string and treat as single pattern
        exclude_patterns_set = {str(exclude_patterns)}
    
    return exclude_patterns_set


def write_to_markdown(summary, tree, content, output_file="repo_analysis.md"):
    """
    Write the repository analysis data to a markdown file
    """
    with open(output_file, "w", encoding='utf-8') as f:
        f.write("# Repository Analysis\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"```\n{summary}\n```\n\n")
        
        f.write("## Important Files\n\n")
        f.write(f"```\n{tree}\n```\n\n")
        
        f.write("## Content\n\n")
        f.write(f"```\n{content}\n```\n\n")
    
    return output_file


def process_repo_with_excludes(repo_path, output_file="repo_analysis.md"):
    """
    Process a repository using gitingest, first getting exclude patterns from Gemini,
    then fetching only the important files
    """
    print(f"Analyzing repository: {repo_path}")
    
    # Get exclude patterns from Gemini (using 50KB limit)
    exclude_patterns = get_important_files(repo_path)
    print(f"Excluding patterns: {exclude_patterns}")
    
    # Fetch repository data without applying exclude patterns to get the full tree
    summary, tree, content = gitingest_ingest(
        source=repo_path,
        exclude_patterns=exclude_patterns
    )
    
    print("\n=== REPOSITORY SUMMARY ===")
    print(summary)
    
    print("\n=== IMPORTANT FILES ===")
    print(tree)
    
    # Write the data to a markdown file
    markdown_file = write_to_markdown(summary, tree, content, output_file)
    print(f"\nRepository analysis written to: {markdown_file}")
    
    return summary, tree, content


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <repository_path_or_url> [output_file]")
        sys.exit(1)
    
    repo_path = sys.argv[1]
    
    # Allow specifying an output file as a second argument
    output_file = "repo_analysis.md"
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    process_repo_with_excludes(repo_path, output_file)
