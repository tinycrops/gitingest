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
                types.Part.from_text(text=f"""write a comma separated list of files to exclude from this repo all but the most important files that contribute to the core logic and functionality
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
    
    # Convert the JSON list to a set of strings
    exclude_patterns = json.loads(response.text)
    # Ensure we have a set of strings, not a list
    if isinstance(exclude_patterns, list):
        exclude_patterns_set = set(exclude_patterns)
    else:
        exclude_patterns_set = {exclude_patterns}
    
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
