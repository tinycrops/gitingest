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

    # First, get repository data
    summary, tree, content = gitingest_ingest(repo_path)
    repo_data = summary + "\n" + tree + "\n" + content

    model = "gemini-2.0-flash"
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


def process_repo_with_excludes(repo_path):
    """
    Process a repository using gitingest, first getting exclude patterns from Gemini,
    then fetching only the important files
    """
    print(f"Analyzing repository: {repo_path}")
    
    # Get exclude patterns from Gemini
    exclude_patterns = get_important_files(repo_path)
    print(f"Excluding patterns: {exclude_patterns}")
    
    # Fetch only the important files using the exclude patterns
    summary, tree, content = gitingest_ingest(
        source=repo_path,
        exclude_patterns=exclude_patterns
    )
    
    print("\n=== REPOSITORY SUMMARY ===")
    print(summary)
    
    print("\n=== IMPORTANT FILES ===")
    print(tree)
    
    return summary, tree, content


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <repository_path_or_url>")
        sys.exit(1)
    
    repo_path = sys.argv[1]
    process_repo_with_excludes(repo_path)
