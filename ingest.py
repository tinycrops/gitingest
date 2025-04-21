import base64
import os
import json
import sys
from google import genai
from google.genai import types
from gitingest import ingest as gitingest_ingest

def get_tool_response(name, args):
    """
    Handler for tool calls from the model
    """
    if name == "analyze_file":
        filepath = args.get("filepath", "")
        # Read the file content
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            return {"content": content, "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}
    
    elif name == "list_directory":
        dirpath = args.get("dirpath", "")
        try:
            files = os.listdir(dirpath)
            return {"files": files, "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}
    
    elif name == "extract_patterns":
        content = args.get("content", "")
        # Extract patterns from content (e.g., extract import statements, function signatures)
        # This is just a simple example - real implementation would be more sophisticated
        patterns = [line for line in content.split('\n') if line.startswith('import') or line.startswith('def ')]
        return {"patterns": patterns, "success": True}
    
    return {"error": f"Unknown tool: {name}", "success": False}

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
    
    tools = [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="analyze_file",
                    description="Analyze a specific file in the repository to determine its importance",
                    parameters=genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        properties={
                            "filepath": genai.types.Schema(
                                type=genai.types.Type.STRING,
                                description="Path to the file to analyze"
                            ),
                        },
                        required=["filepath"]
                    ),
                ),
                types.FunctionDeclaration(
                    name="list_directory",
                    description="List files in a directory to explore repository structure",
                    parameters=genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        properties={
                            "dirpath": genai.types.Schema(
                                type=genai.types.Type.STRING,
                                description="Path to the directory to list"
                            ),
                        },
                        required=["dirpath"]
                    ),
                ),
                types.FunctionDeclaration(
                    name="extract_patterns",
                    description="Extract patterns from file content",
                    parameters=genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        properties={
                            "content": genai.types.Schema(
                                type=genai.types.Type.STRING,
                                description="Content to analyze for patterns"
                            ),
                        },
                        required=["content"]
                    ),
                ),
            ],
        )
    ]
    
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=f"""Analyze this repository data and create a comma-separated list of file patterns to exclude. Keep only the most important files that contribute to the core logic and functionality.
                
Repository information:
{repo_data}"""),
            ],
        ),
    ]
    
    generate_content_config = types.GenerateContentConfig(
        tools=tools,
        response_mime_type="application/json",
    )

    # Instead of a direct call, use streaming to handle potential tool calls
    response_stream = client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    
    exclude_patterns = set()
    aggregated_response = ""
    
    for chunk in response_stream:
        if chunk.function_calls:
            for function_call in chunk.function_calls:
                # Process tool call
                tool_name = function_call.name
                tool_args = function_call.args
                
                # Get response from tool
                tool_response = get_tool_response(tool_name, tool_args)
                
                # Send tool response back to model
                contents.append(
                    types.Content(
                        role="model",
                        parts=[
                            types.Part.from_function_call(
                                name=tool_name,
                                args=tool_args,
                            ),
                        ],
                    )
                )
                
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=tool_name,
                                response=tool_response,
                            ),
                        ],
                    )
                )
                
                # Generate a new response with the tool results
                response_stream = client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=generate_content_config,
                )
                break  # Process one tool call at a time
        else:
            # Normal text response
            if chunk.text:
                aggregated_response += chunk.text
    
    # Process final response to get exclude patterns
    try:
        # Try to parse JSON response
        patterns = json.loads(aggregated_response)
        
        # Ensure we have a set of strings, not a list
        if isinstance(patterns, list):
            exclude_patterns = set(patterns)
        else:
            exclude_patterns = {patterns}
    except json.JSONDecodeError:
        # Fallback if the response is not valid JSON
        # Try to extract a CSV list from the text
        possible_patterns = [p.strip() for p in aggregated_response.split(',')]
        exclude_patterns = set(p for p in possible_patterns if p)
    
    return exclude_patterns


def analyze_repository_with_tools(repo_path):
    """
    Analyze a repository with sophisticated tool calling capabilities
    """
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
    
    # First, get basic repository data
    summary, tree, content = gitingest_ingest(repo_path)
    
    # Define tools for repository analysis
    tools = [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="analyze_file",
                    description="Analyze a specific file in the repository to determine its importance",
                    parameters=genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        properties={
                            "filepath": genai.types.Schema(
                                type=genai.types.Type.STRING,
                                description="Path to the file to analyze"
                            ),
                        },
                        required=["filepath"]
                    ),
                ),
                types.FunctionDeclaration(
                    name="list_directory",
                    description="List files in a directory to explore repository structure",
                    parameters=genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        properties={
                            "dirpath": genai.types.Schema(
                                type=genai.types.Type.STRING,
                                description="Path to the directory to list"
                            ),
                        },
                        required=["dirpath"]
                    ),
                ),
                types.FunctionDeclaration(
                    name="extract_patterns",
                    description="Extract patterns from file content",
                    parameters=genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        properties={
                            "content": genai.types.Schema(
                                type=genai.types.Type.STRING,
                                description="Content to analyze for patterns"
                            ),
                        },
                        required=["content"]
                    ),
                ),
            ],
        )
    ]
    
    model = "gemini-2.0-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=f"""Analyze this repository and provide insights on:
1. Key components and their relationships
2. Main technologies and frameworks used
3. Architecture patterns
4. Code quality assessment
5. Improvement recommendations

Repository summary:
{summary}

Directory structure:
{tree}

Sample content:
{content[:5000] if len(content) > 5000 else content}"""),
            ],
        ),
    ]
    
    generate_content_config = types.GenerateContentConfig(
        tools=tools,
    )
    
    print("Analyzing repository with Gemini...\n")
    
    full_analysis = ""
    
    # Process tool calls in a conversation loop
    response_stream = client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    
    for chunk in response_stream:
        if chunk.function_calls:
            for function_call in chunk.function_calls:
                # Process tool call
                tool_name = function_call.name
                tool_args = function_call.args
                
                print(f"Running tool: {tool_name} with args: {tool_args}")
                
                # Get response from tool
                tool_response = get_tool_response(tool_name, tool_args)
                
                # Send tool response back to model
                contents.append(
                    types.Content(
                        role="model",
                        parts=[
                            types.Part.from_function_call(
                                name=tool_name,
                                args=tool_args,
                            ),
                        ],
                    )
                )
                
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=tool_name,
                                response=tool_response,
                            ),
                        ],
                    )
                )
                
                # Generate a new response with the tool results
                response_stream = client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=generate_content_config,
                )
                break  # Process one tool call at a time
        else:
            # Normal text response
            if chunk.text:
                print(chunk.text, end="")
                full_analysis += chunk.text
    
    return full_analysis


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
        print("Usage: python ingest.py <repository_path_or_url> [--analyze]")
        sys.exit(1)
    
    repo_path = sys.argv[1]
    
    # Check for analysis flag
    if len(sys.argv) > 2 and sys.argv[2] == "--analyze":
        # Run the advanced analysis with tool calling
        analysis = analyze_repository_with_tools(repo_path)
        print("\n=== COMPLETE ANALYSIS ===")
        print(analysis)
    else:
        # Run the standard process
        process_repo_with_excludes(repo_path)
