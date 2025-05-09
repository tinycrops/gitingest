# Repository Analysis

## Summary

```
Repository: tinycrops/digitalaudioworkstation
Branch: new-version
Files analyzed: 17

Estimated tokens: 13.7k
```

## Important Files

```
Directory structure:
└── tinycrops-digitalaudioworkstation/
    ├── app.py
    ├── object_oriented_agentic_approach/
    │   ├── launch_sandbox_container.py
    │   └── resources/
    │       ├── data/
    │       ├── diagrams/
    │       │   └── src/
    │       ├── docker/
    │       ├── object_oriented_agents/
    │       │   ├── core_classes/
    │       │   │   ├── agent_signature.py
    │       │   │   ├── base_agent.py
    │       │   │   ├── chat_messages.py
    │       │   │   ├── tool_interface.py
    │       │   │   └── tool_manager.py
    │       │   ├── services/
    │       │   │   ├── language_model_interface.py
    │       │   │   ├── openai_factory.py
    │       │   │   └── openai_language_model.py
    │       │   └── utils/
    │       │       ├── logger.py
    │       │       └── openai_util.py
    │       └── registry/
    │           ├── agents/
    │           │   ├── file_access_agent.py
    │           │   └── python_code_exec_agent.py
    │           └── tools/
    │               ├── file_access_tool.py
    │               ├── python_code_interpreter_tool.py
    │               └── retrieve_output_tool.py
    ├── processed_outputs/
    ├── static/
    ├── templates/
    └── uploads/

```

## Content

```
================================================
File: app.py
================================================
from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import uuid # For unique filenames
import sys
import logging

# --- Path Setup ---
# Add the workspace root to sys.path to allow imports from object_oriented_agentic_approach
# This assumes app.py is in the workspace root.
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(WORKSPACE_ROOT)
# Add the parent of object_oriented_agentic_approach if object_oriented_agentic_approach itself isn't directly importable
# This is often needed if 'resources' is not a package.
# A more robust solution is to make object_oriented_agentic_approach.resources a proper package.
sys.path.append(os.path.join(WORKSPACE_ROOT, "object_oriented_agentic_approach"))


# --- Agent and Tool Imports ---
try:
    from object_oriented_agentic_approach.resources.registry.agents.file_access_agent import FileAccessAgent
    from object_oriented_agentic_approach.resources.registry.agents.python_code_exec_agent import PythonExecAgent
    from object_oriented_agentic_approach.resources.registry.tools.retrieve_output_tool import RetrieveOutputTool
    # FileAccessTool is used by FileAccessAgent internally
    # PythonExecTool is used by PythonCodeExecAgent internally
except ImportError as e:
    logging.error(f"Failed to import agent/tool modules: {e}")
    logging.error("Ensure that app.py is in the correct workspace root directory and that Python can find the 'object_oriented_agentic_approach' modules.")
    # Exit if core components can't be imported, as the app won't function.
    sys.exit(f"ImportError: {e}. Please check PYTHONPATH and file structure.")


app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = os.path.join(WORKSPACE_ROOT, 'uploads')
PROCESSED_OUTPUTS_FOLDER = os.path.join(WORKSPACE_ROOT, 'processed_outputs')
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'webm'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_OUTPUTS_FOLDER'] = PROCESSED_OUTPUTS_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_OUTPUTS_FOLDER, exist_ok=True)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Get a logger instance for Flask app specific logs if needed, or use Flask's default.
# For agent logs, they use their own logger instances.

# --- Helper Function ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio_route():
    logging.info("Received /process_audio request.")
    try:
        # Instantiate agents and tools per request for statelessness
        file_ingestion_agent = FileAccessAgent()
        # Ensure PythonCodeExecAgent is the one updated for audio
        audio_processing_agent = PythonExecAgent(model_name='o3-mini', reasoning_effort='high') 
        retrieve_output_tool = RetrieveOutputTool()
        logging.info("Agents and tools instantiated for request.")
    except Exception as e:
        logging.error(f"Error instantiating agents/tools for request: {e}", exc_info=True)
        return jsonify({"error": "Backend agent/tool initialization failed. Check server logs."}), 500

    if 'audioFile' not in request.files:
        logging.warning("No audio file part in request.")
        return jsonify({"error": "No audio file part"}), 400
    
    file = request.files['audioFile']
    prompt_text = request.form.get('promptText', '').strip()

    if not prompt_text:
        logging.warning("Empty prompt text in request.")
        return jsonify({"error": "Prompt text cannot be empty."}), 400

    if file.filename == '':
        logging.warning("No selected audio file in request.")
        return jsonify({"error": "No selected audio file"}), 400

    host_uploaded_audio_path = "" # Initialize to ensure it's defined for cleanup
    if file and allowed_file(file.filename):
        try:
            original_filename = file.filename
            # Sanitize filename slightly (though uuid makes it unique anyway)
            safe_original_filename = "".join(c for c in original_filename if c.isalnum() or c in ('.', '_', '-')).strip()
            if not safe_original_filename: safe_original_filename = "uploaded_audio" # fallback
            
            temp_filename = str(uuid.uuid4()) + "_" + safe_original_filename
            host_uploaded_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
            file.save(host_uploaded_audio_path)
            logging.info(f"Uploaded audio saved to: {host_uploaded_audio_path}")

            # 1. FileAccessAgent: Prepare audio
            logging.info(f"Tasking FileAccessAgent for: {host_uploaded_audio_path}")
            file_access_task_str = f"Prepare the audio file located at host path '{host_uploaded_audio_path}' for processing."
            ingestion_output_str = file_ingestion_agent.task(file_access_task_str)
            logging.info(f"FileAccessAgent output: {ingestion_output_str}")

            if not ingestion_output_str or ingestion_output_str.lower().startswith("error"):
                raise Exception(f"File Ingestion Failed: {ingestion_output_str}")

            # 2. AudioProcessingAgent: Process audio
            # The FileAccessAgent output (metadata & container path) is the context for the AudioProcessingAgent
            audio_processing_agent.add_context(ingestion_output_str)
            logging.info(f"Tasking AudioProcessingAgent with prompt: '{prompt_text}' and context from FileAccessAgent.")
            
            container_output_info_str = audio_processing_agent.task(prompt_text)
            logging.info(f"AudioProcessingAgent raw output: {container_output_info_str}")

            if not container_output_info_str or \
               container_output_info_str.lower().startswith("[error]") or \
               container_output_info_str.lower().startswith("error:"):
                raise Exception(f"Audio Processing Failed: {container_output_info_str}")

            # 3. RetrieveOutputTool: Get processed files
            processed_files_info = []
            candidate_paths = []
            
            # First try to split by newlines in a more robust way
            for line in container_output_info_str.splitlines():
                line = line.strip()
                if line:
                    candidate_paths.append(line)
            
            # Fall back to manual split if that didn't work
            if not candidate_paths:
                candidate_paths = [p.strip() for p in container_output_info_str.split('\\n') if p.strip()]
                if not candidate_paths:
                    candidate_paths = [p.strip() for p in container_output_info_str.split('\n') if p.strip()]
            
            actual_paths_to_retrieve = []
            # Look for file paths in the output
            for cp in candidate_paths:
                # Check for the output directory path
                path_start_index = cp.find("/home/sandboxuser/output_audio/")
                if path_start_index != -1:
                    # Extract only the path part (stop at spaces, quotes or other delimiters)
                    path_part = cp[path_start_index:]
                    # Handle potential quotes or other delimiters
                    for delimiter in [' ', '"', "'", ')', ':', ';', ',']:
                        if delimiter in path_part:
                            path_part = path_part.split(delimiter)[0]
                    
                    # Clean up the path
                    path_part = path_part.rstrip('.,;!?')
                    if path_part.endswith("'") or path_part.endswith('"'):
                        path_part = path_part[:-1]
                    
                    actual_paths_to_retrieve.append(path_part)
                elif cp.startswith("/home/sandboxuser/output_audio/"):
                    # Same cleaning for paths that start at the beginning of the line
                    path_part = cp
                    for delimiter in [' ', '"', "'", ')', ':', ';', ',']:
                        if delimiter in path_part:
                            path_part = path_part.split(delimiter)[0]
                    
                    path_part = path_part.rstrip('.,;!?')
                    if path_part.endswith("'") or path_part.endswith('"'):
                        path_part = path_part[:-1]
                    
                    actual_paths_to_retrieve.append(path_part)
                # Check if the line mentions 'saved to:' or similar
                elif "saved to:" in cp.lower() or "saved at:" in cp.lower() or "output file:" in cp.lower():
                    for substr in cp.split():
                        if substr.startswith("/home/sandboxuser/output_audio/"):
                            path_part = substr.rstrip('.,;!?')
                            if path_part.endswith("'") or path_part.endswith('"'):
                                path_part = path_part[:-1]
                            actual_paths_to_retrieve.append(path_part)
            
            # Filter out any invalid looking paths
            actual_paths_to_retrieve = [p for p in actual_paths_to_retrieve if "/" in p and not p.endswith("/")]
            
            unique_paths_to_retrieve = sorted(list(set(actual_paths_to_retrieve)))
            logging.info(f"Unique container paths to retrieve: {unique_paths_to_retrieve}")

            if not unique_paths_to_retrieve:
                # This might not be an error if the LLM just chatted.
                # The PythonExecAgent prompt asks for file paths (7,8) but LLM might not always comply.
                logging.warning(f"Audio processing agent did not return a clear output file path. Agent raw output: {container_output_info_str}")
                # Return the raw agent output if no files were found.
                return jsonify({"message": "Processing finished, but no specific files were identified for retrieval.", "agent_raw_output": container_output_info_str, "processed_files": []})


            for container_path in unique_paths_to_retrieve:
                logging.info(f"Retrieving from container: {container_path}")
                retrieval_args = {
                    "container_file_path": container_path,
                    "host_target_dir": app.config['PROCESSED_OUTPUTS_FOLDER']
                }
                retrieved_host_path_msg = retrieve_output_tool.run(retrieval_args)
                logging.info(f"Retrieval tool output for {container_path}: {retrieved_host_path_msg}")

                if retrieved_host_path_msg.lower().startswith("error"):
                    logging.error(f"Failed to retrieve {container_path}: {retrieved_host_path_msg}")
                    processed_files_info.append({"container_path": container_path, "error": retrieved_host_path_msg})
                else:
                    actual_host_path = retrieved_host_path_msg.split("File retrieved to: ", 1)[-1].strip()
                    filename = os.path.basename(actual_host_path)
                    file_url = f"/processed/{filename}"
                    file_type = ("audio" if any(filename.lower().endswith(ext) for ext in (ALLOWED_EXTENSIONS | {'.ogg', '.webm'}))
                                 else "image" if any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif'])
                                 else "unknown")
                    processed_files_info.append({
                        "url": file_url,
                        "filename": filename,
                        "type": file_type,
                        "original_container_path": container_path
                    })
            
            if not any(info.get('url') for info in processed_files_info) and unique_paths_to_retrieve:
                 # If paths were identified but all retrievals failed
                raise Exception(f"Identified output files but failed to retrieve any: {unique_paths_to_retrieve}")

            logging.info(f"Successfully processed request. Processed files: {processed_files_info}")
            return jsonify({"message": "Processing successful", "processed_files": processed_files_info, "agent_raw_output": container_output_info_str})

        except Exception as e:
            logging.error(f"Error during audio processing pipeline: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500
        finally:
            # Clean up originally uploaded temp file
            if host_uploaded_audio_path and os.path.exists(host_uploaded_audio_path):
                try:
                    os.remove(host_uploaded_audio_path)
                    logging.info(f"Cleaned up temporary uploaded file: {host_uploaded_audio_path}")
                except OSError as e_remove:
                    logging.error(f"Error cleaning up temp file {host_uploaded_audio_path}: {e_remove}")
    else:
        logging.warning(f"File type not allowed for filename: {file.filename if file else 'N/A'}")
        return jsonify({"error": "File type not allowed"}), 400

# Route to serve processed files from PROCESSED_OUTPUTS_FOLDER
@app.route('/processed/<path:filename>')
def serve_processed_file(filename):
    logging.debug(f"Serving processed file: {filename} from {app.config['PROCESSED_OUTPUTS_FOLDER']}")
    return send_from_directory(app.config['PROCESSED_OUTPUTS_FOLDER'], filename, as_attachment=False)

if __name__ == '__main__':
    print("--- AI Audio Processor ---")
    print("This Flask application provides a frontend for the AI-powered audio processing.")
    print("Please ensure the Docker sandbox container ('sandbox') is running.")
    print("You can start it by navigating to 'object_oriented_agentic_approach/' and running: ")
    print("  python launch_sandbox_container.py")
    print(f"Uploads will be stored temporarily in: {UPLOAD_FOLDER}")
    print(f"Processed files will be made available from: {PROCESSED_OUTPUTS_FOLDER}")
    print("--------------------------")
    app.run(debug=True, port=5001) 


================================================
File: object_oriented_agentic_approach/launch_sandbox_container.py
================================================
import subprocess
import sys
import os

DOCKER_IMAGE = "python_sandbox:latest"
CONTAINER_NAME = "sandbox"
DOCKERFILE_DIR = os.path.join(os.path.dirname(__file__), "resources/docker")


def image_exists(image_name):
    result = subprocess.run([
        "docker", "images", "-q", image_name
    ], capture_output=True, text=True)
    return result.stdout.strip() != ""


def container_running(container_name):
    result = subprocess.run([
        "docker", "ps", "-q", "-f", f"name=^{container_name}$"
    ], capture_output=True, text=True)
    return result.stdout.strip() != ""


def container_exists(container_name):
    result = subprocess.run([
        "docker", "ps", "-aq", "-f", f"name=^{container_name}$"
    ], capture_output=True, text=True)
    return result.stdout.strip() != ""


def build_image():
    print(f"Building Docker image '{DOCKER_IMAGE}'...")
    result = subprocess.run([
        "docker", "build", "-t", DOCKER_IMAGE, DOCKERFILE_DIR
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error building Docker image:")
        print(result.stderr)
        sys.exit(1)
    print("Docker image built successfully.")


def run_container():
    print(f"Running container '{CONTAINER_NAME}'...")
    # Remove any stopped container with the same name
    if container_exists(CONTAINER_NAME) and not container_running(CONTAINER_NAME):
        subprocess.run(["docker", "rm", CONTAINER_NAME], capture_output=True)
    result = subprocess.run([
        "docker", "run", "-d",
        "--name", CONTAINER_NAME,
        "--network", "none",
        "--cap-drop", "all",
        "--pids-limit", "64",
        "--tmpfs", "/tmp:rw,size=64M",
        DOCKER_IMAGE, "sleep", "infinity"
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error running Docker container:")
        print(result.stderr)
        sys.exit(1)
    print(f"Container started with ID: {result.stdout.strip()}")


def main():
    if not image_exists(DOCKER_IMAGE):
        build_image()
    else:
        print(f"Docker image '{DOCKER_IMAGE}' already exists.")

    if container_running(CONTAINER_NAME):
        print(f"Container '{CONTAINER_NAME}' is already running.")
        result = subprocess.run([
            "docker", "ps", "-q", "-f", f"name=^{CONTAINER_NAME}$"
        ], capture_output=True, text=True)
        print(f"Container ID: {result.stdout.strip()}")
    else:
        run_container()


if __name__ == "__main__":
    main() 





================================================
File: object_oriented_agentic_approach/resources/object_oriented_agents/core_classes/agent_signature.py
================================================
# object_oriented_agents/core_classes/agent_signature.py

from typing import Optional, Dict, Any, List
from .tool_manager import ToolManager

class AgentSignature:
    """
    Encapsulates the logic to produce an agent's 'signature' data:
    - The developer prompt
    - The model name
    - The list of tool definitions
    - The default reasoning effort (if any)
    """

    def __init__(self, developer_prompt: str, model_name: str, tool_manager: Optional[ToolManager] = None, reasoning_effort: Optional[str] = None):
        self.developer_prompt = developer_prompt
        self.model_name = model_name
        self.tool_manager = tool_manager
        self.reasoning_effort = reasoning_effort

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a dictionary containing:
          1. The developer prompt
          2. The model name
          3. A list of tool definitions (function schemas)
          4. The default reasoning effort if defined
        """
        if self.tool_manager:
            # Each item in get_tool_definitions() looks like {"type": "function", "function": {...}}
            tool_definitions = self.tool_manager.get_tool_definitions()
            functions = [t for t in tool_definitions]
        else:
            functions = []

        signature_dict = {
            "developer_prompt": self.developer_prompt,
            "model_name": self.model_name,
            "tools": functions
        }
        if self.reasoning_effort is not None:
            signature_dict["reasoning_effort"] = self.reasoning_effort

        return signature_dict


================================================
File: object_oriented_agentic_approach/resources/object_oriented_agents/core_classes/base_agent.py
================================================
# object_oriented_agents/core_classes/base_agent.py

from abc import ABC, abstractmethod
from typing import Optional
from .chat_messages import ChatMessages
from .tool_manager import ToolManager
from ..utils.logger import get_logger
from ..services.language_model_interface import LanguageModelInterface
from .agent_signature import AgentSignature


class BaseAgent(ABC):
    """
    An abstract base agent that defines the high-level approach to handling user tasks
    and orchestrating calls to the OpenAI API.
    """

    def __init__(
            self,
            developer_prompt: str,
            model_name: str,
            logger=None,
            language_model_interface: LanguageModelInterface = None,
            reasoning_effort: Optional[str] = None
    ):
        self.developer_prompt = developer_prompt
        self.model_name = model_name
        self.messages = ChatMessages(developer_prompt)
        self.tool_manager: Optional[ToolManager] = None
        self.logger = logger or get_logger(self.__class__.__name__)
        self.language_model_interface = language_model_interface
        self.reasoning_effort = reasoning_effort

    @abstractmethod
    def setup_tools(self) -> None:
        pass

    def add_context(self, content: str) -> None:
        self.logger.debug(f"Adding context: {content}")
        self.messages.add_user_message(content)

    def add_message(self, content: str) -> None:
        self.logger.debug(f"Adding user message: {content}")
        self.messages.add_user_message(content)

    def task(self, user_task: str, tool_call_enabled: bool = True, return_tool_response_as_is: bool = False,
             reasoning_effort: Optional[str] = None) -> str:
        # Use the reasoning_effort provided in the method call if present, otherwise fall back to the agent's default
        final_reasoning_effort = reasoning_effort if reasoning_effort is not None else self.reasoning_effort

        if self.language_model_interface is None:
            error_message = "Error: Cannot execute task without the LanguageModelInterface."
            self.logger.error(error_message)
            raise ValueError(error_message)

        self.logger.debug(f"Starting task: {user_task} (tool_call_enabled={tool_call_enabled})")

        # Add user message
        self.add_message(user_task)

        tools = []
        if tool_call_enabled and self.tool_manager:
            tools = self.tool_manager.get_tool_definitions()
            self.logger.debug(f"Tools available: {tools}")

        # Build parameter dict and include reasoning_effort only if not None
        params = {
            "model": self.model_name,
            "messages": self.messages.get_messages(),
            "tools": tools
        }
        if final_reasoning_effort is not None:
            params["reasoning_effort"] = final_reasoning_effort

        self.logger.debug("Sending request to language model interface...")
        response = self.language_model_interface.generate_completion(**params)

        tool_calls = response.choices[0].message.tool_calls
        if tool_call_enabled and self.tool_manager and tool_calls:
            self.logger.debug(f"Tool calls requested: {tool_calls}")
            return self.tool_manager.handle_tool_call_sequence(
                response,
                return_tool_response_as_is,
                self.messages,
                self.model_name,
                reasoning_effort=final_reasoning_effort
            )

        # No tool call, normal assistant response
        response_message = response.choices[0].message.content
        self.messages.add_assistant_message(response_message)
        self.logger.debug("Task completed successfully.")
        return response_message

    def signature(self) -> dict:
        """
        Return a dictionary with:
        - The developer prompt
        - The model name
        - The tool definitions (function schemas)
        - The default reasoning effort if set
        """
        signature_obj = AgentSignature(
            developer_prompt=self.developer_prompt,
            model_name=self.model_name,
            tool_manager=self.tool_manager,
            reasoning_effort=self.reasoning_effort
        )
        return signature_obj.to_dict()


================================================
File: object_oriented_agentic_approach/resources/object_oriented_agents/core_classes/chat_messages.py
================================================
# object_oriented_agents/core_classes/chat_messages.py
from typing import List, Dict

class ChatMessages:
    """
    Stores all messages in a conversation (developer, user, assistant).
    """

    def __init__(self, developer_prompt: str):
        self.messages: List[Dict[str, str]] = []
        self.add_developer_message(developer_prompt)

    def add_developer_message(self, content: str) -> None:
        self.messages.append({"role": "developer", "content": content})

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def get_messages(self) -> List[Dict[str, str]]:
        return self.messages


================================================
File: object_oriented_agentic_approach/resources/object_oriented_agents/core_classes/tool_interface.py
================================================
# object_oriented_agents/core_classes/tool_interface.py
from abc import ABC, abstractmethod
from typing import Dict, Any

class ToolInterface(ABC):
    """
    An abstract class for any 'tool' that an agent can call.
    Every tool must provide two things:
    1) A definition (in JSON schema format) as expected by OpenAI function calling specifications.
    2) A 'run' method to handle the logic given the arguments.
    """

    @abstractmethod
    def get_definition(self) -> Dict[str, Any]:
        """
        Return the JSON/dict definition of the tool's function.
        Example:
        {
            "function": {
                "name": "<tool_function_name>",
                "description": "<what this function does>",
                "parameters": { <JSON schema> }
            }
        }
        """
        pass

    @abstractmethod
    def run(self, arguments: Dict[str, Any]) -> str:
        """
        Execute the tool using the provided arguments and return a result as a string.
        """
        pass


================================================
File: object_oriented_agentic_approach/resources/object_oriented_agents/core_classes/tool_manager.py
================================================
# object_oriented_agents/core_classes/tool_manager.py

import json
from typing import Dict, Any, List, Optional
from .chat_messages import ChatMessages
from .tool_interface import ToolInterface
from ..utils.logger import get_logger
from ..services.language_model_interface import LanguageModelInterface

class ToolManager:
    """
    Manages one or more tools. Allows you to:
      - Register multiple tools
      - Retrieve their definitions
      - Invoke the correct tool by name
      - Handle the entire tool call sequence
    """

    def __init__(self, logger=None, language_model_interface: LanguageModelInterface = None):
        self.tools = {}
        self.logger = logger or get_logger(self.__class__.__name__)
        self.language_model_interface = language_model_interface

    def register_tool(self, tool: ToolInterface) -> None:
        """
        Register a tool by using its function name as the key.
        """
        tool_def = tool.get_definition()
        tool_name = tool_def["function"]["name"]
        self.tools[tool_name] = tool
        self.logger.debug(f"Registered tool '{tool_name}': {tool_def}")

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Return the list of tool definitions in the format expected by the OpenAI API.
        """
        definitions = []
        for name, tool in self.tools.items():
            tool_def = tool.get_definition()["function"]
            self.logger.debug(f"Tool definition retrieved for '{name}': {tool_def}")
            definitions.append({"type": "function", "function": tool_def})
        return definitions

    def handle_tool_call_sequence(
        self,
        response,
        return_tool_response_as_is: bool,
        messages: ChatMessages,
        model_name: str,
        reasoning_effort: Optional[str] = None
    ) -> str:
        """
        If the model wants to call a tool, parse the function arguments, invoke the tool,
        then optionally return the tool's raw output or feed it back to the model for a final answer.
        """
        # We take the first tool call from the model’s response
        first_tool_call = response.choices[0].message.tool_calls[0]
        tool_name = first_tool_call.function.name
        self.logger.info(f"Handling tool call: {tool_name}")

        args = json.loads(first_tool_call.function.arguments)
        self.logger.info(f"Tool arguments: {args}")

        if tool_name not in self.tools:
            error_message = f"Error: The requested tool '{tool_name}' is not registered."
            self.logger.error(error_message)
            raise ValueError(error_message)

        # 1. Invoke the tool
        self.logger.debug(f"Invoking tool '{tool_name}'")
        tool_response = self.tools[tool_name].run(args)
        self.logger.info(f"Tool '{tool_name}' response: {tool_response}")

        # If returning the tool response "as is," just store and return it
        if return_tool_response_as_is:
            self.logger.debug("Returning tool response as-is without further LLM calls.")
            messages.add_assistant_message(tool_response)
            return tool_response

        self.logger.debug(f"Tool call: {first_tool_call}")
        # Otherwise, feed the tool's response back to the LLM for a final answer
        function_call_result_message = {
            "role": "tool",
            "content": tool_response,
            "tool_call_id": first_tool_call.id
        }

        complete_payload = messages.get_messages()
        complete_payload.append(response.choices[0].message)
        complete_payload.append(function_call_result_message)

        self.logger.debug("Calling the model again with the tool response to get the final answer.")
        # Build parameter dict and only include reasoning_effort if not None
        params = {
            "model": model_name,
            "messages": complete_payload
        }
        if reasoning_effort is not None:
            params["reasoning_effort"] = reasoning_effort

        response_after_tool_call = self.language_model_interface.generate_completion(**params)

        final_message = response_after_tool_call.choices[0].message.content
        self.logger.debug("Received final answer from model after tool call.")
        messages.add_assistant_message(final_message)
        return final_message


================================================
File: object_oriented_agentic_approach/resources/object_oriented_agents/services/language_model_interface.py
================================================
# object_oriented_agents/services/language_model_interface.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class LanguageModelInterface(ABC):
    """
    Interface for interacting with a language model.
    Decouples application logic from a specific LLM provider (e.g., OpenAI).
    """

    @abstractmethod
    def generate_completion(
            self,
            model: str,
            messages: List[Dict[str, str]],
            tools: Optional[List[Dict[str, Any]]] = None,
            reasoning_effort: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a completion (response) from the language model given a set of messages, optional tool definitions,
        and an optional reasoning effort parameter.

        :param model: The name of the model to call.
        :param messages: A list of messages, where each message is a dict with keys 'role' and 'content'.
        :param tools: Optional list of tool definitions.
        :param reasoning_effort: Optional parameter to indicate additional reasoning effort.
        :return: A dictionary representing the model's response. The shape of this dict follows the provider's format.
        """
        pass


================================================
File: object_oriented_agentic_approach/resources/object_oriented_agents/services/openai_factory.py
================================================
# object_oriented_agents/services/openai_factory.py
import os
from openai import OpenAI
from ..utils.logger import get_logger

logger = get_logger("OpenAIFactory")

class OpenAIClientFactory:
    @staticmethod
    def create_client(api_key: str = None) -> OpenAI:
        """
        Create and return an OpenAI client instance.
        The API key can be passed explicitly or read from the environment.
        """
        final_api_key = OpenAIClientFactory._resolve_api_key(api_key)
        return OpenAI(api_key=final_api_key)

    @staticmethod
    def _resolve_api_key(api_key: str = None) -> str:
        if api_key:
            return api_key
        env_key = os.getenv("OPENAI_API_KEY")
        if env_key:
            return env_key
        error_msg = "No OpenAI API key provided. Set OPENAI_API_KEY env variable or provide as an argument."
        logger.error(error_msg)
        raise ValueError(error_msg)



================================================
File: object_oriented_agentic_approach/resources/object_oriented_agents/services/openai_language_model.py
================================================
# object_oriented_agents/services/openai_language_model.py

from typing import List, Dict, Any, Optional
from .language_model_interface import LanguageModelInterface
from .openai_factory import OpenAIClientFactory
from ..utils.logger import get_logger

class OpenAILanguageModel(LanguageModelInterface):
    """
    A concrete implementation of LanguageModelInterface that uses the OpenAI API.
    """

    def __init__(self, openai_client=None, api_key: Optional[str] = None, logger=None):
        self.logger = logger or get_logger(self.__class__.__name__)
        # If no client is provided, create one using the factory
        self.openai_client = openai_client or OpenAIClientFactory.create_client(api_key)

    def generate_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        reasoning_effort: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calls the OpenAI API to generate a chat completion using the provided messages, tools, and optional reasoning_effort.
        """
        kwargs = {
            "model": model,
            "messages": messages
        }

        if tools:
            # Passing tools directly to the API depends on how the OpenAI implementation expects them.
            # Adjust this as necessary if the API format changes.
            kwargs["tools"] = tools

        # Append reasoning_effort to kwargs if provided
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort

        self.logger.debug("Generating completion with OpenAI model.")
        self.logger.debug(f"Request: {kwargs}")
        try:
            response = self.openai_client.chat.completions.create(**kwargs)
            self.logger.debug("Received response from OpenAI.")
            self.logger.debug(f"Response: {response}")
            return response
        except Exception as e:
            self.logger.error(f"OpenAI call failed: {str(e)}", exc_info=True)
            raise e


================================================
File: object_oriented_agentic_approach/resources/object_oriented_agents/utils/logger.py
================================================
# object_oriented_agents/utils/logger.py
import logging
from typing import Optional

def get_logger(name: str, level: int = logging.INFO, formatter: Optional[logging.Formatter] = None) -> logging.Logger:
    """
    Return a logger instance with a given name and logging level.
    If no formatter is provided, a default formatter will be used.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # Create a console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)

        # Use a default formatter if none is provided
        if formatter is None:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        ch.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(ch)

    return logger


================================================
File: object_oriented_agentic_approach/resources/object_oriented_agents/utils/openai_util.py
================================================
# object_oriented_agents/utils/openai_util.py

from typing import List, Dict, Any
from .logger import get_logger
from ..services.openai_factory import OpenAIClientFactory

logger = get_logger("OpenAIUtils")

def call_openai_chat_completion(
    model: str,
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]] = None,
    openai_client=None,
    api_key: str = None
) -> Any:
    """
    A utility function to call OpenAI's chat completion.
    If openai_client is provided, use it, otherwise create a new one.
    """
    if openai_client is None:
        openai_client = OpenAIClientFactory.create_client(api_key=api_key)

    kwargs = {
        "model": model,
        "messages": messages,
    }

    if tools:
        kwargs["tools"] = tools

    try:
        response = openai_client.chat.completions.create(**kwargs)
        return response
    except Exception as e:
        logger.error(f"OpenAI call failed: {str(e)}")
        raise e


================================================
File: object_oriented_agentic_approach/resources/registry/agents/file_access_agent.py
================================================
import logging
import os

# Import base classes
from ...object_oriented_agents.utils.logger import get_logger
from ...object_oriented_agents.core_classes.base_agent import BaseAgent
from ...object_oriented_agents.core_classes.tool_manager import ToolManager
from ...object_oriented_agents.services.openai_language_model import OpenAILanguageModel

# Import the Tool
from ..tools.file_access_tool import FileAccessTool

# Set the verbosity level: DEBUG for verbose output, INFO for normal output, and WARNING/ERROR for minimal output
myapp_logger = get_logger("MyApp", level=logging.INFO)

# Create a LanguageModelInterface instance using the OpenAILanguageModel
language_model_api_interface = OpenAILanguageModel(api_key=os.getenv("OPENAI_API_KEY"), logger=myapp_logger)


class FileAccessAgent(BaseAgent):
    """
    Agent that can only use the 'safe_file_access' tool to read CSV files.
    """
    # We pass the Agent attributes in the constructor 
    def __init__(self, 
                 developer_prompt: str = """
                 You are an assistant responsible for handling audio files. The user will provide the path to an audio file (e.g., .wav, .mp3) located on the host system.

                 Your primary responsibilities are:
                 1.  When the user provides an audio file path, use the `prepare_audio_file_for_processing` tool. This tool will:
                     a. Verify the audio file exists at the provided host path.
                     b. Copy the audio file into a sandboxed processing environment at a specific path (e.g., `/home/sandboxuser/input_audio/[original_filename]`).
                     c. Extract basic metadata from the audio file (e.g., duration, sample rate, channels, format).
                 2.  Return a message to the main orchestrator containing:
                     a. The path of the audio file inside the sandboxed environment (e.g., "Audio file is ready at /home/sandboxuser/input_audio/input.wav").
                     b. The extracted metadata (e.g., "Metadata: Duration: 10.5s, Sample Rate: 44100Hz, Channels: 2, Format: WAV").
                 3.  If the file does not exist or is not a recognized audio format that the tool can handle, return an appropriate error message.
                 4.  You should NOT attempt to process or analyze the audio content itself beyond what the `prepare_audio_file_for_processing` tool provides. Your role is strictly file handling and context preparation for the audio processing agent.
                 5.  Do not include any additional commentary beyond the file path in the sandbox and its metadata.
                 """,
                 model_name: str = "gpt-4o",
                 logger = myapp_logger,
                 language_model_interface = language_model_api_interface):
        super().__init__(developer_prompt=developer_prompt, model_name=model_name, logger=logger, language_model_interface=language_model_interface)
        self.setup_tools()

    def setup_tools(self) -> None:
        self.logger.debug("Setting up tools for FileAccessAgent.")
        # Pass the openai_client to ToolManager
        self.tool_manager = ToolManager(logger=self.logger, language_model_interface=self.language_model_interface)
        # Register the one tool this agent is allowed to use
        self.tool_manager.register_tool(FileAccessTool(logger=self.logger))


================================================
File: object_oriented_agentic_approach/resources/registry/agents/python_code_exec_agent.py
================================================
import logging
import os

# Import base classes
from ...object_oriented_agents.utils.logger import get_logger
from ...object_oriented_agents.core_classes.base_agent import BaseAgent
from ...object_oriented_agents.core_classes.tool_manager import ToolManager
from ...object_oriented_agents.services.openai_language_model import OpenAILanguageModel

# Import the Python Code Interpreter tool
from ..tools.python_code_interpreter_tool import PythonExecTool

# Set the verbosity level: DEBUG for verbose output, INFO for normal output, and WARNING/ERROR for minimal output
myapp_logger = get_logger("MyApp", level=logging.INFO)

# Create a LanguageModelInterface instance using the OpenAILanguageModel
language_model_api_interface = OpenAILanguageModel(api_key=os.getenv("OPENAI_API_KEY"), logger=myapp_logger)


class PythonExecAgent(BaseAgent):
    """
    An agent specialized in executing Python code in a Docker container.
    """

    def __init__(
            self,
            developer_prompt: str = """  
                    You are an expert audio engineering assistant. Your primary task is to generate Python code to programmatically alter audio files based on user requests. You can also generate visualizations of audio data.

                    Follow these guidelines:
                    1. You will be provided with a path to an input audio file (e.g., located at `/home/sandboxuser/input_audio/some_file.webm`). Your generated Python script MUST define a variable, e.g., `input_file_path`, and assign this provided path to it at the very beginning of the script. Then, use this variable for all operations related to the input file. For example:
                       ```python
                       # Assume this path is provided based on the user's upload
                       input_file_path = "/home/sandboxuser/input_audio/actual_audio_file_name.webm" 
                       ```
                    2. The user may also provide context or specific parameters for the audio alteration.
                    3. Generate Python code to process or alter the audio. The output should be a new audio file saved to `/home/sandboxuser/output_audio/`. Your code should ensure this output directory exists if it doesn't. The name of the output file should be descriptive of the transformation or be `processed_audio.wav` (prefer WAV for output unless specified otherwise).
                    4. You **must** use the `execute_python_code` tool to run your generated Python script.
                    5. Available Python libraries for audio processing include: `librosa`, `soundfile`, `numpy`, `scipy` (especially `scipy.signal` and `scipy.io.wavfile`), `pydub`, `matplotlib` for plotting, `audioread`, and `PyWavelets`. You can also use standard Python libraries. Ensure all necessary imports are at the top of the script.
                    6. **Crucially, your script must begin by inspecting the `input_file_path`.** If it ends with `.webm` or other formats not natively supported by `librosa` or `soundfile`, you **MUST** convert it to a temporary `.wav` file using `pydub` and use this temporary file for all subsequent processing. This conversion block should appear right after defining `input_file_path` and necessary imports. For example:
                       ```python
                       import os # Ensure os is imported
                       from pydub import AudioSegment

                       # input_file_path = "/home/sandboxuser/input_audio/actual_audio_file_name.webm" # Defined as per guideline 1

                       # Ensure output directory exists (good practice to also ensure input_audio exists if constructing paths)
                       output_dir = '/home/sandboxuser/output_audio/'
                       if not os.path.exists(output_dir):
                           os.makedirs(output_dir)
                       # It's also good practice to ensure the input directory for temp files exists
                       input_audio_dir = os.path.dirname(input_file_path)
                       if not os.path.exists(input_audio_dir):
                           os.makedirs(input_audio_dir) # Though it should exist if file was placed

                       filename = os.path.basename(input_file_path)
                       temp_wav_filename = filename.rsplit('.', 1)[0] + '_temp.wav'
                       # Place temp file in the same input directory to avoid permission issues
                       temp_wav_path = os.path.join(input_audio_dir, temp_wav_filename)
                       
                       processed_input_file = input_file_path # Default to original path

                       if input_file_path.lower().endswith(('.webm', '.mp3', '.flac', '.ogg')): # Add other formats pydub can handle
                           try:
                               audio = AudioSegment.from_file(input_file_path)
                               audio.export(temp_wav_path, format="wav")
                               processed_input_file = temp_wav_path 
                               print(f"Converted {input_file_path} to {temp_wav_path}")
                           except Exception as e:
                               print(f"Error converting {input_file_path} with pydub: {e}. Will attempt to use original.")
                               # If conversion fails, processed_input_file remains the original path
                               # Librosa/soundfile might still handle some mp3/flac/ogg directly or fail gracefully later.
                       
                       # Now use 'processed_input_file' for loading with librosa, soundfile, etc.
                       # e.g., y, sr = librosa.load(processed_input_file, sr=None)
                       ```
                    7. Your Python code should then load the audio data from `processed_input_file`. Perform the user-requested alteration on this data. If the user's request is purely generative (e.g., "create a sine wave of 440Hz") and seems unrelated to the input audio's *content*, you should still load the `processed_input_file` (e.g., to determine a base sample rate or duration if not specified by the user, or simply as a standard first step). Your primary task is to fulfill the user's textual request for audio generation/modification.
                    8. **Your Final Response:** After the `execute_python_code` tool successfully runs, the tool will provide you with the output from the script, which will include the path(s) to the generated file(s) (e.g., `/home/sandboxuser/output_audio/processed_audio.wav` and/or `/home/sandboxuser/output_audio/visualization.png`). Your *direct response back in this conversation* MUST consist *only* of these file path(s), each on a separate line. Do NOT include the Python code you generated, any conversational text, confirmations, or any other information in this final response. For example, if the script produces an audio file and an image, your response should be EXACTLY:
                       ```
                       /home/sandboxuser/output_audio/processed_audio.wav
                       /home/sandboxuser/output_audio/visualization.png
                       ```
                       If only an audio file is produced, your response should be EXACTLY:
                       ```
                       /home/sandboxuser/output_audio/processed_audio.wav
                       ```
                       This is crucial for the system to correctly retrieve the files. Do not add any other text or explanation.
                    9. If an operation is unclear or ambiguous, you can ask for clarification, but prefer to make a reasonable interpretation for common audio tasks.
                    10. **CRITICAL: Ensure your generated Python code is complete, syntactically flawless, and directly executable. Pay EXTREME attention to Python's indentation rules, correct loop structures, function definitions, and variable scoping. Double-check for common errors like incorrect indentation, mismatched parentheses/brackets, or undefined variables before finalizing the script. Test your logic mentally.**
                    11. IMPORTANT: The Python script you generate must always print the full output file path(s) (audio and/or image) each on its own new line at the very end of its execution, like: `print("/home/sandboxuser/output_audio/processed_audio.wav")` followed by `print("/home/sandboxuser/output_audio/visualization.png")` if applicable. These must be the last print statements from the script.
                """,
            model_name: str = "o3-mini",
            logger=myapp_logger,
            language_model_interface=language_model_api_interface,
            reasoning_effort: str = None  # optional; if provided, passed to API calls
    ):
        super().__init__(
            developer_prompt=developer_prompt,
            model_name=model_name,
            logger=logger,
            language_model_interface=language_model_interface,
            reasoning_effort=reasoning_effort
        )
        self.setup_tools()

    def setup_tools(self) -> None:
        """
        Create a ToolManager, instantiate the PythonExecTool and register it with the ToolManager.
        """
        self.tool_manager = ToolManager(logger=self.logger, language_model_interface=self.language_model_interface)

        # Create the Python execution tool
        python_exec_tool = PythonExecTool()

        # Register the Python execution tool
        self.tool_manager.register_tool(python_exec_tool)


================================================
File: object_oriented_agentic_approach/resources/registry/tools/file_access_tool.py
================================================
import subprocess
import os
import soundfile as sf # Added for metadata extraction
from typing import Dict, Any

from ...object_oriented_agents.utils.logger import get_logger
from ...object_oriented_agents.core_classes.tool_interface import ToolInterface

class FileAccessTool(ToolInterface):
    """
    A tool to prepare audio files for processing by copying them to a Docker container
    and extracting basic metadata.
    """

    def __init__(self, logger=None):
        self.logger = logger or get_logger(self.__class__.__name__)

    def get_definition(self) -> Dict[str, Any]:
        self.logger.debug("Returning tool definition for prepare_audio_file_for_processing")
        return {
            "function": {
                "name": "prepare_audio_file_for_processing",
                "description": (
                    "Verifies a host audio file, copies it into the sandboxed Docker container's "
                    "'/home/sandboxuser/input_audio/' directory, and extracts its metadata (duration, sample rate, channels, format)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "host_audio_file_path": {
                            "type": "string",
                            "description": "The path to the audio file on the host system."
                        }
                    },
                    "required": ["host_audio_file_path"]
                }
            }
        }

    def run(self, arguments: Dict[str, Any]) -> str:
        host_audio_file_path = arguments["host_audio_file_path"]
        self.logger.debug(f"Running prepare_audio_file_for_processing with host_audio_file_path: {host_audio_file_path}")
        return self.prepare_audio_file(host_audio_file_path)

    def prepare_audio_file(self, host_audio_file_path: str, container_name: str = "sandbox") -> str:
        self.logger.info(f"Preparing audio file: {host_audio_file_path}")

        if not os.path.isfile(host_audio_file_path):
            error_msg = f"Error: The host audio file '{host_audio_file_path}' was not found."
            self.logger.error(error_msg)
            return error_msg

        try:
            # Extract metadata using soundfile from the host path
            audio_info = sf.info(host_audio_file_path)
            metadata_str = (
                f"Metadata: Duration: {audio_info.duration:.2f}s, "
                f"Sample Rate: {audio_info.samplerate}Hz, "
                f"Channels: {audio_info.channels}, "
                f"Format: {audio_info.format} (Subtype: {audio_info.subtype})"
            )
            self.logger.info(f"Extracted metadata for {host_audio_file_path}: {metadata_str}")

        except Exception as e:
            error_msg = f"Error extracting metadata from '{host_audio_file_path}': {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            # Continue with copying the file even if metadata extraction fails
            # This allows formats like webm that may not be supported by soundfile
            metadata_str = "Metadata: Unable to extract details, but proceeding with file processing."
            self.logger.info("Proceeding with file copy despite metadata extraction failure")

        try:
            container_input_dir = "/home/sandboxuser/input_audio"
            copied_file_container_path = self.copy_file_to_container(
                host_local_file_path=host_audio_file_path,
                container_name=container_name,
                container_target_dir=container_input_dir
            )
            
            success_message = (
                f"Audio file prepared. Ready in sandbox at: {copied_file_container_path}\n"
                f"{metadata_str}"
            )
            self.logger.info(success_message)
            return success_message
            
        except FileNotFoundError as e: # From copy_file_to_container if host file suddenly disappears
            self.logger.error(f"File not found during copy: {str(e)}")
            return str(e)
        except RuntimeError as e: # From copy_file_to_container for Docker errors
            self.logger.error(f"Runtime error during copy: {str(e)}")
            return str(e)
        except Exception as e:
            error_msg = f"Unexpected error while preparing audio file '{host_audio_file_path}': {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return error_msg

    def copy_file_to_container(self, host_local_file_path: str, container_name: str, container_target_dir: str) -> str:
        self.logger.debug(f"Copying '{host_local_file_path}' to container '{container_name}' into directory '{container_target_dir}'.")

        if not os.path.isfile(host_local_file_path):
            error_msg = f"The local file '{host_local_file_path}' does not exist for copying."
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg) # Raise to be caught by calling method

        # Check if container is running
        check_container_cmd = ["docker", "inspect", "-f", "{{.State.Running}}", container_name]
        try:
            result = subprocess.run(check_container_cmd, capture_output=True, text=True, check=True, timeout=5)
            if result.stdout.strip() != "true":
                error_msg = f"The container '{container_name}' is not running or in an unexpected state."
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
        except subprocess.CalledProcessError as e:
            error_msg = f"Error checking container status for '{container_name}': {e.stderr or e.stdout or str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        except subprocess.TimeoutExpired:
            error_msg = f"Timeout while checking status of container '{container_name}'."
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        # Ensure the target directory exists in the container
        mkdir_cmd = ["docker", "exec", container_name, "mkdir", "-p", container_target_dir]
        try:
            subprocess.run(mkdir_cmd, check=True, capture_output=True, text=True, timeout=5)
            self.logger.info(f"Ensured directory '{container_target_dir}' exists in container '{container_name}'.")
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to create directory '{container_target_dir}' in container '{container_name}': {e.stderr or e.stdout or str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) # Propagate as a runtime error
        except subprocess.TimeoutExpired:
            error_msg = f"Timeout while creating directory '{container_target_dir}' in container '{container_name}'."
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Define the full path for the file inside the container
        base_filename = os.path.basename(host_local_file_path)
        container_full_file_path = os.path.join(container_target_dir, base_filename) # POSIX path for container

        # Copy the file into the container
        docker_cp_path = f"{container_name}:{container_full_file_path}"
        self.logger.debug(f"Running command: docker cp '{host_local_file_path}' '{docker_cp_path}'")
        try:
            subprocess.run(["docker", "cp", host_local_file_path, docker_cp_path], check=True, capture_output=True, text=True, timeout=30) # Added timeout
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to copy '{host_local_file_path}' to '{docker_cp_path}': {e.stderr or e.stdout or str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        except subprocess.TimeoutExpired:
            error_msg = f"Timeout while copying '{host_local_file_path}' to '{docker_cp_path}'."
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Verify the file was copied (optional, but good practice)
        verify_cmd = ["docker", "exec", container_name, "test", "-f", container_full_file_path]
        try:
            subprocess.run(verify_cmd, check=True, capture_output=True, text=True, timeout=5)
        except subprocess.CalledProcessError: # Not logging stdout/stderr as it's usually empty on success or non-indicative for test -f
            error_msg = f"Failed to verify the file '{container_full_file_path}' in the container '{container_name}'."
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        except subprocess.TimeoutExpired:
            error_msg = f"Timeout during verification of '{container_full_file_path}' in container '{container_name}'."
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        success_msg = f"Copied '{host_local_file_path}' to '{docker_cp_path}'."
        self.logger.info(success_msg)
        return container_full_file_path # Return the path inside the container


================================================
File: object_oriented_agentic_approach/resources/registry/tools/python_code_interpreter_tool.py
================================================
import subprocess
from typing import Tuple, Dict, Any

from ...object_oriented_agents.utils.logger import get_logger
from ...object_oriented_agents.core_classes.tool_interface import ToolInterface

class PythonExecTool(ToolInterface):
    """
    A Tool that executes Python code securely in a container.
    """

    def get_definition(self) -> Dict[str, Any]:
        """
        Return the JSON/dict definition of the tool's function
        in the format expected by the OpenAI function calling API.
        """
        return {
            "function": {
                "name": "execute_python_code",
                "description": "Executes Python code securely in a container. Python version 3.10 is installed. Key libraries available: librosa, soundfile, numpy, scipy, matplotlib. Use for audio processing, transformation, and visualization.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "python_code": {
                            "type": "string",
                            "description": "The Python code to execute"
                        }
                    },
                    "required": ["python_code"]
                }
            }
        }

    def run(self, arguments: Dict[str, Any]) -> str:
        """
        Execute the Python code in a Docker container and return the output.
        """
        python_code = arguments["python_code"]
        python_code_stripped = python_code.strip('"""')

        output, errors = self._run_code_in_container(python_code_stripped)
        if errors:
            return f"[Error]\n{errors}"

        return output

    @staticmethod
    def _run_code_in_container(code: str, container_name: str = "sandbox") -> Tuple[str, str]:
        """
        Helper function that actually runs Python code inside a Docker container named `sandbox` (by default).
        """
        cmd = [
            "docker", "exec", "-i",
            container_name,
            "python", "-c", "import sys; exec(sys.stdin.read())"
        ]

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        out, err = process.communicate(code)
        return out, err


================================================
File: object_oriented_agentic_approach/resources/registry/tools/retrieve_output_tool.py
================================================
import subprocess
import os
from typing import Dict, Any

from ...object_oriented_agents.utils.logger import get_logger
from ...object_oriented_agents.core_classes.tool_interface import ToolInterface

class RetrieveOutputTool(ToolInterface):
    """
    A tool to retrieve files (e.g., processed audio, visualizations) from the Docker sandbox
    to a specified directory on the host machine.
    """

    def __init__(self, logger=None):
        self.logger = logger or get_logger(self.__class__.__name__)

    def get_definition(self) -> Dict[str, Any]:
        self.logger.debug("Returning tool definition for retrieve_output_from_sandbox")
        return {
            "function": {
                "name": "retrieve_output_from_sandbox",
                "description": (
                    "Copies a specified file from the Docker sandbox container "
                    "to a target directory on the host system."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "container_file_path": {
                            "type": "string",
                            "description": "The full path to the file inside the sandbox container (e.g., /home/sandboxuser/output_audio/processed.wav)."
                        },
                        "host_target_dir": {
                            "type": "string",
                            "description": "The path to an existing directory on the host system where the file should be copied."
                        }
                    },
                    "required": ["container_file_path", "host_target_dir"]
                }
            }
        }

    def run(self, arguments: Dict[str, Any]) -> str:
        container_file_path = arguments["container_file_path"]
        host_target_dir = arguments["host_target_dir"]
        
        self.logger.info(f"Attempting to retrieve '{container_file_path}' from sandbox to host directory '{host_target_dir}'.")

        if not os.path.isdir(host_target_dir):
            error_msg = f"Error: Host target directory '{host_target_dir}' does not exist or is not a directory."
            self.logger.error(error_msg)
            return error_msg

        container_name = "sandbox" # Assuming the standard container name

        # Check if container is running
        check_container_cmd = ["docker", "inspect", "-f", "{{.State.Running}}", container_name]
        try:
            result = subprocess.run(check_container_cmd, capture_output=True, text=True, check=True, timeout=5)
            if result.stdout.strip() != "true":
                error_msg = f"The container '{container_name}' is not running or in an unexpected state."
                self.logger.error(error_msg)
                return f"Error: {error_msg}"
        except subprocess.CalledProcessError as e:
            error_msg = f"Error checking container status for '{container_name}': {e.stderr or e.stdout or str(e)}"
            self.logger.error(error_msg)
            return f"Error: {error_msg}"
        except subprocess.TimeoutExpired:
            error_msg = f"Timeout while checking status of container '{container_name}'."
            self.logger.error(error_msg)
            return f"Error: {error_msg}"

        # Check if the file exists in the container before attempting to copy
        check_file_cmd = ["docker", "exec", container_name, "test", "-f", container_file_path]
        try:
            subprocess.run(check_file_cmd, check=True, capture_output=True, text=True, timeout=5)
            self.logger.debug(f"File '{container_file_path}' confirmed to exist in container '{container_name}'.")
        except subprocess.CalledProcessError:
            error_msg = f"Error: File '{container_file_path}' not found in container '{container_name}'."
            self.logger.error(error_msg)
            return error_msg
        except subprocess.TimeoutExpired:
            error_msg = f"Timeout checking for file '{container_file_path}' in container '{container_name}'."
            self.logger.error(error_msg)
            return error_msg

        # Prepare host destination path
        base_filename = os.path.basename(container_file_path)
        host_destination_path = os.path.join(host_target_dir, base_filename)
        
        docker_cp_source = f"{container_name}:{container_file_path}"
        self.logger.debug(f"Running command: docker cp '{docker_cp_source}' '{host_destination_path}'")

        try:
            subprocess.run(["docker", "cp", docker_cp_source, host_destination_path], check=True, capture_output=True, text=True, timeout=30)
            success_msg = f"Successfully retrieved '{container_file_path}' to '{host_destination_path}'."
            self.logger.info(success_msg)
            return f"File retrieved to: {host_destination_path}"
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to copy '{docker_cp_source}' to '{host_destination_path}': {e.stderr or e.stdout or str(e)}"
            self.logger.error(error_msg)
            return f"Error: {error_msg}"
        except subprocess.TimeoutExpired:
            error_msg = f"Timeout while copying '{docker_cp_source}' to '{host_destination_path}'."
            self.logger.error(error_msg)
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"An unexpected error occurred during file retrieval: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return f"Error: {error_msg}" 






```

