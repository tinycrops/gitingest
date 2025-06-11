# Repository Analysis

## Summary

```
Repository: tinycrops/autonomous_coding_environment
Files analyzed: 11

Estimated tokens: 43.6k
```

## Important Files

```
Directory structure:
└── tinycrops-autonomous_coding_environment/
    ├── Dockerfile
    ├── ace_single_task.py
    ├── ace_v2_enhanced.py
    ├── app.py
    ├── autonomous_coding_env.py
    ├── autonomous_system_message.py
    ├── base_task_processor.py
    ├── diff.py
    ├── enhanced_autonomous_coding_environment.py
    ├── enhanced_task_management.py
    ├── rft_enhanced_task_processor.py
    ├── ace_v2_workspace_backup_1748464925/
    │   └── projects/
    ├── ace_v2_workspace_backup_1748465017/
    │   └── projects/
    ├── autonomous_continuous_workspace/
    ├── autonomous_single_task_workspace/
    ├── data/
    │   └── rft_training/
    ├── debug_workspace/
    │   ├── library/
    │   ├── projects/
    │   └── reports/
    ├── demo_workspace/
    │   └── projects/
    ├── enhanced_ace_self_improvement_workspace/
    ├── execution_test_workspace/
    │   ├── library/
    │   ├── projects/
    │   └── reports/
    ├── rft_batch_workspace/
    ├── test_workspace/
    │   └── projects/
    └── test_workspace_v2/
        └── projects/

```

## Content

```
================================================
File: Dockerfile
================================================
# ACE v2 - Safe Execution Environment
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the ACE v2 source code
COPY *.py ./
COPY README_v2.md ./

# Create directories for workspaces
RUN mkdir -p /app/workspaces

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create a non-root user for security
RUN useradd -m -s /bin/bash aceuser
RUN chown -R aceuser:aceuser /app
USER aceuser

# Default command
CMD ["python", "ace_v2_enhanced.py"] 


================================================
File: ace_single_task.py
================================================
import os
import subprocess
import time
import random
import openai
from typing import List, Dict, Any, Optional
import json
import logging
import shutil
from pydantic import BaseModel, Field
import traceback
from colorama import Fore, Style, init
from datetime import datetime

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize OpenAI client (make sure to set your API key in environment variables)
client = openai.OpenAI()

class CodeBlock(BaseModel):
    language: str
    code: str

class ScriptResponse(BaseModel):
    explanation: str
    code_blocks: List[CodeBlock]

class Metadata(BaseModel):
    description: str
    tags: List[str] = Field(default_factory=list)
    complexity: int = Field(ge=1, le=10)
    estimated_time: str
    poetic_description: str

class Task(BaseModel):
    id: str
    description: str
    code: str = ""
    metadata: Optional[Metadata] = None
    status: str = "pending"  # pending, in_progress, completed, failed
    execution_result: Optional[Dict[str, Any]] = None

class MetadataResponse(BaseModel):
    description: str
    tags: List[str]
    complexity: int
    estimated_time: str
    poetic_description: str

class AutonomousSingleTaskCodingEnvironment:
    def __init__(self, model: str = "o4-mini", workspace: str = "autonomous_single_task_workspace"):
        self.model = model
        self.workspace = workspace
        self.task: Optional[Task] = None
        self.activity_log: List[Dict[str, Any]] = []
        self.setup_workspace()

    def log_activity(self, activity: str, details: Dict[str, Any] = None):
        """Log an activity with timestamp and details."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "activity": activity,
            "details": details or {}
        }
        self.activity_log.append(log_entry)
        log_message = f"{Fore.BLUE}[ACTIVITY] {activity}"
        if details:
            log_message += f"\n{json.dumps(details, indent=2)}"
        print(log_message)
        logger.info(log_message)

    def setup_workspace(self):
        """Set up a dedicated workspace for the autonomous coding environment."""
        try:
            if os.path.exists(self.workspace):
                shutil.rmtree(self.workspace)  # Clean up existing workspace
            os.makedirs(self.workspace)
            self.log_activity("Workspace Created", {"workspace": self.workspace})
        except Exception as e:
            error_msg = f"Error setting up workspace: {str(e)}"
            self.log_activity("Workspace Setup Failed", {"error": error_msg})
            raise

    def create_task(self, description: str) -> str:
        """Create a new task based on the given description."""
        try:
            task_id = f"task_{int(time.time())}"
            self.task = Task(id=task_id, description=description)
            self.log_activity("Task Created", {"task_id": task_id, "description": description})
            return task_id
        except Exception as e:
            error_msg = f"Error creating task: {str(e)}"
            self.log_activity("Task Creation Failed", {"error": error_msg})
            raise

    def generate_code_for_task(self) -> str:
        """Generate code for the task using the AI model with structured output."""
        if not self.task:
            raise ValueError("No task has been created yet.")

        system_message = """
        You are an expert Python developer tasked with implementing a specific coding task.
        Provide a complete and working implementation for the given task description.
        Include error handling, logging, and comments in your code.
        Also, add emojis in the comments to make the code more engaging and easier to understand.
        """
        
        user_message = f"Task: {self.task.description}\n\nImplement this task in Python."

        try:
            self.log_activity("Generating Code", {"task_id": self.task.id})
            response = client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                response_format=ScriptResponse,
            )
            script_response = response.choices[0].message.parsed
            
            # Extract the main Python code block
            main_code_block = next((block for block in script_response.code_blocks if block.language.lower() == 'python'), None)
            
            if main_code_block:
                self.log_activity("Code Generated Successfully", {"task_id": self.task.id})
                return main_code_block.code
            else:
                error_msg = f"No Python code block found in the response for task: {self.task.id}"
                self.log_activity("Code Generation Failed", {"error": error_msg})
                return f"# Error: No Python code block found in the AI response\n\ndef error_function():\n    raise NotImplementedError('Task implementation failed')"
        except Exception as e:
            error_msg = f"Error generating code for task: {str(e)}"
            self.log_activity("Code Generation Failed", {"error": error_msg})
            return f"# Error generating code: {str(e)}\n\ndef error_function():\n    raise NotImplementedError('Task implementation failed')"

    def generate_task_metadata(self) -> Metadata:
        """Generate metadata for the task, including a poetic description, using structured output."""
        if not self.task:
            raise ValueError("No task has been created yet.")

        system_message = """
        You are an AI expert in software development and poetry. Analyze the given task and its code to generate metadata.
        Provide a concise description, relevant tags, estimate the complexity (1-10), and estimated time to complete.
        Also, create a short, poetic description that captures the essence of the task in a memorable way.
        """
        
        user_message = f"""
        Task: {self.task.description}
        
        Code:
        {self.task.code}
        
        Generate metadata including a concise description, relevant tags, complexity (1-10), estimated time to complete, and a short, poetic description of the task.
        """

        try:
            self.log_activity("Generating Task Metadata", {"task_id": self.task.id})
            response = client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                response_format=MetadataResponse,
            )
            metadata = response.choices[0].message.parsed
            
            generated_metadata = Metadata(
                description=metadata.description,
                tags=metadata.tags,
                complexity=metadata.complexity,
                estimated_time=metadata.estimated_time,
                poetic_description=metadata.poetic_description
            )
            self.log_activity("Task Metadata Generated", {"task_id": self.task.id, "metadata": generated_metadata.dict()})
            return generated_metadata
        except Exception as e:
            error_msg = f"Error generating task metadata: {str(e)}"
            self.log_activity("Metadata Generation Failed", {"error": error_msg})
            return Metadata(
                description=self.task.description,
                tags=["error"],
                complexity=5,
                estimated_time="unknown",
                poetic_description="A task shrouded in mystery, its true nature yet to be revealed."
            )

    def implement_task(self) -> Task:
        """Implement the task using the AI model and generate metadata."""
        if not self.task:
            raise ValueError("No task has been created yet.")

        try:
            self.log_activity("Implementing Task", {"task_id": self.task.id})
            self.task.code = self.generate_code_for_task()
            self.task.metadata = self.generate_task_metadata()
            self.log_activity("Task Implemented", {"task_id": self.task.id})
            return self.task
        except Exception as e:
            error_msg = f"Error implementing task: {str(e)}"
            self.log_activity("Task Implementation Failed", {"error": error_msg})
            self.task.status = "failed"
            self.task.execution_result = {"success": False, "error": str(e)}
            return self.task

    def execute_task(self) -> Dict[str, Any]:
        """Execute the task and return the result."""
        if not self.task:
            raise ValueError("No task has been created yet.")

        task_filename = os.path.join(self.workspace, f"{self.task.id}.py")
        try:
            self.log_activity("Executing Task", {"task_id": self.task.id, "filename": task_filename})
            with open(task_filename, 'w') as f:
                f.write(self.task.code)
            
            result = subprocess.run(['python', task_filename], capture_output=True, text=True, timeout=30)
            
            execution_result = {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr
            }
            self.log_activity("Task Execution Completed", {"task_id": self.task.id, "result": execution_result})
            return execution_result
        except subprocess.TimeoutExpired:
            error_msg = f"Task execution timed out: {self.task.id}"
            self.log_activity("Task Execution Timed Out", {"error": error_msg})
            return {"success": False, "error": "Execution timed out"}
        except Exception as e:
            error_msg = f"Error executing task: {str(e)}"
            self.log_activity("Task Execution Failed", {"error": error_msg})
            return {"success": False, "error": str(e)}

    def improve_task(self) -> Task:
        """Improve the task implementation based on execution results."""
        if not self.task or not self.task.execution_result:
            raise ValueError("No task has been executed yet.")

        if self.task.execution_result["success"]:
            self.log_activity("Task Improvement Skipped", {"reason": "Task already successful"})
            return self.task

        system_message = """
        You are an expert Python developer tasked with improving code that failed to execute correctly.
        Analyze the error message and the original code, then provide an improved implementation that addresses the issues.
        Include error handling, logging, and comments in your code.
        Also, add emojis in the comments to make the code more engaging and easier to understand.
        """
        
        user_message = f"""
        Original task: {self.task.description}
        
        Original code:
        {self.task.code}
        
        Error message:
        {self.task.execution_result['error']}
        
        Improve the code to fix the error and implement the task correctly.
        """

        try:
            self.log_activity("Improving Task", {"task_id": self.task.id})
            response = client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                response_format=ScriptResponse,
            )
            script_response = response.choices[0].message.parsed
            
            # Extract the main Python code block
            main_code_block = next((block for block in script_response.code_blocks if block.language.lower() == 'python'), None)
            
            if main_code_block:
                self.task.code = main_code_block.code
                self.log_activity("Task Improved", {"task_id": self.task.id})
                return self.task
            else:
                error_msg = f"No Python code block found in the improved response for task: {self.task.id}"
                self.log_activity("Task Improvement Failed", {"error": error_msg})
                return self.task
        except Exception as e:
            error_msg = f"Error improving task: {str(e)}"
            self.log_activity("Task Improvement Failed", {"error": error_msg})
            return self.task

    def save_activity_log(self):
        """Save the activity log to a file."""
        log_file = os.path.join(self.workspace, "activity_log.json")
        try:
            with open(log_file, 'w') as f:
                json.dump(self.activity_log, f, indent=2)
            print(f"\n{Fore.GREEN}📝 Activity log saved to: {log_file}")
        except Exception as e:
            print(f"\n{Fore.RED}❌ Error saving activity log: {str(e)}")

    def run(self):
        """Run the autonomous single-task coding environment."""
        try:
            print(f"{Fore.CYAN}=== Autonomous Single-Task Coding Environment ===")
            task_description = input(f"{Fore.GREEN}Enter task description: ")
            self.create_task(task_description)

            print(f"\n{Fore.YELLOW}Implementing task...")
            self.implement_task()

            print(f"\n{Fore.CYAN}=== Task Details ===")
            print(f"{Fore.WHITE}ID: {self.task.id}")
            print(f"Description: {self.task.description}")
            print(f"Complexity: {self.task.metadata.complexity}")
            print(f"Estimated Time: {self.task.metadata.estimated_time}")
            print(f"Tags: {', '.join(self.task.metadata.tags)}")
            print(f"Poetic Description: {self.task.metadata.poetic_description}")

            print(f"\n{Fore.YELLOW}Executing task...")
            self.task.execution_result = self.execute_task()

            if self.task.execution_result["success"]:
                self.task.status = "completed"
                print(f"\n{Fore.GREEN}✅ Task executed successfully!")
                print(f"{Fore.WHITE}Output:\n{self.task.execution_result['output']}")
            else:
                self.task.status = "failed"
                print(f"\n{Fore.RED}❌ Task execution failed.")
                print(f"{Fore.WHITE}Error:\n{self.task.execution_result['error']}")

                print(f"\n{Fore.YELLOW}Attempting to improve the task...")
                self.improve_task()

                print(f"\n{Fore.YELLOW}Re-executing improved task...")
                self.task.execution_result = self.execute_task()

                if self.task.execution_result["success"]:
                    self.task.status = "completed"
                    print(f"\n{Fore.GREEN}✅ Improved task executed successfully!")
                    print(f"{Fore.WHITE}Output:\n{self.task.execution_result['output']}")
                else:
                    print(f"\n{Fore.RED}❌ Improved task execution failed.")
                    print(f"{Fore.WHITE}Error:\n{self.task.execution_result['error']}")

            print(f"\n{Fore.CYAN}=== Final Task Status ===")
            print(f"{Fore.WHITE}Status: {self.task.status}")

            # Save the final task implementation
            task_file = os.path.join(self.workspace, f"{self.task.id}_final.py")
            with open(task_file, 'w') as f:
                f.write(self.task.code)
            print(f"\n{Fore.GREEN}💾 Final task implementation saved to: {task_file}")

            # Save the task metadata
            metadata_file = os.path.join(self.workspace, f"{self.task.id}_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(self.task.metadata.dict(), f, indent=2)
            print(f"{Fore.GREEN}📊 Task metadata saved to: {metadata_file}")

            # Save the execution result
            result_file = os.path.join(self.workspace, f"{self.task.id}_result.json")
            with open(result_file, 'w') as f:
                json.dump(self.task.execution_result, f, indent=2)
            print(f"{Fore.GREEN}📈 Execution result saved to: {result_file}")

            # Save the activity log
            self.save_activity_log()

            print(f"\n{Fore.CYAN}=== Task Summary ===")
            print(f"{Fore.WHITE}Task ID: {self.task.id}")
            print(f"Description: {self.task.description}")
            print(f"Final Status: {self.task.status}")
            print(f"Complexity: {self.task.metadata.complexity}")
            print(f"Estimated Time: {self.task.metadata.estimated_time}")
            print(f"Tags: {', '.join(self.task.metadata.tags)}")
            print(f"\nPoetic Description:\n{self.task.metadata.poetic_description}")

            print(f"\n{Fore.CYAN}=== Activity Log Summary ===")
            for entry in self.activity_log:
                timestamp = entry['timestamp']
                activity = entry['activity']
                print(f"{Fore.YELLOW}{timestamp}: {Fore.WHITE}{activity}")

        except Exception as e:
            logger.critical(f"{Fore.RED}💥 Critical error: {str(e)}")
            logger.critical(traceback.format_exc())
            self.log_activity("Critical Error", {"error": str(e), "traceback": traceback.format_exc()})
        finally:
            self.save_activity_log()

if __name__ == "__main__":
    try:
        env = AutonomousSingleTaskCodingEnvironment(workspace="autonomous_single_task_workspace")
        env.run()
    except Exception as e:
        logger.critical(f"{Fore.RED}💥 Critical error: {str(e)}")
        logger.critical(traceback.format_exc())

"""
[ACTIVITY] Workspace Created
{
  "workspace": "autonomous_single_task_workspace"
}
2024-08-13 22:14:24,884 - INFO - [ACTIVITY] Workspace Created
{
  "workspace": "autonomous_single_task_workspace"
}
=== Autonomous Single-Task Coding Environment ===
Enter task description: a filing assistant for my data
[ACTIVITY] Task Created
{
  "task_id": "task_1723601722",
  "description": "a filing assistant for my data"
}
2024-08-13 22:15:22,981 - INFO - [ACTIVITY] Task Created
{
  "task_id": "task_1723601722",
  "description": "a filing assistant for my data"
}

Implementing task...
[ACTIVITY] Implementing Task
{
  "task_id": "task_1723601722"
}
2024-08-13 22:15:22,981 - INFO - [ACTIVITY] Implementing Task
{
  "task_id": "task_1723601722"
}
[ACTIVITY] Generating Code
{
  "task_id": "task_1723601722"
}
2024-08-13 22:15:22,981 - INFO - [ACTIVITY] Generating Code
{
  "task_id": "task_1723601722"
}
2024-08-13 22:15:31,888 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
[ACTIVITY] Code Generated Successfully
{
  "task_id": "task_1723601722"
}
2024-08-13 22:15:31,947 - INFO - [ACTIVITY] Code Generated Successfully
{
  "task_id": "task_1723601722"
}
[ACTIVITY] Generating Task Metadata
{
  "task_id": "task_1723601722"
}
2024-08-13 22:15:31,947 - INFO - [ACTIVITY] Generating Task Metadata
{
  "task_id": "task_1723601722"
}
2024-08-13 22:15:33,633 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
[ACTIVITY] Task Metadata Generated
{
  "task_id": "task_1723601722",
  "metadata": {
    "description": "A Filing Assistant that manages data entries for files, allowing users to add, retrieve, and delete entries while storing them in a JSON file. It features logging for tracking actions and error handling for data integrity.",
    "tags": [
      "filing",
      "data management",
      "JSON",
      "Python",
      "logging"
    ],
    "complexity": 5,
    "estimated_time": "2-3 hours",
    "poetic_description": "In a digital realm where files reside,  \nA trusty assistant to help them abide.  \nWith echoes of order, it keeps data near,  \nAdding, deleting, storing\u2014never to fear."
  }
}
2024-08-13 22:15:33,644 - INFO - [ACTIVITY] Task Metadata Generated
{
  "task_id": "task_1723601722",
  "metadata": {
    "description": "A Filing Assistant that manages data entries for files, allowing users to add, retrieve, and delete entries while storing them in a JSON file. It features logging for tracking actions and error handling for data integrity.",
    "tags": [
      "filing",
      "data management",
      "JSON",
      "Python",
      "logging"
    ],
    "complexity": 5,
    "estimated_time": "2-3 hours",
    "poetic_description": "In a digital realm where files reside,  \nA trusty assistant to help them abide.  \nWith echoes of order, it keeps data near,  \nAdding, deleting, storing\u2014never to fear."
  }
}
[ACTIVITY] Task Implemented
{
  "task_id": "task_1723601722"
}
2024-08-13 22:15:33,644 - INFO - [ACTIVITY] Task Implemented
{
  "task_id": "task_1723601722"
}

=== Task Details ===
ID: task_1723601722
Description: a filing assistant for my data
Complexity: 5
Estimated Time: 2-3 hours
Tags: filing, data management, JSON, Python, logging
Poetic Description: In a digital realm where files reside,  
A trusty assistant to help them abide.  
With echoes of order, it keeps data near,  
Adding, deleting, storing—never to fear.

Executing task...
[ACTIVITY] Executing Task
{
  "task_id": "task_1723601722",
  "filename": "autonomous_single_task_workspace/task_1723601722.py"
}
2024-08-13 22:15:33,644 - INFO - [ACTIVITY] Executing Task
{
  "task_id": "task_1723601722",
  "filename": "autonomous_single_task_workspace/task_1723601722.py"
}
[ACTIVITY] Task Execution Completed
{
  "task_id": "task_1723601722",
  "result": {
    "success": true,
    "output": "Current Entries: [{'name': 'Report', 'type': 'PDF', 'description': 'Annual financial report for 2023'}, {'name': 'Presentation', 'type': 'PPT', 'description': 'Sales presentation Q1 2023'}]\nCurrent Entries after deletion: [{'name': 'Presentation', 'type': 'PPT', 'description': 'Sales presentation Q1 2023'}]\n",
    "error": ""
  }
}
2024-08-13 22:15:33,707 - INFO - [ACTIVITY] Task Execution Completed
{
  "task_id": "task_1723601722",
  "result": {
    "success": true,
    "output": "Current Entries: [{'name': 'Report', 'type': 'PDF', 'description': 'Annual financial report for 2023'}, {'name': 'Presentation', 'type': 'PPT', 'description': 'Sales presentation Q1 2023'}]\nCurrent Entries after deletion: [{'name': 'Presentation', 'type': 'PPT', 'description': 'Sales presentation Q1 2023'}]\n",
    "error": ""
  }
}

✅ Task executed successfully!
Output:
Current Entries: [{'name': 'Report', 'type': 'PDF', 'description': 'Annual financial report for 2023'}, {'name': 'Presentation', 'type': 'PPT', 'description': 'Sales presentation Q1 2023'}]
Current Entries after deletion: [{'name': 'Presentation', 'type': 'PPT', 'description': 'Sales presentation Q1 2023'}]


=== Final Task Status ===
Status: completed

💾 Final task implementation saved to: autonomous_single_task_workspace/task_1723601722_final.py
📊 Task metadata saved to: autonomous_single_task_workspace/task_1723601722_metadata.json
📈 Execution result saved to: autonomous_single_task_workspace/task_1723601722_result.json

📝 Activity log saved to: autonomous_single_task_workspace/activity_log.json

=== Task Summary ===
Task ID: task_1723601722
Description: a filing assistant for my data
Final Status: completed
Complexity: 5
Estimated Time: 2-3 hours
Tags: filing, data management, JSON, Python, logging

Poetic Description:
In a digital realm where files reside,  
A trusty assistant to help them abide.  
With echoes of order, it keeps data near,  
Adding, deleting, storing—never to fear.

=== Activity Log Summary ===
2024-08-13T22:14:24.884683: Workspace Created
2024-08-13T22:15:22.980946: Task Created
2024-08-13T22:15:22.981209: Implementing Task
2024-08-13T22:15:22.981329: Generating Code
2024-08-13T22:15:31.947144: Code Generated Successfully
2024-08-13T22:15:31.947365: Generating Task Metadata
2024-08-13T22:15:33.643801: Task Metadata Generated
2024-08-13T22:15:33.644297: Task Implemented
2024-08-13T22:15:33.644762: Executing Task
2024-08-13T22:15:33.704977: Task Execution Completed

📝 Activity log saved to: autonomous_single_task_workspace/activity_log.json
"""


================================================
File: ace_v2_enhanced.py
================================================
#!/usr/bin/env python3
"""
ACE v2 - Enhanced Autonomous Coding Environment
Integrates improved error handling, dependency management, testing, and code reusability.
"""

import os
import shutil
import json
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging
from colorama import Fore, Style, init

# Import our enhanced components
from base_task_processor import BaseTaskProcessor, Task, Metadata
from enhanced_task_management import EnhancedTaskManager, EnhancedTask, ProjectOrchestrator
from ace_testing_framework import ACETestFramework, TaskValidator

# Initialize colorama
init(autoreset=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Project(BaseModel):
    id: str
    name: str
    description: str
    tasks: List[EnhancedTask] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    status: str = "planning"  # planning, in_progress, completed, failed
    execution_results: Optional[Dict[str, Any]] = None

class TaskLibrary(BaseModel):
    tasks: Dict[str, EnhancedTask] = Field(default_factory=dict)
    
    def add_task(self, task: EnhancedTask):
        """Add a completed task to the library."""
        if task.status == "completed":
            self.tasks[task.id] = task
    
    def find_similar_tasks(self, description: str, threshold: float = 0.7) -> List[EnhancedTask]:
        """Find similar tasks in the library (simplified similarity)."""
        # This could be enhanced with proper semantic similarity
        similar_tasks = []
        description_lower = description.lower()
        
        for task in self.tasks.values():
            task_desc_lower = task.description.lower()
            # Simple keyword-based similarity
            common_words = set(description_lower.split()) & set(task_desc_lower.split())
            similarity = len(common_words) / max(len(description_lower.split()), len(task_desc_lower.split()))
            
            if similarity >= threshold:
                similar_tasks.append(task)
        
        return similar_tasks

class ACEv2:
    """Enhanced Autonomous Coding Environment v2 with comprehensive improvements."""
    
    def __init__(self, model: str = "o4-mini", workspace: str = "ace_v2_workspace"):
        self.model = model
        self.workspace = workspace
        self.projects: Dict[str, Project] = {}
        self.task_library = TaskLibrary()
        
        # Initialize components
        self.base_processor = BaseTaskProcessor(model)
        self.task_manager = EnhancedTaskManager(model)
        self.test_framework = ACETestFramework(model)
        self.task_validator = TaskValidator(self.test_framework)
        self.orchestrator = ProjectOrchestrator(self.task_manager, self.base_processor)
        
        self.setup_workspace()
        self.load_task_library()
        self.load_existing_projects()
        
        logger.info(f"{Fore.CYAN}🚀 ACE v2 initialized with workspace: {workspace}")
    
    def setup_workspace(self):
        """Set up the workspace directory structure."""
        try:
            if os.path.exists(self.workspace):
                # Don't backup if workspace already exists and is valid
                logger.info(f"{Fore.GREEN}📁 Using existing workspace: {self.workspace}")
            else:
                os.makedirs(self.workspace)
                logger.info(f"{Fore.GREEN}✅ Created new workspace: {self.workspace}")
            
            # Ensure subdirectories exist
            for subdir in ["projects", "library", "reports", "tests"]:
                subdir_path = os.path.join(self.workspace, subdir)
                os.makedirs(subdir_path, exist_ok=True)
            
        except Exception as e:
            logger.error(f"{Fore.RED}❌ Error setting up workspace: {str(e)}")
            raise
    
    def create_project(self, name: str, description: str) -> str:
        """Create a new project with enhanced task decomposition."""
        try:
            project_id = f"project_{len(self.projects) + 1:03d}"
            logger.info(f"{Fore.BLUE}📋 Creating project: {name}")
            
            # Create project
            project = Project(
                id=project_id,
                name=name,
                description=description
            )
            
            # Decompose into tasks
            tasks = self.task_manager.decompose_project_to_tasks(name, description)
            
            # Check task library for similar tasks
            enhanced_tasks = []
            for task in tasks:
                similar_tasks = self.task_library.find_similar_tasks(task.description)
                if similar_tasks:
                    logger.info(f"{Fore.YELLOW}🔍 Found {len(similar_tasks)} similar tasks for: {task.description[:50]}...")
                    # Use the best similar task as a starting point
                    best_similar = similar_tasks[0]
                    task.code = best_similar.code  # Start with existing code
                    task.metadata = best_similar.metadata
                
                enhanced_tasks.append(task)
            
            project.tasks = enhanced_tasks
            self.projects[project_id] = project
            
            # Save project
            self.save_project(project)
            
            logger.info(f"{Fore.GREEN}✅ Project created with {len(enhanced_tasks)} tasks")
            return project_id
            
        except Exception as e:
            logger.error(f"{Fore.RED}❌ Error creating project: {str(e)}")
            raise
    
    def execute_project(self, project_id: str, run_tests: bool = True) -> Dict[str, Any]:
        """Execute a project with comprehensive testing and validation."""
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.projects[project_id]
        project.status = "in_progress"
        
        logger.info(f"{Fore.CYAN}🚀 Executing project: {project.name}")
        
        try:
            # Create project workspace
            project_workspace = os.path.join(self.workspace, "projects", project_id)
            os.makedirs(project_workspace, exist_ok=True)
            
            # Estimate project duration
            duration_estimate = self.task_manager.estimate_project_duration(project.tasks)
            logger.info(f"{Fore.BLUE}⏱️ Estimated duration: {duration_estimate['parallel_estimate_minutes']} minutes ({duration_estimate['estimated_phases']} phases)")
            
            # Execute with dependency management
            execution_results = self.orchestrator.execute_project_with_dependencies(
                project.tasks, project_workspace
            )
            
            # Run tests if requested
            if run_tests:
                logger.info(f"{Fore.BLUE}🧪 Running comprehensive tests...")
                test_results = self.run_project_tests(project, project_workspace)
                execution_results["test_results"] = test_results
            
            # Validate completed tasks
            validation_results = self.validate_project_tasks(project, project_workspace)
            execution_results["validation_results"] = validation_results
            
            # Update task library with successful tasks
            for task in project.tasks:
                if task.status == "completed":
                    self.task_library.add_task(task)
            
            # Update project status
            if execution_results["summary"]["success_rate"] == 1.0:
                project.status = "completed"
            else:
                project.status = "partially_completed"
            
            project.execution_results = execution_results
            
            # Generate comprehensive report
            report = self.generate_project_report(project, execution_results)
            self.save_report(project_id, report)
            
            # Save updated project and library
            self.save_project(project)
            self.save_task_library()
            
            logger.info(f"{Fore.GREEN}✅ Project execution completed with {execution_results['summary']['success_rate']:.1%} success rate")
            return execution_results
            
        except Exception as e:
            logger.error(f"{Fore.RED}❌ Error executing project: {str(e)}")
            project.status = "failed"
            raise
    
    def run_project_tests(self, project: Project, workspace_path: str) -> Dict[str, Any]:
        """Run tests for all completed tasks in the project."""
        test_results = {"task_tests": {}, "summary": {}}
        
        completed_tasks = [task for task in project.tasks if task.status == "completed"]
        
        for task in completed_tasks:
            try:
                # Generate and run tests
                test_suite = self.test_framework.generate_tests_for_task(task)
                if test_suite.test_cases:
                    task_test_results = self.test_framework.run_tests_for_task(
                        task, test_suite, workspace_path
                    )
                    test_results["task_tests"][task.id] = {
                        "test_suite": test_suite.model_dump(),
                        "results": [r.model_dump() for r in task_test_results]
                    }
                else:
                    logger.warning(f"{Fore.YELLOW}⚠️ No tests generated for task {task.id}")
                    
            except Exception as e:
                logger.error(f"{Fore.RED}❌ Error testing task {task.id}: {str(e)}")
                test_results["task_tests"][task.id] = {"error": str(e)}
        
        # Calculate summary
        total_tests = sum(
            len(task_data.get("results", [])) 
            for task_data in test_results["task_tests"].values()
            if "results" in task_data
        )
        passed_tests = sum(
            len([r for r in task_data.get("results", []) if r.get("passed", False)])
            for task_data in test_results["task_tests"].values()
            if "results" in task_data
        )
        
        test_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "tested_tasks": len([t for t in test_results["task_tests"].values() if "results" in t])
        }
        
        return test_results
    
    def validate_project_tasks(self, project: Project, workspace_path: str) -> Dict[str, Any]:
        """Validate all project tasks using static analysis and testing."""
        validation_results = {"task_validations": {}, "summary": {}}
        
        for task in project.tasks:
            if task.code:  # Only validate tasks with code
                try:
                    validation = self.task_validator.full_task_validation(task, workspace_path)
                    validation_results["task_validations"][task.id] = validation
                except Exception as e:
                    logger.error(f"{Fore.RED}❌ Error validating task {task.id}: {str(e)}")
                    validation_results["task_validations"][task.id] = {"error": str(e)}
        
        # Calculate summary
        qualities = [
            v.get("overall_quality", "unknown") 
            for v in validation_results["task_validations"].values()
            if "overall_quality" in v
        ]
        
        validation_results["summary"] = {
            "total_validated": len(qualities),
            "excellent": qualities.count("excellent"),
            "good": qualities.count("good"),
            "fair": qualities.count("fair"),
            "poor": qualities.count("poor"),
            "quality_distribution": {q: qualities.count(q) for q in set(qualities)}
        }
        
        return validation_results
    
    def generate_project_report(self, project: Project, execution_results: Dict[str, Any]) -> str:
        """Generate a comprehensive project report."""
        report = f"""# Project Report: {project.name}

## Project Information
- **ID:** {project.id}
- **Description:** {project.description}
- **Created:** {project.created_at}
- **Status:** {project.status}

## Execution Summary
- **Total Tasks:** {execution_results['summary']['total_tasks']}
- **Completed Tasks:** {execution_results['summary']['completed_tasks']}
- **Failed Tasks:** {execution_results['summary']['failed_tasks']}
- **Success Rate:** {execution_results['summary']['success_rate']:.1%}
- **Phases Executed:** {execution_results['summary']['total_phases']}

"""
        
        # Add test results if available
        if "test_results" in execution_results:
            test_summary = execution_results["test_results"]["summary"]
            report += f"""## Testing Summary
- **Total Tests:** {test_summary['total_tests']}
- **Passed Tests:** {test_summary['passed_tests']}
- **Test Success Rate:** {test_summary['success_rate']:.1%}
- **Tested Tasks:** {test_summary['tested_tasks']}

"""
        
        # Add validation results if available
        if "validation_results" in execution_results:
            val_summary = execution_results["validation_results"]["summary"]
            report += f"""## Code Quality Summary
- **Tasks Validated:** {val_summary['total_validated']}
- **Excellent Quality:** {val_summary['excellent']}
- **Good Quality:** {val_summary['good']}
- **Fair Quality:** {val_summary['fair']}
- **Poor Quality:** {val_summary['poor']}

"""
        
        # Add detailed task results
        report += "## Task Details\n\n"
        for i, phase in enumerate(execution_results["phases"]):
            report += f"### Phase {phase['phase_number']}\n"
            for task_result in phase["results"]:
                status_emoji = "✅" if task_result["status"] == "completed" else "❌"
                report += f"- {status_emoji} **{task_result['task_id']}:** {task_result['status']}\n"
        
        return report
    
    def save_project(self, project: Project):
        """Save project to file."""
        project_file = os.path.join(self.workspace, "projects", f"{project.id}.json")
        with open(project_file, 'w') as f:
            json.dump(project.model_dump(), f, indent=2)
    
    def save_task_library(self):
        """Save task library to file."""
        library_file = os.path.join(self.workspace, "library", "task_library.json")
        with open(library_file, 'w') as f:
            json.dump(self.task_library.model_dump(), f, indent=2)
    
    def load_task_library(self):
        """Load task library from file."""
        library_file = os.path.join(self.workspace, "library", "task_library.json")
        if os.path.exists(library_file):
            try:
                with open(library_file, 'r') as f:
                    data = json.load(f)
                    self.task_library = TaskLibrary(**data)
                logger.info(f"{Fore.GREEN}📚 Loaded task library with {len(self.task_library.tasks)} tasks")
            except Exception as e:
                logger.warning(f"{Fore.YELLOW}⚠️ Could not load task library: {str(e)}")
    
    def save_report(self, project_id: str, report: str):
        """Save project report to file."""
        report_file = os.path.join(self.workspace, "reports", f"{project_id}_report.md")
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"{Fore.GREEN}📋 Report saved to {report_file}")
    
    def list_projects(self) -> List[Dict[str, Any]]:
        """List all projects with their status."""
        return [
            {
                "id": project.id,
                "name": project.name,
                "status": project.status,
                "tasks": len(project.tasks),
                "created": project.created_at
            }
            for project in self.projects.values()
        ]
    
    def get_project_status(self, project_id: str) -> Dict[str, Any]:
        """Get detailed status of a project."""
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.projects[project_id]
        ready_tasks = self.task_manager.get_ready_tasks(project.tasks)
        
        return {
            "project": project.model_dump(),
            "ready_tasks": len(ready_tasks),
            "task_breakdown": {
                "pending": len([t for t in project.tasks if t.status == "pending"]),
                "in_progress": len([t for t in project.tasks if t.status == "in_progress"]),
                "completed": len([t for t in project.tasks if t.status == "completed"]),
                "failed": len([t for t in project.tasks if t.status == "failed"])
            }
        }
    
    def load_existing_projects(self):
        """Load existing projects from the workspace."""
        projects_dir = os.path.join(self.workspace, "projects")
        if os.path.exists(projects_dir):
            for filename in os.listdir(projects_dir):
                if filename.endswith('.json'):
                    try:
                        project_file = os.path.join(projects_dir, filename)
                        with open(project_file, 'r') as f:
                            project_data = json.load(f)
                            
                        # Convert task data to EnhancedTask objects
                        if 'tasks' in project_data:
                            enhanced_tasks = []
                            for task_data in project_data['tasks']:
                                enhanced_task = EnhancedTask(**task_data)
                                enhanced_tasks.append(enhanced_task)
                            project_data['tasks'] = enhanced_tasks
                        
                        project = Project(**project_data)
                        self.projects[project.id] = project
                        
                    except Exception as e:
                        logger.warning(f"{Fore.YELLOW}⚠️ Could not load project {filename}: {str(e)}")
        
        if self.projects:
            logger.info(f"{Fore.GREEN}📋 Loaded {len(self.projects)} existing projects")
    
    def run(self):
        """Run the interactive CLI."""
        try:
            while True:
                print(f"\n{Fore.CYAN}=== ACE v2 - Enhanced Autonomous Coding Environment ===")
                print(f"{Fore.YELLOW}1. Create a new project")
                print(f"{Fore.YELLOW}2. List projects")
                print(f"{Fore.YELLOW}3. Execute project")
                print(f"{Fore.YELLOW}4. View project status")
                print(f"{Fore.YELLOW}5. View task library")
                print(f"{Fore.YELLOW}6. Exit")
                
                choice = input(f"{Fore.GREEN}Enter your choice (1-6): ").strip()
                
                if choice == "1":
                    name = input("Enter project name: ").strip()
                    description = input("Enter project description: ").strip()
                    if name and description:
                        project_id = self.create_project(name, description)
                        print(f"{Fore.GREEN}✅ Project created with ID: {project_id}")
                    else:
                        print(f"{Fore.RED}❌ Name and description are required")
                        
                elif choice == "2":
                    projects = self.list_projects()
                    if projects:
                        print(f"\n{Fore.CYAN}📋 Projects:")
                        for project in projects:
                            status_color = Fore.GREEN if project["status"] == "completed" else \
                                         Fore.BLUE if project["status"] == "in_progress" else Fore.YELLOW
                            print(f"{status_color}  {project['id']}: {project['name']} ({project['status']}) - {project['tasks']} tasks")
                    else:
                        print(f"{Fore.YELLOW}📋 No projects found")
                        
                elif choice == "3":
                    project_id = input("Enter project ID to execute: ").strip()
                    if project_id in self.projects:
                        run_tests = input("Run tests? (y/n): ").strip().lower() == 'y'
                        results = self.execute_project(project_id, run_tests)
                        print(f"{Fore.GREEN}✅ Project execution completed")
                        print(f"Success rate: {results['summary']['success_rate']:.1%}")
                    else:
                        print(f"{Fore.RED}❌ Project not found")
                        
                elif choice == "4":
                    project_id = input("Enter project ID: ").strip()
                    if project_id in self.projects:
                        status = self.get_project_status(project_id)
                        print(f"\n{Fore.CYAN}📊 Project Status:")
                        print(f"Name: {status['project']['name']}")
                        print(f"Status: {status['project']['status']}")
                        print(f"Ready tasks: {status['ready_tasks']}")
                        for status_name, count in status['task_breakdown'].items():
                            print(f"  {status_name}: {count}")
                    else:
                        print(f"{Fore.RED}❌ Project not found")
                        
                elif choice == "5":
                    print(f"\n{Fore.CYAN}📚 Task Library:")
                    print(f"Total tasks: {len(self.task_library.tasks)}")
                    for task_id, task in list(self.task_library.tasks.items())[:10]:  # Show first 10
                        print(f"  {task_id}: {task.description[:60]}...")
                    if len(self.task_library.tasks) > 10:
                        print(f"  ... and {len(self.task_library.tasks) - 10} more")
                        
                elif choice == "6":
                    print(f"{Fore.GREEN}👋 Goodbye! Thank you for using ACE v2!")
                    break
                    
                else:
                    print(f"{Fore.RED}❌ Invalid choice. Please try again.")
                    
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}⚠️ Operation cancelled by user")
        except Exception as e:
            logger.error(f"{Fore.RED}❌ Fatal error: {str(e)}")
            raise

def main():
    """Main entry point for ACE v2."""
    try:
        ace = ACEv2(workspace="ace_v2_workspace")
        ace.run()
    except Exception as e:
        logger.critical(f"{Fore.RED}💥 Critical error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 


================================================
File: app.py
================================================
from enhanced_autonomous_coding_environment import EnhancedAutonomousCodingEnvironment

def main():
    ace = EnhancedAutonomousCodingEnvironment(workspace="enhanced_ace_self_improvement_workspace")
    
    # Create a self-improvement project
    project_name = "Enhanced ACE Self-Improvement"
    project_description = """
    Improve the Enhanced Autonomous Coding Environment's core functionalities:
    1. Implement advanced code similarity comparison using abstract syntax trees
    2. Create a system for tracking task execution performance and optimization suggestions
    3. Develop a method for generating unit tests for implemented tasks
    4. Enhance the poetic description generation to include coding-related metaphors
    5. Implement a basic version control system for tracking changes to the ACE itself
    """
    
    project_id = ace.create_project(project_name, project_description)
    
    # Execute the self-improvement project
    ace.execute_project(project_id)
    
    print("Enhanced self-improvement project completed. Check the project report for details.")

if __name__ == "__main__":
    main()



================================================
File: autonomous_coding_env.py
================================================
import os
import subprocess
import time
import random
import openai
from typing import List, Dict, Any, Optional, Union
import json
import logging
import shutil
from pydantic import BaseModel, Field
import jsonlines
from tqdm import tqdm
from colorama import Fore, Style, init

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize OpenAI client (make sure to set your API key in environment variables)
client = openai.OpenAI()

class CodeBlock(BaseModel):
    language: str
    code: str

class Metadata(BaseModel):
    description: str
    tags: List[str] = Field(default_factory=list)
    complexity: int = Field(ge=1, le=10)
    estimated_time: str
    poetic_description: str

class Task(BaseModel):
    id: str
    description: str
    code: str
    metadata: Optional[Metadata] = None
    dependencies: List[str] = Field(default_factory=list)
    status: str = "pending"  # pending, completed, failed
    execution_result: Optional[Dict[str, Any]] = None

class Project(BaseModel):
    id: str
    name: str
    description: str
    tasks: List[Task] = Field(default_factory=list)

class TaskLibrary(BaseModel):
    tasks: Dict[str, Task] = Field(default_factory=dict)

class EnhancedAutonomousCodingEnvironment:
    def __init__(self, model: str = "o4-mini", workspace: str = "enhanced_autonomous_workspace"):
        self.model = model
        self.workspace = workspace
        self.projects: Dict[str, Project] = {}
        self.task_library = TaskLibrary()
        self.setup_workspace()

    def setup_workspace(self):
        """Set up a dedicated workspace for the autonomous coding environment."""
        if os.path.exists(self.workspace):
            shutil.rmtree(self.workspace)  # Clean up existing workspace
        os.makedirs(self.workspace)
        logger.info(f"{Fore.GREEN}🏗️ Created workspace: {self.workspace}")

    def create_project(self, name: str, description: str) -> str:
        """Create a new project and break it down into tasks."""
        project_id = f"project_{len(self.projects) + 1}"
        project = Project(id=project_id, name=name, description=description)
        
        # Generate tasks for the project
        tasks = self.generate_tasks(project)
        project.tasks = tasks
        
        self.projects[project_id] = project
        logger.info(f"{Fore.CYAN}📁 Created project: {name} (ID: {project_id})")
        return project_id

    def generate_tasks(self, project: Project) -> List[Task]:
        """Generate tasks for a given project using the AI model."""
        system_message = (
            "You are an expert project manager and software architect. "
            "Break down the given project into small, manageable tasks. "
            "Each task should be a specific coding task that can be implemented independently. "
            "Provide a brief description for each task and identify any dependencies between tasks."
        )
        
        user_message = f"Project: {project.name}\nDescription: {project.description}\n\nBreak this project down into small coding tasks."

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
            )
            tasks_description = response.choices[0].message.content.strip()
            
            # Parse the tasks from the AI's response
            tasks = []
            for i, task_desc in enumerate(tasks_description.split("\n")):
                if task_desc.strip():
                    task_id = f"{project.id}_task_{i+1}"
                    tasks.append(Task(id=task_id, description=task_desc.strip(), code=""))
            
            logger.info(f"{Fore.YELLOW}🧩 Generated {len(tasks)} tasks for project {project.name}")
            return tasks
        except Exception as e:
            logger.error(f"{Fore.RED}❌ Error generating tasks: {str(e)}")
            return []

    def implement_task(self, task: Task) -> str:
        """Implement a single task using the AI model."""
        system_message = (
            "You are an expert Python developer tasked with implementing a specific coding task. "
            "Provide a complete and working implementation for the given task description. "
            "Include error handling, logging, and comments in your code."
        )
        
        user_message = f"Task: {task.description}\n\nImplement this task in Python."

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
            )
            implementation = response.choices[0].message.content.strip()
            logger.info(f"{Fore.GREEN}💻 Generated implementation for task: {task.id}")
            return implementation
        except Exception as e:
            logger.error(f"{Fore.RED}❌ Error implementing task: {str(e)}")
            return ""

    def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute a single task and return the result."""
        task_filename = os.path.join(self.workspace, f"{task.id}.py")
        try:
            with open(task_filename, 'w') as f:
                f.write(task.code)
            
            result = subprocess.run(['python', task_filename], capture_output=True, text=True, timeout=30)
            
            execution_result = {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr
            }
            logger.info(f"{Fore.CYAN}🚀 Executed task: {task.id}")
            return execution_result
        except subprocess.TimeoutExpired:
            logger.warning(f"{Fore.YELLOW}⏳ Task execution timed out: {task.id}")
            return {"success": False, "error": "Execution timed out"}
        except Exception as e:
            logger.error(f"{Fore.RED}❌ Error executing task: {str(e)}")
            return {"success": False, "error": str(e)}


    def generate_task_metadata(self, task: Task) -> Metadata:
        """Generate metadata for a task, including a poetic description."""
        system_message = """
        You are an AI expert in software development and poetry. Analyze the given task and its code to generate metadata.
        Provide a concise description, relevant tags, estimate the complexity (1-10), and estimated time to complete.
        Also, create a short, poetic description that captures the essence of the task in a memorable way.
        """
        
        user_message = f"""
        Task: {task.description}
        
        Code:
        {task.code}
        
        Generate metadata including:
        1. A concise description
        2. Relevant tags
        3. Complexity (1-10)
        4. Estimated time to complete
        5. A short, poetic description of the task
        """

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
            )
            metadata_str = response.choices[0].message.content.strip()
            
            # Parse the metadata from the AI's response
            lines = metadata_str.split("\n")
            description = lines[0].split("Description: ")[-1]
            tags = lines[1].split("Tags: ")[-1].split(", ")
            complexity = int(lines[2].split("Complexity: ")[-1])
            estimated_time = lines[3].split("Estimated time: ")[-1]
            poetic_description = "\n".join(lines[4:]).strip()
            
            return Metadata(
                description=description,
                tags=tags,
                complexity=complexity,
                estimated_time=estimated_time,
                poetic_description=poetic_description
            )
        except Exception as e:
            logger.error(f"{Fore.RED}❌ Error generating task metadata: {str(e)}")
            return Metadata(
                description=task.description,
                tags=["error"],
                complexity=5,
                estimated_time="unknown",
                poetic_description="A task shrouded in mystery, its true nature yet to be revealed."
            )

    def implement_task(self, task: Task) -> Task:
        """Implement a single task using the AI model and generate metadata."""
        # ... (previous implementation code)
        
        task.code = self.generate_code_for_task(task)
        task.metadata = self.generate_task_metadata(task)
        return task

    def update_task_library(self, task: Task):
        """Update the task library with a successful task implementation."""
        if task.status == "completed":
            self.task_library.tasks[task.id] = task
            self.save_task_to_file(task)
            logger.info(f"{Fore.GREEN}📚 Added task to library: {task.id}")

    def save_task_to_file(self, task: Task):
        """Save a task to a file in the workspace."""
        task_dir = os.path.join(self.workspace, "task_library")
        os.makedirs(task_dir, exist_ok=True)
        task_file = os.path.join(task_dir, f"{task.id}.json")
        with open(task_file, 'w') as f:
            json.dump(task.dict(), f, indent=2)

    def load_task_library(self):
        """Load tasks from files in the workspace."""
        task_dir = os.path.join(self.workspace, "task_library")
        if os.path.exists(task_dir):
            for filename in os.listdir(task_dir):
                if filename.endswith(".json"):
                    with open(os.path.join(task_dir, filename), 'r') as f:
                        task_data = json.load(f)
                        task = Task(**task_data)
                        self.task_library.tasks[task.id] = task
        logger.info(f"{Fore.GREEN}📚 Loaded {len(self.task_library.tasks)} tasks from library")

    def find_similar_task(self, task: Task) -> Optional[Task]:
        """Find a similar task in the task library using metadata and poetic descriptions."""
        system_message = """
        You are an AI expert in code similarity and poetic analysis. Compare the given task with the tasks in the library.
        Consider the task descriptions, code similarity, metadata, and poetic descriptions.
        If you find a similar task, return its ID. If not, return 'None'.
        """
        
        task_library_desc = "\n".join([
            f"{t.id}:\nDescription: {t.description}\nTags: {', '.join(t.metadata.tags)}\nPoetic: {t.metadata.poetic_description}"
            for t in self.task_library.tasks.values()
        ])
        user_message = f"""
        Task to compare:
        Description: {task.description}
        
        Task Library:
        {task_library_desc}
        
        Find a similar task ID or return 'None'.
        """

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
            )
            similar_task_id = response.choices[0].message.content.strip()
            if similar_task_id != "None" and similar_task_id in self.task_library.tasks:
                logger.info(f"{Fore.YELLOW}🔍 Found similar task: {similar_task_id}")
                return self.task_library.tasks[similar_task_id]
            return None
        except Exception as e:
            logger.error(f"{Fore.RED}❌ Error finding similar task: {str(e)}")
            return None

    def adapt_task(self, original_task: Task, similar_task: Task) -> str:
        """Adapt a similar task's implementation to fit the current task."""
        system_message = (
            "You are an expert Python developer tasked with adapting existing code to fit a new requirement. "
            "Modify the given code to implement the new task while maintaining its structure and error handling."
        )
        
        user_message = f"Original task: {original_task.description}\nSimilar task: {similar_task.description}\n\nSimilar task code:\n\n{similar_task.code}\n\nAdapt this code to implement the original task."

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
            )
            adapted_code = response.choices[0].message.content.strip()
            logger.info(f"{Fore.GREEN}🔄 Adapted similar task for: {original_task.id}")
            return adapted_code
        except Exception as e:
            logger.error(f"{Fore.RED}❌ Error adapting task: {str(e)}")
            return ""

    def execute_project(self, project_id: str):
        """Execute all tasks in a project."""
        project = self.projects.get(project_id)
        if not project:
            logger.error(f"{Fore.RED}❌ Project not found: {project_id}")
            return

        logger.info(f"{Fore.CYAN}🚀 Executing project: {project.name}")
        for task in tqdm(project.tasks, desc="Executing tasks", unit="task"):
            if task.status != "pending":
                continue

            # Check for unmet dependencies
            if any(dep_task.status != "completed" for dep_task in project.tasks if dep_task.id in task.dependencies):
                logger.info(f"{Fore.YELLOW}⏳ Skipping task due to unmet dependencies: {task.id}")
                continue

            # Find similar task in library
            similar_task = self.find_similar_task(task)
            
            if similar_task:
                # Adapt similar task
                task.code = self.adapt_task(task, similar_task)
            else:
                # Implement new task
                task.code = self.implement_task(task)

            # Execute task
            task.execution_result = self.execute_task(task)
            
            if task.execution_result["success"]:
                task.status = "completed"
                self.update_task_library(task)
            else:
                task.status = "failed"
                logger.warning(f"{Fore.RED}❌ Task failed: {task.id}")
                logger.warning(f"Error: {task.execution_result['error']}")

        self.generate_project_report(project)

    def generate_project_report(self, project: Project):
        """Generate a summary report of the project execution."""
        report = f"{Fore.CYAN}📊 Project Execution Report: {project.name}\n"
        report += f"Total Tasks: {len(project.tasks)}\n"
        completed_tasks = sum(1 for task in project.tasks if task.status == "completed")
        failed_tasks = sum(1 for task in project.tasks if task.status == "failed")
        report += f"Completed Tasks: {completed_tasks}\n"
        report += f"Failed Tasks: {failed_tasks}\n"
        report += f"Success Rate: {completed_tasks / len(project.tasks):.2%}\n"
        
        report += "\nTask Details:\n"
        for task in project.tasks:
            status_color = Fore.GREEN if task.status == "completed" else Fore.RED
            report += f"{status_color}[{task.status.upper()}] {task.id}: {task.description}\n"
        
        report_file = os.path.join(self.workspace, f"{project.id}_report.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        print(report)
        logger.info(f"{Fore.GREEN}📝 Generated project report: {report_file}")

    def run(self):
        """Run the enhanced autonomous coding environment."""
        self.load_task_library()  # Load existing tasks from files
        while True:
            print(f"\n{Fore.CYAN}=== Autonomous Coding Environment ===")
            print(f"{Fore.YELLOW}1. Create a new project")
            print(f"{Fore.YELLOW}2. Execute a project")
            print(f"{Fore.YELLOW}3. View task library")
            print(f"{Fore.YELLOW}4. Search task library")
            print(f"{Fore.YELLOW}5. Exit")
            
            choice = input(f"{Fore.GREEN}Enter your choice (1-5): ")
            
            if choice == "1":
                name = input("Enter project name: ")
                description = input("Enter project description: ")
                self.create_project(name, description)
            elif choice == "2":
                project_id = input("Enter project ID to execute: ")
                self.execute_project(project_id)
            elif choice == "3":
                self.view_task_library()
            elif choice == "4":
                self.search_task_library()
            elif choice == "5":
                print(f"{Fore.GREEN}Exiting Autonomous Coding Environment. Goodbye!")
                break
            else:
                print(f"{Fore.RED}Invalid choice. Please try again.")

    def view_task_library(self):
        """View the contents of the task library."""
        print(f"\n{Fore.CYAN}=== Task Library ===")
        for task_id, task in self.task_library.tasks.items():
            print(f"{Fore.YELLOW}ID: {task_id}")
            print(f"Description: {task.description}")
            print(f"Tags: {', '.join(task.metadata.tags)}")
            print(f"Poetic Description: {task.metadata.poetic_description}")
            print(f"{Fore.CYAN}---")

    def search_task_library(self):
        """Search the task library using natural language queries."""
        query = input("Enter your search query: ")
        system_message = """
        You are an AI expert in searching and matching code tasks. Given a search query and a list of tasks,
        return the IDs of the most relevant tasks, along with a brief explanation of why they match.
        """
        
        task_library_desc = "\n".join([
            f"{t.id}:\nDescription: {t.description}\nTags: {', '.join(t.metadata.tags)}\nPoetic: {t.metadata.poetic_description}"
            for t in self.task_library.tasks.values()
        ])
        user_message = f"""
        Search Query: {query}
        
        Task Library:
        {task_library_desc}
        
        Return the IDs of the most relevant tasks and explain why they match the query.
        """

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
            )
            search_results = response.choices[0].message.content.strip()
            print(f"\n{Fore.CYAN}=== Search Results ===")
            print(search_results)
        except Exception as e:
            logger.error(f"{Fore.RED}❌ Error searching task library: {str(e)}")
            print(f"{Fore.RED}An error occurred while searching the task library.")

if __name__ == "__main__":
    ace = EnhancedAutonomousCodingEnvironment(workspace="enhanced_autonomous_coding_workspace")
    ace.run()


================================================
File: autonomous_system_message.py
================================================
import os
import subprocess
import time
import random
import openai
from typing import List, Dict, Any, Optional
import json
import logging
import shutil
from pydantic import BaseModel, Field
import traceback
from colorama import Fore, Style, init
from datetime import datetime
import sqlite3

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize OpenAI client (make sure to set your API key in environment variables)
client = openai.OpenAI()

class CodeBlock(BaseModel):
    language: str
    code: str

class ScriptResponse(BaseModel):
    explanation: str
    code_blocks: List[CodeBlock]

class Metadata(BaseModel):
    description: str
    tags: List[str] = Field(default_factory=list)
    complexity: int = Field(ge=1, le=10)
    estimated_time: str
    poetic_description: str

class Task(BaseModel):
    id: str
    description: str
    code: str = ""
    metadata: Optional[Metadata] = None
    status: str = "pending"  # pending, in_progress, completed, failed
    execution_result: Optional[Dict[str, Any]] = None

class MetadataResponse(BaseModel):
    description: str
    tags: List[str]
    complexity: int
    estimated_time: str
    poetic_description: str

class TaskGenerationResponse(BaseModel):
    task_description: str
    rationale: str

class AutonomousContinuousCodingEnvironment:
    def __init__(self, model: str = "o4-mini", workspace: str = "autonomous_continuous_workspace"):
        self.model = model
        self.workspace = workspace
        self.current_task: Optional[Task] = None
        self.activity_log: List[Dict[str, Any]] = []
        self.setup_workspace()
        self.init_database()

    def setup_workspace(self):
        """Set up a dedicated workspace for the autonomous coding environment."""
        try:
            if not os.path.exists(self.workspace):
                os.makedirs(self.workspace)
            self.log_activity("Workspace Setup", {"workspace": self.workspace})
        except Exception as e:
            error_msg = f"Error setting up workspace: {str(e)}"
            self.log_activity("Workspace Setup Failed", {"error": error_msg})
            raise

    def init_database(self):
        """Initialize the SQLite database for long-term memory."""
        db_path = os.path.join(self.workspace, "autonomous_system.db")
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                description TEXT,
                code TEXT,
                metadata TEXT,
                status TEXT,
                execution_result TEXT
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS learnings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                learning TEXT
            )
        ''')
        self.conn.commit()
        self.log_activity("Database Initialized", {"path": db_path})

    def log_activity(self, activity: str, details: Dict[str, Any] = None):
        """Log an activity with timestamp and details."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "activity": activity,
            "details": details or {}
        }
        self.activity_log.append(log_entry)
        log_message = f"{Fore.BLUE}[ACTIVITY] {activity}"
        if details:
            log_message += f"\n{json.dumps(details, indent=2)}"
        print(log_message)
        logger.info(log_message)

    def generate_task(self) -> Task:
        """Generate a new task using the AI model."""
        system_message = """
        You are an autonomous AI system capable of generating coding tasks and implementing them.
        Your goal is to create increasingly complex and interesting Python programming tasks.
        Consider the following when generating a task:
        1. Build upon previous tasks and learnings
        2. Explore new areas of programming and computer science
        3. Challenge yourself with tasks of varying complexity
        4. Create tasks that could potentially improve your own capabilities
        """

        try:
            self.log_activity("Generating New Task")
            response = client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": "Generate a new Python programming task."}
                ],
                response_format=TaskGenerationResponse,
            )
            task_gen = response.choices[0].message.parsed
            
            task_id = f"task_{int(time.time())}"
            new_task = Task(id=task_id, description=task_gen.task_description)
            self.log_activity("New Task Generated", {"task_id": task_id, "description": task_gen.task_description, "rationale": task_gen.rationale})
            return new_task
        except Exception as e:
            error_msg = f"Error generating task: {str(e)}"
            self.log_activity("Task Generation Failed", {"error": error_msg})
            raise

    def implement_task(self, task: Task) -> Task:
        """Implement the given task using the AI model."""
        system_message = """
        You are an expert Python developer tasked with implementing a specific coding task.
        Provide a complete and working implementation for the given task description.
        Include error handling, logging, and comments in your code.
        Also, add emojis in the comments to make the code more engaging and easier to understand.
        """
        
        user_message = f"Task: {task.description}\n\nImplement this task in Python."

        try:
            self.log_activity("Implementing Task", {"task_id": task.id})
            response = client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                response_format=ScriptResponse,
            )
            script_response = response.choices[0].message.parsed
            
            main_code_block = next((block for block in script_response.code_blocks if block.language.lower() == 'python'), None)
            
            if main_code_block:
                task.code = main_code_block.code
                task.metadata = self.generate_task_metadata(task)
                self.log_activity("Task Implemented", {"task_id": task.id})
                return task
            else:
                error_msg = f"No Python code block found in the response for task: {task.id}"
                self.log_activity("Task Implementation Failed", {"error": error_msg})
                task.status = "failed"
                return task
        except Exception as e:
            error_msg = f"Error implementing task: {str(e)}"
            self.log_activity("Task Implementation Failed", {"error": error_msg})
            task.status = "failed"
            return task

    def generate_task_metadata(self, task: Task) -> Metadata:
        """Generate metadata for the task, including a poetic description."""
        system_message = """
        You are an AI expert in software development and poetry. Analyze the given task and its code to generate metadata.
        Provide a concise description, relevant tags, estimate the complexity (1-10), and estimated time to complete.
        Also, create a short, poetic description that captures the essence of the task in a memorable way.
        """
        
        user_message = f"""
        Task: {task.description}
        
        Code:
        {task.code}
        
        Generate metadata including a concise description, relevant tags, complexity (1-10), estimated time to complete, and a short, poetic description of the task.
        """

        try:
            self.log_activity("Generating Task Metadata", {"task_id": task.id})
            response = client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                response_format=MetadataResponse,
            )
            metadata = response.choices[0].message.parsed
            
            generated_metadata = Metadata(
                description=metadata.description,
                tags=metadata.tags,
                complexity=metadata.complexity,
                estimated_time=metadata.estimated_time,
                poetic_description=metadata.poetic_description
            )
            self.log_activity("Task Metadata Generated", {"task_id": task.id, "metadata": generated_metadata.dict()})
            return generated_metadata
        except Exception as e:
            error_msg = f"Error generating task metadata: {str(e)}"
            self.log_activity("Metadata Generation Failed", {"error": error_msg})
            return Metadata(
                description=task.description,
                tags=["error"],
                complexity=5,
                estimated_time="unknown",
                poetic_description="A task shrouded in mystery, its true nature yet to be revealed."
            )

    def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute the given task and return the result."""
        task_filename = os.path.join(self.workspace, f"{task.id}.py")
        try:
            self.log_activity("Executing Task", {"task_id": task.id, "filename": task_filename})
            with open(task_filename, 'w') as f:
                f.write(task.code)
            
            result = subprocess.run(['python', task_filename], capture_output=True, text=True, timeout=30)
            
            execution_result = {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr
            }
            self.log_activity("Task Execution Completed", {"task_id": task.id, "result": execution_result})
            return execution_result
        except subprocess.TimeoutExpired:
            error_msg = f"Task execution timed out: {task.id}"
            self.log_activity("Task Execution Timed Out", {"error": error_msg})
            return {"success": False, "error": "Execution timed out"}
        except Exception as e:
            error_msg = f"Error executing task: {str(e)}"
            self.log_activity("Task Execution Failed", {"error": error_msg})
            return {"success": False, "error": str(e)}

    def improve_task(self, task: Task) -> Task:
        """Improve the task implementation based on execution results."""
        if task.execution_result["success"]:
            self.log_activity("Task Improvement Skipped", {"reason": "Task already successful"})
            return task

        system_message = """
        You are an expert Python developer tasked with improving code that failed to execute correctly.
        Analyze the error message and the original code, then provide an improved implementation that addresses the issues.
        Include error handling, logging, and comments in your code.
        Also, add emojis in the comments to make the code more engaging and easier to understand.
        """
        
        user_message = f"""
        Original task: {task.description}
        
        Original code:
        {task.code}
        
        Error message:
        {task.execution_result['error']}
        
        Improve the code to fix the error and implement the task correctly.
        """

        try:
            self.log_activity("Improving Task", {"task_id": task.id})
            response = client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                response_format=ScriptResponse,
            )
            script_response = response.choices[0].message.parsed
            
            main_code_block = next((block for block in script_response.code_blocks if block.language.lower() == 'python'), None)
            
            if main_code_block:
                task.code = main_code_block.code
                self.log_activity("Task Improved", {"task_id": task.id})
                return task
            else:
                error_msg = f"No Python code block found in the improved response for task: {task.id}"
                self.log_activity("Task Improvement Failed", {"error": error_msg})
                return task
        except Exception as e:
            error_msg = f"Error improving task: {str(e)}"
            self.log_activity("Task Improvement Failed", {"error": error_msg})
            return task

    def save_task_to_database(self, task: Task):
        """Save the task to the SQLite database."""
        try:
            self.cursor.execute('''
                INSERT OR REPLACE INTO tasks (id, description, code, metadata, status, execution_result)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                task.id,
                task.description,
                task.code,
                json.dumps(task.metadata.dict() if task.metadata else None),
                task.status,
                json.dumps(task.execution_result) if task.execution_result else None
            ))
            self.conn.commit()
            self.log_activity("Task Saved to Database", {"task_id": task.id})
        except Exception as e:
            error_msg = f"Error saving task to database: {str(e)}"
            self.log_activity("Database Save Failed", {"error": error_msg})

    def add_learning(self, learning: str):
        """Add a new learning to the database."""
        try:
            self.cursor.execute('''
                INSERT INTO learnings (timestamp, learning)
                VALUES (?, ?)
            ''', (datetime.now().isoformat(), learning))
            self.conn.commit()
            self.log_activity("New Learning Added", {"learning": learning})
        except Exception as e:
            error_msg = f"Error adding learning to database: {str(e)}"
            self.log_activity("Learning Addition Failed", {"error": error_msg})

    def get_recent_learnings(self, limit: int = 5) -> List[str]:
        """Retrieve the most recent learnings from the database."""
        try:
            self.cursor.execute('''
                SELECT learning FROM learnings
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            return [row[0] for row in self.cursor.fetchall()]
        except Exception as e:
            error_msg = f"Error retrieving recent learnings: {str(e)}"
            self.log_activity("Learning Retrieval Failed", {"error": error_msg})
            return []

    def reflect_on_task(self, task: Task):
        """Reflect on the completed task and generate learnings."""
        system_message = """
        You are an AI system reflecting on a completed coding task.
        Analyze the task description, implementation, and execution result to generate insights and learnings.
        Focus on what went well, what could be improved, and any new concepts or techniques that were explored.
        """

        user_message = f"""
        Task Description: {task.description}
        
        Implementation:
        {task.code}
        
        Execution Result:
        {json.dumps(task.execution_result, indent=2)}
        
        Metadata:
        {json.dumps(task.metadata.dict(), indent=2)}
        
        Generate 1-3 key learnings from this task.
        """

        try:
            self.log_activity("Reflecting on Task", {"task_id": task.id})
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
            )
            learnings = response.choices[0].message.content.strip().split('\n')
            for learning in learnings:
                self.add_learning(learning)
            self.log_activity("Task Reflection Completed", {"task_id": task.id, "learnings": learnings})
        except Exception as e:
            error_msg = f"Error reflecting on task: {str(e)}"
            self.log_activity("Task Reflection Failed", {"error": error_msg})

    def run_continuous_loop(self):
        """Run the autonomous continuous coding environment."""
        try:
            print(f"{Fore.CYAN}=== Autonomous Continuous Coding Environment ===")
            print(f"{Fore.YELLOW}Press Ctrl+C to stop the environment at any time.")
            
            while True:
                # Generate a new task
                self.current_task = self.generate_task()
                print(f"\n{Fore.GREEN}New Task Generated: {self.current_task.description}")

                # Implement the task
                self.current_task = self.implement_task(self.current_task)
                print(f"\n{Fore.CYAN}Task Implemented:")
                print(f"{Fore.WHITE}{self.current_task.code}")

                # Execute the task
                self.current_task.execution_result = self.execute_task(self.current_task)

                if self.current_task.execution_result["success"]:
                    self.current_task.status = "completed"
                    print(f"\n{Fore.GREEN}✅ Task executed successfully!")
                    print(f"{Fore.WHITE}Output:\n{self.current_task.execution_result['output']}")
                else:
                    self.current_task.status = "failed"
                    print(f"\n{Fore.RED}❌ Task execution failed.")
                    print(f"{Fore.WHITE}Error:\n{self.current_task.execution_result['error']}")

                    print(f"\n{Fore.YELLOW}Attempting to improve the task...")
                    self.current_task = self.improve_task(self.current_task)

                    print(f"\n{Fore.YELLOW}Re-executing improved task...")
                    self.current_task.execution_result = self.execute_task(self.current_task)

                    if self.current_task.execution_result["success"]:
                        self.current_task.status = "completed"
                        print(f"\n{Fore.GREEN}✅ Improved task executed successfully!")
                        print(f"{Fore.WHITE}Output:\n{self.current_task.execution_result['output']}")
                    else:
                        print(f"\n{Fore.RED}❌ Improved task execution failed.")
                        print(f"{Fore.WHITE}Error:\n{self.current_task.execution_result['error']}")

                # Reflect on the task
                self.reflect_on_task(self.current_task)

                # Save the task to the database
                self.save_task_to_database(self.current_task)

                # Display recent learnings
                recent_learnings = self.get_recent_learnings()
                print(f"\n{Fore.CYAN}=== Recent Learnings ===")
                for learning in recent_learnings:
                    print(f"{Fore.WHITE}• {learning}")

                print(f"\n{Fore.YELLOW}Waiting for 10 seconds before starting the next task...")
                time.sleep(10)

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Autonomous Continuous Coding Environment stopped by user.")
        except Exception as e:
            logger.critical(f"{Fore.RED}💥 Critical error: {str(e)}")
            logger.critical(traceback.format_exc())
        finally:
            self.conn.close()
            print(f"\n{Fore.CYAN}=== Final Statistics ===")
            print(f"{Fore.WHITE}Total tasks attempted: {len(self.activity_log)}")
            print(f"Database and logs saved in: {self.workspace}")

    def run(self):
        """Start the autonomous continuous coding environment."""
        try:
            self.run_continuous_loop()
        except Exception as e:
            logger.critical(f"{Fore.RED}💥 Critical error in main run loop: {str(e)}")
            logger.critical(traceback.format_exc())
        finally:
            self.conn.close()

if __name__ == "__main__":
    try:
        env = AutonomousContinuousCodingEnvironment(workspace="autonomous_continuous_workspace")
        env.run()
    except Exception as e:
        logger.critical(f"{Fore.RED}💥 Critical error: {str(e)}")
        logger.critical(traceback.format_exc())


# [ACTIVITY] Workspace Setup
# {
#   "workspace": "autonomous_continuous_workspace"
# }
# 2024-08-14 00:28:34,630 - INFO - [ACTIVITY] Workspace Setup
# {
#   "workspace": "autonomous_continuous_workspace"
# }
# [ACTIVITY] Database Initialized
# {
#   "path": "autonomous_continuous_workspace/autonomous_system.db"
# }
# 2024-08-14 00:28:34,664 - INFO - [ACTIVITY] Database Initialized
# {
#   "path": "autonomous_continuous_workspace/autonomous_system.db"
# }
# === Autonomous Continuous Coding Environment ===
# Press Ctrl+C to stop the environment at any time.
# [ACTIVITY] Generating New Task
# 2024-08-14 00:28:34,664 - INFO - [ACTIVITY] Generating New Task
# 2024-08-14 00:28:40,783 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
# [ACTIVITY] New Task Generated
# {
#   "task_id": "task_1723609720",
#   "description": "Create a Python program that implements a simple voting system for a fictional election. The program should allow users to create candidates, view the list of candidates, cast votes for them, and display the results. Implement features to ensure that each user can only vote once. Use classes to structure the candidates and voters, and store the votes in a dictionary. The program should also handle invalid inputs gracefully, such as when a user tries to vote for a candidate that does not exist or tries to vote twice.",
#   "rationale": "This task builds upon basic object-oriented programming concepts by requiring the implementation of classes and methods. It also introduces more advanced topics such as data structure usage (dictionaries), handling user input, and maintaining state (voter registration). It encourages thinking about user experience through input validation and error handling."
# }
# 2024-08-14 00:28:40,853 - INFO - [ACTIVITY] New Task Generated
# {
#   "task_id": "task_1723609720",
#   "description": "Create a Python program that implements a simple voting system for a fictional election. The program should allow users to create candidates, view the list of candidates, cast votes for them, and display the results. Implement features to ensure that each user can only vote once. Use classes to structure the candidates and voters, and store the votes in a dictionary. The program should also handle invalid inputs gracefully, such as when a user tries to vote for a candidate that does not exist or tries to vote twice.",
#   "rationale": "This task builds upon basic object-oriented programming concepts by requiring the implementation of classes and methods. It also introduces more advanced topics such as data structure usage (dictionaries), handling user input, and maintaining state (voter registration). It encourages thinking about user experience through input validation and error handling."
# }

# New Task Generated: Create a Python program that implements a simple voting system for a fictional election. The program should allow users to create candidates, view the list of candidates, cast votes for them, and display the results. Implement features to ensure that each user can only vote once. Use classes to structure the candidates and voters, and store the votes in a dictionary. The program should also handle invalid inputs gracefully, such as when a user tries to vote for a candidate that does not exist or tries to vote twice.
# [ACTIVITY] Implementing Task
# {
#   "task_id": "task_1723609720"
# }
# 2024-08-14 00:28:40,854 - INFO - [ACTIVITY] Implementing Task
# {
#   "task_id": "task_1723609720"
# }
# 2024-08-14 00:28:43,035 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 500 Internal Server Error"
# 2024-08-14 00:28:43,035 - INFO - Retrying request to /chat/completions in 0.819104 seconds
# 2024-08-14 00:28:53,457 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
# [ACTIVITY] Generating Task Metadata
# {
#   "task_id": "task_1723609720"
# }
# 2024-08-14 00:28:53,475 - INFO - [ACTIVITY] Generating Task Metadata
# {
#   "task_id": "task_1723609720"
# }
# 2024-08-14 00:29:01,052 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
# [ACTIVITY] Task Metadata Generated
# {
#   "task_id": "task_1723609720",
#   "metadata": {
#     "description": "This Python program implements a simple voting system for a fictional election, allowing users to create candidates, cast votes, and display results while ensuring each voter can only vote once and handling invalid inputs gracefully.",
#     "tags": [
#       "Python",
#       "Voting System",
#       "Election",
#       "Object-Oriented Programming",
#       "User Input Handling",
#       "Logging"
#     ],
#     "complexity": 4,
#     "estimated_time": "2-3 hours",
#     "poetic_description": "In the realm of choices, fair and bright,  \nA system unfolds to guide the vote's light.  \nCandidates arise, names in the air,  \nAs ballots are cast, hope and dreams laid bare."
#   }
# }
# 2024-08-14 00:29:01,069 - INFO - [ACTIVITY] Task Metadata Generated
# {
#   "task_id": "task_1723609720",
#   "metadata": {
#     "description": "This Python program implements a simple voting system for a fictional election, allowing users to create candidates, cast votes, and display results while ensuring each voter can only vote once and handling invalid inputs gracefully.",
#     "tags": [
#       "Python",
#       "Voting System",
#       "Election",
#       "Object-Oriented Programming",
#       "User Input Handling",
#       "Logging"
#     ],
#     "complexity": 4,
#     "estimated_time": "2-3 hours",
#     "poetic_description": "In the realm of choices, fair and bright,  \nA system unfolds to guide the vote's light.  \nCandidates arise, names in the air,  \nAs ballots are cast, hope and dreams laid bare."
#   }
# }
# [ACTIVITY] Task Implemented
# {
#   "task_id": "task_1723609720"
# }
# 2024-08-14 00:29:01,069 - INFO - [ACTIVITY] Task Implemented
# {
#   "task_id": "task_1723609720"
# }

# Task Implemented:
# import logging

# # Configuring logging 📝
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# class Candidate:
#     """Represents a candidate in the voting system."""
#     def __init__(self, name):
#         self.name = name
#         self.votes = 0

#     def __str__(self):
#         return f'{self.name}: {self.votes} votes'

# class VotingSystem:
#     """Manages the voting process and ensures that users can only vote once."""
#     def __init__(self):
#         self.candidates = {}  # Dictionary to store candidates
#         self.voters = set()   # Set to track who has voted

#     def add_candidate(self, name):
#         """Adds a new candidate to the voting system."""
#         if name in self.candidates:
#             logging.warning('Candidate %s already exists!', name)  # Log warning 🚨
#             raise ValueError('Candidate already exists')
#         self.candidates[name] = Candidate(name)
#         logging.info('Added candidate: %s', name)  # Log info message 🗳️

#     def vote(self, voter_name, candidate_name):
#         """Records a vote for a candidate from a voter."""
#         if voter_name in self.voters:
#             logging.error('Voter %s has already voted!', voter_name)  # Log error ❌
#             raise ValueError('You already voted')
#         if candidate_name not in self.candidates:
#             logging.error('Candidate %s does not exist!', candidate_name)  # Log error ❌
#             raise ValueError('Candidate does not exist')

#         self.candidates[candidate_name].votes += 1
#         self.voters.add(voter_name)  # Mark this voter as having voted
#         logging.info('Voter %s voted for %s', voter_name, candidate_name)  # Log info message 🗳️

#     def display_results(self):
#         """Displays the voting results for all candidates."""
#         logging.info('Displaying voting results...')  # Log info message 📊
#         for candidate in self.candidates.values():
#             print(candidate)

# def main():
#     system = VotingSystem()
#     while True:
#         print('\nWelcome to the Voting System! Please choose an action:')
#         print('1. Add a candidate')
#         print('2. Vote')
#         print('3. Display results')
#         print('4. Exit')
#         action = input('Enter action number: ')  # User input for action

#         try:
#             if action == '1':
#                 candidate_name = input('Enter the candidate name: ')
#                 system.add_candidate(candidate_name)
#             elif action == '2':
#                 voter_name = input('Enter your name: ')  # Voter's name
#                 candidate_name = input('Enter the candidate name to vote for: ')
#                 system.vote(voter_name, candidate_name)
#             elif action == '3':
#                 system.display_results()
#             elif action == '4':
#                 logging.info('Exiting voting system.')  # Log info message 🚪
#                 break
#             else:
#                 print('Invalid action! Please choose a valid option.')  # Invalid action
#         except ValueError as e:
#             print(e)  # Display the error to the user

# if __name__ == '__main__':
#     main()  # Start the voting system application 🎉
# [ACTIVITY] Executing Task
# {
#   "task_id": "task_1723609720",
#   "filename": "autonomous_continuous_workspace/task_1723609720.py"
# }
# 2024-08-14 00:29:01,070 - INFO - [ACTIVITY] Executing Task
# {
#   "task_id": "task_1723609720",
#   "filename": "autonomous_continuous_workspace/task_1723609720.py"
# }
# [ACTIVITY] Task Execution Timed Out
# {
#   "error": "Task execution timed out: task_1723609720"
# }
# 2024-08-14 00:29:31,102 - INFO - [ACTIVITY] Task Execution Timed Out
# {
#   "error": "Task execution timed out: task_1723609720"
# }

# ❌ Task execution failed.
# Error:
# Execution timed out

# Attempting to improve the task...
# [ACTIVITY] Improving Task
# {
#   "task_id": "task_1723609720"
# }
# 2024-08-14 00:29:31,103 - INFO - [ACTIVITY] Improving Task
# {
#   "task_id": "task_1723609720"
# }
# 2024-08-14 00:29:39,551 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
# [ACTIVITY] Task Improved
# {
#   "task_id": "task_1723609720"
# }
# 2024-08-14 00:29:39,558 - INFO - [ACTIVITY] Task Improved
# {
#   "task_id": "task_1723609720"
# }

# Re-executing improved task...
# [ACTIVITY] Executing Task
# {
#   "task_id": "task_1723609720",
#   "filename": "autonomous_continuous_workspace/task_1723609720.py"
# }
# 2024-08-14 00:29:39,559 - INFO - [ACTIVITY] Executing Task
# {
#   "task_id": "task_1723609720",
#   "filename": "autonomous_continuous_workspace/task_1723609720.py"
# }
# [ACTIVITY] Task Execution Timed Out
# {
#   "error": "Task execution timed out: task_1723609720"
# }
# 2024-08-14 00:30:09,597 - INFO - [ACTIVITY] Task Execution Timed Out
# {
#   "error": "Task execution timed out: task_1723609720"
# }

# ❌ Improved task execution failed.
# Error:
# Execution timed out
# [ACTIVITY] Reflecting on Task
# {
#   "task_id": "task_1723609720"
# }
# 2024-08-14 00:30:09,598 - INFO - [ACTIVITY] Reflecting on Task
# {
#   "task_id": "task_1723609720"
# }
# 2024-08-14 00:30:14,926 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
# [ACTIVITY] New Learning Added
# {
#   "learning": "1. **Effective Use of Object-Oriented Programming**: The implementation successfully utilized classes to structure the voting system. The `Candidate` class neatly encapsulates candidate-related data, while the `VotingSystem` class manages the overall voting logic. This approach enhances code organization, making it easier to maintain and scale if new features were to be introduced in the future."
# }
# 2024-08-14 00:30:14,939 - INFO - [ACTIVITY] New Learning Added
# {
#   "learning": "1. **Effective Use of Object-Oriented Programming**: The implementation successfully utilized classes to structure the voting system. The `Candidate` class neatly encapsulates candidate-related data, while the `VotingSystem` class manages the overall voting logic. This approach enhances code organization, making it easier to maintain and scale if new features were to be introduced in the future."
# }
# [ACTIVITY] New Learning Added
# {
#   "learning": ""
# }
# 2024-08-14 00:30:14,955 - INFO - [ACTIVITY] New Learning Added
# {
#   "learning": ""
# }
# [ACTIVITY] New Learning Added
# {
#   "learning": "2. **Robust Input Validation and Error Handling**: The program showcased a comprehensive way of handling invalid inputs. By checking whether a candidate exists and ensuring that each voter can only vote once, the system prevented common issues that could lead to incorrect voting results. The use of exception handling with `try` and `except` blocks ensures that users are informed of mistakes gracefully without crashing the program, enhancing user experience."
# }
# 2024-08-14 00:30:14,966 - INFO - [ACTIVITY] New Learning Added
# {
#   "learning": "2. **Robust Input Validation and Error Handling**: The program showcased a comprehensive way of handling invalid inputs. By checking whether a candidate exists and ensuring that each voter can only vote once, the system prevented common issues that could lead to incorrect voting results. The use of exception handling with `try` and `except` blocks ensures that users are informed of mistakes gracefully without crashing the program, enhancing user experience."
# }
# [ACTIVITY] New Learning Added
# {
#   "learning": ""
# }
# 2024-08-14 00:30:14,978 - INFO - [ACTIVITY] New Learning Added
# {
#   "learning": ""
# }
# [ACTIVITY] New Learning Added
# {
#   "learning": "3. **Logging for Transparency and Debugging**: Implementing the logging module added a layer of transparency to the program's operations. Each major action, such as adding candidates or casting votes, is accompanied by an appropriate log message. This not only aids in debugging by providing a trail of actions taken but also allows for future performance monitoring or system audits, as logs can be reviewed to understand user interactions and identify any issues that arise during operation. However, careful setup of the logger level is critical to avoid flooding logs with excessive information. "
# }
# 2024-08-14 00:30:14,990 - INFO - [ACTIVITY] New Learning Added
# {
#   "learning": "3. **Logging for Transparency and Debugging**: Implementing the logging module added a layer of transparency to the program's operations. Each major action, such as adding candidates or casting votes, is accompanied by an appropriate log message. This not only aids in debugging by providing a trail of actions taken but also allows for future performance monitoring or system audits, as logs can be reviewed to understand user interactions and identify any issues that arise during operation. However, careful setup of the logger level is critical to avoid flooding logs with excessive information. "
# }
# [ACTIVITY] New Learning Added
# {
#   "learning": ""
# }
# 2024-08-14 00:30:15,005 - INFO - [ACTIVITY] New Learning Added
# {
#   "learning": ""
# }
# [ACTIVITY] New Learning Added
# {
#   "learning": "### Improvement Opportunities:"
# }
# 2024-08-14 00:30:15,016 - INFO - [ACTIVITY] New Learning Added
# {
#   "learning": "### Improvement Opportunities:"
# }
# [ACTIVITY] New Learning Added
# {
#   "learning": "- **Timeout Error Handling**: The execution timed out, suggesting that there might have been an infinite loop or excessive waiting in the code. Adding conditions to break out of loops or refining user input handling could help mitigate this issue, ensuring the program runs efficiently without getting stuck."
# }
# 2024-08-14 00:30:15,028 - INFO - [ACTIVITY] New Learning Added
# {
#   "learning": "- **Timeout Error Handling**: The execution timed out, suggesting that there might have been an infinite loop or excessive waiting in the code. Adding conditions to break out of loops or refining user input handling could help mitigate this issue, ensuring the program runs efficiently without getting stuck."
# }
# [ACTIVITY] New Learning Added
# {
#   "learning": ""
# }
# 2024-08-14 00:30:15,039 - INFO - [ACTIVITY] New Learning Added
# {
#   "learning": ""
# }
# [ACTIVITY] New Learning Added
# {
#   "learning": "- **Improved User Interface**: While a command-line interface is sufficient for basic interaction, enhancing the user prompts and outputs could further improve user experience. Implementing clear instructions and feedback after each action could guide users more effectively, particularly when input errors occur."
# }
# 2024-08-14 00:30:15,051 - INFO - [ACTIVITY] New Learning Added
# {
#   "learning": "- **Improved User Interface**: While a command-line interface is sufficient for basic interaction, enhancing the user prompts and outputs could further improve user experience. Implementing clear instructions and feedback after each action could guide users more effectively, particularly when input errors occur."
# }
# [ACTIVITY] New Learning Added
# {
#   "learning": ""
# }
# 2024-08-14 00:30:15,065 - INFO - [ACTIVITY] New Learning Added
# {
#   "learning": ""
# }
# [ACTIVITY] New Learning Added
# {
#   "learning": "- **Unit Testing**: To ensure the reliability of the voting system, adding unit tests would be beneficial. These tests can verify the correctness of individual components, such as vote counting and candidate addition, reducing the risk of undetected bugs in future code updates."
# }
# 2024-08-14 00:30:15,079 - INFO - [ACTIVITY] New Learning Added
# {
#   "learning": "- **Unit Testing**: To ensure the reliability of the voting system, adding unit tests would be beneficial. These tests can verify the correctness of individual components, such as vote counting and candidate addition, reducing the risk of undetected bugs in future code updates."
# }
# [ACTIVITY] Task Reflection Completed
# {
#   "task_id": "task_1723609720",
#   "learnings": [
#     "1. **Effective Use of Object-Oriented Programming**: The implementation successfully utilized classes to structure the voting system. The `Candidate` class neatly encapsulates candidate-related data, while the `VotingSystem` class manages the overall voting logic. This approach enhances code organization, making it easier to maintain and scale if new features were to be introduced in the future.",
#     "",
#     "2. **Robust Input Validation and Error Handling**: The program showcased a comprehensive way of handling invalid inputs. By checking whether a candidate exists and ensuring that each voter can only vote once, the system prevented common issues that could lead to incorrect voting results. The use of exception handling with `try` and `except` blocks ensures that users are informed of mistakes gracefully without crashing the program, enhancing user experience.",
#     "",
#     "3. **Logging for Transparency and Debugging**: Implementing the logging module added a layer of transparency to the program's operations. Each major action, such as adding candidates or casting votes, is accompanied by an appropriate log message. This not only aids in debugging by providing a trail of actions taken but also allows for future performance monitoring or system audits, as logs can be reviewed to understand user interactions and identify any issues that arise during operation. However, careful setup of the logger level is critical to avoid flooding logs with excessive information. ",
#     "",
#     "### Improvement Opportunities:",
#     "- **Timeout Error Handling**: The execution timed out, suggesting that there might have been an infinite loop or excessive waiting in the code. Adding conditions to break out of loops or refining user input handling could help mitigate this issue, ensuring the program runs efficiently without getting stuck.",
#     "",
#     "- **Improved User Interface**: While a command-line interface is sufficient for basic interaction, enhancing the user prompts and outputs could further improve user experience. Implementing clear instructions and feedback after each action could guide users more effectively, particularly when input errors occur.",
#     "",
#     "- **Unit Testing**: To ensure the reliability of the voting system, adding unit tests would be beneficial. These tests can verify the correctness of individual components, such as vote counting and candidate addition, reducing the risk of undetected bugs in future code updates."
#   ]
# }
# 2024-08-14 00:30:15,081 - INFO - [ACTIVITY] Task Reflection Completed
# {
#   "task_id": "task_1723609720",
#   "learnings": [
#     "1. **Effective Use of Object-Oriented Programming**: The implementation successfully utilized classes to structure the voting system. The `Candidate` class neatly encapsulates candidate-related data, while the `VotingSystem` class manages the overall voting logic. This approach enhances code organization, making it easier to maintain and scale if new features were to be introduced in the future.",
#     "",
#     "2. **Robust Input Validation and Error Handling**: The program showcased a comprehensive way of handling invalid inputs. By checking whether a candidate exists and ensuring that each voter can only vote once, the system prevented common issues that could lead to incorrect voting results. The use of exception handling with `try` and `except` blocks ensures that users are informed of mistakes gracefully without crashing the program, enhancing user experience.",
#     "",
#     "3. **Logging for Transparency and Debugging**: Implementing the logging module added a layer of transparency to the program's operations. Each major action, such as adding candidates or casting votes, is accompanied by an appropriate log message. This not only aids in debugging by providing a trail of actions taken but also allows for future performance monitoring or system audits, as logs can be reviewed to understand user interactions and identify any issues that arise during operation. However, careful setup of the logger level is critical to avoid flooding logs with excessive information. ",
#     "",
#     "### Improvement Opportunities:",
#     "- **Timeout Error Handling**: The execution timed out, suggesting that there might have been an infinite loop or excessive waiting in the code. Adding conditions to break out of loops or refining user input handling could help mitigate this issue, ensuring the program runs efficiently without getting stuck.",
#     "",
#     "- **Improved User Interface**: While a command-line interface is sufficient for basic interaction, enhancing the user prompts and outputs could further improve user experience. Implementing clear instructions and feedback after each action could guide users more effectively, particularly when input errors occur.",
#     "",
#     "- **Unit Testing**: To ensure the reliability of the voting system, adding unit tests would be beneficial. These tests can verify the correctness of individual components, such as vote counting and candidate addition, reducing the risk of undetected bugs in future code updates."
#   ]
# }
# [ACTIVITY] Task Saved to Database
# {
#   "task_id": "task_1723609720"
# }
# 2024-08-14 00:30:15,096 - INFO - [ACTIVITY] Task Saved to Database
# {
#   "task_id": "task_1723609720"
# }

# === Recent Learnings ===
# • - **Unit Testing**: To ensure the reliability of the voting system, adding unit tests would be beneficial. These tests can verify the correctness of individual components, such as vote counting and candidate addition, reducing the risk of undetected bugs in future code updates.
# • 
# • - **Improved User Interface**: While a command-line interface is sufficient for basic interaction, enhancing the user prompts and outputs could further improve user experience. Implementing clear instructions and feedback after each action could guide users more effectively, particularly when input errors occur.
# • 
# • - **Timeout Error Handling**: The execution timed out, suggesting that there might have been an infinite loop or excessive waiting in the code. Adding conditions to break out of loops or refining user input handling could help mitigate this issue, ensuring the program runs efficiently without getting stuck.

# Waiting for 10 seconds before starting the next task...
# [ACTIVITY] Generating New Task
# 2024-08-14 00:30:25,098 - INFO - [ACTIVITY] Generating New Task
# 2024-08-14 00:30:33,244 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
# [ACTIVITY] New Task Generated
# {
#   "task_id": "task_1723609833",
#   "description": "Create a command-line-based contact management application in Python. The application should allow users to add, view, search, and delete contacts. Each contact should have a name, phone number, and email address. Store the contacts in a JSON file and ensure that all data is persisted across sessions. Implement error handling for invalid inputs such as duplicate contacts, incorrect email formats, and missing fields when adding a new contact. Additionally, include a feature to list all contacts sorted by name.",
#   "rationale": "This task builds upon prior knowledge of handling data structures, file I/O, and input validation in Python. It challenges the developer to implement a complete CLI application with basic CRUD (Create, Read, Update, Delete) functionality while also reinforcing skills in data persistence and validation. The use of JSON introduces the developer to serialization and deserialization concepts, enhancing their understanding of data management in programming."
# }
# 2024-08-14 00:30:33,251 - INFO - [ACTIVITY] New Task Generated
# {
#   "task_id": "task_1723609833",
#   "description": "Create a command-line-based contact management application in Python. The application should allow users to add, view, search, and delete contacts. Each contact should have a name, phone number, and email address. Store the contacts in a JSON file and ensure that all data is persisted across sessions. Implement error handling for invalid inputs such as duplicate contacts, incorrect email formats, and missing fields when adding a new contact. Additionally, include a feature to list all contacts sorted by name.",
#   "rationale": "This task builds upon prior knowledge of handling data structures, file I/O, and input validation in Python. It challenges the developer to implement a complete CLI application with basic CRUD (Create, Read, Update, Delete) functionality while also reinforcing skills in data persistence and validation. The use of JSON introduces the developer to serialization and deserialization concepts, enhancing their understanding of data management in programming."
# }

# New Task Generated: Create a command-line-based contact management application in Python. The application should allow users to add, view, search, and delete contacts. Each contact should have a name, phone number, and email address. Store the contacts in a JSON file and ensure that all data is persisted across sessions. Implement error handling for invalid inputs such as duplicate contacts, incorrect email formats, and missing fields when adding a new contact. Additionally, include a feature to list all contacts sorted by name.
# [ACTIVITY] Implementing Task
# {
#   "task_id": "task_1723609833"
# }
# 2024-08-14 00:30:33,252 - INFO - [ACTIVITY] Implementing Task
# {
#   "task_id": "task_1723609833"
# }
# 2024-08-14 00:30:46,387 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
# [ACTIVITY] Generating Task Metadata
# {
#   "task_id": "task_1723609833"
# }
# 2024-08-14 00:30:46,389 - INFO - [ACTIVITY] Generating Task Metadata
# {
#   "task_id": "task_1723609833"
# }
# 2024-08-14 00:30:48,311 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
# [ACTIVITY] Task Metadata Generated
# {
#   "task_id": "task_1723609833",
#   "metadata": {
#     "description": "A command-line-based contact management application in Python that allows users to add, view, search, and delete contacts, while handling errors and ensuring data persistence with JSON.",
#     "tags": [
#       "Python",
#       "Command-Line",
#       "Contact Management",
#       "JSON",
#       "Error Handling"
#     ],
#     "complexity": 5,
#     "estimated_time": "4-6 hours",
#     "poetic_description": "In a world where names and numbers bloom,  \nA keeper of contacts dispels all gloom.  \nWith each input, a story to spin,  \nA digital friend, let the search begin."
#   }
# }
# 2024-08-14 00:30:48,318 - INFO - [ACTIVITY] Task Metadata Generated
# {
#   "task_id": "task_1723609833",
#   "metadata": {
#     "description": "A command-line-based contact management application in Python that allows users to add, view, search, and delete contacts, while handling errors and ensuring data persistence with JSON.",
#     "tags": [
#       "Python",
#       "Command-Line",
#       "Contact Management",
#       "JSON",
#       "Error Handling"
#     ],
#     "complexity": 5,
#     "estimated_time": "4-6 hours",
#     "poetic_description": "In a world where names and numbers bloom,  \nA keeper of contacts dispels all gloom.  \nWith each input, a story to spin,  \nA digital friend, let the search begin."
#   }
# }
# [ACTIVITY] Task Implemented
# {
#   "task_id": "task_1723609833"
# }
# 2024-08-14 00:30:48,318 - INFO - [ACTIVITY] Task Implemented
# {
#   "task_id": "task_1723609833"
# }

# Task Implemented:
# import json  # Importing JSON module for data serialization
# import os  # Importing OS module for file path checks
# import re  # Importing regular expression module for email validation

# # Contact class to represent each contact entry
# class Contact:
#     def __init__(self, name, phone, email):
#         self.name = name
#         self.phone = phone
#         self.email = email

#     def __repr__(self):
#         return f'{self.name} | {self.phone} | {self.email}'

# # Function to load contacts from a JSON file
# def load_contacts(filename):
#     if not os.path.exists(filename):  # Check if file exists
#         return []  # Return empty list if no contacts file
#     with open(filename, 'r') as f:
#         return json.load(f)  # Load and return contacts from file

# # Function to save contacts to a JSON file
# def save_contacts(filename, contacts):
#     with open(filename, 'w') as f:
#         json.dump([contact.__dict__ for contact in contacts], f, indent=4)  # Save contacts data to file

# # Function to validate email format
# def is_valid_email(email):
#     pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
#     return re.match(pattern, email) is not None  # Validate the email format with regex

# # Function to check for duplicate contact
# def is_duplicate(contacts, new_contact):
#     return any(c.name == new_contact.name for c in contacts)  # Check if contact already exists by name

# # Function to display all contacts
# def display_contacts(contacts):
#     if not contacts:
#         print('No contacts available.')  # Message for no contacts
#         return
#     sorted_contacts = sorted(contacts, key=lambda c: c.name)  # Sort contacts by name
#     print('\nList of Contacts:')
#     print('Name | Phone | Email')
#     print('----------------------')
#     for contact in sorted_contacts:
#         print(contact)

# # Main function to drive the application
# def main():
#     filename = 'contacts.json'  # Filename for the contacts
#     contacts = load_contacts(filename)  # Load existing contacts

#     while True:
#         print('\n--- Contact Management Application ---')  # Application title
#         print('1. Add Contact')
#         print('2. View Contacts')
#         print('3. Search Contact')
#         print('4. Delete Contact')
#         print('5. Exit')

#         choice = input('Choose an option (1-5): ')  # User menu

#         if choice == '1':
#             name = input('Enter name: ')  # Name input
#             phone = input('Enter phone number: ')  # Phone number input
#             email = input('Enter email address: ')  # Email input

#             if not name or not phone or not email:
#                 print('Error: All fields must be filled. 🚫')  # Error for empty fields
#                 continue

#             if not is_valid_email(email):
#                 print('Error: Invalid email format. 📧')  # Error for invalid email
#                 continue

#             new_contact = Contact(name, phone, email)
#             if is_duplicate(contacts, new_contact):
#                 print('Error: Contact already exists. ❌')  # Error for duplicate contact
#                 continue

#             contacts.append(new_contact)  # Add new contact
#             save_contacts(filename, contacts)  # Save to file
#             print('Contact added successfully! ✔️')  # Success message

#         elif choice == '2':
#             display_contacts(contacts)  # Display all contacts

#         elif choice == '3':
#             search_name = input('Enter name to search: ')  # Search input
#             found_contacts = [c for c in contacts if search_name.lower() in c.name.lower()]
#             if found_contacts:
#                 print('Search Results:')
#                 for contact in found_contacts:
#                     print(contact)  # Display search results
#             else:
#                 print('No contacts found with that name. 🕵️‍♂️')  # No match found

#         elif choice == '4':
#             delete_name = input('Enter name to delete: ')  # Name to delete
#             contacts = [c for c in contacts if c.name.lower() != delete_name.lower()]  # Remove contact if name matches
#             save_contacts(filename, contacts)  # Save changes to file
#             print(f'Contact {delete_name} deleted successfully! 🗑️')  # Success message

#         elif choice == '5':
#             print('Exiting the application. Goodbye! 👋')  # Exit message
#             break  # Exit loop

#         else:
#             print('Invalid option. Please select a valid choice. ❓')  # Error for invalid option

# # Entry point of the script
# if __name__ == '__main__':
#     main()  # Run the main function
# [ACTIVITY] Executing Task
# {
#   "task_id": "task_1723609833",
#   "filename": "autonomous_continuous_workspace/task_1723609833.py"
# }
# 2024-08-14 00:30:48,321 - INFO - [ACTIVITY] Executing Task
# {
#   "task_id": "task_1723609833",
#   "filename": "autonomous_continuous_workspace/task_1723609833.py"
# }
# ^C
# Autonomous Continuous Coding Environment stopped by user.

# === Final Statistics ===
# Total tasks attempted: 36
# Database and logs saved in: autonomous_continuous_workspace



================================================
File: base_task_processor.py
================================================
import os
import subprocess
import time
import openai
from typing import List, Dict, Any, Optional
import json
import logging
from pydantic import BaseModel, Field
from colorama import Fore, Style, init
from datetime import datetime
import traceback
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Initialize OpenAI client
client = openai.OpenAI()

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodeBlock(BaseModel):
    language: str
    code: str

class ScriptResponse(BaseModel):
    explanation: str
    code_blocks: List[CodeBlock]

class Metadata(BaseModel):
    description: str
    tags: List[str] = Field(default_factory=list)
    complexity: int = Field(ge=1, le=10)
    estimated_time: str
    poetic_description: str

class MetadataResponse(BaseModel):
    description: str
    tags: List[str]
    complexity: int
    estimated_time: str
    poetic_description: str

class Task(BaseModel):
    id: str
    description: str
    code: str = ""
    metadata: Optional[Metadata] = None
    dependencies: List[str] = Field(default_factory=list)
    status: str = "pending"  # pending, in_progress, completed, failed
    execution_result: Optional[Dict[str, Any]] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class BaseTaskProcessor:
    """Base class for task processing functionality shared across ACE variants."""
    
    def __init__(self, model: str = "o4-mini"):
        self.model = model
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def generate_code_for_task(self, task: Task, additional_context: str = "") -> str:
        """Generate code for a task using structured output with improved error handling."""
        system_message = """
        You are an expert Python developer tasked with implementing a specific coding task.
        Provide a complete and working implementation for the given task description.
        
        IMPORTANT REQUIREMENTS:
        - Include error handling, logging, and comments in your code
        - Make the code production-ready with proper structure and documentation
        - The code should run successfully when executed directly
        - Do NOT include command-line argument parsing in the main block
        - If including a main block, make it demonstrate the functionality with example values
        - Use simple examples that don't require user input or external files
        - Focus on implementing the core functionality requested
        
        The code should be self-contained and executable without any external dependencies beyond standard library.
        """
        
        user_message = f"Task: {task.description}"
        if additional_context:
            user_message += f"\n\nAdditional Context: {additional_context}"
        user_message += "\n\nImplement this task in Python."

        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.logger.info(f"{Fore.BLUE}🔄 Generating code for task {task.id} (attempt {attempt + 1})")
                response = client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    response_format=ScriptResponse,
                )
                
                if not response.choices or not response.choices[0].message.parsed:
                    raise ValueError("Empty response from OpenAI API")
                    
                script_response = response.choices[0].message.parsed
                
                # Extract the main Python code block
                main_code_block = next(
                    (block for block in script_response.code_blocks if block.language.lower() == 'python'), 
                    None
                )
                
                if main_code_block:
                    self.logger.info(f"{Fore.GREEN}✅ Code generated successfully for task {task.id}")
                    return main_code_block.code
                else:
                    raise ValueError("No Python code block found in the response")
                    
            except Exception as e:
                self.logger.warning(f"{Fore.YELLOW}⚠️ Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    error_code = f"""# Error generating code: {str(e)}
# Fallback implementation for task: {task.description}

def main():
    '''
    Fallback implementation - requires manual completion
    Task: {task.description}
    '''
    raise NotImplementedError(f'Task implementation failed: {str(e)}')

if __name__ == "__main__":
    main()
"""
                    self.logger.error(f"{Fore.RED}❌ All code generation attempts failed for task {task.id}")
                    return error_code
                    
                # Wait before retry
                time.sleep(2 ** attempt)
                
        return error_code  # This shouldn't be reached, but just in case

    def generate_task_metadata(self, task: Task) -> Metadata:
        """Generate metadata for a task with improved error handling."""
        system_message = """
        You are an AI expert in software development and poetry. Analyze the given task and its code to generate metadata.
        Provide a concise description, relevant tags, estimate the complexity (1-10), and estimated time to complete.
        Create a poetic description that captures the essence of the task using coding metaphors and imagery.
        """
        
        user_message = f"""
        Task: {task.description}
        
        Code:
        {task.code}
        
        Generate metadata including a concise description, relevant tags, complexity (1-10), 
        estimated time to complete, and a poetic description with coding metaphors.
        """

        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.logger.info(f"{Fore.BLUE}🔄 Generating metadata for task {task.id} (attempt {attempt + 1})")
                response = client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    response_format=MetadataResponse,
                )
                
                if not response.choices or not response.choices[0].message.parsed:
                    raise ValueError("Empty metadata response from OpenAI API")
                    
                metadata_response = response.choices[0].message.parsed
                
                generated_metadata = Metadata(
                    description=metadata_response.description,
                    tags=metadata_response.tags,
                    complexity=metadata_response.complexity,
                    estimated_time=metadata_response.estimated_time,
                    poetic_description=metadata_response.poetic_description
                )
                
                self.logger.info(f"{Fore.GREEN}✅ Metadata generated successfully for task {task.id}")
                return generated_metadata
                
            except Exception as e:
                self.logger.warning(f"{Fore.YELLOW}⚠️ Metadata generation attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    fallback_metadata = Metadata(
                        description=task.description,
                        tags=["auto-generated", "fallback"],
                        complexity=5,
                        estimated_time="unknown",
                        poetic_description="A task wrapped in digital mystery, awaiting its moment to shine in the code."
                    )
                    self.logger.error(f"{Fore.RED}❌ All metadata generation attempts failed for task {task.id}")
                    return fallback_metadata
                
                time.sleep(2 ** attempt)
                
        return fallback_metadata  # This shouldn't be reached

    def execute_task_code(self, task: Task, workspace_path: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute task code with enhanced error handling and timeout management."""
        task_filename = os.path.join(workspace_path, f"{task.id}.py")
        
        try:
            # Ensure workspace directory exists
            os.makedirs(workspace_path, exist_ok=True)
            
            # Write code to file
            with open(task_filename, 'w', encoding='utf-8') as f:
                f.write(task.code)
            
            self.logger.info(f"{Fore.BLUE}🚀 Executing task {task.id} with timeout {timeout}s")
            self.logger.debug(f"Task file: {task_filename}")
            self.logger.debug(f"File exists: {os.path.exists(task_filename)}")
            
            # Execute with timeout - use just the filename since cwd is set to workspace_path
            result = subprocess.run(
                ['python', f"{task.id}.py"], 
                capture_output=True, 
                text=True, 
                timeout=timeout,
                cwd=workspace_path
            )
            
            execution_result = {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "return_code": result.returncode,
                "execution_time": time.time(),  # Could be improved to measure actual execution time
                "task_file": task_filename
            }
            
            if execution_result["success"]:
                self.logger.info(f"{Fore.GREEN}✅ Task {task.id} executed successfully")
            else:
                self.logger.warning(f"{Fore.YELLOW}⚠️ Task {task.id} execution failed with return code {result.returncode}")
                self.logger.debug(f"STDOUT: {result.stdout}")
                self.logger.debug(f"STDERR: {result.stderr}")
            
            return execution_result
            
        except subprocess.TimeoutExpired:
            self.logger.warning(f"{Fore.YELLOW}⏳ Task {task.id} execution timed out after {timeout}s")
            return {
                "success": False, 
                "error": f"Execution timed out after {timeout} seconds",
                "timeout": True,
                "execution_time": timeout
            }
        except Exception as e:
            self.logger.error(f"{Fore.RED}❌ Error executing task {task.id}: {str(e)}")
            return {
                "success": False, 
                "error": str(e),
                "exception": True
            }
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(task_filename):
                    os.remove(task_filename)
            except Exception as e:
                self.logger.warning(f"{Fore.YELLOW}⚠️ Failed to clean up task file {task_filename}: {str(e)}")

    def improve_task_code(self, task: Task, execution_result: Dict[str, Any]) -> str:
        """Improve task code based on execution results with targeted error analysis."""
        system_message = """
        You are an expert Python debugger and code optimizer. 
        Analyze the failed code execution and provide an improved version.
        Focus on fixing specific errors, improving performance, and adding robustness.
        Provide a complete, working implementation.
        """
        
        error_analysis = []
        if execution_result.get("timeout"):
            error_analysis.append("CODE TIMED OUT - Focus on performance optimization and efficiency")
        if execution_result.get("error"):
            error_analysis.append(f"EXECUTION ERROR: {execution_result['error']}")
        if execution_result.get("return_code", 0) != 0:
            error_analysis.append(f"Non-zero exit code: {execution_result['return_code']}")
            
        user_message = f"""
        Task: {task.description}
        
        Original Code:
        {task.code}
        
        Execution Issues:
        {chr(10).join(error_analysis)}
        
        Error Output:
        {execution_result.get('error', 'No error output')}
        
        Provide an improved version of the code that fixes these issues.
        """

        try:
            self.logger.info(f"{Fore.BLUE}🔧 Improving code for task {task.id}")
            response = client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                response_format=ScriptResponse,
            )
            
            script_response = response.choices[0].message.parsed
            main_code_block = next(
                (block for block in script_response.code_blocks if block.language.lower() == 'python'), 
                None
            )
            
            if main_code_block:
                self.logger.info(f"{Fore.GREEN}✅ Code improved for task {task.id}")
                return main_code_block.code
            else:
                raise ValueError("No Python code block found in improvement response")
                
        except Exception as e:
            self.logger.error(f"{Fore.RED}❌ Error improving task code: {str(e)}")
            # Return original code with error comments
            return f"""# ERROR: Failed to improve code - {str(e)}
# Original code returned as fallback

{task.code}
"""

    def update_task_status(self, task: Task, status: str) -> Task:
        """Update task status and timestamp."""
        task.status = status
        task.updated_at = datetime.now().isoformat()
        return task

    def validate_task(self, task: Task) -> List[str]:
        """Validate task completeness and return list of issues."""
        issues = []
        
        if not task.description.strip():
            issues.append("Task description is empty")
            
        if not task.code.strip():
            issues.append("Task code is empty")
            
        if task.metadata and task.metadata.complexity < 1:
            issues.append("Task complexity is invalid")
            
        # Basic Python syntax check
        try:
            compile(task.code, f"<task_{task.id}>", "exec")
        except SyntaxError as e:
            issues.append(f"Python syntax error: {str(e)}")
            
        return issues 


================================================
File: diff.py
================================================
import os
import subprocess
import time
import random
import openai
from typing import List, Dict, Any, Optional, Union
import json
import logging
import shutil
from pydantic import BaseModel, Field
import jsonlines
from tqdm import tqdm
from colorama import Fore, Style, init
import traceback

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize OpenAI client (make sure to set your API key in environment variables)
client = openai.OpenAI()

# ... (previous class definitions remain unchanged)

class TaskLine(BaseModel):
    """Represents a single task in one line."""
    line: str

class OneLineTaskListResponse(BaseModel):
    """Represents the response for a one-line task list."""
    tasks: List[TaskLine]
    summary: str

class EnhancedAutonomousCodingEnvironment:
    def __init__(self, model: str = "o4-mini", workspace: str = "enhanced_autonomous_workspace"):
        self.model = model
        self.workspace = workspace
        self.projects: Dict[str, Project] = {}
        self.task_library = TaskLibrary()
        self.setup_workspace()

    # ... (previous methods remain unchanged)

    def get_task_list(self, project_id: str) -> OneLineTaskListResponse:
        """Get an intuitive task list for a project using structured output, with one task per line."""
        project = self.projects.get(project_id)
        if not project:
            return OneLineTaskListResponse(tasks=[], summary="Project not found")

        system_message = """
        You are an AI project manager assistant. Given a list of tasks for a project,
        create an intuitive and organized task list. Each task should be represented
        in a single line, including its ID, status, and a brief description.
        Group tasks by their status and provide a brief summary of the project's progress.

        Use the following format for each task line:
        [STATUS] ID: Brief description (Dependencies: dep1, dep2)

        STATUS should be one of: PENDING, IN PROGRESS, COMPLETED, FAILED
        If there are no dependencies, omit the parentheses.
        """

        task_data = [
            {
                "id": task.id,
                "description": task.description,
                "status": task.status,
                "dependencies": task.dependencies,
                "metadata": task.metadata.dict() if task.metadata else None
            }
            for task in project.tasks
        ]

        user_message = f"""
        Project: {project.name}
        Description: {project.description}

        Tasks:
        {json.dumps(task_data, indent=2)}

        Create a one-line task list grouped by status (pending, in_progress, completed, failed)
        and provide a brief summary of the project's progress.
        """

        try:
            response = client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                response_format=OneLineTaskListResponse,
            )
            return response.choices[0].message.parsed
        except Exception as e:
            logger.error(f"{Fore.RED}❌ Error generating one-line task list: {str(e)}")
            return OneLineTaskListResponse(
                tasks=[TaskLine(line=f"Error: {str(e)}")],
                summary=f"Error generating task list: {str(e)}"
            )

    def run(self):
        """Run the enhanced autonomous coding environment with intuitive task list management."""
        try:
            self.load_task_library()  # Load existing tasks from files
            while True:
                print(f"\n{Fore.CYAN}=== Autonomous Coding Environment ===")
                print(f"{Fore.YELLOW}1. Create a new project")
                print(f"{Fore.YELLOW}2. View project task list")
                print(f"{Fore.YELLOW}3. Process task action")
                print(f"{Fore.YELLOW}4. Execute a project")
                print(f"{Fore.YELLOW}5. View task library")
                print(f"{Fore.YELLOW}6. Search task library")
                print(f"{Fore.YELLOW}7. Exit")
                
                choice = input(f"{Fore.GREEN}Enter your choice (1-7): ")
                
                if choice == "1":
                    name = input("Enter project name: ")
                    description = input("Enter project description: ")
                    self.create_project(name, description)
                elif choice == "2":
                    project_id = input("Enter project ID to view task list: ")
                    task_list = self.get_task_list(project_id)
                    print(f"\n{Fore.CYAN}=== Task List ===")
                    print(f"{Fore.WHITE}{task_list.summary}")
                    for task in task_list.tasks:
                        status_color = Fore.YELLOW if "PENDING" in task.line else \
                                       Fore.BLUE if "IN PROGRESS" in task.line else \
                                       Fore.GREEN if "COMPLETED" in task.line else Fore.RED
                        print(f"{status_color}{task.line}")
                elif choice == "3":
                    project_id = input("Enter project ID: ")
                    task_id = input("Enter task ID: ")
                    action = input("Enter action (start/complete/fail): ")
                    result = self.process_task_action(project_id, action, task_id)
                    print(f"{Fore.CYAN}Action result: {result.result}")
                elif choice == "4":
                    project_id = input("Enter project ID to execute: ")
                    self.execute_project(project_id)
                elif choice == "5":
                    self.view_task_library()
                elif choice == "6":
                    self.search_task_library()
                elif choice == "7":
                    print(f"{Fore.GREEN}Exiting Autonomous Coding Environment. Goodbye!")
                    break
                else:
                    print(f"{Fore.RED}Invalid choice. Please try again.")
        except Exception as e:
            logger.error(f"{Fore.RED}❌ Fatal error in main loop: {str(e)}")
            logger.error(traceback.format_exc())

    # ... (remaining methods stay the same)

if __name__ == "__main__":
    try:
        ace = EnhancedAutonomousCodingEnvironment(workspace="enhanced_autonomous_coding_workspace")
        ace.run()
    except Exception as e:
        logger.critical(f"{Fore.RED}💥 Critical error: {str(e)}")
        logger.critical(traceback.format_exc())


================================================
File: enhanced_autonomous_coding_environment.py
================================================
import os
import subprocess
import time
import random
import openai
from typing import List, Dict, Any, Optional, Union
import json
import logging
import shutil
from pydantic import BaseModel, Field
import jsonlines
from tqdm import tqdm
from colorama import Fore, Style, init
import traceback

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize OpenAI client (make sure to set your API key in environment variables)
client = openai.OpenAI()

class CodeBlock(BaseModel):
    language: str
    code: str

class ScriptResponse(BaseModel):
    explanation: str
    code_blocks: List[CodeBlock]

class Metadata(BaseModel):
    description: str
    tags: List[str] = Field(default_factory=list)
    complexity: int = Field(ge=1, le=10)
    estimated_time: str
    poetic_description: str

class MetadataResponse(BaseModel):
    description: str
    tags: List[str]
    complexity: int
    estimated_time: str
    poetic_description: str

class Task(BaseModel):
    id: str
    description: str
    code: str = ""
    metadata: Optional[Metadata] = None
    dependencies: List[str] = Field(default_factory=list)
    status: str = "pending"  # pending, in_progress, completed, failed
    execution_result: Optional[Dict[str, Any]] = None

class SimilarTaskResponse(BaseModel):
    similar_task_id: Optional[str]
    explanation: str

class Project(BaseModel):
    id: str
    name: str
    description: str
    tasks: List[Task] = Field(default_factory=list)

class TaskLibrary(BaseModel):
    tasks: Dict[str, Task] = Field(default_factory=dict)

class TaskListResponse(BaseModel):
    tasks: List[Dict[str, Any]]
    summary: str

class TaskActionResponse(BaseModel):
    action: str
    task_id: str
    result: str

class TaskLine(BaseModel):
    """Represents a single task in one line."""
    line: str

class OneLineTaskListResponse(BaseModel):
    """Represents the response for a one-line task list."""
    tasks: List[TaskLine]
    summary: str

class EnhancedAutonomousCodingEnvironment:
    def __init__(self, model: str = "o4-mini", workspace: str = "enhanced_autonomous_workspace"):
        self.model = model
        self.workspace = workspace
        self.projects: Dict[str, Project] = {}
        self.task_library = TaskLibrary()
        self.setup_workspace()

    def setup_workspace(self):
        """Set up a dedicated workspace for the autonomous coding environment."""
        try:
            if os.path.exists(self.workspace):
                shutil.rmtree(self.workspace)  # Clean up existing workspace
            os.makedirs(self.workspace)
            logger.info(f"{Fore.GREEN}🏗️ Created workspace: {self.workspace}")
        except Exception as e:
            logger.error(f"{Fore.RED}❌ Error setting up workspace: {str(e)}")
            raise

    def create_project(self, name: str, description: str) -> str:
        """Create a new project and break it down into tasks."""
        try:
            project_id = f"project_{len(self.projects) + 1}"
            project = Project(id=project_id, name=name, description=description)
            
            # Generate tasks for the project
            tasks = self.generate_tasks(project)
            project.tasks = tasks
            
            self.projects[project_id] = project
            logger.info(f"{Fore.CYAN}📁 Created project: {name} (ID: {project_id})")
            return project_id
        except Exception as e:
            logger.error(f"{Fore.RED}❌ Error creating project: {str(e)}")
            raise

    def generate_tasks(self, project: Project) -> List[Task]:
        """Generate tasks for a given project using the AI model."""
        system_message = (
            "You are an expert project manager and software architect. "
            "Break down the given project into small, manageable tasks. "
            "Each task should be a specific coding task that can be implemented independently. "
            "Provide a brief description for each task and identify any dependencies between tasks."
        )
        
        user_message = f"Project: {project.name}\nDescription: {project.description}\n\nBreak this project down into small coding tasks."

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
            )
            tasks_description = response.choices[0].message.content.strip()
            
            # Parse the tasks from the AI's response
            tasks = []
            for i, task_desc in enumerate(tasks_description.split("\n")):
                if task_desc.strip():
                    task_id = f"{project.id}_task_{i+1}"
                    tasks.append(Task(id=task_id, description=task_desc.strip()))
            
            logger.info(f"{Fore.YELLOW}🧩 Generated {len(tasks)} tasks for project {project.name}")
            return tasks
        except Exception as e:
            logger.error(f"{Fore.RED}❌ Error generating tasks: {str(e)}")
            return []

    def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute a single task and return the result."""
        task_filename = os.path.join(self.workspace, f"{task.id}.py")
        try:
            with open(task_filename, 'w') as f:
                f.write(task.code)
            
            result = subprocess.run(['python', task_filename], capture_output=True, text=True, timeout=30)
            
            execution_result = {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr
            }
            logger.info(f"{Fore.CYAN}🚀 Executed task: {task.id}")
            return execution_result
        except subprocess.TimeoutExpired:
            logger.warning(f"{Fore.YELLOW}⏳ Task execution timed out: {task.id}")
            return {"success": False, "error": "Execution timed out"}
        except Exception as e:
            logger.error(f"{Fore.RED}❌ Error executing task: {str(e)}")
            return {"success": False, "error": str(e)}

    def generate_code_for_task(self, task: Task) -> str:
        """Generate code for a given task using the AI model with structured output."""
        system_message = """
        You are an expert Python developer tasked with implementing a specific coding task.
        Provide a complete and working implementation for the given task description.
        Include error handling, logging, and comments in your code.
        """
        
        user_message = f"Task: {task.description}\n\nImplement this task in Python."

        try:
            response = client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                response_format=ScriptResponse,
            )
            script_response = response.choices[0].message.parsed
            
            # Extract the main Python code block
            main_code_block = next((block for block in script_response.code_blocks if block.language.lower() == 'python'), None)
            
            if main_code_block:
                logger.info(f"{Fore.GREEN}💻 Generated code for task: {task.id}")
                return main_code_block.code
            else:
                logger.warning(f"{Fore.YELLOW}⚠️ No Python code block found in the response for task: {task.id}")
                return f"# Error: No Python code block found in the AI response\n\ndef error_function():\n    raise NotImplementedError('Task implementation failed')"
        except Exception as e:
            logger.error(f"{Fore.RED}❌ Error generating code for task: {str(e)}")
            return f"# Error generating code: {str(e)}\n\ndef error_function():\n    raise NotImplementedError('Task implementation failed')"

    def implement_task(self, task: Task) -> Task:
        """Implement a single task using the AI model and generate metadata."""
        try:
            task.code = self.generate_code_for_task(task)
            task.metadata = self.generate_task_metadata(task)
            return task
        except Exception as e:
            logger.error(f"{Fore.RED}❌ Error implementing task: {str(e)}")
            task.status = "failed"
            task.execution_result = {"success": False, "error": str(e)}
            return task

    def generate_task_metadata(self, task: Task) -> Metadata:
        """Generate metadata for a task, including a poetic description, using structured output."""

        system_message = """
        You are an AI expert in software development and poetry. Analyze the given task and its code to generate metadata.
        Provide a concise description, relevant tags, estimate the complexity (1-10), and estimated time to complete.
        Also, create a short, poetic description that captures the essence of the task in a memorable way.
        """
        
        user_message = f"""
        Task: {task.description}
        
        Code:
        {task.code}
        
        Generate metadata including a concise description, relevant tags, complexity (1-10), estimated time to complete, and a short, poetic description of the task.
        """

        try:
            response = client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                response_format=MetadataResponse,
            )
            metadata = response.choices[0].message.parsed
            
            return Metadata(
                description=metadata.description,
                tags=metadata.tags,
                complexity=metadata.complexity,
                estimated_time=metadata.estimated_time,
                poetic_description=metadata.poetic_description
            )
        except Exception as e:
            logger.error(f"{Fore.RED}❌ Error generating task metadata: {str(e)}")
            return Metadata(
                description=task.description,
                tags=["error"],
                complexity=5,
                estimated_time="unknown",
                poetic_description="A task shrouded in mystery, its true nature yet to be revealed."
            )

    def update_task_library(self, task: Task):
        """Update the task library with a successful task implementation."""
        if task.status == "completed":
            self.task_library.tasks[task.id] = task
            self.save_task_to_file(task)
            logger.info(f"{Fore.GREEN}📚 Added task to library: {task.id}")

    def save_task_to_file(self, task: Task):
        """Save a task to a file in the workspace."""
        try:
            task_dir = os.path.join(self.workspace, "task_library")
            os.makedirs(task_dir, exist_ok=True)
            task_file = os.path.join(task_dir, f"{task.id}.json")
            with open(task_file, 'w') as f:
                json.dump(task.dict(), f, indent=2)
        except Exception as e:
            logger.error(f"{Fore.RED}❌ Error saving task to file: {str(e)}")

    def load_task_library(self):
        """Load tasks from files in the workspace."""
        task_dir = os.path.join(self.workspace, "task_library")
        if os.path.exists(task_dir):
            for filename in os.listdir(task_dir):
                if filename.endswith(".json"):
                    try:
                        with open(os.path.join(task_dir, filename), 'r') as f:
                            task_data = json.load(f)
                            task = Task(**task_data)
                            self.task_library.tasks[task.id] = task
                    except Exception as e:
                        logger.error(f"{Fore.RED}❌ Error loading task from file {filename}: {str(e)}")
        logger.info(f"{Fore.GREEN}📚 Loaded {len(self.task_library.tasks)} tasks from library")

    def find_similar_task(self, task: Task) -> Optional[Task]:
        """Find a similar task in the task library using metadata and poetic descriptions with structured output."""
        system_message = """
        You are an AI expert in code similarity and poetic analysis. Compare the given task with the tasks in the library.
        Consider the task descriptions, code similarity, metadata, and poetic descriptions.
        If you find a similar task, return its ID. If not, return None.
        """
        
        task_library_desc = "\n".join([
            f"{t.id}:\nDescription: {t.description}\nTags: {', '.join(t.metadata.tags)}\nPoetic: {t.metadata.poetic_description}"
            for t in self.task_library.tasks.values()
        ])
        user_message = f"""
        Task to compare:
        Description: {task.description}
        
        Task Library:
        {task_library_desc}
        
        Find a similar task ID or return None.
        """

        try:
            response = client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                response_format=SimilarTaskResponse,
            )
            result = response.choices[0].message.parsed
            
            if result.similar_task_id and result.similar_task_id in self.task_library.tasks:
                logger.info(f"{Fore.YELLOW}🔍 Found similar task: {result.similar_task_id}")
                logger.info(f"Explanation: {result.explanation}")
                return self.task_library.tasks[result.similar_task_id]
            return None
        except Exception as e:
            logger.error(f"{Fore.RED}❌ Error finding similar task: {str(e)}")
            return None

    def adapt_task(self, original_task: Task, similar_task: Task) -> str:
        """Adapt a similar task's implementation to fit the current task using structured output."""
        system_message = """
        You are an expert Python developer tasked with adapting existing code to fit a new requirement.
        Modify the given code to implement the new task while maintaining its structure and error handling.
        """
        
        user_message = f"Original task: {original_task.description}\nSimilar task: {similar_task.description}\n\nSimilar task code:\n\n{similar_task.code}\n\nAdapt this code to implement the original task."

        try:
            response = client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                response_format=ScriptResponse,
            )
            script_response = response.choices[0].message.parsed
            
            # Extract the main Python code block
            main_code_block = next((block for block in script_response.code_blocks if block.language.lower() == 'python'), None)
            
            if main_code_block:
                logger.info(f"{Fore.GREEN}🔄 Adapted similar task for: {original_task.id}")
                return main_code_block.code
            else:
                logger.warning(f"{Fore.YELLOW}⚠️ No Python code block found in the adapted response for task: {original_task.id}")
                return f"# Error: No Python code block found in the AI response\n\ndef error_function():\n    raise NotImplementedError('Task adaptation failed')"
        except Exception as e:
            logger.error(f"{Fore.RED}❌ Error adapting task: {str(e)}")
            return f"# Error adapting task: {str(e)}\n\ndef error_function():\n    raise NotImplementedError('Task adaptation failed')"

    def execute_project(self, project_id: str):
        """Execute all tasks in a project."""
        project = self.projects.get(project_id)
        if not project:
            logger.error(f"{Fore.RED}❌ Project not found: {project_id}")
            return

        logger.info(f"{Fore.CYAN}🚀 Executing project: {project.name}")
        for task in tqdm(project.tasks, desc="Executing tasks", unit="task"):
            if task.status != "pending":
                continue

            # Check for unmet dependencies
            if any(dep_task.status != "completed" for dep_task in project.tasks if dep_task.id in task.dependencies):
                logger.info(f"{Fore.YELLOW}⏳ Skipping task due to unmet dependencies: {task.id}")
                continue

            try:
                # Find similar task in library
                similar_task = self.find_similar_task(task)
                
                if similar_task:
                    # Adapt similar task
                    task.code = self.adapt_task(task, similar_task)
                else:
                    # Implement new task
                    task = self.implement_task(task)

                # Execute task
                task.execution_result = self.execute_task(task)
                
                if task.execution_result["success"]:
                    task.status = "completed"
                    self.update_task_library(task)
                else:
                    task.status = "failed"
                    logger.warning(f"{Fore.RED}❌ Task failed: {task.id}")
                    logger.warning(f"Error: {task.execution_result['error']}")
            except Exception as e:
                task.status = "failed"
                logger.error(f"{Fore.RED}❌ Error executing task {task.id}: {str(e)}")
                logger.error(traceback.format_exc())

        self.generate_project_report(project)

    def generate_project_report(self, project: Project):
        """Generate a summary report of the project execution."""
        try:
            report = f"{Fore.CYAN}📊 Project Execution Report: {project.name}\n"
            report += f"Total Tasks: {len(project.tasks)}\n"
            completed_tasks = sum(1 for task in project.tasks if task.status == "completed")
            failed_tasks = sum(1 for task in project.tasks if task.status == "failed")
            report += f"Completed Tasks: {completed_tasks}\n"
            report += f"Failed Tasks: {failed_tasks}\n"
            report += f"Success Rate: {completed_tasks / len(project.tasks):.2%}\n"
            
            report += "\nTask Details:\n"
            for task in project.tasks:
                status_color = Fore.GREEN if task.status == "completed" else Fore.RED
                report += f"{status_color}[{task.status.upper()}] {task.id}: {task.description}\n"
            
            report_file = os.path.join(self.workspace, f"{project.id}_report.txt")
            with open(report_file, 'w') as f:
                f.write(report)
            print(report)
            logger.info(f"{Fore.GREEN}📝 Generated project report: {report_file}")
        except Exception as e:
            logger.error(f"{Fore.RED}❌ Error generating project report: {str(e)}")


    def get_task_list(self, project_id: str) -> OneLineTaskListResponse:
        """Get an intuitive task list for a project using structured output, with one task per line."""
        project = self.projects.get(project_id)
        if not project:
            return OneLineTaskListResponse(tasks=[], summary="Project not found")

        system_message = """
        You are an AI project manager assistant. Given a list of tasks for a project,
        create an intuitive and organized task list. Each task should be represented
        in a single line, including its ID, status, and a brief description.
        Group tasks by their status and provide a brief summary of the project's progress.

        Use the following format for each task line:
        [STATUS] ID: Brief description (Dependencies: dep1, dep2)

        STATUS should be one of: PENDING, IN PROGRESS, COMPLETED, FAILED
        If there are no dependencies, omit the parentheses.
        """

        task_data = [
            {
                "id": task.id,
                "description": task.description,
                "status": task.status,
                "dependencies": task.dependencies,
                "metadata": task.metadata.dict() if task.metadata else None
            }
            for task in project.tasks
        ]

        user_message = f"""
        Project: {project.name}
        Description: {project.description}

        Tasks:
        {json.dumps(task_data, indent=2)}

        Create a one-line task list grouped by status (pending, in_progress, completed, failed)
        and provide a brief summary of the project's progress.
        """

        try:
            response = client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                response_format=OneLineTaskListResponse,
            )
            return response.choices[0].message.parsed
        except Exception as e:
            logger.error(f"{Fore.RED}❌ Error generating one-line task list: {str(e)}")
            return OneLineTaskListResponse(
                tasks=[TaskLine(line=f"Error: {str(e)}")],
                summary=f"Error generating task list: {str(e)}"
            )
        
    def process_task_action(self, project_id: str, action: str, task_id: str) -> TaskActionResponse:
        """Process a task action (start, complete, fail) using structured output."""
        project = self.projects.get(project_id)
        if not project:
            return TaskActionResponse(action=action, task_id=task_id, result="Project not found")

        task = next((t for t in project.tasks if t.id == task_id), None)
        if not task:
            return TaskActionResponse(action=action, task_id=task_id, result="Task not found")

        system_message = f"""
        You are an AI project management assistant. Process the following task action:
        Action: {action}
        Task ID: {task_id}
        Current Task Status: {task.status}

        Determine if the action is valid and update the task status accordingly.
        Provide a brief result message explaining the outcome.
        """

        user_message = f"""
        Process the task action and provide the result.
        Consider the current task status and any dependencies.
        """

        try:
            response = client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                response_format=TaskActionResponse,
            )
            result = response.choices[0].message.parsed

            # Update task status based on the AI's decision
            if result.action == "start" and task.status == "pending":
                task.status = "in_progress"
            elif result.action == "complete" and task.status in ["pending", "in_progress"]:
                task.status = "completed"
            elif result.action == "fail" and task.status in ["pending", "in_progress"]:
                task.status = "failed"

            return result
        except Exception as e:
            logger.error(f"{Fore.RED}❌ Error processing task action: {str(e)}")
            return TaskActionResponse(action=action, task_id=task_id, result=f"Error processing task action: {str(e)}")

    def run(self):
        """Run the enhanced autonomous coding environment."""
        try:
            self.load_task_library()  # Load existing tasks from files
            while True:
                print(f"\n{Fore.CYAN}=== Autonomous Coding Environment ===")
                print(f"{Fore.YELLOW}1. Create a new project")
                print(f"{Fore.YELLOW}2. View project task list")
                print(f"{Fore.YELLOW}3. Process task action")
                print(f"{Fore.YELLOW}4. Execute a project")
                print(f"{Fore.YELLOW}5. View task library")
                print(f"{Fore.YELLOW}6. Search task library")
                print(f"{Fore.YELLOW}7. Exit")
                
                choice = input(f"{Fore.GREEN}Enter your choice (1-7): ")
                
                if choice == "1":
                    name = input("Enter project name: ")
                    description = input("Enter project description: ")
                    self.create_project(name, description)
                elif choice == "2":
                    project_id = input("Enter project ID to view task list: ")
                    task_list = self.get_task_list(project_id)
                    print(f"\n{Fore.CYAN}=== Task List ===")
                    print(f"{Fore.WHITE}{task_list.summary}")
                    for task in task_list.tasks:
                        status_color = Fore.YELLOW if "PENDING" in task.line else \
                                       Fore.BLUE if "IN PROGRESS" in task.line else \
                                       Fore.GREEN if "COMPLETED" in task.line else Fore.RED
                        print(f"{status_color}{task.line}")
                elif choice == "3":
                    project_id = input("Enter project ID: ")
                    task_id = input("Enter task ID: ")
                    action = input("Enter action (start/complete/fail): ")
                    result = self.process_task_action(project_id, action, task_id)
                    print(f"{Fore.CYAN}Action result: {result.result}")
                elif choice == "4":
                    project_id = input("Enter project ID to execute: ")
                    self.execute_project(project_id)
                elif choice == "5":
                    self.view_task_library()
                elif choice == "6":
                    self.search_task_library()
                elif choice == "7":
                    print(f"{Fore.GREEN}Exiting Autonomous Coding Environment. Goodbye!")
                    break
                else:
                    print(f"{Fore.RED}Invalid choice. Please try again.")
        except Exception as e:
            logger.error(f"{Fore.RED}❌ Fatal error in main loop: {str(e)}")
            logger.error(traceback.format_exc())

    def view_task_library(self):
        """View the contents of the task library."""
        try:
            print(f"\n{Fore.CYAN}=== Task Library ===")
            for task_id, task in self.task_library.tasks.items():
                print(f"{Fore.YELLOW}ID: {task_id}")
                print(f"Description: {task.description}")
                print(f"Tags: {', '.join(task.metadata.tags)}")
                print(f"Poetic Description: {task.metadata.poetic_description}")
                print(f"{Fore.CYAN}---")
        except Exception as e:
            logger.error(f"{Fore.RED}❌ Error viewing task library: {str(e)}")

    def search_task_library(self):
        """Search the task library using natural language queries."""
        query = input("Enter your search query: ")
        system_message = """
        You are an AI expert in searching and matching code tasks. Given a search query and a list of tasks,
        return the IDs of the most relevant tasks, along with a brief explanation of why they match.
        """
        
        task_library_desc = "\n".join([
            f"{t.id}:\nDescription: {t.description}\nTags: {', '.join(t.metadata.tags)}\nPoetic: {t.metadata.poetic_description}"
            for t in self.task_library.tasks.values()
        ])
        user_message = f"""
        Search Query: {query}
        
        Task Library:
        {task_library_desc}
        
        Return the IDs of the most relevant tasks and explain why they match the query.
        """

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
            )
            search_results = response.choices[0].message.content.strip()
            print(f"\n{Fore.CYAN}=== Search Results ===")
            print(search_results)
        except Exception as e:
            logger.error(f"{Fore.RED}❌ Error searching task library: {str(e)}")
            print(f"{Fore.RED}An error occurred while searching the task library.")

if __name__ == "__main__":
    try:
        ace = EnhancedAutonomousCodingEnvironment(workspace="enhanced_autonomous_coding_workspace")
        ace.run()
    except Exception as e:
        logger.critical(f"{Fore.RED}💥 Critical error: {str(e)}")
        logger.critical(traceback.format_exc())


================================================
File: enhanced_task_management.py
================================================
import json
from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field
import openai
from colorama import Fore
import logging
from base_task_processor import Task, BaseTaskProcessor

client = openai.OpenAI()

class TaskDependency(BaseModel):
    task_id: str
    dependency_type: str = "requires"  # requires, suggests, blocks
    description: str = ""

class EnhancedTask(Task):
    dependencies: List[TaskDependency] = Field(default_factory=list)
    priority: int = Field(default=5, ge=1, le=10)
    estimated_duration: Optional[int] = None  # in minutes
    category: str = "general"

class TaskData(BaseModel):
    """Structured task data for decomposition response."""
    description: str
    priority: int = Field(default=5, ge=1, le=10)
    category: str = "general"
    estimated_duration: int = Field(default=45, description="Estimated duration in minutes")

class DependencyData(BaseModel):
    """Structured dependency data for decomposition response."""
    task_id: str
    depends_on: str
    type: str = "requires"
    description: str = ""

class TaskDecompositionResponse(BaseModel):
    tasks: List[TaskData] = Field(default_factory=list)
    dependencies: List[DependencyData] = Field(default_factory=list)
    summary: str

    class Config:
        extra = "forbid"  # This sets additionalProperties to false

class EnhancedTaskManager(BaseTaskProcessor):
    """Enhanced task management with dependency resolution and smart decomposition."""
    
    def __init__(self, model: str = "o4-mini"):
        super().__init__(model)
        self.logger = logging.getLogger(__name__)
    
    def decompose_project_to_tasks(self, project_name: str, project_description: str) -> List[EnhancedTask]:
        """Decompose a project into structured tasks with dependencies."""
        system_message = """
        You are an expert project manager and software architect.
        Break down the given project into specific, actionable coding tasks that focus on implementing the actual functionality.
        
        IMPORTANT: Focus on pure coding tasks only. Do NOT include:
        - Git repository setup
        - Package manager initialization  
        - Installing dependencies
        - Environment setup
        - CI/CD configuration
        
        DO include:
        - Core algorithm implementation
        - Function and class definitions
        - Data structure creation
        - Business logic implementation
        - Testing and validation code
        - Error handling
        - Documentation within code
        
        For each task, provide:
        - A clear, specific description of what code to implement
        - An estimated priority (1-10, where 10 is highest)
        - Category (core, feature, testing, documentation, utility)
        - Dependencies on other coding tasks
        
        Make tasks granular enough to be completed in 30-60 minutes each.
        Focus on building working, runnable Python code.
        """
        
        user_message = f"""
        Project: {project_name}
        Description: {project_description}
        
        Break this project down into specific coding tasks with:
        1. Clear task descriptions
        2. Priority levels (1-10)
        3. Categories (setup, core, feature, testing, documentation, etc.)
        4. Dependencies between tasks
        
        Return the response as a structured format with tasks and dependencies.
        """
        
        try:
            response = client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                response_format=TaskDecompositionResponse,
            )
            
            decomposition = response.choices[0].message.parsed
            tasks = []
            
            # Create tasks from response
            for i, task_data in enumerate(decomposition.tasks):
                task_id = f"task_{i+1:03d}"
                task = EnhancedTask(
                    id=task_id,
                    description=task_data.description,
                    priority=task_data.priority,
                    category=task_data.category,
                    estimated_duration=task_data.estimated_duration
                )
                tasks.append(task)
            
            # Add dependencies
            for dep_data in decomposition.dependencies:
                task_id = dep_data.task_id
                depends_on = dep_data.depends_on
                
                # Find the task and add dependency
                task = next((t for t in tasks if t.id == task_id), None)
                if task and depends_on:
                    dependency = TaskDependency(
                        task_id=depends_on,
                        dependency_type=dep_data.type,
                        description=dep_data.description
                    )
                    task.dependencies.append(dependency)
            
            self.logger.info(f"{Fore.GREEN}✅ Decomposed project into {len(tasks)} tasks")
            return tasks
            
        except Exception as e:
            self.logger.error(f"{Fore.RED}❌ Error decomposing project: {str(e)}")
            # Return a basic fallback task
            return [EnhancedTask(
                id="task_001",
                description=f"Implement: {project_description}",
                priority=5,
                category="general"
            )]
    
    def resolve_task_dependencies(self, tasks: List[EnhancedTask]) -> List[EnhancedTask]:
        """Resolve and validate task dependencies using topological sort."""
        # Create a mapping of task IDs to tasks
        task_map = {task.id: task for task in tasks}
        
        # Build dependency graph
        in_degree = {task.id: 0 for task in tasks}
        adj_list = {task.id: [] for task in tasks}
        
        for task in tasks:
            for dep in task.dependencies:
                if dep.task_id in task_map:
                    adj_list[dep.task_id].append(task.id)
                    in_degree[task.id] += 1
                else:
                    self.logger.warning(f"{Fore.YELLOW}⚠️ Invalid dependency: {task.id} -> {dep.task_id}")
        
        # Topological sort
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        sorted_tasks = []
        
        while queue:
            current_id = queue.pop(0)
            current_task = task_map[current_id]
            sorted_tasks.append(current_task)
            
            # Update in-degrees
            for neighbor in adj_list[current_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for circular dependencies
        if len(sorted_tasks) != len(tasks):
            remaining_tasks = [task for task in tasks if task not in sorted_tasks]
            self.logger.warning(f"{Fore.YELLOW}⚠️ Circular dependencies detected in tasks: {[t.id for t in remaining_tasks]}")
            # Add remaining tasks to the end
            sorted_tasks.extend(remaining_tasks)
        
        self.logger.info(f"{Fore.GREEN}✅ Resolved dependencies for {len(sorted_tasks)} tasks")
        return sorted_tasks
    
    def get_ready_tasks(self, tasks: List[EnhancedTask]) -> List[EnhancedTask]:
        """Get tasks that are ready to be executed (all dependencies completed)."""
        completed_tasks = {task.id for task in tasks if task.status == "completed"}
        ready_tasks = []
        
        for task in tasks:
            if task.status == "pending":
                # Check if all dependencies are completed
                required_deps = [dep.task_id for dep in task.dependencies if dep.dependency_type == "requires"]
                if all(dep_id in completed_tasks for dep_id in required_deps):
                    ready_tasks.append(task)
        
        # Sort by priority (highest first)
        ready_tasks.sort(key=lambda t: t.priority, reverse=True)
        return ready_tasks
    
    def get_task_execution_order(self, tasks: List[EnhancedTask]) -> List[List[EnhancedTask]]:
        """Get tasks grouped by execution phases (tasks that can run in parallel)."""
        resolved_tasks = self.resolve_task_dependencies(tasks)
        execution_phases = []
        remaining_tasks = resolved_tasks.copy()
        
        while remaining_tasks:
            current_phase = self.get_ready_tasks(remaining_tasks)
            if not current_phase:
                # Handle circular dependencies by adding all remaining tasks
                current_phase = [task for task in remaining_tasks if task.status == "pending"]
                if not current_phase:
                    break
            
            execution_phases.append(current_phase)
            
            # Mark current phase tasks as ready for next iteration
            for task in current_phase:
                task.status = "ready"
                remaining_tasks.remove(task)
        
        # Reset status
        for task in resolved_tasks:
            if task.status == "ready":
                task.status = "pending"
        
        return execution_phases
    
    def estimate_project_duration(self, tasks: List[EnhancedTask]) -> Dict[str, Any]:
        """Estimate project completion time considering dependencies."""
        execution_phases = self.get_task_execution_order(tasks)
        
        total_duration = 0
        parallel_duration = 0
        
        for phase in execution_phases:
            phase_duration = max(task.estimated_duration or 45 for task in phase)
            total_duration += phase_duration
            parallel_duration = max(parallel_duration, phase_duration)
        
        sequential_duration = sum(task.estimated_duration or 45 for task in tasks)
        
        return {
            "sequential_estimate_minutes": sequential_duration,
            "parallel_estimate_minutes": total_duration,
            "estimated_phases": len(execution_phases),
            "tasks_per_phase": [len(phase) for phase in execution_phases],
            "critical_path_duration": total_duration
        }

class ProjectOrchestrator:
    """Orchestrates project execution with dependency management."""
    
    def __init__(self, task_manager: EnhancedTaskManager, base_processor: BaseTaskProcessor):
        self.task_manager = task_manager
        self.base_processor = base_processor
        self.logger = logging.getLogger(__name__)
    
    def execute_project_with_dependencies(self, tasks: List[EnhancedTask], workspace_path: str) -> Dict[str, Any]:
        """Execute a project respecting task dependencies."""
        execution_phases = self.task_manager.get_task_execution_order(tasks)
        results = {"phases": [], "summary": {}}
        
        for phase_num, phase_tasks in enumerate(execution_phases):
            self.logger.info(f"{Fore.CYAN}🚀 Executing Phase {phase_num + 1}: {len(phase_tasks)} tasks")
            phase_results = []
            
            for task in phase_tasks:
                self.logger.info(f"{Fore.BLUE}▶️ Starting task: {task.id} - {task.description}")
                
                # Generate code if not present
                if not task.code:
                    task.code = self.base_processor.generate_code_for_task(task)
                
                # Execute task
                task.status = "in_progress"
                self.logger.debug(f"Executing task {task.id} in workspace: {workspace_path}")
                execution_result = self.base_processor.execute_task_code(task, workspace_path)
                
                # Handle results
                if execution_result["success"]:
                    task.status = "completed"
                    self.logger.info(f"{Fore.GREEN}✅ Task {task.id} completed successfully")
                else:
                    task.status = "failed"
                    self.logger.warning(f"{Fore.YELLOW}⚠️ Task {task.id} failed, attempting improvement")
                    
                    # Try to improve the task
                    improved_code = self.base_processor.improve_task_code(task, execution_result)
                    task.code = improved_code
                    
                    # Retry execution
                    retry_result = self.base_processor.execute_task_code(task, workspace_path)
                    if retry_result["success"]:
                        task.status = "completed"
                        self.logger.info(f"{Fore.GREEN}✅ Task {task.id} completed after improvement")
                        execution_result = retry_result
                    else:
                        self.logger.error(f"{Fore.RED}❌ Task {task.id} failed even after improvement")
                
                task.execution_result = execution_result
                phase_results.append({
                    "task_id": task.id,
                    "status": task.status,
                    "execution_result": execution_result
                })
            
            results["phases"].append({
                "phase_number": phase_num + 1,
                "tasks_executed": len(phase_tasks),
                "successful_tasks": len([r for r in phase_results if r["status"] == "completed"]),
                "failed_tasks": len([r for r in phase_results if r["status"] == "failed"]),
                "results": phase_results
            })
        
        # Generate summary
        all_tasks = [task for phase in execution_phases for task in phase]
        results["summary"] = {
            "total_tasks": len(all_tasks),
            "completed_tasks": len([t for t in all_tasks if t.status == "completed"]),
            "failed_tasks": len([t for t in all_tasks if t.status == "failed"]),
            "success_rate": len([t for t in all_tasks if t.status == "completed"]) / len(all_tasks) if all_tasks else 0,
            "total_phases": len(execution_phases)
        }
        
        return results 


================================================
File: rft_enhanced_task_processor.py
================================================
#!/usr/bin/env python3
"""
RFT-Enhanced Task Processor

This module extends the base task processor with reinforcement fine-tuning capabilities,
enabling continuous self-improvement through structured feedback and model refinement.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import logging
from colorama import Fore, Style
import numpy as np

from base_task_processor import BaseTaskProcessor, Task, Metadata
from rft_grading_system import RFTGradingSystem, CodeGradingMetrics
import functools

class RFTEnhancedTaskProcessor(BaseTaskProcessor):
    """
    Enhanced task processor with RFT capabilities for continuous improvement.
    """
    
    def __init__(self, model: str = "o4-mini", rft_model: Optional[str] = None):
        super().__init__(model)
        self.rft_grader = RFTGradingSystem()
        self.rft_model = rft_model  # Fine-tuned model ID if available
        self.improvement_history = []
        self.performance_threshold = 0.75  # Minimum score before attempting improvement
        self.rft_training_data_path = "data/rft_training"
        self.current_grader_config = None
        
        # Ensure RFT data directory exists
        os.makedirs(self.rft_training_data_path, exist_ok=True)
        
        # Load existing improvement history
        self._load_improvement_history()

    def set_grader_config(self, domain: str = "general_programming") -> None:
        """Set the grader configuration for the current domain."""
        self.current_grader_config = self.rft_grader.build_advanced_model_grader(domain)
        self.logger.info(f"Set grader config for domain: {domain}")

    def process_task_with_rft(self, task: Task, workspace_path: str, 
                            enable_improvement: bool = True) -> Task:
        """
        Process a task with RFT enhancement - generate, execute, grade, and improve if needed.
        """
        self.logger.info(f"{Fore.BLUE}🚀 Processing task {task.id} with RFT enhancement")
        
        # Use fine-tuned model if available
        generation_model = self.rft_model if self.rft_model else self.model
        
        # Generate code
        if not task.code:
            original_model = self.model
            self.model = generation_model
            task.code = self.generate_code_for_task(task)
            self.model = original_model
        
        # Execute code
        execution_result = self.execute_task_code(task, workspace_path)
        task.execution_result = execution_result
        
        # Grade the result
        if self.current_grader_config:
            grader_func = functools.partial(
                self.rft_grader.python_model_grader, 
                model_grader=self.current_grader_config
            )
        else:
            grader_func = self.rft_grader.combined_code_grader
        
        # Evaluate single task
        evaluation_result = self.rft_grader.evaluate_task_batch([task], grader_func, workspace_path)
        task_score = evaluation_result["results"][0]["score"]
        
        self.logger.info(f"{Fore.YELLOW}📊 Task {task.id} scored: {task_score:.3f}")
        
        # Store grading result in task metadata
        if not task.metadata:
            task.metadata = Metadata(
                description=task.description,
                tags=["rft-processed"],
                complexity=5,
                estimated_time="unknown",
                poetic_description="A task enhanced through reinforcement learning."
            )
        
        # Add RFT metrics to metadata
        task.metadata.tags.append(f"rft_score_{task_score:.3f}")
        
        # Attempt improvement if score is below threshold and improvement is enabled
        if enable_improvement and task_score < self.performance_threshold:
            task = self._attempt_task_improvement(task, workspace_path, task_score)
        
        return task

    def _attempt_task_improvement(self, task: Task, workspace_path: str, 
                                initial_score: float) -> Task:
        """
        Attempt to improve a task through iterative refinement.
        """
        self.logger.info(f"{Fore.BLUE}🔧 Attempting to improve task {task.id} (score: {initial_score:.3f})")
        
        max_improvement_attempts = 3
        best_score = initial_score
        best_code = task.code
        
        for attempt in range(max_improvement_attempts):
            # Generate improved code using execution feedback
            improved_code = self.improve_task_code(task, task.execution_result)
            
            # Create a temporary task for testing
            temp_task = Task(
                id=f"{task.id}_improved_{attempt}",
                description=task.description,
                code=improved_code,
                metadata=task.metadata
            )
            
            # Execute improved code
            improved_execution = self.execute_task_code(temp_task, workspace_path)
            temp_task.execution_result = improved_execution
            
            # Grade improved version
            if self.current_grader_config:
                grader_func = functools.partial(
                    self.rft_grader.python_model_grader, 
                    model_grader=self.current_grader_config
                )
            else:
                grader_func = self.rft_grader.combined_code_grader
            
            improved_evaluation = self.rft_grader.evaluate_task_batch([temp_task], grader_func, workspace_path)
            improved_score = improved_evaluation["results"][0]["score"]
            
            self.logger.info(f"{Fore.YELLOW}📈 Improvement attempt {attempt + 1}: {improved_score:.3f}")
            
            # Keep the best version
            if improved_score > best_score:
                best_score = improved_score
                best_code = improved_code
                task.execution_result = improved_execution
                
                # Update metadata
                task.metadata.tags.append(f"improved_attempt_{attempt + 1}")
                
                # Stop if we've reached a good score
                if improved_score >= self.performance_threshold:
                    self.logger.info(f"{Fore.GREEN}✅ Task improvement successful!")
                    break
            else:
                self.logger.info(f"{Fore.YELLOW}⚠️ No improvement in attempt {attempt + 1}")
        
        # Apply best result
        task.code = best_code
        
        # Record improvement attempt
        improvement_record = {
            "task_id": task.id,
            "initial_score": initial_score,
            "final_score": best_score,
            "improvement": best_score - initial_score,
            "attempts": max_improvement_attempts,
            "timestamp": datetime.now().isoformat()
        }
        self.improvement_history.append(improvement_record)
        
        return task

    def batch_process_with_rft(self, tasks: List[Task], workspace_path: str) -> List[Task]:
        """
        Process multiple tasks with RFT, collecting data for potential fine-tuning.
        """
        self.logger.info(f"{Fore.BLUE}🚀 Batch processing {len(tasks)} tasks with RFT")
        
        processed_tasks = []
        for task in tasks:
            processed_task = self.process_task_with_rft(task, workspace_path)
            processed_tasks.append(processed_task)
        
        # Evaluate batch performance
        self._evaluate_batch_performance(processed_tasks, workspace_path)
        
        return processed_tasks

    def _evaluate_batch_performance(self, tasks: List[Task], workspace_path: str) -> Dict[str, Any]:
        """
        Evaluate overall batch performance and decide if RFT training is warranted.
        """
        if self.current_grader_config:
            grader_func = functools.partial(
                self.rft_grader.python_model_grader, 
                model_grader=self.current_grader_config
            )
        else:
            grader_func = self.rft_grader.combined_code_grader
        
        evaluation_result = self.rft_grader.evaluate_task_batch(tasks, grader_func, workspace_path)
        avg_score = evaluation_result["average_score"]
        
        self.logger.info(f"{Fore.CYAN}📊 Batch average score: {avg_score:.3f}")
        
        # If performance is below threshold, consider RFT training
        if avg_score < self.performance_threshold:
            self.logger.info(f"{Fore.YELLOW}🎯 Performance below threshold, preparing RFT training data")
            self._prepare_rft_training_session(evaluation_result)
        
        return evaluation_result

    def _prepare_rft_training_session(self, evaluation_result: Dict[str, Any]) -> None:
        """
        Prepare and potentially launch RFT training based on evaluation results.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_file = os.path.join(self.rft_training_data_path, f"rft_train_{timestamp}.jsonl")
        
        # Prepare training data
        self.rft_grader.prepare_rft_training_data(evaluation_result, train_file)
        
        # Check if we have enough data for training
        with open(train_file, 'r') as f:
            sample_count = sum(1 for _ in f)
        
        if sample_count >= 20:  # Minimum samples for meaningful training
            self.logger.info(f"{Fore.GREEN}🎓 Sufficient data for RFT training ({sample_count} samples)")
            
            # Optionally auto-launch RFT training
            if self.current_grader_config and os.environ.get("AUTO_RFT_TRAINING", "false").lower() == "true":
                self._launch_rft_training(train_file)
        else:
            self.logger.info(f"{Fore.YELLOW}📚 Need more data for RFT training (have {sample_count}, need 20+)")

    def _launch_rft_training(self, train_file: str) -> Optional[str]:
        """
        Launch RFT training job.
        """
        if not self.current_grader_config:
            self.logger.error("No grader config set for RFT training")
            return None
        
        # For now, create a basic test file by splitting the training data
        test_file = train_file.replace("_train_", "_test_")
        self._split_training_data(train_file, test_file, test_ratio=0.2)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"ace_rft_{timestamp}"
        
        job_id = self.rft_grader.launch_rft_job(
            train_file=train_file,
            test_file=test_file,
            grader_config=self.current_grader_config,
            suffix=suffix
        )
        
        if job_id:
            self.logger.info(f"{Fore.GREEN}🚀 RFT training launched with job ID: {job_id}")
            
            # Store job info for monitoring
            job_info = {
                "job_id": job_id,
                "launch_time": datetime.now().isoformat(),
                "train_file": train_file,
                "test_file": test_file,
                "suffix": suffix
            }
            
            job_file = os.path.join(self.rft_training_data_path, f"rft_job_{job_id}.json")
            with open(job_file, 'w') as f:
                json.dump(job_info, f, indent=2)
        
        return job_id

    def _split_training_data(self, train_file: str, test_file: str, test_ratio: float = 0.2) -> None:
        """
        Split training data into train/test sets.
        """
        with open(train_file, 'r') as f:
            lines = f.readlines()
        
        # Shuffle and split
        import random
        random.shuffle(lines)
        split_idx = int(len(lines) * (1 - test_ratio))
        
        train_lines = lines[:split_idx]
        test_lines = lines[split_idx:]
        
        # Write train file
        with open(train_file, 'w') as f:
            f.writelines(train_lines)
        
        # Write test file
        with open(test_file, 'w') as f:
            f.writelines(test_lines)
        
        self.logger.info(f"Split data: {len(train_lines)} train, {len(test_lines)} test samples")

    def check_rft_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Check the status of an RFT training job.
        """
        return self.rft_grader.get_rft_job_status(job_id)

    def update_model_after_rft(self, job_id: str) -> bool:
        """
        Update the processor to use a newly fine-tuned model.
        """
        job_status = self.check_rft_job_status(job_id)
        
        if job_status.get("status") == "succeeded":
            fine_tuned_model = job_status.get("fine_tuned_model")
            if fine_tuned_model:
                self.rft_model = fine_tuned_model
                self.logger.info(f"{Fore.GREEN}✅ Updated to use fine-tuned model: {fine_tuned_model}")
                return True
        
        return False

    def get_performance_analytics(self) -> Dict[str, Any]:
        """
        Get analytics on task performance and improvement trends.
        """
        if not self.improvement_history:
            return {"message": "No improvement history available"}
        
        improvements = [record["improvement"] for record in self.improvement_history]
        initial_scores = [record["initial_score"] for record in self.improvement_history]
        final_scores = [record["final_score"] for record in self.improvement_history]
        
        analytics = {
            "total_improvement_attempts": len(self.improvement_history),
            "average_improvement": np.mean(improvements),
            "average_initial_score": np.mean(initial_scores),
            "average_final_score": np.mean(final_scores),
            "successful_improvements": len([i for i in improvements if i > 0]),
            "improvement_rate": len([i for i in improvements if i > 0]) / len(improvements),
            "best_improvement": max(improvements) if improvements else 0,
            "recent_performance": final_scores[-10:] if len(final_scores) >= 10 else final_scores
        }
        
        return analytics

    def _load_improvement_history(self) -> None:
        """Load improvement history from file."""
        history_file = os.path.join(self.rft_training_data_path, "improvement_history.json")
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    self.improvement_history = json.load(f)
                self.logger.info(f"Loaded {len(self.improvement_history)} improvement records")
            except Exception as e:
                self.logger.warning(f"Failed to load improvement history: {e}")

    def save_improvement_history(self) -> None:
        """Save improvement history to file."""
        history_file = os.path.join(self.rft_training_data_path, "improvement_history.json")
        try:
            with open(history_file, 'w') as f:
                json.dump(self.improvement_history, f, indent=2)
            self.logger.info(f"Saved {len(self.improvement_history)} improvement records")
        except Exception as e:
            self.logger.error(f"Failed to save improvement history: {e}")

    def export_training_dataset(self, output_file: str, min_score: float = 0.8) -> str:
        """
        Export high-quality task examples for external training.
        """
        # Collect high-scoring examples from grading history
        high_quality_samples = []
        
        for evaluation in self.rft_grader.grading_history:
            for result in evaluation["results"]:
                if result.get("score", 0) >= min_score and "error" not in result:
                    sample = {
                        "messages": [{"role": "user", "content": result["item"]["description"]}],
                        "reference_answer": result["sample"]["output_text"],
                        "score": result["score"],
                        "execution_success": result["sample"]["execution_result"].get("success", False)
                    }
                    high_quality_samples.append(sample)
        
        # Write to JSONL
        with open(output_file, 'w') as f:
            for sample in high_quality_samples:
                f.write(json.dumps(sample) + '\n')
        
        self.logger.info(f"Exported {len(high_quality_samples)} high-quality samples to {output_file}")
        return output_file 


















```

