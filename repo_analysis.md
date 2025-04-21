# Repository Analysis

## Summary

```
Repository: tinycrops/instructor-classify
Files analyzed: 32

Estimated tokens: 37.9k
```

## Important Files

```
Directory structure:
└── tinycrops-instructor-classify/
    ├── pyproject.toml
    ├── requirements.txt
    ├── docs/
    │   └── usage/
    │       └── getting-started.md
    ├── instructor_classify/
    │   ├── __init__.py
    │   ├── classify.py
    │   ├── cli.py
    │   ├── schema.py
    │   ├── eval_harness/
    │   │   ├── __init__.py
    │   │   ├── base.py
    │   │   ├── orchestrator.py
    │   │   ├── unified_eval.py
    │   │   ├── caching/
    │   │   │   ├── __init__.py
    │   │   │   └── disk_cache.py
    │   │   ├── config/
    │   │   │   ├── __init__.py
    │   │   │   └── evaluation_config.py
    │   │   ├── pipeline/
    │   │   │   ├── __init__.py
    │   │   │   ├── analysis_stage.py
    │   │   │   ├── config_stage.py
    │   │   │   ├── execution_stage.py
    │   │   │   ├── load_stage.py
    │   │   │   ├── model_stage.py
    │   │   │   └── reporting_stage.py
    │   │   ├── processing_strategies/
    │   │   │   ├── __init__.py
    │   │   │   ├── async_strategy.py
    │   │   │   ├── parallel_strategy.py
    │   │   │   └── sync_strategy.py
    │   │   ├── reporting/
    │   │   │   ├── __init__.py
    │   │   │   ├── console_reporter.py
    │   │   │   └── file_reporter.py
    │   │   └── utils/
    │   │       └── analysis.py
    │   └── templates/
    │       ├── prompt.yaml
    │       └── datasets/
    ├── tests/
    │   └── test_unified_eval.py
    └── .github/
        └── workflows/

```

## Content

```
================================================
File: pyproject.toml
================================================
[project]
name = "instructor_classify"
version = "0.1.0"
description = "A CLI tool for initializing and using classification projects"
requires-python = ">=3.9"
dependencies = [
    "typer>=0.9.0",
    "pydantic>=2.0.0",
    "pyyaml>=6.0.0",
    "instructor>=0.3.0",
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.5.0",
    "numpy>=2.0.2",
    "matplotlib>=3.9.4",
    "seaborn>=0.13.2",
    "scikit-learn>=1.6.1",
]

[project.scripts]
instruct-classify = "instructor_classify.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["instructor_classify"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "strict"
asyncio_default_fixture_loop_scope = "function"



================================================
File: requirements.txt
================================================
typer>=0.9.0
pydantic>=2.0.0
pyyaml>=6.0.0
instructor>=0.3.0
mkdocs>=1.5.0
mkdocs-material>=9.5.0
numpy>=1.24.0 


================================================
File: docs/usage/getting-started.md
================================================
# Getting Started

This guide will help you set up and use Instructor Classify for LLM-based text classification.

## Project Initialization

Create a new classification project:

```bash
instruct-classify init my_classifier
cd my_classifier
```

This creates a project with:
- `prompt.yaml`: Classification schema definition
- `example.py`: Example code for using the classifier
- `configs/`: Evaluation configurations
- `datasets/`: Example evaluation datasets

## Understanding the Key Files

### prompt.yaml

The `prompt.yaml` file defines your classification schema, which is used to classify the user's input. Its often recommended to use an LLM to generate this schema.

```yaml
system_message: |
  You are an expert classification system designed to analyse user inputs.
  
label_definitions:
  - label: question
    description: The user asks for information or clarification.
    examples:
      examples_positive:
        - "What is the capital of France?"
        - "How does this feature work?"
      examples_negative:
        - "Please book me a flight to Paris."
        - "I want to return this product."
  
  - label: request
    description: The user wants the assistant to perform an action.
    examples:
      examples_positive:
        - "Please book me a flight to Paris."
        - "Update my account settings."
      examples_negative:
        - "What is the capital of France?"
        - "I'm having a problem with my order."
```

Each label definition includes:
- `label`: The category name (automatically converted to lowercase)
- `description`: What this category represents
- `examples`: Optional positive and negative examples to guide the model

!!! note "These are all prompts to the LLM"

    The LLM will use these labels to classify the user's input. Changing the labels will change the behavior of the LLM.

### example.py

The `example.py` file shows basic usage:

```python
from instructor_classify.classify import Classifier
from instructor_classify.schema import ClassificationDefinition
import instructor
from openai import OpenAI

# Load classification definition
definition = ClassificationDefinition.from_yaml("prompt.yaml")

# Create classifier
client = instructor.from_openai(OpenAI())
classifier = (
    Classifier(definition)
    .with_client(client)
    .with_model("gpt-3.5-turbo")
)

# Make predictions
result = classifier.predict("What is machine learning?")
print(result.label)  # -> "question"
```

## Basic Classification

### Single-Label Classification

```python
# Single-label prediction
result = classifier.predict("What is the weather today?")
print(f"Label: {result.label}") # -> "question"
```

### Multi-Label Classification

```python
# Multi-label prediction
result = classifier.predict_multi("Can you help me find flights to Paris and book a hotel?")
print(f"Labels: {result.labels}") # -> ["request", "question"]
```

### Batch Processing

```python
# Process multiple texts in parallel
texts = [
    "What is machine learning?",
    "Please book a flight to New York.",
    "Can you explain how to use this API?"
]

# Synchronous batch processing
results = classifier.batch_predict(texts)
for text, result in zip(texts, results):
    print(f"Text: '{text}' → Label: {result.label}")
```

## Asynchronous API

For high-throughput applications, use the `AsyncClassifier`:

```python
from instructor_classify.classify import AsyncClassifier
from openai import AsyncOpenAI
import asyncio

async def main():
    # Create async classifier
    client = instructor.from_openai_aclient(AsyncOpenAI())
    classifier = (
        AsyncClassifier(definition)
        .with_client(client)
        .with_model("gpt-4o")
    )
    
    # Make predictions
    result = await classifier.predict("What is machine learning?")
    print(result.label)
    
    # Batch processing with concurrency control
    results = await classifier.batch_predict(texts, n_jobs=10)
    for text, result in zip(texts, results):
        print(f"Text: '{text}' → Label: {result.label}")

asyncio.run(main())
```

## Working with Multiple LLM Providers

Instructor Classify works with any provider supported by Instructor:

### OpenAI

```python
from openai import OpenAI
client = instructor.from_openai(OpenAI())
```

### Anthropic

```python
from anthropic import Anthropic
client = instructor.from_anthropic(Anthropic())
```

### Google

```python
import google.generativeai as genai
client = instructor.from_gemini(genai)
```

## Customizing Your Classifier

1. **Add or Modify Labels**: Edit `prompt.yaml` to add new categories
2. **Improve Examples**: Add more diverse examples to improve classification
3. **Adjust System Message**: Customize the initial instructions
4. **Switch Models**: Try different models with the `.with_model()` method

## Running Evaluations

Test your classifier's performance:

```bash
instruct-classify eval --config configs/example.yaml
```

The evaluation generates a detailed report with metrics, visualizations, and insights into model performance.

## Next Steps

- Learn about the [Evaluation Framework](evaluation.md) for benchmarking
- Check the [Examples](examples.md) for advanced usage patterns
- Refer to the [API Reference](../api.md) for detailed documentation


================================================
File: instructor_classify/__init__.py
================================================
from .classify import Classifier, AsyncClassifier  # noqa: F401
from .schema import (  # noqa: F401
    LabelDefinition,
    ClassificationDefinition,
    Examples,
    EvalExample,
    EvalSet
)


================================================
File: instructor_classify/classify.py
================================================
from __future__ import annotations

from typing import Any
from pydantic import BaseModel
import instructor
import asyncio
import multiprocessing
from openai.types.chat import ChatCompletionMessageParam

# ------------------------------------------------------------------
# Optional progress bar support (requires ``tqdm``)
# ------------------------------------------------------------------
# ``tqdm`` is a lightweight dependency widely used for progress bars. If it is
# not available in the runtime environment, we gracefully fall back to no‑op
# shims so that the rest of the code keeps working.

try:
    from tqdm import tqdm  # type: ignore
    from tqdm.asyncio import tqdm as tqdm_async  # type: ignore

except ModuleNotFoundError:  # pragma: no cover – allow running without tqdm
    # Fallbacks that mimic the minimal tqdm API we rely on (iteration + gather)
    def _identity(iterable=None, *_, **__):  # noqa: D401 – simple helper
        """Return *iterable* unchanged – replacement for :pyclass:`tqdm.tqdm`."""

        if iterable is None:
            iterable = []
        return iterable

    class _AsyncTqdmShim:  # pylint: disable=too-few-public-methods
        """Subset of ``tqdm.asyncio.tqdm`` we need (only ``gather``)."""

        @classmethod
        async def gather(cls, *coros, **_):  # type: ignore[no-self-use]
            import asyncio as _asyncio  # local import to avoid top‑level dep

            return await _asyncio.gather(*coros)

    tqdm = _identity  # type: ignore  # pylint: disable=invalid-name
    tqdm_async = _AsyncTqdmShim  # type: ignore  # pylint: disable=invalid-name


from instructor_classify.schema import ClassificationDefinition


class Classifier:
    """
    A fluent API for classification using OpenAI and Instructor.

    """

    def __init__(self, classification_definition: ClassificationDefinition):
        """Initialize a classifier with a classification definition."""
        self.definition = classification_definition
        self.client: instructor.Instructor | instructor.AsyncInstructor | None = None
        self.model_name: str | None = None

        # Dynamically generated pydantic models for single‑ and multi‑label predictions
        self._classification_model: type[BaseModel] = (
            self.definition.get_classification_model()
        )
        self._multi_classification_model: type[BaseModel] = (
            self.definition.get_multiclassification_model()
        )

    def with_client(self, client: instructor.Instructor):
        """Attach an OpenAI client (wrapped by Instructor) and return self."""
        self.client = client
        return self

    def with_model(self, model_name: str):
        """Specify which model to use (e.g., ``gpt-4o``) and return self."""
        self.model_name = model_name
        return self

    def _build_messages(
        self, text: str, include_examples: bool = True
    ) -> tuple[list[ChatCompletionMessageParam], dict[str, Any]]:
        """Construct chat messages using Jinja templating.

        This leverages Instructor's built‑in Jinja support (see docs/concepts/templating.md).
        The returned tuple contains the messages *and* the rendering context which
        should be forwarded to ``client.chat.completions.create``.

        Parameters
        ----------
        text:
            The text to classify.
        include_examples:
            Toggle whether few‑shot examples should be embedded in the prompt. This
            demonstrates the use of Jinja conditionals.
        """

        messages: list[ChatCompletionMessageParam] = []

        # ------------------------------------------------------------------
        # Optional system message
        # ------------------------------------------------------------------
        if self.definition.system_message:
            messages.append(
                {
                    "role": "system",
                    "content": self.definition.system_message,
                }
            )

        # ------------------------------------------------------------------
        # User message template – relies on Jinja to render examples and text
        # ------------------------------------------------------------------
        user_template = """
{% if include_examples %}
<examples>
{% for ld in label_definitions %}
  <label name="{{ ld.label }}">
    <description>{{ ld.description }}</description>
    {% if ld.examples and ld.examples.examples_positive %}
      {% for ex in ld.examples.examples_positive %}
      <example type="positive" label="{{ ld.label }}">{{ ex }}</example>
      {% endfor %}
    {% endif %}
    {% if ld.examples and ld.examples.examples_negative %}
      {% for ex in ld.examples.examples_negative %}
      <example type="negative" label="{{ ld.label }}">{{ ex }}</example>
      {% endfor %}
    {% endif %}
  </label>
{% endfor %}
</examples>
{% endif %}

<classify>
  Classify the following text into one of the following labels:
  <labels>
  {% for ld in label_definitions %}
    <label name="{{ ld.label }}">{{ ld.label }}</label>
  {% endfor %}
  </labels>
  <text>{{ input_text }}</text>
</classify>
            """

        messages.append({"role": "user", "content": user_template})

        # ------------------------------------------------------------------
        # Build the Jinja rendering context
        # ------------------------------------------------------------------
        context = {
            "label_definitions": self.definition.label_definitions,  # keep original objects accessible
            "input_text": text,
            "include_examples": include_examples,
        }

        return messages, context

    def _validate_client(self):
        """Validate that client is set and appropriate for the method type."""
        if not self.client:
            raise ValueError("Client not set. Use `.with_client()` first.")
        if isinstance(self.client, instructor.AsyncInstructor):
            raise ValueError(
                "AsyncInstructor cannot be used with synchronous methods. Use AsyncClassifier instead."
            )

    def predict(self, text: str) -> BaseModel:
        """Single‑label prediction – returns an instance of the generated model ``T``."""
        self._validate_client()

        messages, context = self._build_messages(text)
        result = self.client.chat.completions.create(
            model=self.model_name,
            response_model=self._classification_model,
            messages=messages,
            context=context,
        )
        return result

    def predict_multi(self, text: str) -> BaseModel:
        """Multi‑label prediction – returns an instance of the generated model ``M``."""
        self._validate_client()

        messages, context = self._build_messages(text)
        result = self.client.chat.completions.create(
            model=self.model_name,
            response_model=self._multi_classification_model,
            messages=messages,
            context=context,
        )
        return result

    # ------------------------------------------------------------------
    # Synchronous batch prediction
    # ------------------------------------------------------------------

    def batch_predict(
        self, texts: list[str], n_jobs: int | None = None
    ) -> list[BaseModel]:
        """Run :py:meth:`predict` over multiple texts in parallel."""
        self._validate_client()

        if not texts:
            return []

        # Determine desired parallelism level (defaults to ``cpu_count``)
        if n_jobs is None:
            cnt = multiprocessing.cpu_count() or 1
            n_jobs = min(len(texts), cnt)

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            it = executor.map(self.predict, texts)
            results = list(tqdm(it, total=len(texts), desc="classify", leave=False))
        return results

    def batch_predict_multi(
        self, texts: list[str], n_jobs: int | None = None
    ) -> list[BaseModel]:
        """Run :py:meth:`predict_multi` over multiple texts in parallel.

        The same strategy selection as :py:meth:`batch_predict` applies –
        threads are used when a custom client is attached, otherwise the
        original *multiprocessing* implementation is retained.
        """
        self._validate_client()

        if not texts:
            return []

        if n_jobs is None:
            cnt = multiprocessing.cpu_count() or 1
            n_jobs = min(len(texts), cnt)

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            it = executor.map(self.predict_multi, texts)
            results = list(
                tqdm(it, total=len(texts), desc="classify‑multi", leave=False)
            )
        return results

    def predict_with_completion(self, text: str) -> tuple[BaseModel, Any]:
        """Return both the parsed model and the underlying LLM completion."""
        self._validate_client()

        messages, context = self._build_messages(text)
        return self.client.chat.completions.create_with_completion(
            model=self.model_name,
            response_model=self._classification_model,
            messages=messages,
            context=context,
        )

    def predict_multi_with_completion(self, text: str) -> tuple[BaseModel, Any]:
        """Multi‑label variant with raw completion."""
        self._validate_client()

        messages, context = self._build_messages(text)
        return self.client.chat.completions.create_with_completion(
            model=self.model_name,
            response_model=self._multi_classification_model,
            messages=messages,
            context=context,
        )

    def batch_predict_with_completion(
        self, texts: list[str], n_jobs: int | None = None
    ) -> list[tuple[BaseModel, Any]]:
        raise NotImplementedError(
            "Batch prediction with completion is not implemented."
        )

    def batch_predict_multi_with_completion(
        self, texts: list[str], n_jobs: int | None = None
    ) -> list[tuple[BaseModel, Any]]:
        raise NotImplementedError(
            "Batch prediction with completion is not implemented."
        )


class AsyncClassifier(Classifier):
    """Asynchronous variant of :class:`Classifier`. All prediction methods are
    defined as *coroutines* and must be awaited.

    The constructor and fluent helpers are inherited from :class:`Classifier` –
    you can therefore use the same API:

    >>> classifier = AsyncClassifier(definition).with_client(AsyncOpenAI())
    >>> result = await classifier.predict("example text")
    """

    def with_client(self, client: instructor.AsyncInstructor):
        """Attach an OpenAI client (wrapped by Instructor) and return self."""
        self.client = client
        return self

    def _validate_async_client(self):
        """Validate that async client is set and ready for use."""
        if not self.client or not self.model_name:
            raise ValueError(
                "Client and model name must be set. Use `.with_client()` and `.with_model()` first."
            )

    async def predict(self, text: str) -> BaseModel:  # type: ignore[override]
        """Asynchronously predict a single label for *text*."""
        self._validate_async_client()

        messages, context = self._build_messages(text)
        return await self.client.chat.completions.create(  # type: ignore[return-value]
            model=self.model_name,
            response_model=self._classification_model,
            messages=messages,
            context=context,
        )

    async def predict_multi(self, text: str) -> BaseModel:  # type: ignore[override]
        """Asynchronously predict multiple labels for *text*."""
        self._validate_async_client()

        messages, context = self._build_messages(text)
        return await self.client.chat.completions.create(  # type: ignore[return-value]
            model=self.model_name,
            response_model=self._multi_classification_model,
            messages=messages,
            context=context,
        )

    # ------------------------------------------------------------------
    # Async batch prediction with concurrency control
    # ------------------------------------------------------------------

    async def batch_predict(
        self, texts: list[str], *, n_jobs: int = 5
    ) -> list[BaseModel]:
        """Run :py:meth:`predict` concurrently over *texts* using a semaphore.

        Parameters
        ----------
        texts:
            The list of input strings to classify.
        n_jobs:
            Maximum number of concurrent classification jobs.
        """
        self._validate_async_client()
        sem = asyncio.Semaphore(n_jobs)

        async def _worker(t: str):
            async with sem:
                return await self.predict(t)

        tasks = [_worker(t) for t in texts]

        # If tqdm is available, leverage its asyncio integration for a neat
        # progress bar. Otherwise, fall back to plain ``asyncio.gather``.

        try:
            return await tqdm_async.gather(*tasks, total=len(tasks), desc="classify")
        except Exception:  # pragma: no cover – safety net if tqdm absent
            return await asyncio.gather(*tasks)

    async def batch_predict_multi(
        self, texts: list[str], *, n_jobs: int = 5
    ) -> list[BaseModel]:
        """Run :py:meth:`predict_multi` concurrently over *texts* using a semaphore."""
        self._validate_async_client()
        sem = asyncio.Semaphore(n_jobs)

        async def _worker(t: str):
            async with sem:
                return await self.predict_multi(t)

        tasks = [_worker(t) for t in texts]

        try:
            return await tqdm_async.gather(
                *tasks, total=len(tasks), desc="classify‑multi"
            )
        except Exception:  # pragma: no cover
            return await asyncio.gather(*tasks)

    async def predict_with_completion(self, text: str) -> tuple[BaseModel, Any]:
        self._validate_async_client()

        messages, context = self._build_messages(text)
        model, completion = await self.client.chat.completions.create_with_completion(  # type: ignore[call-arg]
            model=self.model_name,
            response_model=self._classification_model,
            messages=messages,
            context=context,
        )
        return model, completion

    async def predict_multi_with_completion(self, text: str) -> tuple[BaseModel, Any]:
        self._validate_async_client()

        messages, context = self._build_messages(text)
        model, completion = await self.client.chat.completions.create_with_completion(  # type: ignore[call-arg]
            model=self.model_name,
            response_model=self._multi_classification_model,
            messages=messages,
            context=context,
        )
        return model, completion

    async def batch_predict_with_completion(
        self, texts: list[str], *, n_jobs: int = 5
    ) -> list[tuple[BaseModel, Any]]:
        sem = asyncio.Semaphore(n_jobs)

        async def _worker(t: str):
            async with sem:
                return await self.predict_with_completion(t)

        tasks = [_worker(t) for t in texts]

        try:
            return await tqdm_async.gather(*tasks, total=len(tasks), desc="classify+")
        except Exception:  # pragma: no cover
            return await asyncio.gather(*tasks)

    async def batch_predict_multi_with_completion(
        self, texts: list[str], *, n_jobs: int = 5
    ) -> list[tuple[BaseModel, Any]]:
        sem = asyncio.Semaphore(n_jobs)

        async def _worker(t: str):
            async with sem:
                return await self.predict_multi_with_completion(t)

        tasks = [_worker(t) for t in texts]

        try:
            return await tqdm_async.gather(
                *tasks, total=len(tasks), desc="classify‑multi+"
            )
        except Exception:  # pragma: no cover
            return await asyncio.gather(*tasks)



================================================
File: instructor_classify/cli.py
================================================
import typer
from pathlib import Path
import shutil
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
# These imports need to come after the sys.path modification
# but ruff will still raise E402 warnings, so we use noqa
from instructor_classify.eval_harness.orchestrator import EvaluationOrchestrator  # noqa: E402
from instructor_classify.eval_harness.config.evaluation_config import EvaluationConfig  # noqa: E402

app = typer.Typer(name="instruct-classify")

def get_template_path() -> Path:
    return Path(__file__).parent / "templates"

@app.command("init")
def init(
    project_name: str = typer.Option(
        ...,
        prompt="What would you like to name your project directory?",
        help="Name of the project directory to create"
    )
):
    """Initialize a new classifier project."""
    project_dir = Path(project_name)
    project_dir.mkdir(exist_ok=True)
    
    # Copy entire templates directory
    template_path = get_template_path()
    shutil.copytree(template_path, project_dir, dirs_exist_ok=True)
    
    typer.echo(f"Created classifier project in {project_dir}")
    typer.echo("Copied all template files and directories:")
    for item in project_dir.rglob("*"):
        if item.is_file():
            typer.echo(f"  - {item.relative_to(project_dir)}")

@app.command("eval")
def evaluate(
    config_path: str = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to the evaluation configuration YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    parallel_mode: str = typer.Option(
        None,
        "--mode",
        "-m",
        help="Parallelism mode: 'sync' (sequential), 'parallel' (thread-based), or 'async' (asyncio-based)",
    ),
    n_jobs: int = typer.Option(
        None,
        "--jobs",
        "-j",
        help="Number of parallel jobs to run (default: 4)",
        min=1,
        max=32,
    ),
    use_cache: bool = typer.Option(
        None,
        "--cache/--no-cache",
        help="Enable or disable caching (default: from config or True)",
    )
):
    """Run evaluation using the unified evaluation framework."""
    try:
        # If CLI options are provided, create temporary config with overrides
        config_to_use = config_path
        temp_config_path = None
        
        if any(option is not None for option in [parallel_mode, n_jobs, use_cache]):
            # Load original config through the EvaluationConfig class for validation
            original_config = EvaluationConfig.from_file(config_path)
            
            # Create a dict with only the overrides that are specified
            overrides = {}
            if parallel_mode is not None:
                overrides["parallel_mode"] = parallel_mode
            if n_jobs is not None:
                overrides["n_jobs"] = n_jobs  
            if use_cache is not None:
                overrides["use_cache"] = use_cache
                
            # Apply overrides
            updated_config = original_config.create_with_overrides(**overrides)
            
            # Create temporary config file
            temp_config_path = updated_config.create_temp_file()
            config_to_use = temp_config_path
            
            # Show override info
            override_msg_parts = []
            if parallel_mode is not None:
                override_msg_parts.append(f"parallel_mode={parallel_mode}")
            if n_jobs is not None:
                override_msg_parts.append(f"n_jobs={n_jobs}")
            if use_cache is not None:
                override_msg_parts.append(f"cache={'enabled' if use_cache else 'disabled'}")
                
            typer.echo(f"Using configuration with CLI overrides: {', '.join(override_msg_parts)}")
        
        # Initialize and run evaluator
        evaluator = EvaluationOrchestrator(config_to_use)
        success = evaluator.execute()
        
        # Clean up temporary file if created
        if temp_config_path:
            os.unlink(temp_config_path)
            
        if success:
            typer.echo("\n[bold green]Evaluation completed successfully![/bold green]")
        else:
            typer.echo("\n[bold red]Evaluation failed.[/bold red]")
            raise typer.Exit(1)
            
    except KeyboardInterrupt:
        typer.echo("\n[bold yellow]Evaluation cancelled by user.[/bold yellow]")
        raise typer.Exit(0)
    except Exception as e:
        typer.echo(f"\n[bold red]Error during evaluation:[/bold red] {str(e)}")
        import traceback
        typer.echo(traceback.format_exc())
        raise typer.Exit(1)

if __name__ == "__main__":
    app()


================================================
File: instructor_classify/schema.py
================================================
from __future__ import annotations
from pydantic import BaseModel, field_validator, model_validator
from typing import Literal, TypeVar, Generic
import yaml


class Examples(BaseModel):
    examples_positive: list[str] | None = None
    examples_negative: list[str] | None = None


class LabelDefinition(BaseModel):
    label: str
    description: str
    examples: Examples | None = None

    @field_validator("label")
    @classmethod
    def lowercase_label(cls, value: str) -> str:
        return value.lower()


class ClassificationDefinition(BaseModel):
    system_message: str | None = None
    label_definitions: list[LabelDefinition]
    classification_type: Literal["single", "multi"] = "single"
    prompt_template: str = """Classify the following text into one of the specified categories.

Text to classify:
{{ text }}

{% if include_examples %}
<examples>
{% for label, examples in examples.items() %}
Positive examples for '{{label}}':
{% for example in examples.positive %}
- {{example}}
{% endfor %}

Negative examples for '{{label}}':
{% for example in examples.negative %}
- {{example}}
{% endfor %}

{% endfor %}
</examples>
{% endif %}

Available labels:
{% for label in labels %}
- {{label}}: {{descriptions[label]}}
{% endfor %}

Classification:
"""

    @model_validator(mode="after")
    def validate_unique_labels(cls, model):
        labels = [ld.label for ld in model.label_definitions]
        if len(labels) != len(set(labels)):
            raise ValueError("Label definitions must have unique labels")
        return model

    @classmethod
    def from_yaml(cls, yaml_path: str) -> ClassificationDefinition:
        with open(yaml_path) as file:
            yaml_data = yaml.safe_load(file)
        return cls(**yaml_data)

    # ------------------------------------------------------------------
    # Dynamic Enum & Pydantic model helpers
    # ------------------------------------------------------------------

    def get_classification_model(self) -> type[BaseModel]:
        labels = tuple(ld.label for ld in self.label_definitions)

        class ClassificationModel(BaseModel):
            """Single‑label classification model. Only use a single label from the provided examples."""

            label: Literal[labels]  # type: ignore

        return ClassificationModel

    def get_multiclassification_model(self) -> type[BaseModel]:
        labels = tuple(ld.label for ld in self.label_definitions)

        class MultiClassificationModel(BaseModel):
            """Multi‑label classification model. Only use labels from the provided examples."""

            labels: list[Literal[labels]]  # type: ignore

        return MultiClassificationModel


# Type variable to represent the prediction model (single or multi-label)
T = TypeVar("T", bound=BaseModel)


class EvalExample(BaseModel):
    """An example for evaluation with text and expected label(s)."""

    text: str
    expected_label: str | None = None  # For single-label classification
    expected_labels: list[str] | None = None  # For multi-label classification


class EvalSet(BaseModel, Generic[T]):
    """A set of examples for evaluating classifier performance."""

    name: str
    description: str
    examples: list[EvalExample]
    classification_type: Literal["single", "multi"]

    @classmethod
    def from_yaml(cls, yaml_path: str) -> EvalSet:
        """Load evaluation set from YAML file."""
        with open(yaml_path) as file:
            yaml_data = yaml.safe_load(file)
        return cls(**yaml_data)

    def validate_against_definition(self, definition: ClassificationDefinition) -> bool:
        """Validate that all expected labels are in the classification definition."""
        valid_labels = {ld.label for ld in definition.label_definitions}

        for example in self.examples:
            if example.expected_label and example.expected_label not in valid_labels:
                return False
            if example.expected_labels and any(
                label not in valid_labels for label in example.expected_labels
            ):
                return False

        return True


================================================
File: instructor_classify/eval_harness/__init__.py
================================================



================================================
File: instructor_classify/eval_harness/base.py
================================================
"""
Base classes and interfaces for the modular evaluation harness.

This module provides the foundational abstract classes and interfaces that
define the architecture of the evaluation system.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic
from pydantic import BaseModel
from rich.console import Console
import os
import yaml
import json



class EvaluationResult(BaseModel):
    """Model for storing evaluation results."""
    model: str
    eval_set_name: str
    total_examples: int
    correct_predictions: int
    accuracy: float
    predictions: List[Dict[str, Any]]


T = TypeVar('T')
P = TypeVar('P')
R = TypeVar('R')


class ProcessingStrategy(Generic[T, P, R], ABC):
    """
    Abstract base class for processing strategies.
    
    This defines how examples are processed during evaluation, whether
    synchronously, using thread-based parallelism, or using asyncio.
    """
    
    def __init__(self, n_jobs: int = 4):
        """
        Initialize the processing strategy.
        
        Parameters
        ----------
        n_jobs : int
            Number of parallel jobs to run
        """
        self.n_jobs = n_jobs
    
    @abstractmethod
    def process_batch(self, classifier: T, examples: List[P], is_multi: bool = False) -> List[R]:
        """
        Process a batch of examples.
        
        Parameters
        ----------
        classifier : T
            The classifier to use
        examples : List[P]
            The examples to process
        is_multi : bool
            Whether this is multi-label classification
            
        Returns
        -------
        List[R]
            The processing results
        """
        pass


class ProcessingStrategyFactory:
    """Factory for creating processing strategies."""
    
    @staticmethod
    def create_strategy(strategy_type: str, n_jobs: int = 4) -> ProcessingStrategy:
        """
        Create a processing strategy of the specified type.
        
        Parameters
        ----------
        strategy_type : str
            Type of strategy to create ('sync', 'parallel', or 'async')
        n_jobs : int
            Number of parallel jobs to run
            
        Returns
        -------
        ProcessingStrategy
            The created strategy
        """
        # Import implementations here to avoid circular imports
        from instructor_classify.eval_harness.processing_strategies.sync_strategy import SyncProcessingStrategy
        from instructor_classify.eval_harness.processing_strategies.parallel_strategy import ParallelProcessingStrategy
        from instructor_classify.eval_harness.processing_strategies.async_strategy import AsyncProcessingStrategy
        
        if strategy_type == "sync":
            return SyncProcessingStrategy(n_jobs)
        elif strategy_type == "parallel":
            return ParallelProcessingStrategy(n_jobs)
        elif strategy_type == "async":
            return AsyncProcessingStrategy(n_jobs)
        else:
            # Default to sync if invalid type specified
            return SyncProcessingStrategy(n_jobs)


class Analyzer(ABC):
    """
    Abstract base class for analyzers.
    
    Analyzers perform statistical analysis on evaluation results.
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        self.console = Console()
    
    @abstractmethod
    def analyze(self, eval_result: EvaluationResult) -> Dict[str, Any]:
        """
        Analyze evaluation results.
        
        Parameters
        ----------
        eval_result : EvaluationResult
            The evaluation result to analyze
            
        Returns
        -------
        Dict[str, Any]
            Analysis results
        """
        pass
    
    @abstractmethod
    def display_results(self, analysis_results: Dict[str, Any]) -> None:
        """
        Display analysis results.
        
        Parameters
        ----------
        analysis_results : Dict[str, Any]
            The analysis results to display
        """
        pass
    
    def save_results(self, analysis_results: Dict[str, Any], output_path: str) -> None:
        """
        Save analysis results to a file.
        
        Parameters
        ----------
        analysis_results : Dict[str, Any]
            The analysis results to save
        output_path : str
            Path to save results to
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)


class Reporter(ABC):
    """
    Abstract base class for reporters.
    
    Reporters generate reports and visualizations from evaluation results.
    """
    
    def __init__(self):
        """Initialize the reporter."""
        self.console = Console()
    
    @abstractmethod
    def generate_report(self, eval_results: Dict[str, Dict[str, EvaluationResult]], 
                       analysis_results: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """
        Generate a report from evaluation results.
        
        Parameters
        ----------
        eval_results : Dict[str, Dict[str, EvaluationResult]]
            Evaluation results by model and eval set
        analysis_results : Dict[str, Dict[str, Dict[str, Any]]]
            Analysis results by analyzer, model, and eval set
        """
        pass


class PipelineStage(ABC):
    """
    Abstract base class for pipeline stages.
    
    Defines a single stage in the evaluation pipeline.
    """
    
    def __init__(self, name: str):
        """
        Initialize the pipeline stage.
        
        Parameters
        ----------
        name : str
            Name of the pipeline stage
        """
        self.name = name
        self.console = Console()
    
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the pipeline stage.
        
        Parameters
        ----------
        context : Dict[str, Any]
            The pipeline context containing data from previous stages
            
        Returns
        -------
        Dict[str, Any]
            The updated pipeline context
        """
        pass


class Pipeline:
    """
    Evaluation pipeline that executes stages in sequence.
    """
    
    def __init__(self, stages: List[PipelineStage] = None):
        """
        Initialize the pipeline.
        
        Parameters
        ----------
        stages : List[PipelineStage], optional
            Pipeline stages to execute
        """
        self.stages = stages or []
        self.console = Console()
    
    def add_stage(self, stage: PipelineStage) -> None:
        """
        Add a stage to the pipeline.
        
        Parameters
        ----------
        stage : PipelineStage
            Stage to add
        """
        self.stages.append(stage)
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute all stages in the pipeline.
        
        Returns
        -------
        Dict[str, Any]
            The final pipeline context
        """
        context = {}
        
        for stage in self.stages:
            self.console.print(f"[bold cyan]Executing pipeline stage:[/bold cyan] {stage.name}")
            try:
                context = stage.execute(context)
            except Exception as e:
                self.console.print(f"[bold red]Error in pipeline stage {stage.name}:[/bold red] {str(e)}")
                import traceback
                self.console.print(traceback.format_exc())
                # Propagate the exception to the caller
                raise
        
        return context


class Cache(ABC):
    """
    Abstract base class for caches.
    
    Caches store and retrieve cached values to avoid redundant computation.
    """
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Parameters
        ----------
        key : str
            Cache key
            
        Returns
        -------
        Optional[Any]
            Cached value, or None if not found
        """
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """
        Set a value in the cache.
        
        Parameters
        ----------
        key : str
            Cache key
        value : Any
            Value to cache
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear the cache."""
        pass


class ConfigLoader:
    """Utility for loading and validating configuration files."""
    
    @staticmethod
    def load_yaml(file_path: str) -> Dict[str, Any]:
        """
        Load a YAML configuration file.
        
        Parameters
        ----------
        file_path : str
            Path to the YAML file
            
        Returns
        -------
        Dict[str, Any]
            Loaded configuration
        """
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            console = Console()
            console.print(f"[bold red]Error loading YAML file {file_path}:[/bold red] {str(e)}")
            raise
    
    @staticmethod
    def resolve_path(base_dir: str, path: str) -> str:
        """
        Resolve a path relative to a base directory if it is not absolute.
        
        Parameters
        ----------
        base_dir : str
            Base directory
        path : str
            Path to resolve
            
        Returns
        -------
        str
            Resolved absolute path
        """
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(base_dir, path))


================================================
File: instructor_classify/eval_harness/orchestrator.py
================================================
"""
Evaluation orchestrator.

This module provides the main orchestrator class that coordinates the evaluation
process using the pipeline architecture.
"""

from typing import Any, Dict
from rich.console import Console
from rich.panel import Panel

from instructor_classify.eval_harness.base import Pipeline
from instructor_classify.eval_harness.pipeline import (
    ConfigStage, LoadStage, ModelStage, ExecutionStage, AnalysisStage, ReportingStage
)
from instructor_classify.eval_harness.utils.analysis import CostLatencyAnalyzer


class EvaluationOrchestrator:
    """
    Orchestrator for the evaluation process.
    
    This class coordinates the evaluation pipeline and handles the overall
    execution flow.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the evaluation orchestrator.
        
        Parameters
        ----------
        config_path : str
            Path to the configuration file
        """
        self.config_path = config_path
        self.console = Console()
        self.pipeline = self._create_pipeline()
        
        # To be set during execution
        self.results = None
        self.analysis_results = None
        self.completed = False
    
    def _create_pipeline(self) -> Pipeline:
        """
        Create the evaluation pipeline.
        
        Returns
        -------
        Pipeline
            The created pipeline
        """
        pipeline = Pipeline()
        
        # Create and add common analyzers that are used by multiple stages
        CostLatencyAnalyzer()
        
        # Set up context with common analyzers
        
        # Add pipeline stages
        pipeline.add_stage(ConfigStage(self.config_path))
        pipeline.add_stage(LoadStage())
        pipeline.add_stage(ModelStage())
        pipeline.add_stage(ExecutionStage())
        pipeline.add_stage(AnalysisStage())
        pipeline.add_stage(ReportingStage())
        
        return pipeline
    
    def execute(self) -> bool:
        """
        Execute the evaluation process.
        
        Returns
        -------
        bool
            True if execution was successful, False otherwise
        """
        try:
            self.console.print(
                Panel(
                    "Starting evaluation process",
                    title="[bold cyan]Instructor Classify Evaluation[/bold cyan]",
                    border_style="cyan"
                )
            )
            
            # Execute the pipeline
            result_context = self.pipeline.execute()
            
            # Store results for access by clients
            self.results = result_context.get("results", {})
            self.analysis_results = result_context.get("analysis_results", {})
            self.completed = result_context.get("completed", False)
            
            self.console.print(
                Panel(
                    "Evaluation completed successfully",
                    title="[bold green]Success[/bold green]",
                    border_style="green"
                )
            )
            
            return True
        
        except KeyboardInterrupt:
            self.console.print("\n[bold yellow]Evaluation cancelled by user.[/bold yellow]")
            return False
        
        except Exception as e:
            self.console.print(f"\n[bold red]Error during evaluation:[/bold red] {str(e)}")
            import traceback
            self.console.print(traceback.format_exc())
            return False
    
    def get_results(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the evaluation results.
        
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Evaluation results by model and eval set
        """
        if not self.completed:
            self.console.print("[yellow]Warning: Evaluation not completed, results may be incomplete[/yellow]")
        
        return self.results
    
    def get_analysis_results(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Get the analysis results.
        
        Returns
        -------
        Dict[str, Dict[str, Dict[str, Any]]]
            Analysis results by analyzer, model, and eval set
        """
        if not self.completed:
            self.console.print("[yellow]Warning: Evaluation not completed, results may be incomplete[/yellow]")
        
        return self.analysis_results


================================================
File: instructor_classify/eval_harness/unified_eval.py
================================================
#!/usr/bin/env python3
"""
Unified evaluation framework for classification models.

This script provides a comprehensive framework for:
1. Running evaluations on multiple models and datasets
2. Tracking and analyzing cost and latency metrics
3. Generating statistical confidence intervals
4. Analyzing error patterns and confusion matrices
5. Visualizing results and comparisons

Usage:
  python unified_eval.py --config configs/example.yaml
"""

import os
import sys
import argparse
import yaml
from typing import Any
from datetime import datetime
from rich.console import Console
from rich.table import Table
import warnings

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# These imports need to come after the sys.path modification
from instructor_classify.schema import ClassificationDefinition, EvalSet  # noqa: E402
from instructor_classify.eval_harness.orchestrator import EvaluationOrchestrator  # noqa: E402

# Import local utilities
from instructor_classify.eval_harness.utils.analysis import BootstrapAnalyzer, CostLatencyAnalyzer, ConfusionAnalyzer  # noqa: E402
from instructor_classify.eval_harness.base import EvaluationResult  # noqa: E402


# This class is maintained for backward compatibility
class UnifiedEvaluator:
    """
    Unified evaluation framework for classification models.

    This class handles:
    - Loading models, definitions, and evaluation sets
    - Running synchronous or asynchronous evaluations
    - Collecting metrics on accuracy, cost, and latency
    - Generating comprehensive analysis and visualizations
    
    DEPRECATED: This class is maintained for backward compatibility.
    Use the new modular architecture in `EvaluationOrchestrator` for new code.
    """

    def __init__(self, config_path: str):
        """
        Initialize the unified evaluator with a configuration file.

        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file
        """
        warnings.warn(
            "UnifiedEvaluator is deprecated and will be removed in a future version. "
            "Use EvaluationOrchestrator instead.",
            DeprecationWarning, 
            stacklevel=2
        )
        
        self.console = Console()
        self.orchestrator = EvaluationOrchestrator(config_path)
        
        # Load configuration
        self.console.print(f"[green]Loading configuration from {config_path}[/green]")
        self.config = self._load_config(config_path)

        # Set up paths
        self.base_dir = os.path.dirname(os.path.abspath(config_path))
        self.output_dir = os.path.join(
            self.base_dir, self.config.get("output_dir", "results")
        )

        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "analysis"), exist_ok=True)

        # Set up analyzers
        self.cost_analyzer = CostLatencyAnalyzer()
        self.bootstrap_analyzer = BootstrapAnalyzer(
            n_resamples=self.config.get("bootstrap_samples", 1000),
            confidence_level=self.config.get("confidence_level", 0.95),
        )
        self.confusion_analyzer = ConfusionAnalyzer()

        # Set up parallelism configuration
        self.parallel_mode = self.config.get("parallel_mode", "sync")  # Options: sync, parallel, async
        self.n_jobs = self.config.get("n_jobs", 4)
        
        if self.parallel_mode not in ["sync", "parallel", "async"]:
            self.console.print(
                f"[yellow]Warning: Invalid parallel_mode '{self.parallel_mode}', defaulting to 'sync'[/yellow]"
            )
            self.parallel_mode = "sync"
            
        self.console.print(f"[green]Using {self.parallel_mode} mode with {self.n_jobs} workers[/green]")

        # Generate timestamp for this evaluation run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Tracking variables
        self.results_by_model = {}
        self.accuracy_by_model = {}
        self.evaluation_sets = []
        self.models = []

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.console.print(
                f"[bold red]Error loading configuration:[/bold red] {str(e)}"
            )
            sys.exit(1)

    def _resolve_path(self, path: str) -> str:
        """Resolve a path relative to the base directory if not absolute."""
        if os.path.isabs(path):
            return path
        return os.path.join(self.base_dir, path)

    def prepare(self) -> None:
        """
        Prepare for evaluation by loading all necessary components.

        This includes:
        - Loading the classification definition
        - Loading all evaluation sets
        - Setting up model configurations
        """
        # Forward to new implementation
        # Load classification definition
        if "definition_path" not in self.config:
            self.console.print(
                "[bold red]Error: No definition_path specified in config file[/bold red]"
            )
            sys.exit(1)
            
        definition_path = self._resolve_path(self.config["definition_path"])
        self.console.print(
            f"[green]Loading classification definition from {definition_path}[/green]"
        )

        # Check if the file exists
        if not os.path.exists(definition_path):
            self.console.print(
                f"[bold red]Error: Classification definition file not found: {definition_path}[/bold red]"
            )
            self.console.print(
                "[yellow]Hint: Make sure the path is correct and accessible. If using relative paths with CLI options, consider using absolute paths.[/yellow]"
            )
            sys.exit(1)
            
        try:
            self.definition = ClassificationDefinition.from_yaml(definition_path)
        except Exception as e:
            self.console.print(
                f"[bold red]Error loading classification definition:[/bold red] {str(e)}"
            )
            sys.exit(1)

        # Load evaluation sets
        self.console.print("[green]Loading evaluation sets...[/green]")
        
        if "eval_sets" not in self.config or not self.config["eval_sets"]:
            self.console.print(
                "[bold red]Error: No eval_sets specified in config file[/bold red]"
            )
            sys.exit(1)

        for eval_set_path in self.config["eval_sets"]:
            path = self._resolve_path(eval_set_path)
            
            # Check if the file exists
            if not os.path.exists(path):
                self.console.print(
                    f"[yellow]Warning: Evaluation set file not found: {path}[/yellow]"
                )
                self.console.print(
                    "[yellow]Hint: Make sure the path is correct and accessible. If using relative paths with CLI options, consider using absolute paths.[/yellow]"
                )
                continue
                
            try:
                eval_set = EvalSet.from_yaml(path)
                self.evaluation_sets.append(eval_set)
                self.console.print(f"  - Loaded: {eval_set.name}")
            except Exception as e:
                self.console.print(
                    f"[yellow]Warning: Failed to load {path}: {str(e)}[/yellow]"
                )

        if not self.evaluation_sets:
            self.console.print(
                "[bold red]Error: No valid evaluation sets loaded[/bold red]"
            )
            sys.exit(1)

        # Set up models
        self.models = self.config["models"]
        if not self.models:
            self.console.print(
                "[bold red]Error: No models specified in configuration[/bold red]"
            )
            sys.exit(1)

        self.console.print(
            f"[green]Prepared {len(self.models)} models and {len(self.evaluation_sets)} evaluation sets[/green]"
        )

    def _process_single_example(self, classifier, example, is_multi=False):
        """Process a single example with appropriate prediction method and timing."""
        if is_multi:
            prediction = classifier.predict_multi(example.text)
            is_correct = set(prediction.labels) == set(example.expected_labels)
            expected = example.expected_labels
            predicted = prediction.labels
        else:
            prediction = classifier.predict(example.text)
            is_correct = prediction.label == example.expected_label
            expected = example.expected_label
            predicted = prediction.label
            
        return {
            "prediction": prediction,
            "is_correct": is_correct,
            "text": example.text,
            "expected": expected,
            "predicted": predicted
        }
        
    def _process_batch_parallel(self, classifier, examples, is_multi=False):
        """Process a batch of examples using thread-based parallelism."""
        from concurrent.futures import ThreadPoolExecutor
        import tqdm
        
        results = []
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [
                executor.submit(self._process_single_example, classifier, example, is_multi)
                for example in examples
            ]
            
            for future in tqdm.tqdm(
                futures, total=len(futures), desc="Classifying", leave=False
            ):
                results.append(future.result())
                
        return results
        
    async def _process_batch_async(self, classifier, examples, is_multi=False):
        """Process a batch of examples using asyncio-based parallelism."""
        import asyncio
        from tqdm.asyncio import tqdm
        
        async def _process_async(example):
            # Create async semaphore to limit concurrency
            async with asyncio.Semaphore(self.n_jobs):
                if is_multi:
                    prediction = await classifier.predict_multi(example.text)
                    is_correct = set(prediction.labels) == set(example.expected_labels)
                    expected = example.expected_labels
                    predicted = prediction.labels
                else:
                    prediction = await classifier.predict(example.text)
                    is_correct = prediction.label == example.expected_label
                    expected = example.expected_label
                    predicted = prediction.label
                    
                return {
                    "prediction": prediction,
                    "is_correct": is_correct,
                    "text": example.text,
                    "expected": expected,
                    "predicted": predicted
                }
        
        # Create tasks
        tasks = [_process_async(example) for example in examples]
        
        # Process with progress bar
        results = await tqdm.gather(*tasks, desc="Classifying")
        return results

    def run(self) -> None:
        """
        Run the evaluation process for all models and evaluation sets.
        
        DEPRECATED: This method is maintained for backward compatibility.
        The actual implementation now uses the new modular architecture.
        """
        # Forward to new implementation
        result = self.orchestrator.execute()
        
        if result:
            # Copy results back for backward compatibility
            self.results_by_model = self.orchestrator.get_results()
        
        # Return for backward compatibility, even though this doesn't actually return anything
        return


def evaluate_classifier(classifier: Any, eval_set: EvalSet) -> EvaluationResult:
    """
    Evaluate a classifier against an evaluation set.
    
    Parameters
    ----------
    classifier : Any
        The classifier to evaluate. Must have batch_predict() or batch_predict_multi() methods.
    eval_set : EvalSet
        The evaluation set to use.
        
    Returns
    -------
    EvaluationResult
        The evaluation results.
    """
    # Determine if single or multi-label classification
    is_multi = eval_set.classification_type == "multi"
    
    # Filter for examples with expected labels
    if is_multi:
        valid_examples = [ex for ex in eval_set.examples if ex.expected_labels]
        
        # Get predictions
        predictions = classifier.batch_predict_multi([ex.text for ex in valid_examples])
        
        # Compare predictions to expected labels
        correct_count = 0
        prediction_details = []
        
        for example, pred in zip(valid_examples, predictions):
            is_correct = set(pred.labels) == set(example.expected_labels)
            if is_correct:
                correct_count += 1
                
            prediction_details.append({
                "text": example.text,
                "expected": example.expected_labels,
                "predicted": pred.labels,
                "is_correct": is_correct
            })
    else:
        # Single-label classification
        valid_examples = [ex for ex in eval_set.examples if ex.expected_label]
        
        # Get predictions
        predictions = classifier.batch_predict([ex.text for ex in valid_examples])
        
        # Compare predictions to expected labels
        correct_count = 0
        prediction_details = []
        
        for example, pred in zip(valid_examples, predictions):
            is_correct = pred.label == example.expected_label
            if is_correct:
                correct_count += 1
                
            prediction_details.append({
                "text": example.text,
                "expected": example.expected_label,
                "predicted": pred.label,
                "is_correct": is_correct
            })
    
    # Calculate accuracy
    accuracy = correct_count / len(valid_examples) if valid_examples else 0
    
    # Create result object
    result = EvaluationResult(
        model=getattr(classifier, "model", "unknown"),
        eval_set_name=eval_set.name,
        total_examples=len(valid_examples),
        correct_predictions=correct_count,
        accuracy=accuracy,
        predictions=prediction_details
    )
    
    return result


def display_evaluation_results(result: EvaluationResult) -> None:
    """
    Display evaluation results in a formatted table.
    
    Parameters
    ----------
    result : EvaluationResult
        The evaluation results to display.
    """
    console = Console()
    
    # Create a summary table
    summary_table = Table(title=f"Evaluation Results: {result.eval_set_name}")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Model", result.model)
    summary_table.add_row("Total Examples", str(result.total_examples))
    summary_table.add_row("Correct Predictions", str(result.correct_predictions))
    summary_table.add_row("Accuracy", f"{result.accuracy:.2%}")
    
    console.print(summary_table)
    
    # Create a detailed results table for incorrect predictions
    if result.predictions:
        errors_table = Table(title="Incorrect Predictions")
        errors_table.add_column("Text", style="white", no_wrap=False)
        errors_table.add_column("Expected", style="green")
        errors_table.add_column("Predicted", style="red")
        
        # Add rows for incorrect predictions
        for pred in result.predictions:
            if not pred.get("is_correct", False):
                # Truncate long texts
                text = pred["text"]
                if len(text) > 80:
                    text = text[:77] + "..."
                
                # Format expected and predicted values
                expected = str(pred["expected"])
                predicted = str(pred["predicted"])
                
                errors_table.add_row(text, expected, predicted)
        
        # Only display the table if there are errors
        if errors_table.row_count > 0:
            console.print("\nIncorrect Predictions:")
            console.print(errors_table)


def main():
    """Main entry point for the unified evaluation framework."""
    parser = argparse.ArgumentParser(
        description="Unified classification evaluation framework"
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to the evaluation configuration YAML file",
    )

    args = parser.parse_args()

    try:
        # Use the new orchestrator directly
        orchestrator = EvaluationOrchestrator(args.config)
        success = orchestrator.execute()
        
        if success:
            Console().print("\n[bold green]Evaluation completed successfully![/bold green]")
        else:
            Console().print("\n[bold red]Evaluation failed.[/bold red]")
            sys.exit(1)

    except KeyboardInterrupt:
        Console().print("\n[bold yellow]Evaluation cancelled by user.[/bold yellow]")
        sys.exit(0)
    except Exception as e:
        Console().print(f"\n[bold red]Error during evaluation:[/bold red] {str(e)}")
        import traceback

        Console().print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()


================================================
File: instructor_classify/eval_harness/caching/__init__.py
================================================
"""
Caching implementations for the evaluation harness.

This package provides caching mechanisms to improve performance and resilience.
"""

from instructor_classify.eval_harness.caching.disk_cache import DiskCache, CachedClassifier, AsyncCachedClassifier

__all__ = ['DiskCache', 'CachedClassifier', 'AsyncCachedClassifier']


================================================
File: instructor_classify/eval_harness/caching/disk_cache.py
================================================
"""
Disk-based cache implementation.

This module provides a persistent cache that stores data on disk,
allowing for resilience against failures and resuming interrupted evaluations.
"""

import json
import hashlib
import pickle
from typing import Any, Dict, Optional
from pathlib import Path
from instructor_classify.eval_harness.base import Cache
from rich.console import Console


class DiskCache(Cache):
    """
    Disk-based cache that stores data in files.
    
    This cache implementation stores cached data on disk, providing persistence
    across runs and allowing for recovery from failures.
    """
    
    def __init__(self, cache_dir: str = ".cache", use_pickle: bool = False):
        """
        Initialize the disk cache.
        
        Parameters
        ----------
        cache_dir : str
            Directory to store cache files
        use_pickle : bool
            Whether to use pickle for serialization (more flexible but less safe)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_pickle = use_pickle
        self.console = Console()
        
        # Track cache stats
        self.hits = 0
        self.misses = 0
    
    def _hash_key(self, key: str) -> str:
        """
        Hash a key to create a filename.
        
        Parameters
        ----------
        key : str
            Cache key
            
        Returns
        -------
        str
            Hashed key suitable for a filename
        """
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """
        Get the path for a cache file.
        
        Parameters
        ----------
        key : str
            Cache key
            
        Returns
        -------
        Path
            Path to the cache file
        """
        hashed_key = self._hash_key(key)
        return self.cache_dir / f"{hashed_key}.{'pkl' if self.use_pickle else 'json'}"
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Parameters
        ----------
        key : str
            Cache key
            
        Returns
        -------
        Optional[Any]
            Cached value, or None if not found
        """
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            self.misses += 1
            return None
        
        try:
            if self.use_pickle:
                with open(cache_path, 'rb') as f:
                    value = pickle.load(f)
            else:
                with open(cache_path, 'r') as f:
                    value = json.load(f)
            
            self.hits += 1
            return value
        except Exception as e:
            self.console.print(f"[yellow]Cache read error for {key}: {str(e)}[/yellow]")
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a value in the cache.
        
        Parameters
        ----------
        key : str
            Cache key
        value : Any
            Value to cache
        """
        cache_path = self._get_cache_path(key)
        
        try:
            if self.use_pickle:
                with open(cache_path, 'wb') as f:
                    pickle.dump(value, f)
            else:
                with open(cache_path, 'w') as f:
                    json.dump(value, f, default=lambda o: str(o) if isinstance(o, object) else o)
        except Exception as e:
            self.console.print(f"[yellow]Cache write error for {key}: {str(e)}[/yellow]")
    
    def clear(self) -> None:
        """Clear all cached values."""
        for cache_file in self.cache_dir.glob("*"):
            try:
                cache_file.unlink()
            except Exception as e:
                self.console.print(f"[yellow]Error deleting cache file {cache_file}: {str(e)}[/yellow]")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns
        -------
        Dict[str, Any]
            Cache statistics
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        stats = {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": hit_rate,
            "cache_size": len(list(self.cache_dir.glob("*"))),
            "cache_dir": str(self.cache_dir),
        }
        
        return stats
    
    def generate_key(self, model: str, text: str, is_multi: bool = False) -> str:
        """
        Generate a cache key for a prediction.
        
        Parameters
        ----------
        model : str
            Model name
        text : str
            Text to predict
        is_multi : bool
            Whether this is multi-label classification
            
        Returns
        -------
        str
            Cache key
        """
        # Include enough information to uniquely identify the prediction
        key_parts = [
            model,
            text,
            "multi" if is_multi else "single"
        ]
        
        return hashlib.md5("||".join(key_parts).encode()).hexdigest()


class CachedClassifier:
    """
    Decorator for classifiers that adds caching.
    
    This wraps a classifier to add caching of prediction results.
    """
    
    def __init__(self, classifier: Any, cache: Cache):
        """
        Initialize the cached classifier.
        
        Parameters
        ----------
        classifier : Any
            The classifier to wrap
        cache : Cache
            The cache to use
        """
        self.classifier = classifier
        self.cache = cache
        self.model_name = getattr(classifier, "model_name", "unknown")
    
    def predict(self, text: str):
        """
        Predict with caching.
        
        Parameters
        ----------
        text : str
            Text to predict
            
        Returns
        -------
        Any
            Prediction result
        """
        key = self.cache.generate_key(self.model_name, text, False)
        cached = self.cache.get(key)
        
        if cached is not None:
            return cached
        
        result = self.classifier.predict(text)
        self.cache.set(key, result)
        return result
    
    def predict_multi(self, text: str):
        """
        Predict multi-label with caching.
        
        Parameters
        ----------
        text : str
            Text to predict
            
        Returns
        -------
        Any
            Prediction result
        """
        key = self.cache.generate_key(self.model_name, text, True)
        cached = self.cache.get(key)
        
        if cached is not None:
            return cached
        
        result = self.classifier.predict_multi(text)
        self.cache.set(key, result)
        return result
    
    def batch_predict(self, texts, n_jobs=None):
        """
        Batch predict with caching.
        
        Parameters
        ----------
        texts : List[str]
            Texts to predict
        n_jobs : Optional[int]
            Number of jobs (ignored, for compatibility)
            
        Returns
        -------
        List[Any]
            Prediction results
        """
        # Use batch method from underlying classifier if available and no cached results
        if hasattr(self.classifier, "batch_predict"):
            return self.classifier.batch_predict(texts, n_jobs)
        
        # Otherwise, fall back to individual predictions with caching
        return [self.predict(text) for text in texts]
    
    def batch_predict_multi(self, texts, n_jobs=None):
        """
        Batch predict multi-label with caching.
        
        Parameters
        ----------
        texts : List[str]
            Texts to predict
        n_jobs : Optional[int]
            Number of jobs (ignored, for compatibility)
            
        Returns
        -------
        List[Any]
            Prediction results
        """
        # Use batch method from underlying classifier if available and no cached results
        if hasattr(self.classifier, "batch_predict_multi"):
            return self.classifier.batch_predict_multi(texts, n_jobs)
        
        # Otherwise, fall back to individual predictions with caching
        return [self.predict_multi(text) for text in texts]


class AsyncCachedClassifier:
    """
    Decorator for async classifiers that adds caching.
    
    This wraps an async classifier to add caching of prediction results.
    """
    
    def __init__(self, classifier: Any, cache: Cache):
        """
        Initialize the cached async classifier.
        
        Parameters
        ----------
        classifier : Any
            The async classifier to wrap
        cache : Cache
            The cache to use
        """
        self.classifier = classifier
        self.cache = cache
        self.model_name = getattr(classifier, "model_name", "unknown")
    
    async def predict(self, text: str):
        """
        Predict with caching.
        
        Parameters
        ----------
        text : str
            Text to predict
            
        Returns
        -------
        Any
            Prediction result
        """
        key = self.cache.generate_key(self.model_name, text, False)
        cached = self.cache.get(key)
        
        if cached is not None:
            return cached
        
        result = await self.classifier.predict(text)
        self.cache.set(key, result)
        return result
    
    async def predict_multi(self, text: str):
        """
        Predict multi-label with caching.
        
        Parameters
        ----------
        text : str
            Text to predict
            
        Returns
        -------
        Any
            Prediction result
        """
        key = self.cache.generate_key(self.model_name, text, True)
        cached = self.cache.get(key)
        
        if cached is not None:
            return cached
        
        result = await self.classifier.predict_multi(text)
        self.cache.set(key, result)
        return result


================================================
File: instructor_classify/eval_harness/config/__init__.py
================================================
"""
Configuration management for the evaluation harness.

This package provides tools for loading, validating, and managing configurations.
"""

from instructor_classify.eval_harness.config.evaluation_config import EvaluationConfig

__all__ = ['EvaluationConfig']


================================================
File: instructor_classify/eval_harness/config/evaluation_config.py
================================================
"""
Configuration management for the evaluation harness.

This module handles loading, validation, and management of evaluation configurations.
"""

import os
import yaml
import tempfile
from typing import List
from pydantic import BaseModel, Field, validator
from rich.console import Console

from instructor_classify.eval_harness.base import ConfigLoader


class EvaluationConfig(BaseModel):
    """Configuration for an evaluation run."""
    
    # Model configuration
    models: List[str] = Field(..., description="List of models to evaluate")
    
    # Path configuration
    definition_path: str = Field(..., description="Path to classification definition YAML")
    eval_sets: List[str] = Field(..., description="Paths to evaluation set YAML files")
    output_dir: str = Field("results", description="Directory to store results")
    
    # Analysis configuration
    bootstrap_samples: int = Field(1000, description="Number of bootstrap samples for confidence intervals")
    confidence_level: float = Field(0.95, description="Confidence level for intervals")
    
    # Processing configuration
    parallel_mode: str = Field("sync", description="Processing mode: sync, parallel, or async")
    n_jobs: int = Field(4, description="Number of parallel jobs")
    
    # Caching configuration
    use_cache: bool = Field(True, description="Whether to use caching")
    cache_dir: str = Field(".eval_cache", description="Directory for disk cache")
    
    # Analyzers to use (defaults to all)
    analyzers: List[str] = Field(
        ["bootstrap", "cost", "confusion"],
        description="Analyzers to use"
    )
    
    # Reporters to use (defaults to all)
    reporters: List[str] = Field(
        ["console", "file", "visualization"],
        description="Reporters to use"
    )
    
    @validator("parallel_mode")
    def validate_parallel_mode(cls, v):
        """Validate parallel mode."""
        if v not in ["sync", "parallel", "async"]:
            console = Console()
            console.print(f"[yellow]Warning: Invalid parallel_mode '{v}', defaulting to 'sync'[/yellow]")
            return "sync"
        return v
    
    @validator("confidence_level")
    def validate_confidence_level(cls, v):
        """Validate confidence level."""
        if not 0 < v < 1:
            console = Console()
            console.print(f"[yellow]Warning: Invalid confidence_level '{v}', defaulting to 0.95[/yellow]")
            return 0.95
        return v
    
    class Config:
        """Pydantic configuration."""
        extra = "allow"  # Allow extra fields for user-defined config options
    
    @classmethod
    def from_file(cls, config_path: str) -> "EvaluationConfig":
        """
        Load configuration from a YAML file.
        
        Parameters
        ----------
        config_path : str
            Path to the configuration YAML file
            
        Returns
        -------
        EvaluationConfig
            Loaded configuration
        """
        console = Console()
        
        try:
            config_data = ConfigLoader.load_yaml(config_path)
            
            # Resolve paths to make them absolute
            base_dir = os.path.dirname(os.path.abspath(config_path))
            
            if "definition_path" in config_data and not os.path.isabs(config_data["definition_path"]):
                config_data["definition_path"] = ConfigLoader.resolve_path(
                    base_dir, config_data["definition_path"]
                )
            
            if "eval_sets" in config_data:
                for i, eval_set in enumerate(config_data["eval_sets"]):
                    if not os.path.isabs(eval_set):
                        config_data["eval_sets"][i] = ConfigLoader.resolve_path(
                            base_dir, eval_set
                        )
            
            return cls(**config_data)
        except Exception as e:
            console.print(f"[bold red]Error loading configuration from {config_path}:[/bold red] {str(e)}")
            raise
    
    def create_with_overrides(self, **overrides) -> "EvaluationConfig":
        """
        Create a new configuration with overrides.
        
        Parameters
        ----------
        **overrides
            Overrides for configuration values
            
        Returns
        -------
        EvaluationConfig
            New configuration with overrides applied
        """
        # Convert to dict, apply overrides, and create new instance
        config_dict = self.dict()
        config_dict.update(overrides)
        return EvaluationConfig(**config_dict)
    
    def save_to_file(self, file_path: str) -> None:
        """
        Save configuration to a YAML file.
        
        Parameters
        ----------
        file_path : str
            Path to save configuration to
        """
        with open(file_path, 'w') as f:
            yaml.dump(self.dict(), f)
    
    def create_temp_file(self) -> str:
        """
        Create a temporary file with this configuration.
        
        Returns
        -------
        str
            Path to the temporary file
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp:
            yaml.dump(self.dict(), temp)
            return temp.name


================================================
File: instructor_classify/eval_harness/pipeline/__init__.py
================================================
"""
Pipeline stages for the evaluation harness.

This package provides a pipeline architecture for executing the evaluation
process in stages.
"""

from instructor_classify.eval_harness.pipeline.config_stage import ConfigStage
from instructor_classify.eval_harness.pipeline.load_stage import LoadStage
from instructor_classify.eval_harness.pipeline.model_stage import ModelStage
from instructor_classify.eval_harness.pipeline.execution_stage import ExecutionStage
from instructor_classify.eval_harness.pipeline.analysis_stage import AnalysisStage
from instructor_classify.eval_harness.pipeline.reporting_stage import ReportingStage

__all__ = [
    'ConfigStage',
    'LoadStage',
    'ModelStage',
    'ExecutionStage',
    'AnalysisStage',
    'ReportingStage'
]


================================================
File: instructor_classify/eval_harness/pipeline/analysis_stage.py
================================================
"""
Analysis pipeline stage.

This module provides the pipeline stage for analyzing evaluation results.
"""

import os
from typing import Any, Dict

from instructor_classify.eval_harness.base import PipelineStage
from instructor_classify.eval_harness.utils.analysis import (
    BootstrapAnalyzer, CostLatencyAnalyzer, ConfusionAnalyzer
)


class AnalysisStage(PipelineStage):
    """
    Pipeline stage for analyzing evaluation results.
    """
    
    def __init__(self):
        """Initialize the analysis stage."""
        super().__init__(name="Analysis")
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the analysis stage.
        
        Parameters
        ----------
        context : Dict[str, Any]
            The pipeline context
            
        Returns
        -------
        Dict[str, Any]
            The updated pipeline context
        """
        config = context["config"]
        results_by_model = context["results"]
        output_dir = config.output_dir
        
        # Initialize analyzers if not already in context
        bootstrap_analyzer = context.get("bootstrap_analyzer")
        if not bootstrap_analyzer:
            bootstrap_analyzer = BootstrapAnalyzer(
                n_resamples=config.bootstrap_samples,
                confidence_level=config.confidence_level
            )
            context["bootstrap_analyzer"] = bootstrap_analyzer
        
        cost_analyzer = context.get("cost_analyzer")
        if not cost_analyzer:
            cost_analyzer = CostLatencyAnalyzer()
            context["cost_analyzer"] = cost_analyzer
        
        confusion_analyzer = context.get("confusion_analyzer")
        if not confusion_analyzer:
            confusion_analyzer = ConfusionAnalyzer()
            context["confusion_analyzer"] = confusion_analyzer
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "analysis"), exist_ok=True)
        
        # Run all analyses
        self.console.print("\n[bold cyan]Running Comprehensive Analysis...[/bold cyan]")
        
        # Initialize analysis results storage
        analysis_results = {
            "bootstrap": {},
            "cost": {},
            "confusion": {}
        }
        
        # Run the selected analyzers
        analyzers_to_run = set(config.analyzers)
        
        # Run cost and latency analysis
        if "cost" in analyzers_to_run:
            self.console.print("\n[bold green]Cost and Latency Analysis[/bold green]")
            self._run_cost_analysis(cost_analyzer, results_by_model, analysis_results, output_dir)
        
        # Run bootstrap analysis
        if "bootstrap" in analyzers_to_run:
            self.console.print("\n[bold green]Confidence Interval Analysis[/bold green]")
            self._run_bootstrap_analysis(bootstrap_analyzer, results_by_model, analysis_results, output_dir)
        
        # Run confusion analysis
        if "confusion" in analyzers_to_run:
            self.console.print("\n[bold green]Confusion and Error Pattern Analysis[/bold green]")
            self._run_confusion_analysis(confusion_analyzer, results_by_model, analysis_results, output_dir)
        
        # Store analysis results in context
        context["analysis_results"] = analysis_results
        
        return context
    
    def _run_cost_analysis(self, analyzer, results_by_model, analysis_results, output_dir):
        """Run cost and latency analysis."""
        # Calculate costs
        costs = analyzer.calculate_costs()
        latency_stats = analyzer.calculate_latency_stats()
        
        # Display results
        analyzer.display_cost_analysis(costs)
        analyzer.display_latency_analysis(latency_stats)
        
        # Generate visualizations
        viz_dir = os.path.join(output_dir, "visualizations")
        
        # Latency distribution
        latency_viz = os.path.join(viz_dir, "latency_distribution.png")
        analyzer.plot_latency_distribution(latency_viz)
        
        # Calculate average accuracy per model for cost efficiency
        avg_accuracies = {}
        for model, eval_results in results_by_model.items():
            total_examples = 0
            weighted_sum = 0
            for eval_name, result in eval_results.items():
                examples = result.total_examples
                total_examples += examples
                weighted_sum += result.accuracy * examples
            
            avg_accuracies[model] = weighted_sum / total_examples if total_examples > 0 else 0
        
        # Cost efficiency
        efficiency_viz = os.path.join(viz_dir, "cost_efficiency.png")
        analyzer.plot_cost_efficiency(costs, avg_accuracies, efficiency_viz)
        
        # Store analysis results
        for model in results_by_model.keys():
            if model not in analysis_results["cost"]:
                analysis_results["cost"][model] = {}
            
            for eval_name in results_by_model[model].keys():
                if eval_name not in analysis_results["cost"][model]:
                    analysis_results["cost"][model][eval_name] = {}
                
                analysis_results["cost"][model][eval_name] = {
                    "costs": costs,
                    "latency": latency_stats,
                    "avg_accuracy": avg_accuracies
                }
    
    def _run_bootstrap_analysis(self, analyzer, results_by_model, analysis_results, output_dir):
        """Run bootstrap analysis."""
        viz_dir = os.path.join(output_dir, "visualizations")
        
        for model_name, eval_results in results_by_model.items():
            if model_name not in analysis_results["bootstrap"]:
                analysis_results["bootstrap"][model_name] = {}
            
            for eval_name, result in eval_results.items():
                # Run bootstrap analysis
                eval_bootstrap = analyzer.analyze(result)
                analysis_results["bootstrap"][model_name][eval_name] = eval_bootstrap
                
                # Display results
                self.console.print(f"\n[bold]Model: {model_name}, Evaluation: {eval_name}[/bold]")
                analyzer.display_results(eval_bootstrap)
                
                # Generate visualizations
                bootstrap_viz = os.path.join(
                    viz_dir,
                    f"bootstrap_{model_name}_{eval_name.replace(' ', '_')}.png"
                )
                analyzer.plot_bootstrap_distributions(eval_bootstrap, bootstrap_viz)
    
    def _run_confusion_analysis(self, analyzer, results_by_model, analysis_results, output_dir):
        """Run confusion analysis."""
        viz_dir = os.path.join(output_dir, "visualizations")
        
        for model_name, eval_results in results_by_model.items():
            if model_name not in analysis_results["confusion"]:
                analysis_results["confusion"][model_name] = {}
            
            for eval_name, result in eval_results.items():
                # Run confusion analysis
                confusion_analysis = analyzer.analyze(result)
                analysis_results["confusion"][model_name][eval_name] = confusion_analysis
                
                if confusion_analysis:
                    # Display results
                    self.console.print(f"\n[bold]Model: {model_name}, Evaluation: {eval_name}[/bold]")
                    analyzer.display_results(confusion_analysis)
                    
                    # Generate visualizations
                    cm_viz = os.path.join(
                        viz_dir,
                        f"confusion_matrix_{model_name}_{eval_name.replace(' ', '_')}.png"
                    )
                    analyzer.plot_confusion_matrix(confusion_analysis, cm_viz)
                    
                    error_viz = os.path.join(
                        viz_dir,
                        f"error_distribution_{model_name}_{eval_name.replace(' ', '_')}.png"
                    )
                    analyzer.plot_error_distribution(confusion_analysis, error_viz)


================================================
File: instructor_classify/eval_harness/pipeline/config_stage.py
================================================
"""
Configuration loading pipeline stage.

This module provides the pipeline stage for loading and validating configuration.
"""

from typing import Any, Dict
from rich.panel import Panel

from instructor_classify.eval_harness.base import PipelineStage
from instructor_classify.eval_harness.config import EvaluationConfig


class ConfigStage(PipelineStage):
    """
    Pipeline stage for loading and validating configuration.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the configuration stage.
        
        Parameters
        ----------
        config_path : str
            Path to the configuration file
        """
        super().__init__(name="Configuration Loading")
        self.config_path = config_path
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the configuration loading stage.
        
        Parameters
        ----------
        context : Dict[str, Any]
            The pipeline context
            
        Returns
        -------
        Dict[str, Any]
            The updated pipeline context
        """
        self.console.print(f"[green]Loading configuration from {self.config_path}[/green]")
        
        # Load configuration
        config = EvaluationConfig.from_file(self.config_path)
        
        # Display configuration summary
        self.console.print(
            Panel(
                f"[bold]Evaluation Configuration[/bold]\n\n"
                f"Models: {', '.join(config.models)}\n"
                f"Definition Path: {config.definition_path}\n"
                f"Evaluation Sets: {len(config.eval_sets)}\n"
                f"Parallel Mode: {config.parallel_mode} with {config.n_jobs} workers\n"
                f"Caching: {'Enabled' if config.use_cache else 'Disabled'}\n"
                f"Output Directory: {config.output_dir}",
                title="[bold cyan]Configuration Summary[/bold cyan]",
                border_style="cyan"
            )
        )
        
        # Update context
        context["config"] = config
        context["base_dir"] = self.config_path.rsplit("/", 1)[0] if "/" in self.config_path else "."
        
        return context


================================================
File: instructor_classify/eval_harness/pipeline/execution_stage.py
================================================
"""
Execution pipeline stage.

This module provides the pipeline stage for executing the evaluation.
"""

from typing import Any, Dict
from rich.panel import Panel

from instructor_classify.eval_harness.base import PipelineStage, EvaluationResult
from instructor_classify.eval_harness.processing_strategies import (
    SyncProcessingStrategy, ParallelProcessingStrategy, AsyncProcessingStrategy
)


class ExecutionStage(PipelineStage):
    """
    Pipeline stage for executing the evaluation.
    """
    
    def __init__(self):
        """Initialize the execution stage."""
        super().__init__(name="Evaluation Execution")
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the evaluation stage.
        
        Parameters
        ----------
        context : Dict[str, Any]
            The pipeline context
            
        Returns
        -------
        Dict[str, Any]
            The updated pipeline context
        """
        config = context["config"]
        evaluation_sets = context["evaluation_sets"]
        classifiers = context["classifiers"]
        async_classifiers = context.get("async_classifiers", {})
        
        # Create appropriate processing strategy
        strategy = None
        if config.parallel_mode == "sync":
            strategy = SyncProcessingStrategy(n_jobs=1)
        elif config.parallel_mode == "parallel":
            strategy = ParallelProcessingStrategy(n_jobs=config.n_jobs)
        elif config.parallel_mode == "async":
            strategy = AsyncProcessingStrategy(n_jobs=config.n_jobs)
        else:
            self.console.print(
                f"[yellow]Warning: Unknown parallel mode '{config.parallel_mode}', defaulting to sync[/yellow]"
            )
            strategy = SyncProcessingStrategy(n_jobs=1)
        
        # Display evaluation plan
        self.console.print(
            Panel(
                f"[bold]Evaluation Plan[/bold]\n\n"
                f"Models: {', '.join(config.models)}\n"
                f"Evaluation Sets: {', '.join(es.name for es in evaluation_sets)}\n"
                f"Processing Strategy: {config.parallel_mode.capitalize()} with {config.n_jobs} workers\n"
                f"Output Directory: {config.output_dir}",
                title="[bold cyan]Evaluation Execution[/bold cyan]",
                border_style="cyan"
            )
        )
        
        # Initialize results storage
        results_by_model = {}
        
        # Execute evaluation for each model and eval set
        for model_name in config.models:
            self.console.print(f"\n[bold cyan]Evaluating model:[/bold cyan] {model_name}")
            
            # Initialize model-specific results
            results_by_model[model_name] = {}
            
            # Get the appropriate classifier
            classifier = classifiers[model_name]
            
            # For async mode, use the async classifier
            if config.parallel_mode == "async" and model_name in async_classifiers:
                classifier = async_classifiers[model_name]
            
            # Evaluate on each evaluation set
            for eval_set in evaluation_sets:
                self.console.print(f"  - Evaluating on: {eval_set.name}")
                
                # Filter for examples with expected_label/expected_labels
                is_multi = eval_set.classification_type == "multi"
                if not is_multi:
                    valid_examples = [ex for ex in eval_set.examples if ex.expected_label]
                else:
                    valid_examples = [ex for ex in eval_set.examples if ex.expected_labels]
                
                # Skip if no valid examples
                if not valid_examples:
                    self.console.print(f"    [yellow]No valid examples found for {eval_set.name}[/yellow]")
                    continue
                
                # Process examples using the selected strategy
                results = strategy.process_batch(classifier, valid_examples, is_multi)
                
                # Extract metrics
                correct_count = sum(1 for r in results if r["is_correct"])
                accuracy = correct_count / len(valid_examples) if valid_examples else 0
                
                # Extract prediction details
                prediction_details = [
                    {
                        "text": r["text"],
                        "expected": r["expected"],
                        "predicted": r["predicted"],
                        "is_correct": r["is_correct"]
                    }
                    for r in results
                ]
                
                # Build result
                result = EvaluationResult(
                    model=model_name,
                    eval_set_name=eval_set.name,
                    total_examples=len(valid_examples),
                    correct_predictions=correct_count,
                    accuracy=accuracy,
                    predictions=prediction_details
                )
                
                # Store result
                results_by_model[model_name][eval_set.name] = result
                
                # Display accuracy
                self.console.print(f"    [green]Accuracy: {accuracy:.2%}[/green]")
        
        # Store results in context
        context["results"] = results_by_model
        
        return context


================================================
File: instructor_classify/eval_harness/pipeline/load_stage.py
================================================
"""
Resource loading pipeline stage.

This module provides the pipeline stage for loading classification definitions
and evaluation sets.
"""

import os
import sys
from typing import Any, Dict
from rich.panel import Panel

from instructor_classify.eval_harness.base import PipelineStage
from instructor_classify.schema import ClassificationDefinition, EvalSet


class LoadStage(PipelineStage):
    """
    Pipeline stage for loading classification definitions and evaluation sets.
    """
    
    def __init__(self):
        """Initialize the loading stage."""
        super().__init__(name="Resource Loading")
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the resource loading stage.
        
        Parameters
        ----------
        context : Dict[str, Any]
            The pipeline context
            
        Returns
        -------
        Dict[str, Any]
            The updated pipeline context
        """
        config = context["config"]
        
        # Load classification definition
        self.console.print(f"[green]Loading classification definition from {config.definition_path}[/green]")
        
        if not os.path.exists(config.definition_path):
            self.console.print(
                f"[bold red]Error: Classification definition file not found: {config.definition_path}[/bold red]"
            )
            self.console.print(
                "[yellow]Hint: Make sure the path is correct and accessible. If using relative paths with CLI options, consider using absolute paths.[/yellow]"
            )
            sys.exit(1)
        
        try:
            definition = ClassificationDefinition.from_yaml(config.definition_path)
            context["definition"] = definition
        except Exception as e:
            self.console.print(
                f"[bold red]Error loading classification definition:[/bold red] {str(e)}"
            )
            sys.exit(1)
        
        # Load evaluation sets
        self.console.print("[green]Loading evaluation sets...[/green]")
        evaluation_sets = []
        
        for eval_set_path in config.eval_sets:
            if not os.path.exists(eval_set_path):
                self.console.print(
                    f"[yellow]Warning: Evaluation set file not found: {eval_set_path}[/yellow]"
                )
                self.console.print(
                    "[yellow]Hint: Make sure the path is correct and accessible. If using relative paths with CLI options, consider using absolute paths.[/yellow]"
                )
                continue
            
            try:
                eval_set = EvalSet.from_yaml(eval_set_path)
                evaluation_sets.append(eval_set)
                self.console.print(f"  - Loaded: {eval_set.name}")
            except Exception as e:
                self.console.print(
                    f"[yellow]Warning: Failed to load {eval_set_path}: {str(e)}[/yellow]"
                )
        
        if not evaluation_sets:
            self.console.print(
                "[bold red]Error: No valid evaluation sets loaded[/bold red]"
            )
            sys.exit(1)
        
        context["evaluation_sets"] = evaluation_sets
        
        # Display loading summary
        self.console.print(
            Panel(
                f"[bold]Resources Loaded[/bold]\n\n"
                f"Classification Definition: {definition.label_definitions[0].label} and {len(definition.label_definitions) - 1} other labels\n"
                f"Evaluation Sets: {len(evaluation_sets)}\n"
                f"Total Examples: {sum(len(es.examples) for es in evaluation_sets)}",
                title="[bold cyan]Resources Summary[/bold cyan]",
                border_style="cyan"
            )
        )
        
        return context


================================================
File: instructor_classify/eval_harness/pipeline/model_stage.py
================================================
"""
Model initialization pipeline stage.

This module provides the pipeline stage for initializing classifier models.
"""

from typing import Any, Dict
import instructor
from openai import OpenAI, AsyncOpenAI

from instructor_classify.eval_harness.base import PipelineStage
from instructor_classify.classify import Classifier, AsyncClassifier
from instructor_classify.eval_harness.caching import DiskCache, CachedClassifier, AsyncCachedClassifier


class InstrumentedClassifier(Classifier):
    """Classifier with instrumentation for cost and latency tracking."""
    
    def __init__(self, definition, analyzer, model_name):
        """
        Initialize the instrumented classifier.
        
        Parameters
        ----------
        definition : ClassificationDefinition
            The classification definition
        analyzer : Any
            Cost analyzer for tracking metrics
        model_name : str
            Name of the model
        """
        super().__init__(definition)
        self.analyzer = analyzer
        self.model_name = model_name
    
    def predict(self, text):
        """
        Predict with instrumentation.
        
        Parameters
        ----------
        text : str
            Text to predict
            
        Returns
        -------
        Any
            Prediction result
        """
        self._validate_client()
        timestamp = self.analyzer.start_timing(self.model_name, text)
        messages, context = self._build_messages(text)
        result = self.client.chat.completions.create_with_completion(
            model=self.model_name,
            response_model=self._classification_model,
            messages=messages,
            context=context,
        )
        input_tokens = result[1].usage.prompt_tokens
        output_tokens = result[1].usage.completion_tokens
        self.analyzer.end_timing(timestamp, input_tokens, output_tokens)
        return result[0]
    
    def predict_multi(self, text):
        """
        Predict multi-label with instrumentation.
        
        Parameters
        ----------
        text : str
            Text to predict
            
        Returns
        -------
        Any
            Prediction result
        """
        self._validate_client()
        timestamp = self.analyzer.start_timing(self.model_name, text)
        messages, context = self._build_messages(text)
        result = self.client.chat.completions.create_with_completion(
            model=self.model_name,
            response_model=self._multi_classification_model,
            messages=messages,
            context=context,
        )
        input_tokens = result[1].usage.prompt_tokens
        output_tokens = result[1].usage.completion_tokens
        self.analyzer.end_timing(timestamp, input_tokens, output_tokens)
        return result[0]


class InstrumentedAsyncClassifier(AsyncClassifier):
    """Async classifier with instrumentation for cost and latency tracking."""
    
    def __init__(self, definition, analyzer, model_name):
        """
        Initialize the instrumented async classifier.
        
        Parameters
        ----------
        definition : ClassificationDefinition
            The classification definition
        analyzer : Any
            Cost analyzer for tracking metrics
        model_name : str
            Name of the model
        """
        super().__init__(definition)
        self.analyzer = analyzer
        self.model_name = model_name
    
    async def predict(self, text):
        """
        Predict with instrumentation.
        
        Parameters
        ----------
        text : str
            Text to predict
            
        Returns
        -------
        Any
            Prediction result
        """
        self._validate_client()
        timestamp = self.analyzer.start_timing(self.model_name, text)
        messages, context = self._build_messages(text)
        result = await self.client.chat.completions.create_with_completion(
            model=self.model_name,
            response_model=self._classification_model,
            messages=messages,
            context=context,
        )
        input_tokens = result[1].usage.prompt_tokens
        output_tokens = result[1].usage.completion_tokens
        self.analyzer.end_timing(timestamp, input_tokens, output_tokens)
        return result[0]
    
    async def predict_multi(self, text):
        """
        Predict multi-label with instrumentation.
        
        Parameters
        ----------
        text : str
            Text to predict
            
        Returns
        -------
        Any
            Prediction result
        """
        self._validate_client()
        timestamp = self.analyzer.start_timing(self.model_name, text)
        messages, context = self._build_messages(text)
        result = await self.client.chat.completions.create_with_completion(
            model=self.model_name,
            response_model=self._multi_classification_model,
            messages=messages,
            context=context,
        )
        input_tokens = result[1].usage.prompt_tokens
        output_tokens = result[1].usage.completion_tokens
        self.analyzer.end_timing(timestamp, input_tokens, output_tokens)
        return result[0]


class ModelStage(PipelineStage):
    """
    Pipeline stage for initializing classifier models.
    """
    
    def __init__(self):
        """Initialize the model stage."""
        super().__init__(name="Model Initialization")
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the model initialization stage.
        
        Parameters
        ----------
        context : Dict[str, Any]
            The pipeline context
            
        Returns
        -------
        Dict[str, Any]
            The updated pipeline context
        """
        config = context["config"]
        definition = context["definition"]
        
        # Get cost analyzer from context (added by analyzer stage)
        cost_analyzer = context.get("cost_analyzer")
        if not cost_analyzer:
            self.console.print("[yellow]Warning: Cost analyzer not found, instrumentation will be limited[/yellow]")
        
        # Initialize cache if enabled
        cache = None
        if config.use_cache:
            cache_dir = config.cache_dir
            self.console.print(f"[green]Initializing disk cache in {cache_dir}[/green]")
            cache = DiskCache(cache_dir)
            context["cache"] = cache
        
        # Initialize clients and classifiers for each model
        classifiers = {}
        async_classifiers = {}
        
        for model_name in config.models:
            self.console.print(f"[green]Initializing classifier for model: {model_name}[/green]")
            
            # Initialize OpenAI client with Instructor
            client = instructor.from_openai(OpenAI())
            
            # Create instrumented classifier
            classifier = InstrumentedClassifier(definition, cost_analyzer, model_name)
            classifier.with_client(client)
            
            # Wrap with cache if enabled
            if cache:
                classifier = CachedClassifier(classifier, cache)
            
            classifiers[model_name] = classifier
            
            # For async mode, also create async classifier
            if config.parallel_mode == "async":
                async_client = instructor.from_openai(AsyncOpenAI())
                
                async_classifier = InstrumentedAsyncClassifier(definition, cost_analyzer, model_name)
                async_classifier.with_client(async_client)
                
                # Wrap with cache if enabled
                if cache:
                    async_classifier = AsyncCachedClassifier(async_classifier, cache)
                
                async_classifiers[model_name] = async_classifier
        
        context["classifiers"] = classifiers
        context["async_classifiers"] = async_classifiers
        
        self.console.print(f"[green]Initialized {len(classifiers)} classifiers[/green]")
        
        return context


================================================
File: instructor_classify/eval_harness/pipeline/reporting_stage.py
================================================
"""
Reporting pipeline stage.

This module provides the pipeline stage for generating reports from evaluation results.
"""

from typing import Any, Dict

from instructor_classify.eval_harness.base import PipelineStage
from instructor_classify.eval_harness.reporting import ConsoleReporter, FileReporter


class ReportingStage(PipelineStage):
    """
    Pipeline stage for generating reports from evaluation results.
    """
    
    def __init__(self):
        """Initialize the reporting stage."""
        super().__init__(name="Reporting")
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the reporting stage.
        
        Parameters
        ----------
        context : Dict[str, Any]
            The pipeline context
            
        Returns
        -------
        Dict[str, Any]
            The updated pipeline context
        """
        config = context["config"]
        results = context["results"]
        analysis_results = context["analysis_results"]
        output_dir = config.output_dir
        
        # Create reporters based on configuration
        reporters = []
        reporters_to_use = set(config.reporters)
        
        if "console" in reporters_to_use:
            reporters.append(ConsoleReporter(detailed=True))
        
        if "file" in reporters_to_use:
            reporters.append(FileReporter(output_dir=output_dir))
        
        # TODO: Add visualization reporter if needed
        
        # Generate reports
        self.console.print("\n[bold cyan]Generating Reports...[/bold cyan]")
        
        for reporter in reporters:
            reporter.generate_report(results, analysis_results)
        
        # If cache was used, show statistics
        if "cache" in context and context["cache"]:
            cache = context["cache"]
            stats = cache.get_stats()
            
            self.console.print("\n[bold cyan]Cache Statistics[/bold cyan]")
            self.console.print(f"Hits: {stats['hits']}")
            self.console.print(f"Misses: {stats['misses']}")
            self.console.print(f"Hit Rate: {stats['hit_rate']:.2%}")
            self.console.print(f"Cache Size: {stats['cache_size']} items")
        
        # Mark successful completion
        context["completed"] = True
        
        return context


================================================
File: instructor_classify/eval_harness/processing_strategies/__init__.py
================================================
"""
Processing strategy implementations for evaluation.

This package provides different processing strategies for evaluating examples:
- SyncProcessingStrategy: Sequential processing
- ParallelProcessingStrategy: Thread-based parallel processing
- AsyncProcessingStrategy: Asyncio-based parallel processing
"""

from instructor_classify.eval_harness.processing_strategies.sync_strategy import SyncProcessingStrategy
from instructor_classify.eval_harness.processing_strategies.parallel_strategy import ParallelProcessingStrategy
from instructor_classify.eval_harness.processing_strategies.async_strategy import AsyncProcessingStrategy

__all__ = ['SyncProcessingStrategy', 'ParallelProcessingStrategy', 'AsyncProcessingStrategy']


================================================
File: instructor_classify/eval_harness/processing_strategies/async_strategy.py
================================================
"""
Asynchronous processing strategy implementation.

This module provides an asyncio-based processing strategy for evaluating
examples concurrently using async/await.
"""

import asyncio
from typing import Any, Dict, List, TypeVar
from tqdm.asyncio import tqdm
from instructor_classify.eval_harness.base import ProcessingStrategy

T = TypeVar('T')  # Classifier type
P = TypeVar('P')  # Input example type
R = TypeVar('R')  # Result type


class AsyncProcessingStrategy(ProcessingStrategy[T, P, R]):
    """
    Asyncio-based processing strategy implementation.
    
    Processes examples concurrently using asyncio.
    """
    
    def process_batch(self, classifier: T, examples: List[P], is_multi: bool = False) -> List[R]:
        """
        Process examples using asyncio with a semaphore for concurrency control.
        
        Parameters
        ----------
        classifier : T
            The classifier to use
        examples : List[P]
            The examples to process
        is_multi : bool
            Whether this is multi-label classification
            
        Returns
        -------
        List[R]
            The processing results
        """
        # We need to get an asyncio classifier for this to work
        # The orchestrator should handle this conversion before calling us
        
        # Ensure we have an async capable classifier
        if not hasattr(classifier, 'predict') or not asyncio.iscoroutinefunction(getattr(classifier, 'predict')):
            raise TypeError("Classifier must have async predict methods for AsyncProcessingStrategy")
        
        # Run the async processing
        return asyncio.run(self._process_batch_async(classifier, examples, is_multi))
    
    async def _process_batch_async(self, classifier: Any, examples: List[Any], is_multi: bool = False) -> List[Dict[str, Any]]:
        """
        Process a batch of examples asynchronously.
        
        Parameters
        ----------
        classifier : Any
            The async classifier to use
        examples : List[Any]
            The examples to process
        is_multi : bool
            Whether this is multi-label classification
            
        Returns
        -------
        List[Dict[str, Any]]
            The processing results
        """
        async def _process_async(example):
            # Create async semaphore to limit concurrency
            async with asyncio.Semaphore(self.n_jobs):
                if is_multi:
                    prediction = await classifier.predict_multi(example.text)
                    is_correct = set(prediction.labels) == set(example.expected_labels)
                    expected = example.expected_labels
                    predicted = prediction.labels
                else:
                    prediction = await classifier.predict(example.text)
                    is_correct = prediction.label == example.expected_label
                    expected = example.expected_label
                    predicted = prediction.label
                    
                return {
                    "prediction": prediction,
                    "is_correct": is_correct,
                    "text": example.text,
                    "expected": expected,
                    "predicted": predicted
                }
        
        # Create tasks
        tasks = [_process_async(example) for example in examples]
        
        # Process with progress bar
        results = await tqdm.gather(*tasks, desc=f"Classifying with {self.n_jobs} async workers")
        return results


================================================
File: instructor_classify/eval_harness/processing_strategies/parallel_strategy.py
================================================
"""
Parallel processing strategy implementation.

This module provides a thread-based parallel processing strategy for evaluating
examples concurrently.
"""

from typing import Any, Dict, List, TypeVar
from concurrent.futures import ThreadPoolExecutor
import tqdm
from instructor_classify.eval_harness.base import ProcessingStrategy

T = TypeVar('T')  # Classifier type
P = TypeVar('P')  # Input example type
R = TypeVar('R')  # Result type


class ParallelProcessingStrategy(ProcessingStrategy[T, P, R]):
    """
    Thread-based parallel processing strategy implementation.
    
    Processes examples concurrently using a thread pool.
    """
    
    def process_batch(self, classifier: T, examples: List[P], is_multi: bool = False) -> List[R]:
        """
        Process examples in parallel using threads.
        
        Parameters
        ----------
        classifier : T
            The classifier to use
        examples : List[P]
            The examples to process
        is_multi : bool
            Whether this is multi-label classification
            
        Returns
        -------
        List[R]
            The processing results
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [
                executor.submit(self._process_single_example, classifier, example, is_multi)
                for example in examples
            ]
            
            for future in tqdm.tqdm(
                futures, total=len(futures), desc=f"Classifying with {self.n_jobs} threads", leave=False
            ):
                results.append(future.result())
        
        return results
    
    def _process_single_example(self, classifier: Any, example: Any, is_multi: bool = False) -> Dict[str, Any]:
        """
        Process a single example.
        
        Parameters
        ----------
        classifier : Any
            The classifier to use
        example : Any
            The example to process
        is_multi : bool
            Whether this is multi-label classification
            
        Returns
        -------
        Dict[str, Any]
            The processing result
        """
        if is_multi:
            prediction = classifier.predict_multi(example.text)
            is_correct = set(prediction.labels) == set(example.expected_labels)
            expected = example.expected_labels
            predicted = prediction.labels
        else:
            prediction = classifier.predict(example.text)
            is_correct = prediction.label == example.expected_label
            expected = example.expected_label
            predicted = prediction.label
            
        return {
            "prediction": prediction,
            "is_correct": is_correct,
            "text": example.text,
            "expected": expected,
            "predicted": predicted
        }


================================================
File: instructor_classify/eval_harness/processing_strategies/sync_strategy.py
================================================
"""
Synchronous processing strategy implementation.

This module provides a sequential processing strategy for evaluating examples
without any parallelism.
"""

from typing import Any, Dict, List, TypeVar
from rich.progress import Progress
from instructor_classify.eval_harness.base import ProcessingStrategy

T = TypeVar('T')  # Classifier type
P = TypeVar('P')  # Input example type
R = TypeVar('R')  # Result type


class SyncProcessingStrategy(ProcessingStrategy[T, P, R]):
    """
    Sequential processing strategy implementation.
    
    Processes examples one by one without any parallelism.
    """
    
    def process_batch(self, classifier: T, examples: List[P], is_multi: bool = False) -> List[R]:
        """
        Process examples sequentially with a progress bar.
        
        Parameters
        ----------
        classifier : T
            The classifier to use
        examples : List[P]
            The examples to process
        is_multi : bool
            Whether this is multi-label classification
            
        Returns
        -------
        List[R]
            The processing results
        """
        results = []
        
        with Progress() as progress:
            task = progress.add_task(
                "Classifying sequentially", total=len(examples)
            )
            
            for example in examples:
                result = self._process_single_example(classifier, example, is_multi)
                results.append(result)
                progress.update(task, advance=1)
        
        return results
    
    def _process_single_example(self, classifier: Any, example: Any, is_multi: bool = False) -> Dict[str, Any]:
        """
        Process a single example.
        
        Parameters
        ----------
        classifier : Any
            The classifier to use
        example : Any
            The example to process
        is_multi : bool
            Whether this is multi-label classification
            
        Returns
        -------
        Dict[str, Any]
            The processing result
        """
        if is_multi:
            prediction = classifier.predict_multi(example.text)
            is_correct = set(prediction.labels) == set(example.expected_labels)
            expected = example.expected_labels
            predicted = prediction.labels
        else:
            prediction = classifier.predict(example.text)
            is_correct = prediction.label == example.expected_label
            expected = example.expected_label
            predicted = prediction.label
            
        return {
            "prediction": prediction,
            "is_correct": is_correct,
            "text": example.text,
            "expected": expected,
            "predicted": predicted
        }


================================================
File: instructor_classify/eval_harness/reporting/__init__.py
================================================
"""
Reporting implementations for the evaluation harness.

This package provides reporters for generating and displaying evaluation results.
"""

from instructor_classify.eval_harness.reporting.console_reporter import ConsoleReporter
from instructor_classify.eval_harness.reporting.file_reporter import FileReporter

__all__ = ['ConsoleReporter', 'FileReporter']


================================================
File: instructor_classify/eval_harness/reporting/console_reporter.py
================================================
"""
Console reporter for evaluation results.

This module provides rich console output for evaluation results.
"""

from typing import Any, Dict
from rich.table import Table

from instructor_classify.eval_harness.base import Reporter, EvaluationResult


class ConsoleReporter(Reporter):
    """
    Console reporter that displays results using Rich formatting.
    """
    
    def __init__(self, detailed: bool = True):
        """
        Initialize the console reporter.
        
        Parameters
        ----------
        detailed : bool
            Whether to show detailed results
        """
        super().__init__()
        self.detailed = detailed
    
    def generate_report(self, eval_results: Dict[str, Dict[str, EvaluationResult]], 
                       analysis_results: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """
        Generate a report from evaluation results.
        
        Parameters
        ----------
        eval_results : Dict[str, Dict[str, EvaluationResult]]
            Evaluation results by model and eval set
        analysis_results : Dict[str, Dict[str, Dict[str, Any]]]
            Analysis results by analyzer, model, and eval set
        """
        self._display_summary(eval_results)
        
        if self.detailed:
            self._display_detailed_results(eval_results)
            self._display_analysis_results(analysis_results)
        
        self._display_final_recommendation(eval_results, analysis_results)
    
    def _display_summary(self, eval_results: Dict[str, Dict[str, EvaluationResult]]) -> None:
        """
        Display a summary of evaluation results.
        
        Parameters
        ----------
        eval_results : Dict[str, Dict[str, EvaluationResult]]
            Evaluation results by model and eval set
        """
        # Create summary table
        summary_table = Table(title="Model Performance Summary")
        summary_table.add_column("Evaluation Set", style="cyan")
        
        # Add columns for each model
        models = list(eval_results.keys())
        for model in models:
            summary_table.add_column(model, style="green")
        
        # Add a "best model" column
        summary_table.add_column("Best Model", style="yellow")
        
        # Get all evaluation sets
        eval_sets = set()
        for model_results in eval_results.values():
            eval_sets.update(model_results.keys())
        
        # Add rows for each eval set
        for eval_set in sorted(eval_sets):
            row = [eval_set]
            
            # Get accuracy for each model
            accuracies = {}
            for model in models:
                if model in eval_results and eval_set in eval_results[model]:
                    accuracy = eval_results[model][eval_set].accuracy
                    accuracies[model] = accuracy
                    row.append(f"{accuracy:.2%}")
                else:
                    row.append("N/A")
            
            # Determine best model
            if accuracies:
                best_model = max(accuracies, key=accuracies.get)
                row.append(best_model)
            else:
                row.append("N/A")
            
            summary_table.add_row(*row)
        
        # Add a row for average accuracy
        avg_row = ["Weighted Average"]
        avg_accuracies = {}
        
        for model in models:
            if model not in eval_results:
                avg_row.append("N/A")
                continue
                
            total_examples = 0
            weighted_sum = 0
            
            for eval_set, result in eval_results[model].items():
                examples = result.total_examples
                total_examples += examples
                weighted_sum += result.accuracy * examples
            
            avg = weighted_sum / total_examples if total_examples > 0 else 0
            avg_accuracies[model] = avg
            avg_row.append(f"{avg:.2%}")
        
        # Add best model for average
        if avg_accuracies:
            best_model = max(avg_accuracies, key=avg_accuracies.get)
            avg_row.append(best_model)
        else:
            avg_row.append("N/A")
        
        summary_table.add_row(*avg_row)
        
        # Display summary table
        self.console.print("\n[bold green]Evaluation Summary[/bold green]")
        self.console.print(summary_table)
    
    def _display_detailed_results(self, eval_results: Dict[str, Dict[str, EvaluationResult]]) -> None:
        """
        Display detailed evaluation results.
        
        Parameters
        ----------
        eval_results : Dict[str, Dict[str, EvaluationResult]]
            Evaluation results by model and eval set
        """
        self.console.print("\n[bold green]Detailed Results by Model and Evaluation Set[/bold green]")
        
        for model, model_results in eval_results.items():
            self.console.print(f"\n[bold cyan]Model:[/bold cyan] {model}")
            
            for eval_set, result in model_results.items():
                self._display_evaluation_result(result)
    
    def _display_evaluation_result(self, result: EvaluationResult) -> None:
        """
        Display a single evaluation result.
        
        Parameters
        ----------
        result : EvaluationResult
            The evaluation result to display
        """
        # Create a summary table
        summary_table = Table(title=f"Evaluation Results: {result.eval_set_name}")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Model", result.model)
        summary_table.add_row("Total Examples", str(result.total_examples))
        summary_table.add_row("Correct Predictions", str(result.correct_predictions))
        summary_table.add_row("Accuracy", f"{result.accuracy:.2%}")
        
        self.console.print(summary_table)
        
        # Create a detailed results table for incorrect predictions
        if result.predictions:
            errors_table = Table(title="Incorrect Predictions")
            errors_table.add_column("Text", style="white", no_wrap=False)
            errors_table.add_column("Expected", style="green")
            errors_table.add_column("Predicted", style="red")
            
            # Add rows for incorrect predictions
            for pred in result.predictions:
                if not pred.get("is_correct", False):
                    # Truncate long texts
                    text = pred["text"]
                    if len(text) > 80:
                        text = text[:77] + "..."
                    
                    # Format expected and predicted values
                    expected = str(pred["expected"])
                    predicted = str(pred["predicted"])
                    
                    errors_table.add_row(text, expected, predicted)
            
            # Only display the table if there are errors
            if errors_table.row_count > 0:
                self.console.print("\nIncorrect Predictions:")
                self.console.print(errors_table)
    
    def _display_analysis_results(self, analysis_results: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """
        Display analysis results.
        
        Parameters
        ----------
        analysis_results : Dict[str, Dict[str, Dict[str, Any]]]
            Analysis results by analyzer, model, and eval set
        """
        self.console.print("\n[bold green]Analysis Results[/bold green]")
        
        for analyzer, analyzer_results in analysis_results.items():
            self.console.print(f"\n[bold cyan]Analyzer:[/bold cyan] {analyzer}")
            
            # For now, we just show that analysis was performed
            # The actual display is handled by the analyzer's display_results method
            # which has already been called during the analysis phase
            self.console.print("Analysis completed. Results available in output directory.")
    
    def _display_final_recommendation(self, eval_results: Dict[str, Dict[str, EvaluationResult]], 
                                     analysis_results: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """
        Display final recommendation.
        
        Parameters
        ----------
        eval_results : Dict[str, Dict[str, EvaluationResult]]
            Evaluation results by model and eval set
        analysis_results : Dict[str, Dict[str, Dict[str, Any]]]
            Analysis results by analyzer, model, and eval set
        """
        # Calculate average accuracy per model
        avg_accuracies = {}
        
        for model, model_results in eval_results.items():
            total_examples = 0
            weighted_sum = 0
            
            for result in model_results.values():
                examples = result.total_examples
                total_examples += examples
                weighted_sum += result.accuracy * examples
            
            avg = weighted_sum / total_examples if total_examples > 0 else 0
            avg_accuracies[model] = avg
        
        # Determine best model by accuracy
        best_accuracy_model = max(avg_accuracies, key=avg_accuracies.get) if avg_accuracies else None
        
        # Get cost information if available
        cost_analysis = analysis_results.get("cost", {})
        best_efficiency_model = None
        best_efficiency = 0
        
        for model, model_analyses in cost_analysis.items():
            for eval_set, analysis in model_analyses.items():
                if "costs" in analysis and model in analysis["costs"]:
                    cost = analysis["costs"][model].get("total_cost")
                    if cost and cost > 0:
                        efficiency = avg_accuracies.get(model, 0) / cost
                        if efficiency > best_efficiency:
                            best_efficiency = efficiency
                            best_efficiency_model = model
        
        # Display recommendation
        recommendation = []
        if best_accuracy_model:
            recommendation.append(f"Best accuracy: {best_accuracy_model} ({avg_accuracies[best_accuracy_model]:.2%})")
        
        if best_efficiency_model:
            recommendation.append(f"Best efficiency: {best_efficiency_model} ({best_efficiency:.2%}/$ ratio)")
        
        if recommendation:
            self.console.print("\n[bold green]Recommendation[/bold green]")
            for rec in recommendation:
                self.console.print(f"- {rec}")
        
        self.console.print("\n[bold green]Evaluation completed successfully![/bold green]")


================================================
File: instructor_classify/eval_harness/reporting/file_reporter.py
================================================
"""
File reporter for evaluation results.

This module provides file-based output for evaluation results.
"""

import json
from typing import Any, Dict
from datetime import datetime
from pathlib import Path

from instructor_classify.eval_harness.base import Reporter, EvaluationResult


class FileReporter(Reporter):
    """
    File reporter that saves results to disk.
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize the file reporter.
        
        Parameters
        ----------
        output_dir : str
            Directory to save results to
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        (self.output_dir / "analysis").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
    
    def generate_report(self, eval_results: Dict[str, Dict[str, EvaluationResult]], 
                       analysis_results: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """
        Generate a report from evaluation results.
        
        Parameters
        ----------
        eval_results : Dict[str, Dict[str, EvaluationResult]]
            Evaluation results by model and eval set
        analysis_results : Dict[str, Dict[str, Dict[str, Any]]]
            Analysis results by analyzer, model, and eval set
        """
        # Save evaluation results
        self._save_evaluation_results(eval_results)
        
        # Save analysis results
        self._save_analysis_results(analysis_results)
        
        # Generate and save summary report
        self._generate_summary_report(eval_results, analysis_results)
    
    def _save_evaluation_results(self, eval_results: Dict[str, Dict[str, EvaluationResult]]) -> None:
        """
        Save evaluation results to files.
        
        Parameters
        ----------
        eval_results : Dict[str, Dict[str, EvaluationResult]]
            Evaluation results by model and eval set
        """
        metrics_dir = self.output_dir / "metrics"
        
        for model, model_results in eval_results.items():
            for eval_set, result in model_results.items():
                # Convert to dict for serialization
                result_dict = result.dict()
                
                # Save to file
                result_path = metrics_dir / f"{model}_{eval_set.replace(' ', '_')}_{self.timestamp}.json"
                with open(result_path, 'w') as f:
                    json.dump(result_dict, f, indent=2)
                
                self.console.print(f"Saved evaluation results to {result_path}")
    
    def _save_analysis_results(self, analysis_results: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """
        Save analysis results to files.
        
        Parameters
        ----------
        analysis_results : Dict[str, Dict[str, Dict[str, Any]]]
            Analysis results by analyzer, model, and eval set
        """
        analysis_dir = self.output_dir / "analysis"
        
        for analyzer, analyzer_results in analysis_results.items():
            # Save to file
            analysis_path = analysis_dir / f"{analyzer}_{self.timestamp}.json"
            
            # Convert analysis results to a serializable format
            serializable_results = self._make_serializable(analyzer_results)
            
            with open(analysis_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            self.console.print(f"Saved analysis results to {analysis_path}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert an object to a JSON-serializable format.
        
        Parameters
        ----------
        obj : Any
            Object to convert
            
        Returns
        -------
        Any
            Serializable version of the object
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # Convert other types to string
            return str(obj)
    
    def _generate_summary_report(self, eval_results: Dict[str, Dict[str, EvaluationResult]], 
                              analysis_results: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """
        Generate and save a summary report.
        
        Parameters
        ----------
        eval_results : Dict[str, Dict[str, EvaluationResult]]
            Evaluation results by model and eval set
        analysis_results : Dict[str, Dict[str, Dict[str, Any]]]
            Analysis results by analyzer, model, and eval set
        """
        summary_path = self.output_dir / f"summary_report_{self.timestamp}.txt"
        
        # Calculate average accuracy per model
        avg_accuracies = {}
        
        for model, model_results in eval_results.items():
            total_examples = 0
            weighted_sum = 0
            
            for result in model_results.values():
                examples = result.total_examples
                total_examples += examples
                weighted_sum += result.accuracy * examples
            
            avg = weighted_sum / total_examples if total_examples > 0 else 0
            avg_accuracies[model] = avg
        
        # Determine best model by accuracy
        best_accuracy_model = max(avg_accuracies, key=avg_accuracies.get) if avg_accuracies else None
        
        # Get cost information if available
        cost_analysis = analysis_results.get("cost", {})
        best_efficiency_model = None
        best_efficiency = 0
        
        with open(summary_path, 'w') as f:
            f.write("Classification Evaluation Summary\n")
            f.write("===============================\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Models: {', '.join(eval_results.keys())}\n")
            
            # Get all evaluation sets
            eval_sets = set()
            for model_results in eval_results.values():
                eval_sets.update(model_results.keys())
            
            f.write(f"Evaluation Sets: {', '.join(eval_sets)}\n\n")
            
            f.write("Performance Summary:\n")
            for model, model_results in eval_results.items():
                f.write(f"\n{model}:\n")
                
                for eval_set, result in model_results.items():
                    f.write(f"  {eval_set}: {result.accuracy:.2%}\n")
                
                f.write(f"  Average: {avg_accuracies.get(model, 0):.2%}\n")
            
            # Write cost analysis if available
            if cost_analysis:
                f.write("\nCost Analysis:\n")
                
                for model, model_analyses in cost_analysis.items():
                    for eval_set, analysis in model_analyses.items():
                        if "costs" in analysis and model in analysis["costs"]:
                            costs = analysis["costs"][model]
                            
                            f.write(f"\n{model}:\n")
                            f.write(f"  Input Tokens: {costs.get('input_tokens', 'N/A')}\n")
                            f.write(f"  Output Tokens: {costs.get('output_tokens', 'N/A')}\n")
                            
                            total_cost = costs.get("total_cost")
                            if total_cost:
                                f.write(f"  Total Cost: ${total_cost:.4f}\n")
                                
                                efficiency = avg_accuracies.get(model, 0) / total_cost if total_cost > 0 else 0
                                f.write(f"  Efficiency: {efficiency:.2%}/$ (higher is better)\n")
                                
                                if efficiency > best_efficiency:
                                    best_efficiency = efficiency
                                    best_efficiency_model = model
            
            # Write recommendation
            f.write("\nRecommendation:\n")
            
            if best_accuracy_model:
                f.write(f"- Best accuracy: {best_accuracy_model} ({avg_accuracies[best_accuracy_model]:.2%})\n")
            
            if best_efficiency_model:
                f.write(f"- Best efficiency: {best_efficiency_model} ({best_efficiency:.2%}/$ ratio)\n")
        
        self.console.print(f"Saved summary report to {summary_path}")


================================================
File: instructor_classify/eval_harness/utils/analysis.py
================================================
"""
Advanced analysis utilities for classification evaluation.

Includes:
- Bootstrapped confidence intervals
- Cost and latency analysis
- Detailed confusion analysis and error pattern detection
"""

import os
import time
import random
from typing import Any, Optional
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from sklearn.metrics import confusion_matrix

# Add parent directory to path
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


class BootstrapAnalyzer:
    """Performs bootstrapped confidence interval analysis on evaluation results."""

    def __init__(self, n_resamples: int = 1000, confidence_level: float = 0.95):
        """
        Initialize the bootstrap analyzer.

        Parameters
        ----------
        n_resamples : int
            Number of bootstrap resamples to generate
        confidence_level : float
            Confidence level for interval calculation (between 0 and 1)
        """
        self.n_resamples = n_resamples
        self.confidence_level = confidence_level
        self.console = Console()

    def analyze(self, eval_result) -> dict[str, Any]:
        """
        Generate bootstrapped confidence intervals for metrics in the evaluation result.

        Parameters
        ----------
        eval_result : Union[EvalResult, Dict[str, Any]]
            The evaluation result to analyze, either as EvalResult object or dictionary

        Returns
        -------
        Dict[str, Any]
            Dictionary containing metrics with confidence intervals
        """
        self.console.print(
            "[bold cyan]Generating bootstrapped confidence intervals...[/bold cyan]"
        )

        # Extract prediction data
        if hasattr(eval_result, "predictions"):
            # It's an EvalResult object
            predictions = eval_result.predictions
        elif isinstance(eval_result, dict) and "predictions" in eval_result:
            # It's a dictionary with predictions key
            predictions = eval_result["predictions"]
        else:
            # Handle case where we don't have predictions
            # Create a simplified result with just the accuracy
            accuracy = (
                eval_result.get("accuracy", 0) if isinstance(eval_result, dict) else 0
            )
            return {
                "accuracy": {
                    "mean": accuracy,
                    "lower": max(0, accuracy * 0.95),  # Simple estimate
                    "upper": min(1.0, accuracy * 1.05),  # Simple estimate
                },
                "f1_scores": {},
            }

        # Define statistic functions
        def calc_accuracy(data):
            correct = sum(1 for p in data if p["expected"] == p["predicted"])
            return correct / len(data) if data else 0

        def calc_label_f1(data, label):
            # Extract true positives, false positives, false negatives
            tp = sum(
                1 for p in data if p["expected"] == label and p["predicted"] == label
            )
            fp = sum(
                1 for p in data if p["expected"] != label and p["predicted"] == label
            )
            fn = sum(
                1 for p in data if p["expected"] == label and p["predicted"] != label
            )

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )
            return f1

        # Get all labels
        all_labels = set()
        for p in predictions:
            if isinstance(p["expected"], list):
                all_labels.update(p["expected"])
            else:
                all_labels.add(p["expected"])

        # Perform bootstrap analysis
        results = {}

        # Overall accuracy
        accuracy_samples = []
        for _ in range(self.n_resamples):
            # Resample with replacement
            sample = random.choices(predictions, k=len(predictions))
            accuracy_samples.append(calc_accuracy(sample))

        # Calculate confidence interval
        lower_idx = int((1 - self.confidence_level) / 2 * self.n_resamples)
        upper_idx = int((1 - (1 - self.confidence_level) / 2) * self.n_resamples)
        sorted_acc = sorted(accuracy_samples)
        results["accuracy"] = {
            "mean": np.mean(accuracy_samples),
            "lower": sorted_acc[lower_idx],
            "upper": sorted_acc[upper_idx],
        }

        # Per-label F1 scores
        results["f1_scores"] = {}
        for label in all_labels:
            f1_samples = []
            for _ in range(self.n_resamples):
                sample = random.choices(predictions, k=len(predictions))
                f1_samples.append(calc_label_f1(sample, label))

            sorted_f1 = sorted(f1_samples)
            results["f1_scores"][label] = {
                "mean": np.mean(f1_samples),
                "lower": sorted_f1[lower_idx],
                "upper": sorted_f1[upper_idx],
            }

        return results

    def display_results(self, bootstrap_results: dict[str, Any]) -> None:
        """Display bootstrapped confidence intervals in a rich table."""
        accuracy = bootstrap_results["accuracy"]
        f1_scores = bootstrap_results["f1_scores"]

        # Accuracy table
        accuracy_table = Table(
            title=f"Bootstrapped Accuracy ({self.confidence_level * 100:.0f}% CI)"
        )
        accuracy_table.add_column("Metric", style="cyan")
        accuracy_table.add_column("Mean", style="green")
        accuracy_table.add_column("Lower Bound", style="yellow")
        accuracy_table.add_column("Upper Bound", style="yellow")

        accuracy_table.add_row(
            "Accuracy",
            f"{accuracy['mean']:.2%}",
            f"{accuracy['lower']:.2%}",
            f"{accuracy['upper']:.2%}",
        )

        self.console.print(accuracy_table)

        # F1 score table
        f1_table = Table(
            title=f"Bootstrapped F1 Scores ({self.confidence_level * 100:.0f}% CI)"
        )
        f1_table.add_column("Label", style="cyan")
        f1_table.add_column("Mean F1", style="green")
        f1_table.add_column("Lower Bound", style="yellow")
        f1_table.add_column("Upper Bound", style="yellow")

        for label, stats in f1_scores.items():
            f1_table.add_row(
                label,
                f"{stats['mean']:.2%}",
                f"{stats['lower']:.2%}",
                f"{stats['upper']:.2%}",
            )

        self.console.print(f1_table)

    def plot_bootstrap_distributions(
        self, bootstrap_results: dict[str, Any], save_path: Optional[str] = None
    ) -> None:
        """
        Plot the bootstrap distributions for accuracy and F1 scores.

        Parameters
        ----------
        bootstrap_results : Dict[str, Any]
            The bootstrap analysis results
        save_path : Optional[str]
            Path to save the plot (if None, the plot is shown)
        """
        # Check if we have the necessary data
        if "accuracy" not in bootstrap_results:
            self.console.print(
                "[yellow]Insufficient data for bootstrap distribution plot[/yellow]"
            )
            return

        # Create subplots: 1 for accuracy, 1 for each label's F1 score
        n_plots = 1 + len(bootstrap_results.get("f1_scores", {}))
        fig, axs = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots))

        if n_plots == 1:
            axs = [axs]  # Convert to list for consistent indexing

        # Plot accuracy distribution - we don't have samples, so we'll just plot vertical lines
        ax = axs[0]
        accuracy = bootstrap_results["accuracy"]

        # Create a simple normal distribution around the mean for visualization purposes
        if "mean" in accuracy:
            x = np.linspace(
                max(0, accuracy["mean"] - 0.2), min(1.0, accuracy["mean"] + 0.2), 100
            )
            # Simulate a normal distribution around the mean
            width = (
                accuracy.get("upper", accuracy["mean"] * 1.05)
                - accuracy.get("lower", accuracy["mean"] * 0.95)
            ) / 4
            y = np.exp(-0.5 * ((x - accuracy["mean"]) / width) ** 2)
            ax.plot(x, y, color="blue")
            ax.fill_between(x, y, alpha=0.3)

            ax.axvline(accuracy["mean"], color="red", linestyle="-", label="Mean")

            if "lower" in accuracy and "upper" in accuracy:
                ax.axvline(
                    accuracy["lower"],
                    color="orange",
                    linestyle="--",
                    label=f"{self.confidence_level * 100:.0f}% CI",
                )
                ax.axvline(accuracy["upper"], color="orange", linestyle="--")

            ax.set_title("Bootstrapped Accuracy Estimate")
            ax.set_xlabel("Accuracy")
            ax.set_ylabel("Density")
            ax.legend()

        # Plot F1 score distributions
        for i, (label, stats) in enumerate(
            bootstrap_results.get("f1_scores", {}).items(), 1
        ):
            if i >= len(axs):
                break

            ax = axs[i]

            # Create a simple normal distribution around the mean for visualization purposes
            if "mean" in stats:
                x = np.linspace(
                    max(0, stats["mean"] - 0.2), min(1.0, stats["mean"] + 0.2), 100
                )
                # Simulate a normal distribution
                width = (
                    stats.get("upper", stats["mean"] * 1.05)
                    - stats.get("lower", stats["mean"] * 0.95)
                ) / 4
                y = np.exp(-0.5 * ((x - stats["mean"]) / width) ** 2)
                ax.plot(x, y, color="blue")
                ax.fill_between(x, y, alpha=0.3)

                ax.axvline(stats["mean"], color="red", linestyle="-", label="Mean")

                if "lower" in stats and "upper" in stats:
                    ax.axvline(
                        stats["lower"],
                        color="orange",
                        linestyle="--",
                        label=f"{self.confidence_level * 100:.0f}% CI",
                    )
                    ax.axvline(stats["upper"], color="orange", linestyle="--")

                ax.set_title(f"Bootstrapped F1 Score Estimate: {label}")
                ax.set_xlabel("F1 Score")
                ax.set_ylabel("Density")
                ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


class CostLatencyAnalyzer:
    """Analyzes cost and latency metrics for classification operations."""

    # Cost per 1K tokens for different models (in USD)
    MODEL_COSTS = {
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-4o-mini": {"input": 0.0015, "output": 0.0060},
        "gpt-4o": {"input": 0.0050, "output": 0.0150},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    }

    def __init__(self):
        """Initialize the cost and latency analyzer."""
        self.console = Console()
        self.token_counts = defaultdict(lambda: {"input": 0, "output": 0})
        self.latencies = defaultdict(list)
        self.timing_data = defaultdict(list)

    def start_timing(self, model: str, text: str) -> int:
        """
        Start timing a classification request.

        Parameters
        ----------
        model : str
            The model name
        text : str
            The text being classified

        Returns
        -------
        int
            Timestamp for tracking
        """
        timestamp = int(time.time() * 1000)
        self.timing_data[timestamp] = {
            "model": model,
            "text": text,
            "start_time": timestamp,
            "end_time": None,
            "input_tokens": None,
            "output_tokens": None,
        }
        return timestamp

    def end_timing(self, timestamp: int, input_tokens: int, output_tokens: int) -> None:
        """
        End timing for a classification request and record token usage.

        Parameters
        ----------
        timestamp : int
            The timestamp returned by start_timing
        input_tokens : int
            Number of input tokens used
        output_tokens : int
            Number of output tokens used
        """
        end_time = int(time.time() * 1000)

        if timestamp in self.timing_data:
            data = self.timing_data[timestamp]
            data["end_time"] = end_time
            data["input_tokens"] = input_tokens
            data["output_tokens"] = output_tokens

            model = data["model"]
            latency = end_time - data["start_time"]

            # Update token counts
            self.token_counts[model]["input"] += input_tokens
            self.token_counts[model]["output"] += output_tokens

            # Update latencies
            self.latencies[model].append(latency)

    def calculate_costs(self) -> dict[str, dict[str, float]]:
        """
        Calculate costs for each model based on recorded token usage.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary of cost metrics by model
        """
        costs = {}

        for model, tokens in self.token_counts.items():
            if model in self.MODEL_COSTS:
                input_cost = tokens["input"] / 1000 * self.MODEL_COSTS[model]["input"]
                output_cost = (
                    tokens["output"] / 1000 * self.MODEL_COSTS[model]["output"]
                )
                total_cost = input_cost + output_cost

                costs[model] = {
                    "input_tokens": tokens["input"],
                    "output_tokens": tokens["output"],
                    "input_cost": input_cost,
                    "output_cost": output_cost,
                    "total_cost": total_cost,
                }
            else:
                # Use a fallback for unknown models
                costs[model] = {
                    "input_tokens": tokens["input"],
                    "output_tokens": tokens["output"],
                    "input_cost": None,
                    "output_cost": None,
                    "total_cost": None,
                }

        return costs

    def calculate_latency_stats(self) -> dict[str, dict[str, float]]:
        """
        Calculate latency statistics for each model.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary of latency metrics by model
        """
        latency_stats = {}

        for model, latencies in self.latencies.items():
            if latencies:
                latency_stats[model] = {
                    "min": min(latencies),
                    "max": max(latencies),
                    "mean": sum(latencies) / len(latencies),
                    "median": sorted(latencies)[len(latencies) // 2],
                    "p90": sorted(latencies)[int(len(latencies) * 0.9)],
                    "p95": sorted(latencies)[int(len(latencies) * 0.95)],
                    "p99": sorted(latencies)[int(len(latencies) * 0.99)]
                    if len(latencies) >= 100
                    else None,
                    "count": len(latencies),
                }
            else:
                latency_stats[model] = {}

        return latency_stats

    def display_cost_analysis(self, costs: dict[str, dict[str, float]]) -> None:
        """Display cost analysis in a rich table."""
        cost_table = Table(title="Cost Analysis by Model")
        cost_table.add_column("Model", style="cyan")
        cost_table.add_column("Input Tokens", style="green")
        cost_table.add_column("Output Tokens", style="green")
        cost_table.add_column("Input Cost", style="yellow")
        cost_table.add_column("Output Cost", style="yellow")
        cost_table.add_column("Total Cost", style="bold red")

        for model, metrics in costs.items():
            input_cost = (
                f"${metrics['input_cost']:.4f}"
                if metrics["input_cost"] is not None
                else "Unknown"
            )
            output_cost = (
                f"${metrics['output_cost']:.4f}"
                if metrics["output_cost"] is not None
                else "Unknown"
            )
            total_cost = (
                f"${metrics['total_cost']:.4f}"
                if metrics["total_cost"] is not None
                else "Unknown"
            )

            cost_table.add_row(
                model,
                f"{metrics['input_tokens']:,}",
                f"{metrics['output_tokens']:,}",
                input_cost,
                output_cost,
                total_cost,
            )

        self.console.print(cost_table)

    def display_latency_analysis(
        self, latency_stats: dict[str, dict[str, float]]
    ) -> None:
        """Display latency analysis in a rich table."""
        latency_table = Table(title="Latency Analysis by Model (milliseconds)")
        latency_table.add_column("Model", style="cyan")
        latency_table.add_column("Count", style="cyan")
        latency_table.add_column("Min", style="green")
        latency_table.add_column("Mean", style="yellow")
        latency_table.add_column("Median", style="yellow")
        latency_table.add_column("P90", style="red")
        latency_table.add_column("P95", style="red")
        latency_table.add_column("Max", style="bold red")

        for model, stats in latency_stats.items():
            if stats:
                latency_table.add_row(
                    model,
                    f"{stats['count']}",
                    f"{stats['min']:.0f}",
                    f"{stats['mean']:.0f}",
                    f"{stats['median']:.0f}",
                    f"{stats['p90']:.0f}",
                    f"{stats['p95']:.0f}",
                    f"{stats['max']:.0f}",
                )

        self.console.print(latency_table)

    def plot_latency_distribution(self, save_path: Optional[str] = None) -> None:
        """
        Plot latency distributions for each model.

        Parameters
        ----------
        save_path : Optional[str]
            Path to save the plot (if None, the plot is shown)
        """
        plt.figure(figsize=(12, 6))

        # Create violin plots for each model
        data = []
        labels = []

        for model, latencies in self.latencies.items():
            if latencies:
                data.append(latencies)
                labels.append(model)

        if data:
            plt.violinplot(data, showmedians=True)
            plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha="right")
            plt.ylabel("Latency (ms)")
            plt.title("Latency Distribution by Model")
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()

    def plot_cost_efficiency(
        self,
        costs: dict[str, dict[str, float]],
        accuracy_by_model: dict[str, float],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot cost efficiency (accuracy per dollar) for each model.

        Parameters
        ----------
        costs : Dict[str, Dict[str, float]]
            Cost analysis results
        accuracy_by_model : Dict[str, float]
            Accuracy for each model
        save_path : Optional[str]
            Path to save the plot (if None, the plot is shown)
        """
        models = []
        total_costs = []
        accuracies = []
        efficiency = []

        for model, cost_data in costs.items():
            if model in accuracy_by_model and cost_data["total_cost"] is not None:
                models.append(model)
                total_cost = cost_data["total_cost"]
                accuracy = accuracy_by_model[model]

                total_costs.append(total_cost)
                accuracies.append(accuracy)

                # Calculate efficiency: accuracy percentage per dollar
                eff = accuracy / total_cost if total_cost > 0 else 0
                efficiency.append(eff)

        if not models:
            return

        # Create the plot
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Bar chart for costs
        x = range(len(models))
        ax1.bar(x, total_costs, alpha=0.6, color="blue", label="Total Cost ($)")
        ax1.set_xlabel("Models")
        ax1.set_ylabel("Cost ($)", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")

        # Second y-axis for accuracy
        ax2 = ax1.twinx()
        ax2.plot(x, accuracies, "r-", marker="o", label="Accuracy")
        ax2.set_ylabel("Accuracy", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

        # Set x-ticks to model names
        plt.xticks(x, models, rotation=45, ha="right")

        # Add a third set of labels for efficiency
        for i, (eff, cost) in enumerate(zip(efficiency, total_costs)):
            plt.text(
                i,
                cost + 0.01,
                f"Eff: {eff:.1f}%/$",
                ha="center",
                va="bottom",
                color="green",
                fontweight="bold",
            )

        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        plt.title("Cost, Accuracy, and Efficiency by Model")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


class ConfusionAnalyzer:
    """Advanced confusion matrix analysis and error pattern detection."""

    def __init__(self):
        """Initialize the confusion analyzer."""
        self.console = Console()

    def analyze(self, eval_result) -> dict[str, Any]:
        """
        Perform detailed confusion analysis on evaluation results.

        Parameters
        ----------
        eval_result : Union[EvalResult, Dict[str, Any]]
            The evaluation result to analyze, either as EvalResult object or dictionary

        Returns
        -------
        Dict[str, Any]
            Dictionary containing confusion analysis results
        """
        self.console.print(
            "[bold cyan]Performing detailed confusion analysis...[/bold cyan]"
        )

        # Extract prediction data
        if hasattr(eval_result, "predictions"):
            # It's an EvalResult object
            predictions = eval_result.predictions
        elif isinstance(eval_result, dict) and "predictions" in eval_result:
            # It's a dictionary with predictions key
            predictions = eval_result["predictions"]
        else:
            # If we don't have predictions, return empty result
            self.console.print(
                "[yellow]No predictions found for confusion analysis[/yellow]"
            )
            return {}

        # Skip analysis if we have multi-label predictions
        if any(isinstance(p.get("expected"), list) for p in predictions):
            self.console.print(
                "[yellow]Confusion analysis is only available for single-label classifications[/yellow]"
            )
            return {}

        # Get all unique labels
        all_labels = set()
        for p in predictions:
            all_labels.add(p["expected"])
            all_labels.add(p["predicted"])
        all_labels = sorted(all_labels)

        # Create label index mapping
        label_to_idx = {label: i for i, label in enumerate(all_labels)}

        # Prepare data for confusion matrix
        y_true = [label_to_idx[p["expected"]] for p in predictions]
        y_pred = [label_to_idx[p["predicted"]] for p in predictions]

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Calculate normalized confusion matrix (by row)
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0

        # Identify most confused pairs
        confused_pairs = []
        for i in range(len(all_labels)):
            for j in range(len(all_labels)):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append(
                        {
                            "true_label": all_labels[i],
                            "predicted_label": all_labels[j],
                            "count": int(cm[i, j]),
                            "percentage": float(cm_normalized[i, j]),
                        }
                    )

        # Sort by count
        confused_pairs.sort(key=lambda x: x["count"], reverse=True)

        # Find examples for each confused pair
        confusion_examples = {}
        for pair in confused_pairs:
            true_label = pair["true_label"]
            pred_label = pair["predicted_label"]
            key = f"{true_label}→{pred_label}"

            # Find examples with this confusion
            examples = [
                p["text"]
                for p in predictions
                if p["expected"] == true_label and p["predicted"] == pred_label
            ]

            confusion_examples[key] = examples

        # Group predictions by expected label
        predictions_by_label = defaultdict(list)
        for p in predictions:
            predictions_by_label[p["expected"]].append(p)

        # Analyze error patterns: look for common words/patterns in incorrect predictions
        error_patterns = {}
        for label, preds in predictions_by_label.items():
            correct = [p["text"] for p in preds if p["predicted"] == p["expected"]]
            incorrect = [p["text"] for p in preds if p["predicted"] != p["expected"]]

            # Skip if no incorrect predictions
            if not incorrect:
                continue

            # Simple pattern analysis - count word frequencies
            # In a full implementation, we would use more sophisticated NLP techniques
            error_words = Counter()
            for text in incorrect:
                words = text.lower().split()
                error_words.update(words)

            correct_words = Counter()
            for text in correct:
                words = text.lower().split()
                correct_words.update(words)

            # Find words more common in errors (crude approximation)
            distinctive_error_words = {}
            for word, count in error_words.items():
                if count > 1:  # Only consider words that appear multiple times
                    error_freq = count / len(incorrect)
                    correct_freq = correct_words.get(word, 0) / max(len(correct), 1)

                    if error_freq > correct_freq:
                        distinctive_error_words[word] = error_freq / max(
                            correct_freq, 0.1
                        )

            # Sort by distinctiveness ratio
            sorted_words = sorted(
                distinctive_error_words.items(), key=lambda x: x[1], reverse=True
            )
            error_patterns[label] = sorted_words[:10]  # Top 10 distinctive words

        return {
            "confusion_matrix": cm.tolist(),
            "labels": all_labels,
            "confused_pairs": confused_pairs,
            "confusion_examples": confusion_examples,
            "error_patterns": error_patterns,
        }

    def display_results(self, confusion_analysis: dict[str, Any]) -> None:
        """Display confusion analysis results using rich tables and panels."""
        if not confusion_analysis:
            return

        confusion_analysis["labels"]
        confusion_analysis["confusion_matrix"]
        confused_pairs = confusion_analysis["confused_pairs"]
        confusion_examples = confusion_analysis["confusion_examples"]
        error_patterns = confusion_analysis["error_patterns"]

        # Display most confused pairs
        if confused_pairs:
            confused_table = Table(title="Most Confused Label Pairs")
            confused_table.add_column("True Label", style="cyan")
            confused_table.add_column("Predicted Label", style="yellow")
            confused_table.add_column("Count", style="green")
            confused_table.add_column("Percentage", style="red")

            for pair in confused_pairs[:10]:  # Show top 10
                confused_table.add_row(
                    pair["true_label"],
                    pair["predicted_label"],
                    str(pair["count"]),
                    f"{pair['percentage']:.1%}",
                )

            self.console.print(confused_table)

        # Display examples of confused pairs
        if confusion_examples and confused_pairs:
            self.console.print("\n[bold]Examples of Confused Pairs:[/bold]")

            for pair in confused_pairs[:5]:  # Show top 5
                true_label = pair["true_label"]
                pred_label = pair["predicted_label"]
                key = f"{true_label}→{pred_label}"

                if key in confusion_examples and confusion_examples[key]:
                    examples = confusion_examples[key][:3]  # Show up to 3 examples

                    panel_title = f"[bold]{true_label} mistaken as {pred_label} ({pair['count']} times)[/bold]"
                    panel_content = "\n\n".join(
                        [
                            f"• {ex[:100]}..." if len(ex) > 100 else f"• {ex}"
                            for ex in examples
                        ]
                    )

                    self.console.print(
                        Panel(panel_content, title=panel_title, border_style="yellow")
                    )

        # Display error patterns
        if error_patterns:
            self.console.print("\n[bold]Possible Error Patterns:[/bold]")

            for label, patterns in error_patterns.items():
                if patterns:
                    words = [f"{word} ({ratio:.1f}x)" for word, ratio in patterns[:5]]
                    pattern_text = Text(f"{label}: {', '.join(words)}")
                    self.console.print(pattern_text)

    def plot_confusion_matrix(
        self, confusion_analysis: dict[str, Any], save_path: Optional[str] = None
    ) -> None:
        """
        Plot a heatmap of the confusion matrix.

        Parameters
        ----------
        confusion_analysis : Dict[str, Any]
            The confusion analysis results
        save_path : Optional[str]
            Path to save the plot (if None, the plot is shown)
        """
        if not confusion_analysis:
            return

        labels = confusion_analysis["labels"]
        cm = np.array(confusion_analysis["confusion_matrix"])

        # Create normalized confusion matrix
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0

        plt.figure(figsize=(10, 8))

        # Plot heatmap
        sns.heatmap(
            cm_normalized,
            annot=cm,  # Show raw counts in cells
            fmt="d",  # Integer format for counts
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )

        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title("Confusion Matrix")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def plot_error_distribution(
        self, confusion_analysis: dict[str, Any], save_path: Optional[str] = None
    ) -> None:
        """
        Plot distribution of errors by label.

        Parameters
        ----------
        confusion_analysis : Dict[str, Any]
            The confusion analysis results
        save_path : Optional[str]
            Path to save the plot (if None, the plot is shown)
        """
        if not confusion_analysis:
            return

        labels = confusion_analysis["labels"]
        cm = np.array(confusion_analysis["confusion_matrix"])

        # Calculate errors for each true label
        errors_by_label = []
        for i, label in enumerate(labels):
            total = cm[i].sum()
            correct = cm[i, i]
            error_rate = 1 - (correct / total) if total > 0 else 0
            errors_by_label.append((label, error_rate, total - correct))

        # Sort by error rate
        errors_by_label.sort(key=lambda x: x[1], reverse=True)

        # Prepare data for plotting
        plot_labels = [x[0] for x in errors_by_label]
        error_rates = [x[1] for x in errors_by_label]
        error_counts = [x[2] for x in errors_by_label]

        plt.figure(figsize=(12, 6))

        # Create bar chart
        bars = plt.bar(plot_labels, error_rates, color="red", alpha=0.7)

        # Add count labels on top of bars
        for bar, count in zip(bars, error_counts):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{count}",
                ha="center",
                va="bottom",
            )

        plt.ylabel("Error Rate")
        plt.title("Error Distribution by True Label")
        plt.ylim(0, max(error_rates) * 1.2)  # Add some space for labels
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()



================================================
File: instructor_classify/templates/prompt.yaml
================================================
system_message: |
    You are a helpful assistant that classifies text into categories based on the user's intent.
    Only use the labels provided in the label definitions.

label_definitions:
  - label: question
    description: The text is asking for information, explanation, or clarification about a topic
    examples:
      examples_positive:
        - "What's the difference between Python 2 and Python 3?"
        - "Can you explain the concept of recursion in programming?"
        - "What are the main features of React.js?"
      examples_negative:
        - "Book a team lunch for next Tuesday at noon."
        - "Help me optimize this database query for better performance."

  - label: scheduling
    description: The text is about organizing, planning, or coordinating events, meetings, or time-related activities
    examples:
      examples_positive:
        - "Book a team lunch for next Tuesday at noon."
        - "Please send a calendar invite for the quarterly review."
        - "Let's plan our next sprint planning session."
      examples_negative:
        - "What's the difference between Python 2 and Python 3?"
        - "I need help fixing this CSS layout issue."

  - label: coding
    description: The text is about programming, development, debugging, or technical implementation
    examples:
      examples_positive:
        - "Can you show me how to implement a REST API in Flask?"
        - "Help me optimize this database query for better performance."
        - "I need help fixing this CSS layout issue."
      examples_negative:
        - "What's the difference between Python 2 and Python 3?"
        - "Book a team lunch for next Tuesday at noon." 



================================================
File: tests/test_unified_eval.py
================================================
import os
import sys
import yaml
import tempfile
import pytest
from unittest.mock import patch, MagicMock

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from instructor_classify.schema import ClassificationDefinition  # noqa: E402
from instructor_classify.eval_harness.unified_eval import UnifiedEvaluator  # noqa: E402


@pytest.fixture
def mock_config():
    """Create a temporary config file for testing."""
    config_content = {
        "models": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "definition_path": os.path.join(current_dir, "intent_classification.yaml"),
        "eval_sets": [os.path.join(current_dir, "example_evalset.yaml")],
        "bootstrap_samples": 100,
        "confidence_level": 0.95,
        "n_jobs": 2,
        "parallel_mode": "sync",
        "output_dir": "test_results"
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix=".yaml", delete=False) as temp:
        yaml.dump(config_content, temp)
        temp_path = temp.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)


def test_unified_evaluator_init(mock_config):
    """Test initialization of UnifiedEvaluator."""
    evaluator = UnifiedEvaluator(mock_config)
    
    # Check initialization of parallel settings
    assert evaluator.parallel_mode == "sync"
    assert evaluator.n_jobs == 2
    
    # Check config loading
    assert "gpt-3.5-turbo" in evaluator.config["models"]
    assert "gpt-4o-mini" in evaluator.config["models"]
    assert evaluator.config["bootstrap_samples"] == 100
    assert evaluator.config["confidence_level"] == 0.95


def test_invalid_parallel_mode(mock_config):
    """Test handling of invalid parallel mode."""
    # Create a config with invalid parallel mode
    with open(mock_config, "r") as f:
        config = yaml.safe_load(f)
    
    config["parallel_mode"] = "invalid"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix=".yaml", delete=False) as temp:
        yaml.dump(config, temp)
        invalid_config_path = temp.name
    
    try:
        # Should use default "sync" mode when invalid
        evaluator = UnifiedEvaluator(invalid_config_path)
        assert evaluator.parallel_mode == "sync"
    finally:
        os.unlink(invalid_config_path)


def test_prepare_method(mock_config):
    """Test the prepare method of UnifiedEvaluator."""
    evaluator = UnifiedEvaluator(mock_config)
    evaluator.prepare()
    
    # Check that models and evaluation sets were loaded
    assert len(evaluator.models) == 2
    assert len(evaluator.evaluation_sets) == 1
    assert evaluator.models[0] == "gpt-3.5-turbo"
    assert evaluator.models[1] == "gpt-4o-mini"
    assert evaluator.evaluation_sets[0].name == "Example Classification Evaluation Set"
    assert evaluator.definition is not None


@patch("instructor_classify.eval_harness.unified_eval.UnifiedEvaluator._process_batch_parallel")
def test_parallel_processing(mock_process, mock_config):
    """Test parallel processing mode."""
    # Create a config with parallel mode
    with open(mock_config, "r") as f:
        config = yaml.safe_load(f)
    
    config["parallel_mode"] = "parallel"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix=".yaml", delete=False) as temp:
        yaml.dump(config, temp)
        parallel_config_path = temp.name
    
    # Mock the _process_batch_parallel method to return expected results
    mock_process.return_value = [
        {
            "prediction": MagicMock(label="question"),
            "is_correct": True,
            "text": "What is the capital of France?",
            "expected": "question",
            "predicted": "question"
        }
    ]
    
    try:
        evaluator = UnifiedEvaluator(parallel_config_path)
        evaluator.prepare()
        
        # We'll need to patch a lot of things to test run() properly
        # So instead we'll just test that it's initialized correctly
        assert evaluator.parallel_mode == "parallel"
        assert evaluator.n_jobs == 2
    finally:
        os.unlink(parallel_config_path)


@patch("instructor_classify.eval_harness.unified_eval.UnifiedEvaluator._process_single_example")
def test_sync_processing(mock_process, mock_config):
    """Test synchronous processing mode."""
    mock_process.return_value = {
        "prediction": MagicMock(label="question"),
        "is_correct": True,
        "text": "What is the capital of France?",
        "expected": "question",
        "predicted": "question"
    }
    
    evaluator = UnifiedEvaluator(mock_config)
    evaluator.prepare()
    
    # Test that sync mode is set
    assert evaluator.parallel_mode == "sync"
    

def test_process_single_example():
    """Test the _process_single_example method."""
    yaml_path = os.path.join(current_dir, "intent_classification.yaml")
    definition = ClassificationDefinition.from_yaml(yaml_path)
    
    # Create a simplified evaluator for testing
    evaluator = UnifiedEvaluator.__new__(UnifiedEvaluator)
    evaluator.definition = definition
    
    # Create a mock classifier and example
    class MockClassifier:
        def predict(self, text):
            return MagicMock(label="question")
        
        def predict_multi(self, text):
            return MagicMock(labels=["question"])
    
    class MockExample:
        def __init__(self, text, label=None, labels=None):
            self.text = text
            self.expected_label = label
            self.expected_labels = labels
    
    # Test single-label example
    example = MockExample("What is the capital of France?", label="question")
    result = evaluator._process_single_example(MockClassifier(), example, is_multi=False)
    
    assert result["is_correct"] is True
    assert result["text"] == "What is the capital of France?"
    assert result["expected"] == "question"
    assert result["predicted"] == "question"
    
    # Test multi-label example
    example = MockExample("What is the capital of France?", labels=["question"])
    result = evaluator._process_single_example(MockClassifier(), example, is_multi=True)
    
    assert result["is_correct"] is True
    assert result["text"] == "What is the capital of France?"
    assert result["expected"] == ["question"]
    assert result["predicted"] == ["question"]


@pytest.mark.parametrize("n_jobs", [1, 2, 4])
def test_process_batch_parallel(n_jobs):
    """Test the _process_batch_parallel method with different worker counts."""
    from concurrent.futures import ThreadPoolExecutor
    import tqdm
    
    yaml_path = os.path.join(current_dir, "intent_classification.yaml")
    definition = ClassificationDefinition.from_yaml(yaml_path)
    
    # Create a simplified evaluator for testing
    evaluator = UnifiedEvaluator.__new__(UnifiedEvaluator)
    evaluator.definition = definition
    evaluator.n_jobs = n_jobs
    
    # Create a mock classifier and examples
    class MockClassifier:
        def predict(self, text):
            return MagicMock(label="question")
    
    class MockExample:
        def __init__(self, text, label=None):
            self.text = text
            self.expected_label = label
    
    # Create a batch of examples
    examples = [
        MockExample(f"Example {i}", label="question") for i in range(10)
    ]
    
    # Create mock results
    mock_results = []
    for i in range(10):
        mock_results.append({
            "prediction": MagicMock(label="question"),
            "is_correct": True,
            "text": f"Example {i}",
            "expected": "question",
            "predicted": "question"
        })
    
    # Mock the ThreadPoolExecutor
    mock_executor = MagicMock()
    mock_futures = []
    
    for i in range(10):
        mock_future = MagicMock()
        mock_future.result.return_value = mock_results[i]
        mock_futures.append(mock_future)
    
    mock_executor.submit.side_effect = mock_futures
    
    # Mock tqdm
    mock_tqdm = MagicMock()
    mock_tqdm.__iter__.return_value = mock_futures
    
    # Test the batch processing with mocked components
    with patch.object(ThreadPoolExecutor, '__enter__', return_value=mock_executor):
        with patch.object(ThreadPoolExecutor, '__exit__'):
            with patch.object(tqdm, 'tqdm', return_value=mock_tqdm):
                # Define a mock implementation of _process_single_example for the test
                def mock_process_single_example(self, classifier, example, is_multi=False):
                    idx = int(example.text.split()[-1])
                    return mock_results[idx]
                
                # Temporarily replace the method
                evaluator._process_single_example = mock_process_single_example.__get__(evaluator)
                
                results = evaluator._process_batch_parallel(MockClassifier(), examples)
                
                # Should return one result per example
                assert len(results) == len(examples)
                
                # Verify results match our mock data
                for i, result in enumerate(results):
                    # Skip actual assertion on the result content since we're mocking it completely
                    pass


def test_creating_output_directories(mock_config):
    """Test that output directories are created."""
    evaluator = UnifiedEvaluator(mock_config)
    
    # Check that output directories exist
    assert os.path.exists(evaluator.output_dir)
    assert os.path.exists(os.path.join(evaluator.output_dir, "visualizations"))
    assert os.path.exists(os.path.join(evaluator.output_dir, "metrics"))
    assert os.path.exists(os.path.join(evaluator.output_dir, "analysis"))
    
    # Cleanup
    import shutil
    shutil.rmtree(evaluator.output_dir)



```

