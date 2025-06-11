# Repository Analysis

## Summary

```
Repository: frdel/agent-zero
Files analyzed: 131

Estimated tokens: 96.9k
```

## Important Files

```
Directory structure:
└── frdel-agent-zero/
    ├── agent.py
    ├── initialize.py
    ├── models.py
    ├── preload.py
    ├── prepare.py
    ├── docker/
    │   └── run/
    │       └── fs/
    │           ├── etc/
    │           │   ├── searxng/
    │           │   └── supervisor/
    │           │       └── conf.d/
    │           ├── exe/
    │           ├── ins/
    │           └── per/
    │               └── root/
    ├── docs/
    │   └── res/
    │       ├── a0-vector-graphics/
    │       └── setup/
    │           └── settings/
    ├── instruments/
    │   ├── custom/
    │   └── default/
    │       └── yt_download/
    ├── knowledge/
    │   ├── custom/
    │   │   ├── main/
    │   │   └── solutions/
    │   └── default/
    │       ├── main/
    │       │   └── about/
    │       └── solutions/
    ├── lib/
    │   └── browser/
    │       ├── click.js
    │       ├── extract_dom.js
    │       └── init_override.js
    ├── logs/
    ├── memory/
    ├── prompts/
    │   ├── default/
    │   └── reflection/
    ├── python/
    │   ├── __init__.py
    │   ├── api/
    │   │   ├── chat_export.py
    │   │   ├── chat_load.py
    │   │   ├── chat_remove.py
    │   │   ├── chat_reset.py
    │   │   ├── ctx_window_get.py
    │   │   ├── delete_work_dir_file.py
    │   │   ├── download_work_dir_file.py
    │   │   ├── file_info.py
    │   │   ├── get_work_dir_files.py
    │   │   ├── health.py
    │   │   ├── history_get.py
    │   │   ├── image_get.py
    │   │   ├── import_knowledge.py
    │   │   ├── message.py
    │   │   ├── message_async.py
    │   │   ├── nudge.py
    │   │   ├── pause.py
    │   │   ├── poll.py
    │   │   ├── restart.py
    │   │   ├── rfc.py
    │   │   ├── scheduler_task_create.py
    │   │   ├── scheduler_task_delete.py
    │   │   ├── scheduler_task_run.py
    │   │   ├── scheduler_task_update.py
    │   │   ├── scheduler_tasks_list.py
    │   │   ├── scheduler_tick.py
    │   │   ├── settings_get.py
    │   │   ├── settings_set.py
    │   │   ├── transcribe.py
    │   │   ├── tunnel.py
    │   │   ├── tunnel_proxy.py
    │   │   ├── upload.py
    │   │   └── upload_work_dir_files.py
    │   ├── extensions/
    │   │   ├── message_loop_end/
    │   │   │   ├── _10_organize_history.py
    │   │   │   ├── _90_save_chat.py
    │   │   │   └── .gitkeep
    │   │   ├── message_loop_prompts_after/
    │   │   │   ├── _50_recall_memories.py
    │   │   │   ├── _51_recall_solutions.py
    │   │   │   ├── _60_include_current_datetime.py
    │   │   │   ├── _91_recall_wait.py
    │   │   │   └── .gitkeep
    │   │   ├── message_loop_prompts_before/
    │   │   │   ├── _90_organize_history_wait.py
    │   │   │   └── .gitkeep
    │   │   ├── message_loop_start/
    │   │   │   ├── _10_iteration_no.py
    │   │   │   └── .gitkeep
    │   │   ├── monologue_end/
    │   │   │   ├── _50_memorize_fragments.py
    │   │   │   ├── _51_memorize_solutions.py
    │   │   │   ├── _90_waiting_for_input_msg.py
    │   │   │   └── .gitkeep
    │   │   ├── monologue_start/
    │   │   │   ├── _60_rename_chat.py
    │   │   │   └── .gitkeep
    │   │   └── system_prompt/
    │   │       ├── _10_system_prompt.py
    │   │       ├── _20_behaviour_prompt.py
    │   │       └── .gitkeep
    │   ├── helpers/
    │   │   ├── api.py
    │   │   ├── attachment_manager.py
    │   │   ├── browser.py
    │   │   ├── browser_use.py
    │   │   ├── call_llm.py
    │   │   ├── cloudflare_tunnel.py
    │   │   ├── crypto.py
    │   │   ├── defer.py
    │   │   ├── dirty_json.py
    │   │   ├── docker.py
    │   │   ├── dotenv.py
    │   │   ├── duckduckgo_search.py
    │   │   ├── errors.py
    │   │   ├── extension.py
    │   │   ├── extract_tools.py
    │   │   ├── file_browser.py
    │   │   ├── files.py
    │   │   ├── git.py
    │   │   ├── history.py
    │   │   ├── images.py
    │   │   ├── job_loop.py
    │   │   ├── knowledge_import.py
    │   │   ├── localization.py
    │   │   ├── log.py
    │   │   ├── memory.py
    │   │   ├── messages.py
    │   │   ├── perplexity_search.py
    │   │   ├── persist_chat.py
    │   │   ├── print_catch.py
    │   │   ├── print_style.py
    │   │   ├── process.py
    │   │   ├── rag.py
    │   │   ├── rate_limiter.py
    │   │   ├── rfc.py
    │   │   ├── rfc_exchange.py
    │   │   ├── runtime.py
    │   │   ├── searxng.py
    │   │   ├── settings.py
    │   │   ├── shell_local.py
    │   │   ├── shell_ssh.py
    │   │   ├── strings.py
    │   │   ├── task_scheduler.py
    │   │   ├── timed_input.py
    │   │   ├── tokens.py
    │   │   ├── tool.py
    │   │   ├── tunnel_manager.py
    │   │   ├── vector_db.py
    │   │   └── whisper.py
    │   └── tools/
    │       ├── behaviour_adjustment.py
    │       ├── browser.py
    │       ├── browser_agent.py
    │       ├── browser_do.py
    │       ├── browser_open.py
    │       ├── call_subordinate.py
    │       ├── code_execution_tool.py
    │       ├── input.py
    │       ├── knowledge_tool.py
    │       ├── memory_delete.py
    │       ├── memory_forget.py
    │       ├── memory_load.py
    │       ├── memory_save.py
    │       ├── response.py
    │       ├── scheduler.py
    │       ├── search_engine.py
    │       ├── task_done.py
    │       ├── unknown.py
    │       ├── vision_load.py
    │       └── webpage_content_tool.py
    ├── tmp/
    ├── webui/
    │   ├── css/
    │   ├── js/
    │   └── public/
    └── .github/

```

## Content

```
================================================
File: agent.py
================================================
import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
import json
from typing import Any, Awaitable, Coroutine, Optional, Dict, TypedDict
import uuid
import models

from python.helpers import extract_tools, rate_limiter, files, errors, history, tokens
from python.helpers import dirty_json
from python.helpers.print_style import PrintStyle
from langchain_core.prompts import (
    ChatPromptTemplate,
)
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage

import python.helpers.log as Log
from python.helpers.dirty_json import DirtyJson
from python.helpers.defer import DeferredTask
from typing import Callable
from python.helpers.localization import Localization


class AgentContext:

    _contexts: dict[str, "AgentContext"] = {}
    _counter: int = 0

    def __init__(
        self,
        config: "AgentConfig",
        id: str | None = None,
        name: str | None = None,
        agent0: "Agent|None" = None,
        log: Log.Log | None = None,
        paused: bool = False,
        streaming_agent: "Agent|None" = None,
        created_at: datetime | None = None,
    ):
        # build context
        self.id = id or str(uuid.uuid4())
        self.name = name
        self.config = config
        self.log = log or Log.Log()
        self.agent0 = agent0 or Agent(0, self.config, self)
        self.paused = paused
        self.streaming_agent = streaming_agent
        self.task: DeferredTask | None = None
        self.created_at = created_at or datetime.now()
        AgentContext._counter += 1
        self.no = AgentContext._counter

        existing = self._contexts.get(self.id, None)
        if existing:
            AgentContext.remove(self.id)
        self._contexts[self.id] = self

    @staticmethod
    def get(id: str):
        return AgentContext._contexts.get(id, None)

    @staticmethod
    def first():
        if not AgentContext._contexts:
            return None
        return list(AgentContext._contexts.values())[0]

    @staticmethod
    def remove(id: str):
        context = AgentContext._contexts.pop(id, None)
        if context and context.task:
            context.task.kill()
        return context

    def serialize(self):
        return {
            "id": self.id,
            "name": self.name,
            "created_at": (
                Localization.get().serialize_datetime(self.created_at)
                if self.created_at else Localization.get().serialize_datetime(datetime.fromtimestamp(0))
            ),
            "no": self.no,
            "log_guid": self.log.guid,
            "log_version": len(self.log.updates),
            "log_length": len(self.log.logs),
            "paused": self.paused,
        }

    def get_created_at(self):
        return self.created_at

    def kill_process(self):
        if self.task:
            self.task.kill()

    def reset(self):
        self.kill_process()
        self.log.reset()
        self.agent0 = Agent(0, self.config, self)
        self.streaming_agent = None
        self.paused = False

    def nudge(self):
        self.kill_process()
        self.paused = False
        if self.streaming_agent:
            current_agent = self.streaming_agent
        else:
            current_agent = self.agent0

        self.task = self.run_task(current_agent.monologue)
        return self.task

    def communicate(self, msg: "UserMessage", broadcast_level: int = 1):
        self.paused = False  # unpause if paused

        if self.streaming_agent:
            current_agent = self.streaming_agent
        else:
            current_agent = self.agent0

        if self.task and self.task.is_alive():
            # set intervention messages to agent(s):
            intervention_agent = current_agent
            while intervention_agent and broadcast_level != 0:
                intervention_agent.intervention = msg
                broadcast_level -= 1
                intervention_agent = intervention_agent.data.get(
                    Agent.DATA_NAME_SUPERIOR, None
                )
        else:
            self.task = self.run_task(self._process_chain, current_agent, msg)

        return self.task

    def run_task(
        self, func: Callable[..., Coroutine[Any, Any, Any]], *args: Any, **kwargs: Any
    ):
        if not self.task:
            self.task = DeferredTask(
                thread_name=self.__class__.__name__,
            )
        self.task.start_task(func, *args, **kwargs)
        return self.task

    # this wrapper ensures that superior agents are called back if the chat was loaded from file and original callstack is gone
    async def _process_chain(self, agent: "Agent", msg: "UserMessage|str", user=True):
        try:
            msg_template = (
                agent.hist_add_user_message(msg)  # type: ignore
                if user
                else agent.hist_add_tool_result(
                    tool_name="call_subordinate", tool_result=msg  # type: ignore
                )
            )
            response = await agent.monologue()  # type: ignore
            superior = agent.data.get(Agent.DATA_NAME_SUPERIOR, None)
            if superior:
                response = await self._process_chain(superior, response, False)  # type: ignore
            return response
        except Exception as e:
            agent.handle_critical_exception(e)


@dataclass
class ModelConfig:
    provider: models.ModelProvider
    name: str
    ctx_length: int = 0
    limit_requests: int = 0
    limit_input: int = 0
    limit_output: int = 0
    vision: bool = False
    kwargs: dict = field(default_factory=dict)


@dataclass
class AgentConfig:
    chat_model: ModelConfig
    utility_model: ModelConfig
    embeddings_model: ModelConfig
    browser_model: ModelConfig
    prompts_subdir: str = ""
    memory_subdir: str = ""
    knowledge_subdirs: list[str] = field(default_factory=lambda: ["default", "custom"])
    code_exec_docker_enabled: bool = False
    code_exec_docker_name: str = "A0-dev"
    code_exec_docker_image: str = "frdel/agent-zero-run:development"
    code_exec_docker_ports: dict[str, int] = field(
        default_factory=lambda: {"22/tcp": 55022, "80/tcp": 55080}
    )
    code_exec_docker_volumes: dict[str, dict[str, str]] = field(
        default_factory=lambda: {
            files.get_base_dir(): {"bind": "/a0", "mode": "rw"},
            files.get_abs_path("work_dir"): {"bind": "/root", "mode": "rw"},
        }
    )
    code_exec_ssh_enabled: bool = True
    code_exec_ssh_addr: str = "localhost"
    code_exec_ssh_port: int = 55022
    code_exec_ssh_user: str = "root"
    code_exec_ssh_pass: str = ""
    additional: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserMessage:
    message: str
    attachments: list[str] = field(default_factory=list[str])
    system_message: list[str] = field(default_factory=list[str])


class LoopData:
    def __init__(self, **kwargs):
        self.iteration = -1
        self.system = []
        self.user_message: history.Message | None = None
        self.history_output: list[history.OutputMessage] = []
        self.extras_temporary: OrderedDict[str, history.MessageContent] = OrderedDict()
        self.extras_persistent: OrderedDict[str, history.MessageContent] = OrderedDict()
        self.last_response = ""

        # override values with kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


# intervention exception class - skips rest of message loop iteration
class InterventionException(Exception):
    pass


# killer exception class - not forwarded to LLM, cannot be fixed on its own, ends message loop
class RepairableException(Exception):
    pass


class HandledException(Exception):
    pass


class Agent:

    DATA_NAME_SUPERIOR = "_superior"
    DATA_NAME_SUBORDINATE = "_subordinate"
    DATA_NAME_CTX_WINDOW = "ctx_window"

    def __init__(
        self, number: int, config: AgentConfig, context: AgentContext | None = None
    ):

        # agent config
        self.config = config

        # agent context
        self.context = context or AgentContext(config)

        # non-config vars
        self.number = number
        self.agent_name = f"Agent {self.number}"

        self.history = history.History(self)
        self.last_user_message: history.Message | None = None
        self.intervention: UserMessage | None = None
        self.data = {}  # free data object all the tools can use

    async def monologue(self):
        while True:
            try:
                # loop data dictionary to pass to extensions
                self.loop_data = LoopData(user_message=self.last_user_message)
                # call monologue_start extensions
                await self.call_extensions("monologue_start", loop_data=self.loop_data)

                printer = PrintStyle(italic=True, font_color="#b3ffd9", padding=False)

                # let the agent run message loop until he stops it with a response tool
                while True:

                    self.context.streaming_agent = self  # mark self as current streamer
                    self.loop_data.iteration += 1

                    # call message_loop_start extensions
                    await self.call_extensions("message_loop_start", loop_data=self.loop_data)

                    try:
                        # prepare LLM chain (model, system, history)
                        prompt = await self.prepare_prompt(loop_data=self.loop_data)

                        # output that the agent is starting
                        PrintStyle(
                            bold=True,
                            font_color="green",
                            padding=True,
                            background_color="white",
                        ).print(f"{self.agent_name}: Generating")
                        log = self.context.log.log(
                            type="agent", heading=f"{self.agent_name}: Generating"
                        )

                        async def stream_callback(chunk: str, full: str):
                            # output the agent response stream
                            if chunk:
                                printer.stream(chunk)
                                self.log_from_stream(full, log)

                        agent_response = await self.call_chat_model(
                            prompt, callback=stream_callback
                        )  # type: ignore

                        await self.handle_intervention(agent_response)

                        if (
                            self.loop_data.last_response == agent_response
                        ):  # if assistant_response is the same as last message in history, let him know
                            # Append the assistant's response to the history
                            self.hist_add_ai_response(agent_response)
                            # Append warning message to the history
                            warning_msg = self.read_prompt("fw.msg_repeat.md")
                            self.hist_add_warning(message=warning_msg)
                            PrintStyle(font_color="orange", padding=True).print(
                                warning_msg
                            )
                            self.context.log.log(type="warning", content=warning_msg)

                        else:  # otherwise proceed with tool
                            # Append the assistant's response to the history
                            self.hist_add_ai_response(agent_response)
                            # process tools requested in agent message
                            tools_result = await self.process_tools(agent_response)
                            if tools_result:  # final response of message loop available
                                return tools_result  # break the execution if the task is done

                    # exceptions inside message loop:
                    except InterventionException as e:
                        pass  # intervention message has been handled in handle_intervention(), proceed with conversation loop
                    except RepairableException as e:
                        # Forward repairable errors to the LLM, maybe it can fix them
                        error_message = errors.format_error(e)
                        self.hist_add_warning(error_message)
                        PrintStyle(font_color="red", padding=True).print(error_message)
                        self.context.log.log(type="error", content=error_message)
                    except Exception as e:
                        # Other exception kill the loop
                        self.handle_critical_exception(e)

                    finally:
                        # call message_loop_end extensions
                        await self.call_extensions(
                            "message_loop_end", loop_data=self.loop_data
                        )

            # exceptions outside message loop:
            except InterventionException as e:
                pass  # just start over
            except Exception as e:
                self.handle_critical_exception(e)
            finally:
                self.context.streaming_agent = None  # unset current streamer
                # call monologue_end extensions
                await self.call_extensions("monologue_end", loop_data=self.loop_data)  # type: ignore

    async def prepare_prompt(self, loop_data: LoopData) -> ChatPromptTemplate:
        # call extensions before setting prompts
        await self.call_extensions("message_loop_prompts_before", loop_data=loop_data)

        # set system prompt and message history
        loop_data.system = await self.get_system_prompt(self.loop_data)
        loop_data.history_output = self.history.output()

        # and allow extensions to edit them
        await self.call_extensions("message_loop_prompts_after", loop_data=loop_data)

        # extras (memory etc.)
        # extras: list[history.OutputMessage] = []
        # for extra in loop_data.extras_persistent.values():
        #     extras += history.Message(False, content=extra).output()
        # for extra in loop_data.extras_temporary.values():
        #     extras += history.Message(False, content=extra).output()
        extras = history.Message(
            False, 
            content=self.read_prompt("agent.context.extras.md", extras=dirty_json.stringify(
                {**loop_data.extras_persistent, **loop_data.extras_temporary}
                ))).output()
        loop_data.extras_temporary.clear()

        # convert history + extras to LLM format
        history_langchain: list[BaseMessage] = history.output_langchain(
            loop_data.history_output + extras
        )

        # build chain from system prompt, message history and model
        system_text = "\n\n".join(loop_data.system)
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_text),
                *history_langchain,
                # AIMessage(content="JSON:"), # force the LLM to start with json
            ]
        )

        # store as last context window content
        self.set_data(
            Agent.DATA_NAME_CTX_WINDOW,
            {
                "text": prompt.format(),
                "tokens": self.history.get_tokens()
                + tokens.approximate_tokens(system_text)
                + tokens.approximate_tokens(history.output_text(extras)),
            },
        )

        return prompt

    def handle_critical_exception(self, exception: Exception):
        if isinstance(exception, HandledException):
            raise exception  # Re-raise the exception to kill the loop
        elif isinstance(exception, asyncio.CancelledError):
            # Handling for asyncio.CancelledError
            PrintStyle(font_color="white", background_color="red", padding=True).print(
                f"Context {self.context.id} terminated during message loop"
            )
            raise HandledException(
                exception
            )  # Re-raise the exception to cancel the loop
        else:
            # Handling for general exceptions
            error_text = errors.error_text(exception)
            error_message = errors.format_error(exception)
            PrintStyle(font_color="red", padding=True).print(error_message)
            self.context.log.log(
                type="error",
                heading="Error",
                content=error_message,
                kvps={"text": error_text},
            )
            raise HandledException(exception)  # Re-raise the exception to kill the loop

    async def get_system_prompt(self, loop_data: LoopData) -> list[str]:
        system_prompt = []
        await self.call_extensions(
            "system_prompt", system_prompt=system_prompt, loop_data=loop_data
        )
        return system_prompt

    def parse_prompt(self, file: str, **kwargs):
        prompt_dir = files.get_abs_path("prompts/default")
        backup_dir = []
        if (
            self.config.prompts_subdir
        ):  # if agent has custom folder, use it and use default as backup
            prompt_dir = files.get_abs_path("prompts", self.config.prompts_subdir)
            backup_dir.append(files.get_abs_path("prompts/default"))
        prompt = files.parse_file(
            files.get_abs_path(prompt_dir, file), _backup_dirs=backup_dir, **kwargs
        )
        return prompt

    def read_prompt(self, file: str, **kwargs) -> str:
        prompt_dir = files.get_abs_path("prompts/default")
        backup_dir = []
        if (
            self.config.prompts_subdir
        ):  # if agent has custom folder, use it and use default as backup
            prompt_dir = files.get_abs_path("prompts", self.config.prompts_subdir)
            backup_dir.append(files.get_abs_path("prompts/default"))
        prompt = files.read_file(
            files.get_abs_path(prompt_dir, file), _backup_dirs=backup_dir, **kwargs
        )
        prompt = files.remove_code_fences(prompt)
        return prompt

    def get_data(self, field: str):
        return self.data.get(field, None)

    def set_data(self, field: str, value):
        self.data[field] = value

    def hist_add_message(
        self, ai: bool, content: history.MessageContent, tokens: int = 0
    ):
        return self.history.add_message(ai=ai, content=content, tokens=tokens)

    def hist_add_user_message(self, message: UserMessage, intervention: bool = False):
        self.history.new_topic()  # user message starts a new topic in history

        # load message template based on intervention
        if intervention:
            content = self.parse_prompt(
                "fw.intervention.md",
                message=message.message,
                attachments=message.attachments,
                system_message=message.system_message
            )
        else:
            content = self.parse_prompt(
                "fw.user_message.md",
                message=message.message,
                attachments=message.attachments,
                system_message=message.system_message
            )

        # remove empty parts from template
        if isinstance(content, dict):
            content = {k: v for k, v in content.items() if v}

        # add to history
        msg = self.hist_add_message(False, content=content)  # type: ignore
        self.last_user_message = msg
        return msg

    def hist_add_ai_response(self, message: str):
        self.loop_data.last_response = message
        content = self.parse_prompt("fw.ai_response.md", message=message)
        return self.hist_add_message(True, content=content)

    def hist_add_warning(self, message: history.MessageContent):
        content = self.parse_prompt("fw.warning.md", message=message)
        return self.hist_add_message(False, content=content)

    def hist_add_tool_result(self, tool_name: str, tool_result: str):
        content = self.parse_prompt(
            "fw.tool_result.md", tool_name=tool_name, tool_result=tool_result
        )
        return self.hist_add_message(False, content=content)

    def concat_messages(
        self, messages
    ):  # TODO add param for message range, topic, history
        return self.history.output_text(human_label="user", ai_label="assistant")

    def get_chat_model(self):
        return models.get_model(
            models.ModelType.CHAT,
            self.config.chat_model.provider,
            self.config.chat_model.name,
            **self.config.chat_model.kwargs,
        )

    def get_utility_model(self):
        return models.get_model(
            models.ModelType.CHAT,
            self.config.utility_model.provider,
            self.config.utility_model.name,
            **self.config.utility_model.kwargs,
        )

    def get_embedding_model(self):
        return models.get_model(
            models.ModelType.EMBEDDING,
            self.config.embeddings_model.provider,
            self.config.embeddings_model.name,
            **self.config.embeddings_model.kwargs,
        )

    async def call_utility_model(
        self,
        system: str,
        message: str,
        callback: Callable[[str], Awaitable[None]] | None = None,
        background: bool = False,
    ):
        prompt = ChatPromptTemplate.from_messages(
            [SystemMessage(content=system), HumanMessage(content=message)]
        )

        response = ""

        # model class
        model = self.get_utility_model()

        # rate limiter
        limiter = await self.rate_limiter(
            self.config.utility_model, prompt.format(), background
        )

        async for chunk in (prompt | model).astream({}):
            await self.handle_intervention()  # wait for intervention and handle it, if paused

            content = models.parse_chunk(chunk)
            limiter.add(output=tokens.approximate_tokens(content))
            response += content

            if callback:
                await callback(content)

        return response

    async def call_chat_model(
        self,
        prompt: ChatPromptTemplate,
        callback: Callable[[str, str], Awaitable[None]] | None = None,
    ):
        response = ""

        # model class
        model = self.get_chat_model()

        # rate limiter
        limiter = await self.rate_limiter(self.config.chat_model, prompt.format())

        async for chunk in (prompt | model).astream({}):
            await self.handle_intervention()  # wait for intervention and handle it, if paused

            content = models.parse_chunk(chunk)
            limiter.add(output=tokens.approximate_tokens(content))
            response += content

            if callback:
                await callback(content, response)

        return response

    async def rate_limiter(
        self, model_config: ModelConfig, input: str, background: bool = False
    ):
        # rate limiter log
        wait_log = None

        async def wait_callback(msg: str, key: str, total: int, limit: int):
            nonlocal wait_log
            if not wait_log:
                wait_log = self.context.log.log(
                    type="util",
                    update_progress="none",
                    heading=msg,
                    model=f"{model_config.provider.value}\\{model_config.name}",
                )
            wait_log.update(heading=msg, key=key, value=total, limit=limit)
            if not background:
                self.context.log.set_progress(msg, -1)

        # rate limiter
        limiter = models.get_rate_limiter(
            model_config.provider,
            model_config.name,
            model_config.limit_requests,
            model_config.limit_input,
            model_config.limit_output,
        )
        limiter.add(input=tokens.approximate_tokens(input))
        limiter.add(requests=1)
        await limiter.wait(callback=wait_callback)
        return limiter

    async def handle_intervention(self, progress: str = ""):
        while self.context.paused:
            await asyncio.sleep(0.1)  # wait if paused
        if (
            self.intervention
        ):  # if there is an intervention message, but not yet processed
            msg = self.intervention
            self.intervention = None  # reset the intervention message
            if progress.strip():
                self.hist_add_ai_response(progress)
            # append the intervention message
            self.hist_add_user_message(msg, intervention=True)
            raise InterventionException(msg)

    async def wait_if_paused(self):
        while self.context.paused:
            await asyncio.sleep(0.1)

    async def process_tools(self, msg: str):
        # search for tool usage requests in agent message
        tool_request = extract_tools.json_parse_dirty(msg)

        if tool_request is not None:
            tool_name = tool_request.get("tool_name", "")
            tool_method = None
            tool_args = tool_request.get("tool_args", {})

            if ":" in tool_name:
                tool_name, tool_method = tool_name.split(":", 1)

            tool = self.get_tool(name=tool_name, method=tool_method, args=tool_args, message=msg)

            await self.handle_intervention()  # wait if paused and handle intervention message if needed
            await tool.before_execution(**tool_args)
            await self.handle_intervention()  # wait if paused and handle intervention message if needed
            response = await tool.execute(**tool_args)
            await self.handle_intervention()  # wait if paused and handle intervention message if needed
            await tool.after_execution(response)
            await self.handle_intervention()  # wait if paused and handle intervention message if needed
            if response.break_loop:
                return response.message
        else:
            msg = self.read_prompt("fw.msg_misformat.md")
            self.hist_add_warning(msg)
            PrintStyle(font_color="red", padding=True).print(msg)
            self.context.log.log(
                type="error", content=f"{self.agent_name}: Message misformat"
            )

    def log_from_stream(self, stream: str, logItem: Log.LogItem):
        try:
            if len(stream) < 25:
                return  # no reason to try
            response = DirtyJson.parse_string(stream)
            if isinstance(response, dict):
                # log if result is a dictionary already
                logItem.update(content=stream, kvps=response)
        except Exception as e:
            pass

    def get_tool(self, name: str, method: str | None, args: dict, message: str, **kwargs):
        from python.tools.unknown import Unknown
        from python.helpers.tool import Tool

        classes = extract_tools.load_classes_from_folder(
            "python/tools", name + ".py", Tool
        )
        tool_class = classes[0] if classes else Unknown
        return tool_class(agent=self, name=name, method=method, args=args, message=message, **kwargs)

    async def call_extensions(self, folder: str, **kwargs) -> Any:
        from python.helpers.extension import Extension

        classes = extract_tools.load_classes_from_folder(
            "python/extensions/" + folder, "*", Extension
        )
        for cls in classes:
            await cls(agent=self).execute(**kwargs)



================================================
File: initialize.py
================================================
import asyncio
import models
from agent import AgentConfig, ModelConfig
from python.helpers import dotenv, files, rfc_exchange, runtime, settings, docker, log


def initialize():

    current_settings = settings.get_settings()

    # chat model from user settings
    chat_llm = ModelConfig(
        provider=models.ModelProvider[current_settings["chat_model_provider"]],
        name=current_settings["chat_model_name"],
        ctx_length=current_settings["chat_model_ctx_length"],
        vision=current_settings["chat_model_vision"],
        limit_requests=current_settings["chat_model_rl_requests"],
        limit_input=current_settings["chat_model_rl_input"],
        limit_output=current_settings["chat_model_rl_output"],
        kwargs=current_settings["chat_model_kwargs"],
    )

    # utility model from user settings
    utility_llm = ModelConfig(
        provider=models.ModelProvider[current_settings["util_model_provider"]],
        name=current_settings["util_model_name"],
        ctx_length=current_settings["util_model_ctx_length"],
        limit_requests=current_settings["util_model_rl_requests"],
        limit_input=current_settings["util_model_rl_input"],
        limit_output=current_settings["util_model_rl_output"],
        kwargs=current_settings["util_model_kwargs"],
    )
    # embedding model from user settings
    embedding_llm = ModelConfig(
        provider=models.ModelProvider[current_settings["embed_model_provider"]],
        name=current_settings["embed_model_name"],
        limit_requests=current_settings["embed_model_rl_requests"],
        kwargs=current_settings["embed_model_kwargs"],
    )
    # browser model from user settings
    browser_llm = ModelConfig(
        provider=models.ModelProvider[current_settings["browser_model_provider"]],
        name=current_settings["browser_model_name"],
        vision=current_settings["browser_model_vision"],
        kwargs=current_settings["browser_model_kwargs"],
    )
    # agent configuration
    config = AgentConfig(
        chat_model=chat_llm,
        utility_model=utility_llm,
        embeddings_model=embedding_llm,
        browser_model=browser_llm,
        prompts_subdir=current_settings["agent_prompts_subdir"],
        memory_subdir=current_settings["agent_memory_subdir"],
        knowledge_subdirs=["default", current_settings["agent_knowledge_subdir"]],
        code_exec_docker_enabled=False,
        # code_exec_docker_name = "A0-dev",
        # code_exec_docker_image = "frdel/agent-zero-run:development",
        # code_exec_docker_ports = { "22/tcp": 55022, "80/tcp": 55080 }
        # code_exec_docker_volumes = {
        # files.get_base_dir(): {"bind": "/a0", "mode": "rw"},
        # files.get_abs_path("work_dir"): {"bind": "/root", "mode": "rw"},
        # },
        # code_exec_ssh_enabled = True,
        # code_exec_ssh_addr = "localhost",
        # code_exec_ssh_port = 55022,
        # code_exec_ssh_user = "root",
        # code_exec_ssh_pass = "",
        # additional = {},
    )

    # update SSH and docker settings
    set_runtime_config(config, current_settings)

    # update config with runtime args
    args_override(config)

    # return config object
    return config


def args_override(config):
    # update config with runtime args
    for key, value in runtime.args.items():
        if hasattr(config, key):
            # conversion based on type of config[key]
            if isinstance(getattr(config, key), bool):
                value = value.lower().strip() == "true"
            elif isinstance(getattr(config, key), int):
                value = int(value)
            elif isinstance(getattr(config, key), float):
                value = float(value)
            elif isinstance(getattr(config, key), str):
                value = str(value)
            else:
                raise Exception(
                    f"Unsupported argument type of '{key}': {type(getattr(config, key))}"
                )

            setattr(config, key, value)


def set_runtime_config(config: AgentConfig, set: settings.Settings):
    ssh_conf = settings.get_runtime_config(set)
    for key, value in ssh_conf.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # if config.code_exec_docker_enabled:
    #     config.code_exec_docker_ports["22/tcp"] = ssh_conf["code_exec_ssh_port"]
    #     config.code_exec_docker_ports["80/tcp"] = ssh_conf["code_exec_http_port"]
    #     config.code_exec_docker_name = f"{config.code_exec_docker_name}-{ssh_conf['code_exec_ssh_port']}-{ssh_conf['code_exec_http_port']}"

    #     dman = docker.DockerContainerManager(
    #         logger=log.Log(),
    #         name=config.code_exec_docker_name,
    #         image=config.code_exec_docker_image,
    #         ports=config.code_exec_docker_ports,
    #         volumes=config.code_exec_docker_volumes,
    #     )
    #     dman.start_container()

    # config.code_exec_ssh_pass = asyncio.run(rfc_exchange.get_root_password())



================================================
File: models.py
================================================
from enum import Enum
import os
from typing import Any
from langchain_openai import (
    ChatOpenAI,
    OpenAI,
    OpenAIEmbeddings,
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
    AzureOpenAI,
)
from langchain_community.llms.ollama import Ollama
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    ChatHuggingFace,
    HuggingFaceEndpoint,
)
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
    embeddings as google_embeddings,
)
from langchain_mistralai import ChatMistralAI

# from pydantic.v1.types import SecretStr
from python.helpers import dotenv, runtime
from python.helpers.dotenv import load_dotenv
from python.helpers.rate_limiter import RateLimiter

# environment variables
load_dotenv()


class ModelType(Enum):
    CHAT = "Chat"
    EMBEDDING = "Embedding"


class ModelProvider(Enum):
    ANTHROPIC = "Anthropic"
    CHUTES = "Chutes"
    DEEPSEEK = "DeepSeek"
    GOOGLE = "Google"
    GROQ = "Groq"
    HUGGINGFACE = "HuggingFace"
    LMSTUDIO = "LM Studio"
    MISTRALAI = "Mistral AI"
    OLLAMA = "Ollama"
    OPENAI = "OpenAI"
    OPENAI_AZURE = "OpenAI Azure"
    OPENROUTER = "OpenRouter"
    SAMBANOVA = "Sambanova"
    OTHER = "Other"


rate_limiters: dict[str, RateLimiter] = {}


# Utility function to get API keys from environment variables
def get_api_key(service):
    return (
        dotenv.get_dotenv_value(f"API_KEY_{service.upper()}")
        or dotenv.get_dotenv_value(f"{service.upper()}_API_KEY")
        or dotenv.get_dotenv_value(
            f"{service.upper()}_API_TOKEN"
        )  # Added for CHUTES_API_TOKEN
        or "None"
    )


def get_model(type: ModelType, provider: ModelProvider, name: str, **kwargs):
    fnc_name = f"get_{provider.name.lower()}_{type.name.lower()}"  # function name of model getter
    model = globals()[fnc_name](name, **kwargs)  # call function by name
    return model


def get_rate_limiter(
    provider: ModelProvider, name: str, requests: int, input: int, output: int
) -> RateLimiter:
    # get or create
    key = f"{provider.name}\\{name}"
    rate_limiters[key] = limiter = rate_limiters.get(key, RateLimiter(seconds=60))
    # always update
    limiter.limits["requests"] = requests or 0
    limiter.limits["input"] = input or 0
    limiter.limits["output"] = output or 0
    return limiter


def parse_chunk(chunk: Any):
    if isinstance(chunk, str):
        content = chunk
    elif hasattr(chunk, "content"):
        content = str(chunk.content)
    else:
        content = str(chunk)
    return content


# Ollama models
def get_ollama_base_url():
    return (
        dotenv.get_dotenv_value("OLLAMA_BASE_URL")
        or f"http://{runtime.get_local_url()}:11434"
    )


def get_ollama_chat(
    model_name: str,
    base_url=None,
    num_ctx=8192,
    **kwargs,
):
    if not base_url:
        base_url = get_ollama_base_url()
    return ChatOllama(
        model=model_name,
        base_url=base_url,
        num_ctx=num_ctx,
        **kwargs,
    )


def get_ollama_embedding(
    model_name: str,
    base_url=None,
    num_ctx=8192,
    **kwargs,
):
    if not base_url:
        base_url = get_ollama_base_url()
    return OllamaEmbeddings(
        model=model_name, base_url=base_url, num_ctx=num_ctx, **kwargs
    )


# HuggingFace models
def get_huggingface_chat(
    model_name: str,
    api_key=None,
    **kwargs,
):
    # different naming convention here
    if not api_key:
        api_key = get_api_key("huggingface") or os.environ["HUGGINGFACEHUB_API_TOKEN"]

    # Initialize the HuggingFaceEndpoint with the specified model and parameters
    llm = HuggingFaceEndpoint(
        repo_id=model_name,
        task="text-generation",
        do_sample=True,
        **kwargs,
    )

    # Initialize the ChatHuggingFace with the configured llm
    return ChatHuggingFace(llm=llm)


def get_huggingface_embedding(model_name: str, **kwargs):
    return HuggingFaceEmbeddings(model_name=model_name, **kwargs)


# LM Studio and other OpenAI compatible interfaces
def get_lmstudio_base_url():
    return (
        dotenv.get_dotenv_value("LM_STUDIO_BASE_URL")
        or f"http://{runtime.get_local_url()}:1234/v1"
    )


def get_lmstudio_chat(
    model_name: str,
    base_url=None,
    **kwargs,
):
    if not base_url:
        base_url = get_lmstudio_base_url()
    return ChatOpenAI(model_name=model_name, base_url=base_url, api_key="none", **kwargs)  # type: ignore


def get_lmstudio_embedding(
    model_name: str,
    base_url=None,
    **kwargs,
):
    if not base_url:
        base_url = get_lmstudio_base_url()
    return OpenAIEmbeddings(model=model_name, api_key="none", base_url=base_url, check_embedding_ctx_length=False, **kwargs)  # type: ignore


# Anthropic models
def get_anthropic_chat(
    model_name: str,
    api_key=None,
    base_url=None,
    **kwargs,
):
    if not api_key:
        api_key = get_api_key("anthropic")
    if not base_url:
        base_url = (
            dotenv.get_dotenv_value("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
        )
    return ChatAnthropic(model_name=model_name, api_key=api_key, base_url=base_url, **kwargs)  # type: ignore


# right now anthropic does not have embedding models, but that might change
def get_anthropic_embedding(
    model_name: str,
    api_key=None,
    **kwargs,
):
    if not api_key:
        api_key = get_api_key("anthropic")
    return OpenAIEmbeddings(model=model_name, api_key=api_key, **kwargs)  # type: ignore


# OpenAI models
def get_openai_chat(
    model_name: str,
    api_key=None,
    **kwargs,
):
    if not api_key:
        api_key = get_api_key("openai")
    return ChatOpenAI(model_name=model_name, api_key=api_key, **kwargs)  # type: ignore


def get_openai_embedding(model_name: str, api_key=None, **kwargs):
    if not api_key:
        api_key = get_api_key("openai")
    return OpenAIEmbeddings(model=model_name, api_key=api_key, **kwargs)  # type: ignore


def get_openai_azure_chat(
    deployment_name: str,
    api_key=None,
    azure_endpoint=None,
    **kwargs,
):
    if not api_key:
        api_key = get_api_key("openai_azure")
    if not azure_endpoint:
        azure_endpoint = dotenv.get_dotenv_value("OPENAI_AZURE_ENDPOINT")
    return AzureChatOpenAI(deployment_name=deployment_name, api_key=api_key, azure_endpoint=azure_endpoint, **kwargs)  # type: ignore


def get_openai_azure_embedding(
    deployment_name: str,
    api_key=None,
    azure_endpoint=None,
    **kwargs,
):
    if not api_key:
        api_key = get_api_key("openai_azure")
    if not azure_endpoint:
        azure_endpoint = dotenv.get_dotenv_value("OPENAI_AZURE_ENDPOINT")
    return AzureOpenAIEmbeddings(deployment_name=deployment_name, api_key=api_key, azure_endpoint=azure_endpoint, **kwargs)  # type: ignore


# Google models
def get_google_chat(
    model_name: str,
    api_key=None,
    **kwargs,
):
    if not api_key:
        api_key = get_api_key("google")
    return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}, **kwargs)  # type: ignore


def get_google_embedding(
    model_name: str,
    api_key=None,
    **kwargs,
):
    if not api_key:
        api_key = get_api_key("google")
    return google_embeddings.GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key, **kwargs)  # type: ignore


# Mistral models
def get_mistralai_chat(
    model_name: str,
    api_key=None,
    **kwargs,
):
    if not api_key:
        api_key = get_api_key("mistral")
    return ChatMistralAI(model=model_name, api_key=api_key, **kwargs)  # type: ignore


# Groq models
def get_groq_chat(
    model_name: str,
    api_key=None,
    **kwargs,
):
    if not api_key:
        api_key = get_api_key("groq")
    return ChatGroq(model_name=model_name, api_key=api_key, **kwargs)  # type: ignore


# DeepSeek models
def get_deepseek_chat(
    model_name: str,
    api_key=None,
    base_url=None,
    **kwargs,
):
    if not api_key:
        api_key = get_api_key("deepseek")
    if not base_url:
        base_url = (
            dotenv.get_dotenv_value("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
        )

    return ChatOpenAI(api_key=api_key, model=model_name, base_url=base_url, **kwargs)  # type: ignore


# OpenRouter models
def get_openrouter_chat(
    model_name: str,
    api_key=None,
    base_url=None,
    **kwargs,
):
    if not api_key:
        api_key = get_api_key("openrouter")
    if not base_url:
        base_url = (
            dotenv.get_dotenv_value("OPEN_ROUTER_BASE_URL")
            or "https://openrouter.ai/api/v1"
        )
    return ChatOpenAI(
        api_key=api_key, # type: ignore
        model=model_name,
        base_url=base_url,
        stream_usage=True,
        model_kwargs={
            "extra_headers": {
                "HTTP-Referer": "https://agent-zero.ai",
                "X-Title": "Agent Zero",
            }
        },
        **kwargs,
    )


def get_openrouter_embedding(
    model_name: str,
    api_key=None,
    base_url=None,
    **kwargs,
):
    if not api_key:
        api_key = get_api_key("openrouter")
    if not base_url:
        base_url = (
            dotenv.get_dotenv_value("OPEN_ROUTER_BASE_URL")
            or "https://openrouter.ai/api/v1"
        )
    return OpenAIEmbeddings(model=model_name, api_key=api_key, base_url=base_url, **kwargs)  # type: ignore


# Sambanova models
def get_sambanova_chat(
    model_name: str,
    api_key=None,
    base_url=None,
    max_tokens=1024,
    **kwargs,
):
    if not api_key:
        api_key = get_api_key("sambanova")
    if not base_url:
        base_url = (
            dotenv.get_dotenv_value("SAMBANOVA_BASE_URL")
            or "https://fast-api.snova.ai/v1"
        )
    return ChatOpenAI(api_key=api_key, model=model_name, base_url=base_url, max_tokens=max_tokens, **kwargs)  # type: ignore


# right now sambanova does not have embedding models, but that might change
def get_sambanova_embedding(
    model_name: str,
    api_key=None,
    base_url=None,
    **kwargs,
):
    if not api_key:
        api_key = get_api_key("sambanova")
    if not base_url:
        base_url = (
            dotenv.get_dotenv_value("SAMBANOVA_BASE_URL")
            or "https://fast-api.snova.ai/v1"
        )
    return OpenAIEmbeddings(model=model_name, api_key=api_key, base_url=base_url, **kwargs)  # type: ignore


# Other OpenAI compatible models
def get_other_chat(
    model_name: str,
    api_key=None,
    base_url=None,
    **kwargs,
):
    return ChatOpenAI(api_key=api_key, model=model_name, base_url=base_url, **kwargs)  # type: ignore


def get_other_embedding(model_name: str, api_key=None, base_url=None, **kwargs):
    return OpenAIEmbeddings(model=model_name, api_key=api_key, base_url=base_url, **kwargs)  # type: ignore


# Chutes models
def get_chutes_chat(
    model_name: str,
    api_key=None,
    base_url=None,
    **kwargs,
):
    if not api_key:
        api_key = get_api_key("chutes")
    if not base_url:
        base_url = (
            dotenv.get_dotenv_value("CHUTES_BASE_URL") or "https://llm.chutes.ai/v1"
        )
    return ChatOpenAI(api_key=api_key, model=model_name, base_url=base_url, **kwargs)  # type: ignore



================================================
File: preload.py
================================================
import asyncio
from python.helpers import runtime, whisper, settings
from python.helpers.print_style import PrintStyle
import models

PrintStyle().print("Running preload...")
runtime.initialize()


async def preload():
    try:
        set = settings.get_default_settings()

        # preload whisper model
        async def preload_whisper():
            try:
                return await whisper.preload(set["stt_model_size"])
            except Exception as e:
                PrintStyle().error(f"Error in preload_whisper: {e}")

        # preload embedding model
        async def preload_embedding():
            if set["embed_model_provider"] == models.ModelProvider.HUGGINGFACE.name:
                try:
                    emb_mod = models.get_huggingface_embedding(set["embed_model_name"])
                    emb_txt = await emb_mod.aembed_query("test")
                    return emb_txt
                except Exception as e:
                    PrintStyle().error(f"Error in preload_embedding: {e}")


        # async tasks to preload
        tasks = [preload_whisper(), preload_embedding()]

        await asyncio.gather(*tasks, return_exceptions=True)
        PrintStyle().print("Preload completed")
    except Exception as e:
        PrintStyle().error(f"Error in preload: {e}")


# preload transcription model
asyncio.run(preload())



================================================
File: prepare.py
================================================
from python.helpers import dotenv, runtime, settings
import string
import random
from python.helpers.print_style import PrintStyle


PrintStyle.standard("Preparing environment...")

try:

    runtime.initialize()

    # generate random root password if not set (for SSH)
    root_pass = dotenv.get_dotenv_value(dotenv.KEY_ROOT_PASSWORD)
    if not root_pass:
        root_pass = "".join(random.choices(string.ascii_letters + string.digits, k=32))
        PrintStyle.standard("Changing root password...")
    settings.set_root_password(root_pass)

except Exception as e:
    PrintStyle.error(f"Error in preload: {e}")
















================================================
File: lib/browser/click.js
================================================
function click(selector){
  {
    const element = document.querySelector(selector);
    if (element) {
      element.click();
      return true;
    }
    return false;
  }
}


================================================
File: lib/browser/extract_dom.js
================================================
function extractDOM([
  selectorLabel = "",
  selectorName = "data-a0sel3ct0r",
  guidName = "data-a0gu1d",
]) {
  let elementCounter = 0;
  const time = new Date().toISOString().slice(11, -1).replace(/[:.]/g, "");
  const ignoredTags = [
    "style",
    "script",
    "meta",
    "link",
    "svg",
    "noscript",
    "path",
  ];

  // Convert number to base64 and trim unnecessary chars
  function toBase64(num) {
    const chars =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let result = "";

    do {
      result = chars[num & 63] + result;
      num = num >> 6;
    } while (num > 0);

    return result;
  }

  function isElementVisible(element) {
    // Return true for non-element nodes
    if (element.nodeType !== Node.ELEMENT_NODE) {
      return true;
    }

    const computedStyle = window.getComputedStyle(element);

    // Check if element is hidden via CSS
    if (
      computedStyle.display === "none" ||
      computedStyle.visibility === "hidden" ||
      computedStyle.opacity === "0"
    ) {
      return false;
    }

    // Check for hidden input type
    if (element.tagName === "INPUT" && element.type === "hidden") {
      return false;
    }

    // Check for hidden attribute
    if (
      element.hasAttribute("hidden") ||
      element.getAttribute("aria-hidden") === "true"
    ) {
      return false;
    }

    return true;
  }

  function convertAttribute(tag, attr) {
    let out = {
      name: attr.name,
      value:
        typeof attr.value == "string" ? attr.value : JSON.stringify(attr.value),
    };

    //excluded attributes
    if (["srcset"].includes(out.name)) return null;
    if (out.name.startsWith("data-") && out.name != selectorName) return null;

    if (out.name == "src" && out.value.startsWith("data:"))
      out.value = "data...";

    return out;
  }

  function traverseNodes(node, depth = 0, visited = new Set()) {
    // Safety checks
    if (!node) return "";
    if (depth > 1000) return "<!-- Max depth exceeded -->";

    const guid = node.getAttribute?.(guidName);
    if (guid && visited.has(guid)) {
      return `<!-- Circular reference detected at guid: ${guid} -->`;
    }

    let content = "";
    const tagName = node.tagName ? node.tagName.toLowerCase() : "";

    // Skip ignored tags
    if (tagName && ignoredTags.includes(tagName)) {
      return "";
    }

    if (node.nodeType === Node.ELEMENT_NODE) {
      // Add unique ID to the actual DOM element
      if (tagName) {
        const no = elementCounter++;
        const selector = `${no}${selectorLabel}`;
        const guid = `${time}-${selector}`;
        node.setAttribute(selectorName, selector);
        node.setAttribute(guidName, guid);
        visited.add(guid);
      }

      content += `<${tagName}`;

      // Add invisible attribute if element is not visible
      if (!isElementVisible(node)) {
        content += " invisible";
      }

      for (let attr of node.attributes) {
        const out = convertAttribute(tagName, attr);
        if (out) content += ` ${out.name}="${out.value}"`;
      }

      content += ">";

      // Handle iframes
      if (tagName === "iframe") {
        try {
          const frameId = elementCounter++;
          node.setAttribute(selectorName, frameId);
          content += `<!-- IFrame Content Placeholder ${frameId} -->`;
        } catch (e) {
          console.warn("Error marking iframe:", e);
        }
      }

      if (node.shadowRoot) {
        content += "<!-- Shadow DOM Start -->";
        for (let shadowChild of node.shadowRoot.childNodes) {
          content += traverseNodes(shadowChild, depth + 1, visited);
        }
        content += "<!-- Shadow DOM End -->";
      }

      for (let child of node.childNodes) {
        content += traverseNodes(child, depth + 1, visited);
      }

      content += `</${tagName}>`;
    } else if (node.nodeType === Node.TEXT_NODE) {
      content += node.textContent;
    } else if (node.nodeType === Node.COMMENT_NODE) {
      content += `<!--${node.textContent}-->`;
    }

    return content;
  }

  const fullHTML = traverseNodes(document.documentElement);
  return fullHTML;
}



================================================
File: lib/browser/init_override.js
================================================
// open all shadow doms
(function () {
  const originalAttachShadow = Element.prototype.attachShadow;
  Element.prototype.attachShadow = function attachShadow(options) {
    return originalAttachShadow.call(this, { ...options, mode: "open" });
  };
})();

// // Create a global bridge for iframe communication
// (function() {
//   let elementCounter = 0;
//   const ignoredTags = [
//     "style",
//     "script",
//     "meta",
//     "link",
//     "svg",
//     "noscript",
//     "path",
//   ];

//   function isElementVisible(element) {
//     // Return true for non-element nodes
//     if (element.nodeType !== Node.ELEMENT_NODE) {
//       return true;
//     }

//     const computedStyle = window.getComputedStyle(element);

//     // Check if element is hidden via CSS
//     if (
//       computedStyle.display === "none" ||
//       computedStyle.visibility === "hidden" ||
//       computedStyle.opacity === "0"
//     ) {
//       return false;
//     }

//     // Check for hidden input type
//     if (element.tagName === "INPUT" && element.type === "hidden") {
//       return false;
//     }

//     // Check for hidden attribute
//     if (
//       element.hasAttribute("hidden") ||
//       element.getAttribute("aria-hidden") === "true"
//     ) {
//       return false;
//     }

//     return true;
//   }

//   function convertAttribute(tag, attr) {
//     let out = {
//       name: attr.name,
//       value: attr.value,
//     };

//     if (["srcset"].includes(out.name)) return null;
//     if (out.name.startsWith("data-") && out.name != "data-A0UID" && out.name != "data-a0-frame-id") return null;

//     if (tag === "img" && out.value.startsWith("data:")) out.value = "data...";

//     return out;
//   }

//   // This function will be available in all frames
//   window.__A0_extractFrameContent = function() {
//     // Get the current frame's DOM content
//     const extractContent = (node) => {
//       if (!node) return "";
      
//       let content = "";
//       const tagName = node.tagName ? node.tagName.toLowerCase() : "";
      
//       // Skip ignored tags
//       if (tagName && ignoredTags.includes(tagName)) {
//         return "";
//       }

//       if (node.nodeType === Node.ELEMENT_NODE) {
//         // Add unique ID to the actual DOM element
//         if (tagName) {
//           const uid = elementCounter++;
//           node.setAttribute("data-A0UID", uid);
//         }

//         content += `<${tagName}`;

//         // Add invisible attribute if element is not visible
//         if (!isElementVisible(node)) {
//           content += " invisible";
//         }

//         // Add attributes with conversion
//         for (let attr of node.attributes) {
//           const out = convertAttribute(tagName, attr);
//           if (out) content += ` ${out.name}="${out.value}"`;
//         }

//         if (tagName) {
//           content += ` selector="${node.getAttribute("data-A0UID")}"`;
//         }
        
//         content += ">";
        
//         // Handle shadow DOM
//         if (node.shadowRoot) {
//           content += "<!-- Shadow DOM Start -->";
//           for (let shadowChild of node.shadowRoot.childNodes) {
//             content += extractContent(shadowChild);
//           }
//           content += "<!-- Shadow DOM End -->";
//         }
        
//         // Handle child nodes
//         for (let child of node.childNodes) {
//           content += extractContent(child);
//         }
        
//         content += `</${tagName}>`;
//       } else if (node.nodeType === Node.TEXT_NODE) {
//         content += node.textContent;
//       } else if (node.nodeType === Node.COMMENT_NODE) {
//         content += `<!--${node.textContent}-->`;
//       }
      
//       return content;
//     };

//     return extractContent(document.documentElement);
//   };

//   // Setup message listener in each frame
//   window.addEventListener('message', function(event) {
//     if (event.data === 'A0_REQUEST_CONTENT') {
//       // Extract content and send it back to parent
//       const content = window.__A0_extractFrameContent();
//       // Use '*' as targetOrigin since we're in a controlled environment
//       window.parent.postMessage({
//         type: 'A0_FRAME_CONTENT',
//         content: content,
//         frameId: window.frameElement?.getAttribute('data-a0-frame-id')
//       }, '*');
//     }
//   });

//   // Function to extract content from all frames
//   window.__A0_extractAllFramesContent = async function(rootNode = document) {
//     let content = "";
    
//     // Extract content from current document
//     content += window.__A0_extractFrameContent();
    
//     // Find all iframes
//     const iframes = rootNode.getElementsByTagName('iframe');
    
//     // Create a map to store frame contents
//     const frameContents = new Map();
    
//     // Setup promise for each iframe
//     const framePromises = Array.from(iframes).map((iframe) => {
//       return new Promise((resolve) => {
//         const frameId = 'frame_' + Math.random().toString(36).substr(2, 9);
//         iframe.setAttribute('data-a0-frame-id', frameId);
        
//         // Setup one-time message listener for this specific frame
//         const listener = function(event) {
//           if (event.data?.type === 'A0_FRAME_CONTENT' && 
//               event.data?.frameId === frameId) {
//             frameContents.set(frameId, event.data.content);
//             window.removeEventListener('message', listener);
//             resolve();
//           }
//         };
//         window.addEventListener('message', listener);
        
//         // Request content from frame
//         iframe.contentWindow.postMessage('A0_REQUEST_CONTENT', '*');
        
//         // Timeout after 2 seconds
//         setTimeout(resolve, 2000);
//       });
//     });
    
//     // Wait for all frames to respond or timeout
//     await Promise.all(framePromises);
    
//     // Add frame contents in order
//     for (let iframe of iframes) {
//       const frameId = iframe.getAttribute('data-a0-frame-id');
//       const frameContent = frameContents.get(frameId);
//       if (frameContent) {
//         content += `<!-- IFrame ${iframe.src || 'unnamed'} Content Start -->`;
//         content += frameContent;
//         content += `<!-- IFrame Content End -->`;
//       }
//     }
    
//     return content;
//   };
// })();

// // override iframe creation to inject our script into them
// (function() {
//   // Store the original createElement to use for iframe creation
//   const originalCreateElement = document.createElement;

//   // Override createElement to catch iframe creation
//   document.createElement = function(tagName, options) {
//     const element = originalCreateElement.call(document, tagName, options);
//     if (tagName.toLowerCase() === 'iframe') {
//       // Override the src setter
//       const originalSrcSetter = Object.getOwnPropertyDescriptor(HTMLIFrameElement.prototype, 'src').set;
//       Object.defineProperty(element, 'src', {
//         set: function(value) {
//           // Call original setter
//           originalSrcSetter.call(this, value);
          
//           // Wait for load and inject our script
//           this.addEventListener('load', () => {
//             try {
//               // Try to inject our script into the iframe
//               const iframeDoc = this.contentWindow.document;
//               const script = iframeDoc.createElement('script');
//               script.textContent = `
//                 // Make iframe accessible
//                 document.domain = document.domain;
//                 // Disable security policies if possible
//                 if (window.SecurityPolicyViolationEvent) {
//                   window.SecurityPolicyViolationEvent = undefined;
//                 }
//               `;
//               iframeDoc.head.appendChild(script);
//             } catch(e) {
//               console.warn('Could not inject into iframe:', e);
//             }
//           }, { once: true });
//         }
//       });
//     }
//     return element;
//   };
// })();






================================================
File: python/__init__.py
================================================



================================================
File: python/api/chat_export.py
================================================
from python.helpers.api import ApiHandler, Input, Output, Request, Response

from python.helpers import persist_chat

class ExportChat(ApiHandler):
    async def process(self, input: Input, request: Request) -> Output:
        ctxid = input.get("ctxid", "")
        if not ctxid:
            raise Exception("No context id provided")

        context = self.get_context(ctxid)
        content = persist_chat.export_json_chat(context)
        return {
            "message": "Chats exported.",
            "ctxid": context.id,
            "content": content,
        }


================================================
File: python/api/chat_load.py
================================================
from python.helpers.api import ApiHandler, Input, Output, Request, Response


from python.helpers import persist_chat

class LoadChats(ApiHandler):
    async def process(self, input: Input, request: Request) -> Output:
        chats = input.get("chats", [])
        if not chats:
            raise Exception("No chats provided")

        ctxids = persist_chat.load_json_chats(chats)

        return {
            "message": "Chats loaded.",
            "ctxids": ctxids,
        }



================================================
File: python/api/chat_remove.py
================================================
from python.helpers.api import ApiHandler, Input, Output, Request, Response
from agent import AgentContext
from python.helpers import persist_chat
from python.helpers.task_scheduler import TaskScheduler


class RemoveChat(ApiHandler):
    async def process(self, input: Input, request: Request) -> Output:
        ctxid = input.get("context", "")

        context = AgentContext.get(ctxid)
        if context:
            # stop processing any tasks
            context.reset()

        AgentContext.remove(ctxid)
        persist_chat.remove_chat(ctxid)

        scheduler = TaskScheduler.get()
        await scheduler.reload()

        tasks = scheduler.get_tasks_by_context_id(ctxid)
        for task in tasks:
            await scheduler.remove_task_by_uuid(task.uuid)

        return {
            "message": "Context removed.",
        }



================================================
File: python/api/chat_reset.py
================================================
from python.helpers.api import ApiHandler, Input, Output, Request, Response


from python.helpers import persist_chat


class Reset(ApiHandler):
    async def process(self, input: Input, request: Request) -> Output:
        ctxid = input.get("context", "")

        # context instance - get or create
        context = self.get_context(ctxid)
        context.reset()
        persist_chat.save_tmp_chat(context)

        return {
            "message": "Agent restarted.",
        }



================================================
File: python/api/ctx_window_get.py
================================================
from python.helpers.api import ApiHandler, Input, Output, Request, Response

from python.helpers import tokens


class GetCtxWindow(ApiHandler):
    async def process(self, input: Input, request: Request) -> Output:
        ctxid = input.get("context", [])
        context = self.get_context(ctxid)
        agent = context.streaming_agent or context.agent0
        window = agent.get_data(agent.DATA_NAME_CTX_WINDOW)
        if not window or not isinstance(window, dict):
            return {"content": "", "tokens": 0}

        text = window["text"]
        tokens = window["tokens"]

        return {"content": text, "tokens": tokens}



================================================
File: python/api/delete_work_dir_file.py
================================================
from python.helpers.api import ApiHandler, Input, Output, Request, Response


from python.helpers.file_browser import FileBrowser
from python.helpers import files, runtime
from python.api import get_work_dir_files


class DeleteWorkDirFile(ApiHandler):
    async def process(self, input: Input, request: Request) -> Output:
        file_path = input.get("path", "")
        if not file_path.startswith("/"):
            file_path = f"/{file_path}"

        current_path = input.get("currentPath", "")

        # browser = FileBrowser()
        res = await runtime.call_development_function(delete_file, file_path)

        if res:
            # Get updated file list
            # result = browser.get_files(current_path)
            result = await runtime.call_development_function(get_work_dir_files.get_files, current_path)
            return {"data": result}
        else:
            raise Exception("File not found or could not be deleted")


async def delete_file(file_path: str):
    browser = FileBrowser()
    return browser.delete_file(file_path)



================================================
File: python/api/download_work_dir_file.py
================================================
import base64
from io import BytesIO

from python.helpers.api import ApiHandler, Input, Output, Request, Response
from flask import send_file

from python.helpers import files, runtime
from python.api import file_info
import os


class DownloadFile(ApiHandler):
    async def process(self, input: Input, request: Request) -> Output:
        file_path = request.args.get("path", input.get("path", ""))
        if not file_path:
            raise ValueError("No file path provided")
        if not file_path.startswith("/"):
            file_path = f"/{file_path}"

        file = await runtime.call_development_function(
            file_info.get_file_info, file_path
        )

        if not file["exists"]:
            raise Exception(f"File {file_path} not found")

        if file["is_dir"]:
            zip_file = await runtime.call_development_function(files.zip_dir, file["abs_path"])
            if runtime.is_development():
                b64 = await runtime.call_development_function(fetch_file, zip_file)
                file_data = BytesIO(base64.b64decode(b64))
                return send_file(
                    file_data,
                    as_attachment=True,
                    download_name=os.path.basename(zip_file),
                )
            else:
                return send_file(
                    zip_file,
                    as_attachment=True,
                    download_name=f"{os.path.basename(file_path)}.zip",
                )
        elif file["is_file"]:
            if runtime.is_development():
                b64 = await runtime.call_development_function(fetch_file, file["abs_path"])
                file_data = BytesIO(base64.b64decode(b64))
                return send_file(
                    file_data,
                    as_attachment=True,
                    download_name=os.path.basename(file_path),
                )
            else:
                return send_file(
                    file["abs_path"],
                    as_attachment=True,
                    download_name=os.path.basename(file["file_name"]),
                )
        raise Exception(f"File {file_path} not found")


async def fetch_file(path):
    with open(path, "rb") as file:
        file_content = file.read()
        return base64.b64encode(file_content).decode("utf-8")



================================================
File: python/api/file_info.py
================================================
import os
from python.helpers.api import ApiHandler, Input, Output, Request, Response
from python.helpers import files, runtime
from typing import TypedDict

class FileInfoApi(ApiHandler):
    async def process(self, input: Input, request: Request) -> Output:
        path = input.get("path", "")
        info = await runtime.call_development_function(get_file_info, path)
        return info

class FileInfo(TypedDict):
    input_path: str
    abs_path: str
    exists: bool
    is_dir: bool
    is_file: bool
    is_link: bool
    size: int
    modified: float
    created: float
    permissions: int
    dir_path: str
    file_name: str
    file_ext: str
    message: str

async def get_file_info(path: str) -> FileInfo:
    abs_path = files.get_abs_path(path)
    exists = os.path.exists(abs_path)
    message = ""

    if not exists:
        message = f"File {path} not found."

    return {
        "input_path": path,
        "abs_path": abs_path,
        "exists": exists,
        "is_dir": os.path.isdir(abs_path) if exists else False,
        "is_file": os.path.isfile(abs_path) if exists else False,
        "is_link": os.path.islink(abs_path) if exists else False,
        "size": os.path.getsize(abs_path) if exists else 0,
        "modified": os.path.getmtime(abs_path) if exists else 0,
        "created": os.path.getctime(abs_path) if exists else 0,
        "permissions": os.stat(abs_path).st_mode if exists else 0,
        "dir_path": os.path.dirname(abs_path),
        "file_name": os.path.basename(abs_path),
        "file_ext": os.path.splitext(abs_path)[1],
        "message": message
    }


================================================
File: python/api/get_work_dir_files.py
================================================
from python.helpers.api import ApiHandler
from flask import Request, Response

from python.helpers.file_browser import FileBrowser
from python.helpers import files, runtime


class GetWorkDirFiles(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        current_path = request.args.get("path", "")
        if current_path == "$WORK_DIR":
            # if runtime.is_development():
            #     current_path = "work_dir"
            # else:
            #     current_path = "root"
            current_path = "root"

        # browser = FileBrowser()
        # result = browser.get_files(current_path)
        result = await runtime.call_development_function(get_files, current_path)

        return {"data": result}

async def get_files(path):
    browser = FileBrowser()
    return browser.get_files(path)


================================================
File: python/api/health.py
================================================
from python.helpers.api import ApiHandler
from flask import Request, Response
from python.helpers import errors

from python.helpers import git

class HealthCheck(ApiHandler):

    async def process(self, input: dict, request: Request) -> dict | Response:
        gitinfo = None
        error = None
        try:
            gitinfo = git.get_git_info()
        except Exception as e:
            error = errors.error_text(e)

        return {"gitinfo": gitinfo, "error": error}



================================================
File: python/api/history_get.py
================================================
from python.helpers import tokens
from python.helpers.api import ApiHandler
from flask import Request, Response


class GetHistory(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        ctxid = input.get("context", [])
        context = self.get_context(ctxid)
        agent = context.streaming_agent or context.agent0
        history = agent.history.output_text()
        size = agent.history.get_tokens()

        return {
            "history": history,
            "tokens": size
        }


================================================
File: python/api/image_get.py
================================================
import os
import re
from python.helpers.api import ApiHandler
from python.helpers import files
from flask import Request, Response, send_file


class ImageGet(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
            # input data
            path = input.get("path", request.args.get("path", ""))
            if not path:
                raise ValueError("No path provided")
            
            # check if path is within base directory
            if not files.is_in_base_dir(path):
                raise ValueError("Path is outside of allowed directory")
            
            # check if file has an image extension
            # list of allowed image extensions
            allowed_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg"]
            # get file extension
            file_ext = os.path.splitext(path)[1].lower()
            if file_ext not in allowed_extensions:
                raise ValueError(f"File type not allowed. Allowed types: {', '.join(allowed_extensions)}")
            
            # check if file exists
            if not os.path.exists(path):
                raise ValueError("File not found")
            
            # send file
            return send_file(path)

            


================================================
File: python/api/import_knowledge.py
================================================
from python.helpers.api import ApiHandler
from flask import Request, Response

from python.helpers.file_browser import FileBrowser
from python.helpers import files, memory
import os
from werkzeug.utils import secure_filename


class ImportKnowledge(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        if "files[]" not in request.files:
            raise Exception("No files part")

        ctxid = request.form.get("ctxid", "")
        if not ctxid:
            raise Exception("No context id provided")

        context = self.get_context(ctxid)

        file_list = request.files.getlist("files[]")
        KNOWLEDGE_FOLDER = files.get_abs_path(memory.get_custom_knowledge_subdir_abs(context.agent0),"main")

        saved_filenames = []

        for file in file_list:
            if file:
                filename = secure_filename(file.filename)  # type: ignore
                file.save(os.path.join(KNOWLEDGE_FOLDER, filename))
                saved_filenames.append(filename)

        #reload memory to re-import knowledge
        await memory.Memory.reload(context.agent0)
        context.log.set_initial_progress()

        return {
            "message": "Knowledge Imported",
            "filenames": saved_filenames[:5]
        }


================================================
File: python/api/message.py
================================================
from agent import AgentContext, UserMessage
from python.helpers.api import ApiHandler
from flask import Request, Response

from python.helpers import files
import os
from werkzeug.utils import secure_filename
from python.helpers.defer import DeferredTask
from python.helpers.print_style import PrintStyle


class Message(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        task, context = await self.communicate(input=input, request=request)
        return await self.respond(task, context)

    async def respond(self, task: DeferredTask, context: AgentContext):
        result = await task.result()  # type: ignore
        return {
            "message": result,
            "context": context.id,
        }

    async def communicate(self, input: dict, request: Request):
        # Handle both JSON and multipart/form-data
        if request.content_type.startswith("multipart/form-data"):
            text = request.form.get("text", "")
            ctxid = request.form.get("context", "")
            message_id = request.form.get("message_id", None)
            attachments = request.files.getlist("attachments")
            attachment_paths = []

            upload_folder_int = "/a0/tmp/uploads"
            upload_folder_ext = files.get_abs_path("tmp/uploads")

            if attachments:
                os.makedirs(upload_folder_ext, exist_ok=True)
                for attachment in attachments:
                    if attachment.filename is None:
                        continue
                    filename = secure_filename(attachment.filename)
                    save_path = files.get_abs_path(upload_folder_ext, filename)
                    attachment.save(save_path)
                    attachment_paths.append(os.path.join(upload_folder_int, filename))
        else:
            # Handle JSON request as before
            input_data = request.get_json()
            text = input_data.get("text", "")
            ctxid = input_data.get("context", "")
            message_id = input_data.get("message_id", None)
            attachment_paths = []

        # Now process the message
        message = text

        # Obtain agent context
        context = self.get_context(ctxid)

        # Store attachments in agent data
        # context.agent0.set_data("attachments", attachment_paths)

        # Prepare attachment filenames for logging
        attachment_filenames = (
            [os.path.basename(path) for path in attachment_paths]
            if attachment_paths
            else []
        )

        # Print to console and log
        PrintStyle(
            background_color="#6C3483", font_color="white", bold=True, padding=True
        ).print(f"User message:")
        PrintStyle(font_color="white", padding=False).print(f"> {message}")
        if attachment_filenames:
            PrintStyle(font_color="white", padding=False).print("Attachments:")
            for filename in attachment_filenames:
                PrintStyle(font_color="white", padding=False).print(f"- {filename}")

        # Log the message with message_id and attachments
        context.log.log(
            type="user",
            heading="User message",
            content=message,
            kvps={"attachments": attachment_filenames},
            id=message_id,
        )

        return context.communicate(UserMessage(message, attachment_paths)), context


================================================
File: python/api/message_async.py
================================================
from agent import AgentContext
from python.helpers.api import ApiHandler
from flask import Request, Response

from python.helpers import files
import os
from werkzeug.utils import secure_filename
from python.helpers.defer import DeferredTask
from python.api.message import Message


class MessageAsync(Message):
    async def respond(self, task: DeferredTask, context: AgentContext):
        return {
            "message": "Message received.",
            "context": context.id,
        }



================================================
File: python/api/nudge.py
================================================
from python.helpers.api import ApiHandler
from flask import Request, Response

class Nudge(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        ctxid = input.get("ctxid", "")
        if not ctxid:
            raise Exception("No context id provided")

        context = self.get_context(ctxid)
        context.nudge()

        msg = "Process reset, agent nudged."
        context.log.log(type="info", content=msg)
        
        return {
            "message": msg,
            "ctxid": context.id,
        }


================================================
File: python/api/pause.py
================================================
from python.helpers.api import ApiHandler
from flask import Request, Response


class Pause(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
            # input data
            paused = input.get("paused", False)
            ctxid = input.get("context", "")

            # context instance - get or create
            context = self.get_context(ctxid)

            context.paused = paused

            return {
                "message": "Agent paused." if paused else "Agent unpaused.",
                "pause": paused,
            }    



================================================
File: python/api/poll.py
================================================
import time
from datetime import datetime
from python.helpers.api import ApiHandler
from flask import Request, Response

from agent import AgentContext

from python.helpers import persist_chat
from python.helpers.task_scheduler import TaskScheduler
from python.helpers.localization import Localization
from python.helpers.dotenv import get_dotenv_value


class Poll(ApiHandler):

    async def process(self, input: dict, request: Request) -> dict | Response:
        ctxid = input.get("context", None)
        from_no = input.get("log_from", 0)

        # Get timezone from input (default to dotenv default or UTC if not provided)
        timezone = input.get("timezone", get_dotenv_value("DEFAULT_USER_TIMEZONE", "UTC"))
        Localization.get().set_timezone(timezone)

        # context instance - get or create
        context = self.get_context(ctxid)

        logs = context.log.output(start=from_no)

        # loop AgentContext._contexts

        # Get a task scheduler instance
        scheduler = TaskScheduler.get()

        # Always reload the scheduler on each poll to ensure we have the latest task state
        # await scheduler.reload() # does not seem to be needed

        # loop AgentContext._contexts and divide into contexts and tasks

        ctxs = []
        tasks = []
        processed_contexts = set()  # Track processed context IDs

        all_ctxs = list(AgentContext._contexts.values())
        # First, identify all tasks
        for ctx in all_ctxs:
            # Skip if already processed
            if ctx.id in processed_contexts:
                continue

            # Create the base context data that will be returned
            context_data = ctx.serialize()

            context_task = scheduler.get_task_by_uuid(ctx.id)
            # Determine if this is a task-dedicated context by checking if a task with this UUID exists
            is_task_context = (
                context_task is not None and context_task.context_id == ctx.id
            )

            if not is_task_context:
                ctxs.append(context_data)
            else:
                # If this is a task, get task details from the scheduler
                task_details = scheduler.serialize_task(ctx.id)
                if task_details:
                    # Add task details to context_data with the same field names
                    # as used in scheduler endpoints to maintain UI compatibility
                    context_data.update({
                        "task_name": task_details.get("name"), # name is for context, task_name for the task name
                        "uuid": task_details.get("uuid"),
                        "state": task_details.get("state"),
                        "type": task_details.get("type"),
                        "system_prompt": task_details.get("system_prompt"),
                        "prompt": task_details.get("prompt"),
                        "last_run": task_details.get("last_run"),
                        "last_result": task_details.get("last_result"),
                        "attachments": task_details.get("attachments", []),
                        "context_id": task_details.get("context_id"),
                    })

                    # Add type-specific fields
                    if task_details.get("type") == "scheduled":
                        context_data["schedule"] = task_details.get("schedule")
                    elif task_details.get("type") == "planned":
                        context_data["plan"] = task_details.get("plan")
                    else:
                        context_data["token"] = task_details.get("token")

                tasks.append(context_data)

            # Mark as processed
            processed_contexts.add(ctx.id)

        # Sort tasks and chats by their creation date, descending
        ctxs.sort(key=lambda x: x["created_at"], reverse=True)
        tasks.sort(key=lambda x: x["created_at"], reverse=True)

        # data from this server
        return {
            "context": context.id,
            "contexts": ctxs,
            "tasks": tasks,
            "logs": logs,
            "log_guid": context.log.guid,
            "log_version": len(context.log.updates),
            "log_progress": context.log.progress,
            "log_progress_active": context.log.progress_active,
            "paused": context.paused,
        }



================================================
File: python/api/restart.py
================================================
from python.helpers.api import ApiHandler
from flask import Request, Response

from python.helpers import process

class Restart(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        process.reload()
        return Response(status=200)


================================================
File: python/api/rfc.py
================================================
from python.helpers.api import ApiHandler
from flask import Request, Response

from python.helpers import runtime

class RFC(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        result = await runtime.handle_rfc(input) # type: ignore
        return result



================================================
File: python/api/scheduler_task_create.py
================================================
from python.helpers.api import ApiHandler, Input, Output, Request
from python.helpers.task_scheduler import (
    TaskScheduler, ScheduledTask, AdHocTask, PlannedTask, TaskSchedule,
    serialize_task, parse_task_schedule, parse_task_plan, TaskType
)
from python.helpers.localization import Localization
from python.helpers.print_style import PrintStyle
import random


class SchedulerTaskCreate(ApiHandler):
    async def process(self, input: Input, request: Request) -> Output:
        """
        Create a new task in the scheduler
        """
        printer = PrintStyle(italic=True, font_color="blue", padding=False)

        # Get timezone from input (do not set if not provided, we then rely on poll() to set it)
        if timezone := input.get("timezone", None):
            Localization.get().set_timezone(timezone)

        scheduler = TaskScheduler.get()
        await scheduler.reload()

        # Get common fields from input
        name = input.get("name")
        system_prompt = input.get("system_prompt", "")
        prompt = input.get("prompt")
        attachments = input.get("attachments", [])
        context_id = input.get("context_id", None)

        # Check if schedule is provided (for ScheduledTask)
        schedule = input.get("schedule", {})
        token: str = input.get("token", "")

        # Debug log the token value
        printer.print(f"Token received from frontend: '{token}' (type: {type(token)}, length: {len(token) if token else 0})")

        # Generate a random token if empty or not provided
        if not token:
            token = str(random.randint(1000000000000000000, 9999999999999999999))
            printer.print(f"Generated new token: '{token}'")

        plan = input.get("plan", {})

        # Validate required fields
        if not name or not prompt:
            # return {"error": "Missing required fields: name, system_prompt, prompt"}
            raise ValueError("Missing required fields: name, system_prompt, prompt")

        task = None
        if schedule:
            # Create a scheduled task
            # Handle different schedule formats (string or object)
            if isinstance(schedule, str):
                # Parse the string schedule
                parts = schedule.split(' ')
                task_schedule = TaskSchedule(
                    minute=parts[0] if len(parts) > 0 else "*",
                    hour=parts[1] if len(parts) > 1 else "*",
                    day=parts[2] if len(parts) > 2 else "*",
                    month=parts[3] if len(parts) > 3 else "*",
                    weekday=parts[4] if len(parts) > 4 else "*"
                )
            elif isinstance(schedule, dict):
                # Use our standardized parsing function
                try:
                    task_schedule = parse_task_schedule(schedule)
                except ValueError as e:
                    raise ValueError(str(e))
            else:
                raise ValueError("Invalid schedule format. Must be string or object.")

            task = ScheduledTask.create(
                name=name,
                system_prompt=system_prompt,
                prompt=prompt,
                schedule=task_schedule,
                attachments=attachments,
                context_id=context_id,
                timezone=timezone
            )
        elif plan:
            # Create a planned task
            try:
                # Use our standardized parsing function
                task_plan = parse_task_plan(plan)
            except ValueError as e:
                return {"error": str(e)}

            task = PlannedTask.create(
                name=name,
                system_prompt=system_prompt,
                prompt=prompt,
                plan=task_plan,
                attachments=attachments,
                context_id=context_id
            )
        else:
            # Create an ad-hoc task
            printer.print(f"Creating AdHocTask with token: '{token}'")
            task = AdHocTask.create(
                name=name,
                system_prompt=system_prompt,
                prompt=prompt,
                token=token,
                attachments=attachments,
                context_id=context_id
            )
            # Verify token after creation
            if isinstance(task, AdHocTask):
                printer.print(f"AdHocTask created with token: '{task.token}'")

        # Add the task to the scheduler
        await scheduler.add_task(task)

        # Verify the task was added correctly - retrieve by UUID to check persistence
        saved_task = scheduler.get_task_by_uuid(task.uuid)
        if saved_task:
            if saved_task.type == TaskType.AD_HOC and isinstance(saved_task, AdHocTask):
                printer.print(f"Task verified after save, token: '{saved_task.token}'")
            else:
                printer.print("Task verified after save, not an adhoc task")
        else:
            printer.print("WARNING: Task not found after save!")

        # Return the created task using our standardized serialization function
        task_dict = serialize_task(task)

        # Debug log the serialized task
        if task_dict and task_dict.get('type') == 'adhoc':
            printer.print(f"Serialized adhoc task, token in response: '{task_dict.get('token')}'")

        return {
            "task": task_dict
        }



================================================
File: python/api/scheduler_task_delete.py
================================================
from python.helpers.api import ApiHandler, Input, Output, Request
from python.helpers.task_scheduler import TaskScheduler, TaskState
from python.helpers.localization import Localization
from agent import AgentContext
from python.helpers import persist_chat


class SchedulerTaskDelete(ApiHandler):
    async def process(self, input: Input, request: Request) -> Output:
        """
        Delete a task from the scheduler by ID
        """
        # Get timezone from input (do not set if not provided, we then rely on poll() to set it)
        if timezone := input.get("timezone", None):
            Localization.get().set_timezone(timezone)

        scheduler = TaskScheduler.get()
        await scheduler.reload()

        # Get task ID from input
        task_id: str = input.get("task_id", "")

        if not task_id:
            return {"error": "Missing required field: task_id"}

        # Check if the task exists first
        task = scheduler.get_task_by_uuid(task_id)
        if not task:
            return {"error": f"Task with ID {task_id} not found"}

        context = None
        if task.context_id:
            context = self.get_context(task.context_id)

        # If the task is running, update its state to IDLE first
        if task.state == TaskState.RUNNING:
            if context:
                context.reset()
            # Update the state to IDLE so any ongoing processes know to terminate
            await scheduler.update_task(task_id, state=TaskState.IDLE)
            # Force a save to ensure the state change is persisted
            await scheduler.save()

        # This is a dedicated context for the task, so we remove it
        if context and context.id == task.uuid:
            AgentContext.remove(context.id)
            persist_chat.remove_chat(context.id)

        # Remove the task
        await scheduler.remove_task_by_uuid(task_id)

        return {"success": True, "message": f"Task {task_id} deleted successfully"}



================================================
File: python/api/scheduler_task_run.py
================================================
from python.helpers.api import ApiHandler, Input, Output, Request
from python.helpers.task_scheduler import TaskScheduler, TaskState
from python.helpers.print_style import PrintStyle
from python.helpers.localization import Localization


class SchedulerTaskRun(ApiHandler):

    _printer: PrintStyle = PrintStyle(italic=True, font_color="green", padding=False)

    async def process(self, input: Input, request: Request) -> Output:
        """
        Manually run a task from the scheduler by ID
        """
        # Get timezone from input (do not set if not provided, we then rely on poll() to set it)
        if timezone := input.get("timezone", None):
            Localization.get().set_timezone(timezone)

        # Get task ID from input
        task_id: str = input.get("task_id", "")

        if not task_id:
            return {"error": "Missing required field: task_id"}

        self._printer.print(f"SchedulerTaskRun: On-Demand running task {task_id}")

        scheduler = TaskScheduler.get()
        await scheduler.reload()

        # Check if the task exists first
        task = scheduler.get_task_by_uuid(task_id)
        if not task:
            self._printer.error(f"SchedulerTaskRun: Task with ID '{task_id}' not found")
            return {"error": f"Task with ID '{task_id}' not found"}

        # Check if task is already running
        if task.state == TaskState.RUNNING:
            # Return task details along with error for better frontend handling
            serialized_task = scheduler.serialize_task(task_id)
            self._printer.error(f"SchedulerTaskRun: Task '{task_id}' is in state '{task.state}' and cannot be run")
            return {
                "error": f"Task '{task_id}' is in state '{task.state}' and cannot be run",
                "task": serialized_task
            }

        # Run the task, which now includes atomic state checks and updates
        try:
            await scheduler.run_task_by_uuid(task_id)
            self._printer.print(f"SchedulerTaskRun: Task '{task_id}' started successfully")
            # Get updated task after run starts
            serialized_task = scheduler.serialize_task(task_id)
            if serialized_task:
                return {
                    "success": True,
                    "message": f"Task '{task_id}' started successfully",
                    "task": serialized_task
                }
            else:
                return {"success": True, "message": f"Task '{task_id}' started successfully"}
        except ValueError as e:
            self._printer.error(f"SchedulerTaskRun: Task '{task_id}' failed to start: {str(e)}")
            return {"error": str(e)}
        except Exception as e:
            self._printer.error(f"SchedulerTaskRun: Task '{task_id}' failed to start: {str(e)}")
            return {"error": f"Failed to run task '{task_id}': {str(e)}"}



================================================
File: python/api/scheduler_task_update.py
================================================
from python.helpers.api import ApiHandler, Input, Output, Request
from python.helpers.task_scheduler import (
    TaskScheduler, ScheduledTask, AdHocTask, PlannedTask, TaskState,
    serialize_task, parse_task_schedule, parse_task_plan
)
from python.helpers.localization import Localization


class SchedulerTaskUpdate(ApiHandler):
    async def process(self, input: Input, request: Request) -> Output:
        """
        Update an existing task in the scheduler
        """
        # Get timezone from input (do not set if not provided, we then rely on poll() to set it)
        if timezone := input.get("timezone", None):
            Localization.get().set_timezone(timezone)

        scheduler = TaskScheduler.get()
        await scheduler.reload()

        # Get task ID from input
        task_id: str = input.get("task_id", "")

        if not task_id:
            return {"error": "Missing required field: task_id"}

        # Get the task to update
        task = scheduler.get_task_by_uuid(task_id)

        if not task:
            return {"error": f"Task with ID {task_id} not found"}

        # Update fields if provided using the task's update method
        update_params = {}

        if "name" in input:
            update_params["name"] = input.get("name", "")

        if "state" in input:
            update_params["state"] = TaskState(input.get("state", TaskState.IDLE))

        if "system_prompt" in input:
            update_params["system_prompt"] = input.get("system_prompt", "")

        if "prompt" in input:
            update_params["prompt"] = input.get("prompt", "")

        if "attachments" in input:
            update_params["attachments"] = input.get("attachments", [])

        # Update schedule if this is a scheduled task and schedule is provided
        if isinstance(task, ScheduledTask) and "schedule" in input:
            schedule_data = input.get("schedule", {})
            try:
                # Parse the schedule with timezone handling
                task_schedule = parse_task_schedule(schedule_data)

                # Set the timezone from the request if not already in schedule_data
                if not schedule_data.get('timezone', None) and timezone:
                    task_schedule.timezone = timezone

                update_params["schedule"] = task_schedule
            except ValueError as e:
                return {"error": f"Invalid schedule format: {str(e)}"}
        elif isinstance(task, AdHocTask) and "token" in input:
            token_value = input.get("token", "")
            if token_value:  # Only update if non-empty
                update_params["token"] = token_value
        elif isinstance(task, PlannedTask) and "plan" in input:
            plan_data = input.get("plan", {})
            try:
                # Parse the plan data
                task_plan = parse_task_plan(plan_data)
                update_params["plan"] = task_plan
            except ValueError as e:
                return {"error": f"Invalid plan format: {str(e)}"}

        # Use atomic update method to apply changes
        updated_task = await scheduler.update_task(task_id, **update_params)

        if not updated_task:
            return {"error": f"Task with ID {task_id} not found or could not be updated"}

        # Return the updated task using our standardized serialization function
        task_dict = serialize_task(updated_task)

        return {
            "task": task_dict
        }



================================================
File: python/api/scheduler_tasks_list.py
================================================
from python.helpers.api import ApiHandler, Input, Output, Request
from python.helpers.task_scheduler import TaskScheduler
import traceback
from python.helpers.print_style import PrintStyle
from python.helpers.localization import Localization


class SchedulerTasksList(ApiHandler):
    async def process(self, input: Input, request: Request) -> Output:
        """
        List all tasks in the scheduler with their types
        """
        try:
            # Get timezone from input (do not set if not provided, we then rely on poll() to set it)
            if timezone := input.get("timezone", None):
                Localization.get().set_timezone(timezone)

            # Get task scheduler
            scheduler = TaskScheduler.get()
            await scheduler.reload()

            # Use the scheduler's convenience method for task serialization
            tasks_list = scheduler.serialize_all_tasks()

            return {"tasks": tasks_list}

        except Exception as e:
            PrintStyle.error(f"Failed to list tasks: {str(e)} {traceback.format_exc()}")
            return {"error": f"Failed to list tasks: {str(e)} {traceback.format_exc()}", "tasks": []}



================================================
File: python/api/scheduler_tick.py
================================================
from datetime import datetime

from python.helpers.api import ApiHandler, Input, Output, Request
from python.helpers.print_style import PrintStyle
from python.helpers.task_scheduler import TaskScheduler
from python.helpers.localization import Localization


class SchedulerTick(ApiHandler):
    @classmethod
    def requires_loopback(cls) -> bool:
        return True

    async def process(self, input: Input, request: Request) -> Output:
        # Get timezone from input (do not set if not provided, we then rely on poll() to set it)
        if timezone := input.get("timezone", None):
            Localization.get().set_timezone(timezone)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        printer = PrintStyle(font_color="green", padding=False)
        printer.print(f"Scheduler tick - API: {timestamp}")

        # Get the task scheduler instance and print detailed debug info
        scheduler = TaskScheduler.get()
        await scheduler.reload()

        tasks = scheduler.get_tasks()
        tasks_count = len(tasks)

        # Log information about the tasks
        printer.print(f"Scheduler has {tasks_count} task(s)")
        if tasks_count > 0:
            for task in tasks:
                printer.print(f"Task: {task.name} (UUID: {task.uuid}, State: {task.state})")

        # Run the scheduler tick
        await scheduler.tick()

        # Get updated tasks after tick
        serialized_tasks = scheduler.serialize_all_tasks()

        return {
            "scheduler": "tick",
            "timestamp": timestamp,
            "tasks_count": tasks_count,
            "tasks": serialized_tasks
        }



================================================
File: python/api/settings_get.py
================================================
from python.helpers.api import ApiHandler
from flask import Request, Response

from python.helpers import settings

class GetSettings(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        set = settings.convert_out(settings.get_settings())
        return {"settings": set}



================================================
File: python/api/settings_set.py
================================================
from python.helpers.api import ApiHandler
from flask import Request, Response

from python.helpers import settings


class SetSettings(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        set = settings.convert_in(input)
        set = settings.set_settings(set)
        return {"settings": set}



================================================
File: python/api/transcribe.py
================================================
from python.helpers.api import ApiHandler
from flask import Request, Response

from python.helpers import runtime, settings, whisper

class Transcribe(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        audio = input.get("audio")
        ctxid = input.get("ctxid", "")

        context = self.get_context(ctxid)
        if await whisper.is_downloading():
            context.log.log(type="info", content="Whisper model is currently being downloaded, please wait...")

        set = settings.get_settings()
        result = await whisper.transcribe(set["stt_model_size"], audio) # type: ignore
        return result



================================================
File: python/api/tunnel.py
================================================
from flask import Request, Response
from python.helpers import runtime
from python.helpers.api import ApiHandler
from python.helpers.tunnel_manager import TunnelManager

class Tunnel(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        action = input.get("action", "get")
        
        tunnel_manager = TunnelManager.get_instance()

        if action == "health":
            return {"success": True}
        
        if action == "create":
            port = runtime.get_web_ui_port()
            tunnel_url = tunnel_manager.start_tunnel(port)
            if tunnel_url is None:
                # Add a little delay and check again - tunnel might be starting
                import time
                time.sleep(2)
                tunnel_url = tunnel_manager.get_tunnel_url()
            
            return {
                "success": tunnel_url is not None,
                "tunnel_url": tunnel_url,
                "message": "Tunnel creation in progress" if tunnel_url is None else "Tunnel created successfully"
            }
        
        elif action == "stop":
            return self.stop()
        
        elif action == "get":
            tunnel_url = tunnel_manager.get_tunnel_url()
            return {
                "success": tunnel_url is not None,
                "tunnel_url": tunnel_url,
                "is_running": tunnel_manager.is_running
            }
        
        return {
            "success": False,
            "error": "Invalid action. Use 'create', 'stop', or 'get'."
        } 

    def stop(self):
        tunnel_manager = TunnelManager.get_instance()
        tunnel_manager.stop_tunnel()
        return {
            "success": True
        }



================================================
File: python/api/tunnel_proxy.py
================================================
from flask import Request, Response
from python.helpers import dotenv, runtime
from python.helpers.api import ApiHandler
from python.helpers.tunnel_manager import TunnelManager
import requests


class TunnelProxy(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        # Get configuration from environment
        tunnel_api_port = (
            runtime.get_arg("tunnel_api_port")
            or int(dotenv.get_dotenv_value("TUNNEL_API_PORT", 0))
            or 5070
        )

        # first verify the service is running:
        service_ok = False
        try:
            response = requests.post(f"http://localhost:{tunnel_api_port}/", json={"action": "health"})
            if response.status_code == 200:
                service_ok = True
        except Exception as e:
            service_ok = False

        # forward this request to the tunnel service if OK
        if service_ok:
            try:
                response = requests.post(f"http://localhost:{tunnel_api_port}/", json=input)
                return response.json()
            except Exception as e:
                return {"error": str(e)}
        else:
            # forward to API handler directly
            from python.api.tunnel import Tunnel
            return await Tunnel(self.app, self.thread_lock).process(input, request)



================================================
File: python/api/upload.py
================================================
from python.helpers.api import ApiHandler
from flask import Request, Response

from python.helpers import files
from werkzeug.utils import secure_filename


class UploadFile(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        if "file" not in request.files:
            raise Exception("No file part")

        file_list = request.files.getlist("file")  # Handle multiple files
        saved_filenames = []

        for file in file_list:
            if file and self.allowed_file(file.filename):  # Check file type
                filename = secure_filename(file.filename) # type: ignore
                file.save(files.get_abs_path("tmp/upload", filename))
                saved_filenames.append(filename)

        return {"filenames": saved_filenames}  # Return saved filenames


    def allowed_file(self,filename):
        return True
        # ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "txt", "pdf", "csv", "html", "json", "md"}
        # return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


================================================
File: python/api/upload_work_dir_files.py
================================================
import base64
from werkzeug.datastructures import FileStorage
from python.helpers.api import ApiHandler
from flask import Request, Response, send_file

from python.helpers.file_browser import FileBrowser
from python.helpers import files, runtime
from python.api import get_work_dir_files
import os


class UploadWorkDirFiles(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        if "files[]" not in request.files:
            raise Exception("No files uploaded")

        current_path = request.form.get("path", "")
        uploaded_files = request.files.getlist("files[]")

        # browser = FileBrowser()
        # successful, failed = browser.save_files(uploaded_files, current_path)

        successful, failed = await upload_files(uploaded_files, current_path)

        if not successful and failed:
            raise Exception("All uploads failed")

        # result = browser.get_files(current_path)
        result = await runtime.call_development_function(get_work_dir_files.get_files, current_path)

        return {
            "message": (
                "Files uploaded successfully"
                if not failed
                else "Some files failed to upload"
            ),
            "data": result,
            "successful": successful,
            "failed": failed,
        }


async def upload_files(uploaded_files: list[FileStorage], current_path: str):
    if runtime.is_development():
        successful = []
        failed = []
        for file in uploaded_files:
            file_content = file.stream.read()
            base64_content = base64.b64encode(file_content).decode("utf-8")
            if await runtime.call_development_function(
                upload_file, current_path, file.filename, base64_content
            ):
                successful.append(file.filename)
            else:
                failed.append(file.filename)
    else:
        browser = FileBrowser()
        successful, failed = browser.save_files(uploaded_files, current_path)

    return successful, failed


async def upload_file(current_path: str, filename: str, base64_content: str):
    browser = FileBrowser()
    return browser.save_file_b64(current_path, filename, base64_content)




================================================
File: python/extensions/message_loop_end/_10_organize_history.py
================================================
import asyncio
from python.helpers.extension import Extension
from agent import LoopData

DATA_NAME_TASK = "_organize_history_task"


class OrganizeHistory(Extension):
    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        # is there a running task? if yes, skip this round, the wait extension will double check the context size
        task = self.agent.get_data(DATA_NAME_TASK)
        if task and not task.done():
            return

        # start task
        task = asyncio.create_task(self.agent.history.compress())
        # set to agent to be able to wait for it
        self.agent.set_data(DATA_NAME_TASK, task)



================================================
File: python/extensions/message_loop_end/_90_save_chat.py
================================================
from python.helpers.extension import Extension
from agent import LoopData
from python.helpers import persist_chat


class SaveChat(Extension):
    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        persist_chat.save_tmp_chat(self.agent.context)


================================================
File: python/extensions/message_loop_end/.gitkeep
================================================



================================================
File: python/extensions/message_loop_prompts_after/_50_recall_memories.py
================================================
import asyncio
from python.helpers.extension import Extension
from python.helpers.memory import Memory
from agent import LoopData

DATA_NAME_TASK = "_recall_memories_task"

class RecallMemories(Extension):

    INTERVAL = 3
    HISTORY = 10000
    RESULTS = 3
    THRESHOLD = 0.6

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):

        # every 3 iterations (or the first one) recall memories
        if loop_data.iteration % RecallMemories.INTERVAL == 0:
            task = asyncio.create_task(self.search_memories(loop_data=loop_data, **kwargs))
        else:
            task = None

        # set to agent to be able to wait for it
        self.agent.set_data(DATA_NAME_TASK, task)
            

    async def search_memories(self, loop_data: LoopData, **kwargs):

        # cleanup
        extras = loop_data.extras_persistent
        if "memories" in extras:
            del extras["memories"]

        # try:
        # show temp info message
        self.agent.context.log.log(
            type="info", content="Searching memories...", temp=True
        )

        # show full util message, this will hide temp message immediately if turned on
        log_item = self.agent.context.log.log(
            type="util",
            heading="Searching memories...",
        )

        # get system message and chat history for util llm
        # msgs_text = self.agent.concat_messages(
        #     self.agent.history[-RecallMemories.HISTORY :]
        # )  # only last X messages
        msgs_text = self.agent.history.output_text()[-RecallMemories.HISTORY:]
        system = self.agent.read_prompt(
            "memory.memories_query.sys.md", history=msgs_text
        )

        # log query streamed by LLM
        async def log_callback(content):
            log_item.stream(query=content)

        # call util llm to summarize conversation
        query = await self.agent.call_utility_model(
            system=system,
            message=loop_data.user_message.output_text() if loop_data.user_message else "",
            callback=log_callback,
        )

        # get solutions database
        db = await Memory.get(self.agent)

        memories = await db.search_similarity_threshold(
            query=query,
            limit=RecallMemories.RESULTS,
            threshold=RecallMemories.THRESHOLD,
            filter=f"area == '{Memory.Area.MAIN.value}' or area == '{Memory.Area.FRAGMENTS.value}'",  # exclude solutions
        )

        # log the short result
        if not isinstance(memories, list) or len(memories) == 0:
            log_item.update(
                heading="No useful memories found",
            )
            return
        else:
            log_item.update(
                heading=f"{len(memories)} memories found",
            )

        # concatenate memory.page_content in memories:
        memories_text = ""
        for memory in memories:
            memories_text += memory.page_content + "\n\n"
        memories_text = memories_text.strip()

        # log the full results
        log_item.update(memories=memories_text)

        # place to prompt
        memories_prompt = self.agent.parse_prompt(
            "agent.system.memories.md", memories=memories_text
        )

        # append to prompt
        extras["memories"] = memories_prompt

    # except Exception as e:čč
    #     err = errors.format_error(e)
    #     self.agent.context.log.log(
    #         type="error", heading="Recall memories extension error:", content=err
    #     )



================================================
File: python/extensions/message_loop_prompts_after/_51_recall_solutions.py
================================================
import asyncio
from python.helpers.extension import Extension
from python.helpers.memory import Memory
from agent import LoopData

DATA_NAME_TASK = "_recall_solutions_task"

class RecallSolutions(Extension):

    INTERVAL = 3
    HISTORY = 10000
    SOLUTIONS_COUNT = 2
    INSTRUMENTS_COUNT = 2
    THRESHOLD = 0.6

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):

        # every 3 iterations (or the first one) recall memories
        if loop_data.iteration % RecallSolutions.INTERVAL == 0:
            task = asyncio.create_task(self.search_solutions(loop_data=loop_data, **kwargs))
        else:
            task = None

        # set to agent to be able to wait for it
        self.agent.set_data(DATA_NAME_TASK, task)

    async def search_solutions(self, loop_data: LoopData, **kwargs):

        #cleanup
        extras = loop_data.extras_persistent
        if "solutions" in extras:
            del extras["solutions"]
        
        # try:
        # show temp info message
        self.agent.context.log.log(
            type="info", content="Searching memory for solutions...", temp=True
        )

        # show full util message, this will hide temp message immediately if turned on
        log_item = self.agent.context.log.log(
            type="util",
            heading="Searching memory for solutions...",
        )

        # get system message and chat history for util llm
        # msgs_text = self.agent.concat_messages(
        #     self.agent.history[-RecallSolutions.HISTORY :]
        # )  # only last X messages
        # msgs_text = self.agent.history.current.output_text()
        msgs_text = self.agent.history.output_text()[-RecallSolutions.HISTORY:]

        system = self.agent.read_prompt(
            "memory.solutions_query.sys.md", history=msgs_text
        )

        # log query streamed by LLM
        async def log_callback(content):
            log_item.stream(query=content)

        # call util llm to summarize conversation
        query = await self.agent.call_utility_model(
            system=system, message=loop_data.user_message.output_text() if loop_data.user_message else "", callback=log_callback
        )

        # get solutions database
        db = await Memory.get(self.agent)

        solutions = await db.search_similarity_threshold(
            query=query,
            limit=RecallSolutions.SOLUTIONS_COUNT,
            threshold=RecallSolutions.THRESHOLD,
            filter=f"area == '{Memory.Area.SOLUTIONS.value}'",
        )
        instruments = await db.search_similarity_threshold(
            query=query,
            limit=RecallSolutions.INSTRUMENTS_COUNT,
            threshold=RecallSolutions.THRESHOLD,
            filter=f"area == '{Memory.Area.INSTRUMENTS.value}'",
        )

        log_item.update(
            heading=f"{len(instruments)} instruments, {len(solutions)} solutions found",
        )

        if instruments:
            instruments_text = ""
            for instrument in instruments:
                instruments_text += instrument.page_content + "\n\n"
            instruments_text = instruments_text.strip()
            log_item.update(instruments=instruments_text)
            instruments_prompt = self.agent.read_prompt(
                "agent.system.instruments.md", instruments=instruments_text
            )
            loop_data.system.append(instruments_prompt)

        if solutions:
            solutions_text = ""
            for solution in solutions:
                solutions_text += solution.page_content + "\n\n"
            solutions_text = solutions_text.strip()
            log_item.update(solutions=solutions_text)
            solutions_prompt = self.agent.parse_prompt(
                "agent.system.solutions.md", solutions=solutions_text
            )

            # append to prompt
            extras["solutions"] = solutions_prompt

    # except Exception as e:
    #     err = errors.format_error(e)
    #     self.agent.context.log.log(
    #         type="error", heading="Recall solutions extension error:", content=err
    #     )



================================================
File: python/extensions/message_loop_prompts_after/_60_include_current_datetime.py
================================================
from datetime import datetime, timezone
from python.helpers.extension import Extension
from agent import LoopData
from python.helpers.localization import Localization


class IncludeCurrentDatetime(Extension):
    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        # get current datetime
        current_datetime = Localization.get().utc_dt_to_localtime_str(
            datetime.now(timezone.utc), sep=" ", timespec="seconds"
        )
        # remove timezone offset
        if current_datetime and "+" in current_datetime:
            current_datetime = current_datetime.split("+")[0]

        # read prompt
        datetime_prompt = self.agent.read_prompt(
            "agent.system.datetime.md", date_time=current_datetime
        )

        # add current datetime to the loop data
        loop_data.extras_temporary["current_datetime"] = datetime_prompt



================================================
File: python/extensions/message_loop_prompts_after/_91_recall_wait.py
================================================
from python.helpers.extension import Extension
from agent import LoopData
from python.extensions.message_loop_prompts_after._50_recall_memories import DATA_NAME_TASK as DATA_NAME_TASK_MEMORIES
from python.extensions.message_loop_prompts_after._51_recall_solutions import DATA_NAME_TASK as DATA_NAME_TASK_SOLUTIONS


class RecallWait(Extension):
    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):

            task = self.agent.get_data(DATA_NAME_TASK_MEMORIES)
            if task and not task.done():
                # self.agent.context.log.set_progress("Recalling memories...")
                await task

            task = self.agent.get_data(DATA_NAME_TASK_SOLUTIONS)
            if task and not task.done():
                # self.agent.context.log.set_progress("Recalling solutions...")
                await task




================================================
File: python/extensions/message_loop_prompts_after/.gitkeep
================================================



================================================
File: python/extensions/message_loop_prompts_before/_90_organize_history_wait.py
================================================
from python.helpers.extension import Extension
from agent import LoopData
from python.extensions.message_loop_end._10_organize_history import DATA_NAME_TASK
import asyncio


class OrganizeHistoryWait(Extension):
    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):

        # sync action only required if the history is too large, otherwise leave it in background
        while self.agent.history.is_over_limit():
            # get task
            task = self.agent.get_data(DATA_NAME_TASK)

            # Check if the task is already done
            if task:
                if not task.done():
                    self.agent.context.log.set_progress("Compressing history...")

                # Wait for the task to complete
                await task

                # Clear the coroutine data after it's done
                self.agent.set_data(DATA_NAME_TASK, None)
            else:
                # no task running, start and wait
                self.agent.context.log.set_progress("Compressing history...")
                await self.agent.history.compress()




================================================
File: python/extensions/message_loop_prompts_before/.gitkeep
================================================



================================================
File: python/extensions/message_loop_start/_10_iteration_no.py
================================================
from python.helpers.extension import Extension
from agent import Agent, LoopData

DATA_NAME_ITER_NO = "iteration_no"

class IterationNo(Extension):
    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        # total iteration number
        no = self.agent.get_data(DATA_NAME_ITER_NO) or 0
        self.agent.set_data(DATA_NAME_ITER_NO, no + 1)


def get_iter_no(agent: Agent) -> int:
    return agent.get_data(DATA_NAME_ITER_NO) or 0


================================================
File: python/extensions/message_loop_start/.gitkeep
================================================



================================================
File: python/extensions/monologue_end/_50_memorize_fragments.py
================================================
import asyncio
from python.helpers.extension import Extension
from python.helpers.memory import Memory
from python.helpers.dirty_json import DirtyJson
from agent import LoopData
from python.helpers.log import LogItem


class MemorizeMemories(Extension):

    REPLACE_THRESHOLD = 0.9

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        # try:

        # show temp info message
        self.agent.context.log.log(
            type="info", content="Memorizing new information...", temp=True
        )

        # show full util message, this will hide temp message immediately if turned on
        log_item = self.agent.context.log.log(
            type="util",
            heading="Memorizing new information...",
        )

        # memorize in background
        asyncio.create_task(self.memorize(loop_data, log_item))

    async def memorize(self, loop_data: LoopData, log_item: LogItem, **kwargs):

        # get system message and chat history for util llm
        system = self.agent.read_prompt("memory.memories_sum.sys.md")
        msgs_text = self.agent.concat_messages(self.agent.history)

        # log query streamed by LLM
        async def log_callback(content):
            log_item.stream(content=content)

        # call util llm to find info in history
        memories_json = await self.agent.call_utility_model(
            system=system,
            message=msgs_text,
            callback=log_callback,
            background=True,
        )

        memories = DirtyJson.parse_string(memories_json)

        if not isinstance(memories, list) or len(memories) == 0:
            log_item.update(heading="No useful information to memorize.")
            return
        else:
            log_item.update(heading=f"{len(memories)} entries to memorize.")

        # save chat history
        db = await Memory.get(self.agent)

        memories_txt = ""
        rem = []
        for memory in memories:
            # solution to plain text:
            txt = f"{memory}"
            memories_txt += "\n\n" + txt
            log_item.update(memories=memories_txt.strip())

            # remove previous fragments too similiar to this one
            if self.REPLACE_THRESHOLD > 0:
                rem += await db.delete_documents_by_query(
                    query=txt,
                    threshold=self.REPLACE_THRESHOLD,
                    filter=f"area=='{Memory.Area.FRAGMENTS.value}'",
                )
                if rem:
                    rem_txt = "\n\n".join(Memory.format_docs_plain(rem))
                    log_item.update(replaced=rem_txt)

            # insert new solution
            await db.insert_text(text=txt, metadata={"area": Memory.Area.FRAGMENTS.value})

        log_item.update(
            result=f"{len(memories)} entries memorized.",
            heading=f"{len(memories)} entries memorized.",
        )
        if rem:
            log_item.stream(result=f"\nReplaced {len(rem)} previous memories.")

    # except Exception as e:
    #     err = errors.format_error(e)
    #     self.agent.context.log.log(
    #         type="error", heading="Memorize memories extension error:", content=err
    #     )



================================================
File: python/extensions/monologue_end/_51_memorize_solutions.py
================================================
import asyncio
from python.helpers.extension import Extension
from python.helpers.memory import Memory
from python.helpers.dirty_json import DirtyJson
from agent import LoopData
from python.helpers.log import LogItem


class MemorizeSolutions(Extension):

    REPLACE_THRESHOLD = 0.9

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        # try:

        # show temp info message
        self.agent.context.log.log(
            type="info", content="Memorizing succesful solutions...", temp=True
        )

        # show full util message, this will hide temp message immediately if turned on
        log_item = self.agent.context.log.log(
            type="util",
            heading="Memorizing succesful solutions...",
        )

        #memorize in background
        asyncio.create_task(self.memorize(loop_data, log_item))        

    async def memorize(self, loop_data: LoopData, log_item: LogItem, **kwargs):
        # get system message and chat history for util llm
        system = self.agent.read_prompt("memory.solutions_sum.sys.md")
        msgs_text = self.agent.concat_messages(self.agent.history)

        # log query streamed by LLM
        async def log_callback(content):
            log_item.stream(content=content)

        # call util llm to find solutions in history
        solutions_json = await self.agent.call_utility_model(
            system=system,
            message=msgs_text,
            callback=log_callback,
            background=True,
        )

        solutions = DirtyJson.parse_string(solutions_json)

        if not isinstance(solutions, list) or len(solutions) == 0:
            log_item.update(heading="No successful solutions to memorize.")
            return
        else:
            log_item.update(
                heading=f"{len(solutions)} successful solutions to memorize."
            )

        # save chat history
        db = await Memory.get(self.agent)

        solutions_txt = ""
        rem = []
        for solution in solutions:
            # solution to plain text:
            txt = f"# Problem\n {solution['problem']}\n# Solution\n {solution['solution']}"
            solutions_txt += txt + "\n\n"

            # remove previous solutions too similiar to this one
            if self.REPLACE_THRESHOLD > 0:
                rem += await db.delete_documents_by_query(
                    query=txt,
                    threshold=self.REPLACE_THRESHOLD,
                    filter=f"area=='{Memory.Area.SOLUTIONS.value}'",
                )
                if rem:
                    rem_txt = "\n\n".join(Memory.format_docs_plain(rem))
                    log_item.update(replaced=rem_txt)

            # insert new solution
            await db.insert_text(text=txt, metadata={"area": Memory.Area.SOLUTIONS.value})

        solutions_txt = solutions_txt.strip()
        log_item.update(solutions=solutions_txt)
        log_item.update(
            result=f"{len(solutions)} solutions memorized.",
            heading=f"{len(solutions)} solutions memorized.",
        )
        if rem:
            log_item.stream(result=f"\nReplaced {len(rem)} previous solutions.")

    # except Exception as e:
    #     err = errors.format_error(e)
    #     self.agent.context.log.log(
    #         type="error", heading="Memorize solutions extension error:", content=err
    #     )



================================================
File: python/extensions/monologue_end/_90_waiting_for_input_msg.py
================================================
from python.helpers.extension import Extension
from agent import LoopData

class WaitingForInputMsg(Extension):

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        # show temp info message
        if self.agent.number == 0:
            self.agent.context.log.set_initial_progress()




================================================
File: python/extensions/monologue_end/.gitkeep
================================================



================================================
File: python/extensions/monologue_start/_60_rename_chat.py
================================================
from python.helpers import persist_chat, tokens
from python.helpers.extension import Extension
from agent import LoopData
import asyncio


class RenameChat(Extension):

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        asyncio.create_task(self.change_name())

    async def change_name(self):
        try:
            # prepare history
            history_text = self.agent.history.output_text()
            ctx_length = int(self.agent.config.utility_model.ctx_length * 0.3)
            history_text = tokens.trim_to_tokens(history_text, ctx_length, "start")
            # prepare system and user prompt
            system = self.agent.read_prompt("fw.rename_chat.sys.md")
            current_name = self.agent.context.name
            message = self.agent.read_prompt(
                "fw.rename_chat.msg.md", current_name=current_name, history=history_text
            )
            # call utility model
            new_name = await self.agent.call_utility_model(
                system=system, message=message, background=True
            )
            # update name
            if new_name:
                # trim name to max length if needed
                if len(new_name) > 40:
                    new_name = new_name[:40] + "..."
                # apply to context and save
                self.agent.context.name = new_name
                persist_chat.save_tmp_chat(self.agent.context)
        except Exception as e:
            pass  # non-critical



================================================
File: python/extensions/monologue_start/.gitkeep
================================================



================================================
File: python/extensions/system_prompt/_10_system_prompt.py
================================================
from datetime import datetime, timezone
from python.helpers.extension import Extension
from agent import Agent, LoopData
from python.helpers.localization import Localization


class SystemPrompt(Extension):

    async def execute(self, system_prompt: list[str]=[], loop_data: LoopData = LoopData(), **kwargs):
        # append main system prompt and tools
        main = get_main_prompt(self.agent)
        tools = get_tools_prompt(self.agent)
        system_prompt.append(main)
        system_prompt.append(tools)


def get_main_prompt(agent: Agent):
    return agent.read_prompt("agent.system.main.md")


def get_tools_prompt(agent: Agent):
    prompt = agent.read_prompt("agent.system.tools.md")
    if agent.config.chat_model.vision:
        prompt += '\n' + agent.read_prompt("agent.system.tools_vision.md")
    return prompt


================================================
File: python/extensions/system_prompt/_20_behaviour_prompt.py
================================================
from datetime import datetime
from python.helpers.extension import Extension
from agent import Agent, LoopData
from python.helpers import files, memory


class BehaviourPrompt(Extension):

    async def execute(self, system_prompt: list[str]=[], loop_data: LoopData = LoopData(), **kwargs):
        prompt = read_rules(self.agent)
        system_prompt.insert(0, prompt) #.append(prompt)

def get_custom_rules_file(agent: Agent):
    return memory.get_memory_subdir_abs(agent) + f"/behaviour.md"

def read_rules(agent: Agent):
    rules_file = get_custom_rules_file(agent)
    if files.exists(rules_file):
        rules = files.read_file(rules_file)
        return agent.read_prompt("agent.system.behaviour.md", rules=rules)
    else:
        rules = agent.read_prompt("agent.system.behaviour_default.md")
        return agent.read_prompt("agent.system.behaviour.md", rules=rules)
  


================================================
File: python/extensions/system_prompt/.gitkeep
================================================



================================================
File: python/helpers/api.py
================================================
from abc import abstractmethod
import json
import threading
from typing import Union, TypedDict, Dict, Any
from attr import dataclass
from flask import Request, Response, jsonify, Flask
from agent import AgentContext
from initialize import initialize
from python.helpers.print_style import PrintStyle
from python.helpers.errors import format_error
from werkzeug.serving import make_server


Input = dict
Output = Union[Dict[str, Any], Response, TypedDict]  # type: ignore


class ApiHandler:
    def __init__(self, app: Flask, thread_lock: threading.Lock):
        self.app = app
        self.thread_lock = thread_lock

    @classmethod
    def requires_loopback(cls) -> bool:
        return False

    @classmethod
    def requires_api_key(cls) -> bool:
        return False

    @classmethod
    def requires_auth(cls) -> bool:
        return True

    @abstractmethod
    async def process(self, input: Input, request: Request) -> Output:
        pass

    async def handle_request(self, request: Request) -> Response:
        try:
            # input data from request based on type
            input_data: Input = {}
            if request.is_json:
                try:
                    if request.data:  # Check if there's any data
                        input_data = request.get_json()
                    # If empty or not valid JSON, use empty dict
                except Exception as e:
                    # Just log the error and continue with empty input
                    PrintStyle().print(f"Error parsing JSON: {str(e)}")
                    input_data = {}
            else:
                input_data = {"data": request.get_data(as_text=True)}

            # process via handler
            output = await self.process(input_data, request)

            # return output based on type
            if isinstance(output, Response):
                return output
            else:
                response_json = json.dumps(output)
                return Response(
                    response=response_json, status=200, mimetype="application/json"
                )

            # return exceptions with 500
        except Exception as e:
            error = format_error(e)
            PrintStyle.error(f"API error: {error}")
            return Response(response=error, status=500, mimetype="text/plain")

    # get context to run agent zero in
    def get_context(self, ctxid: str):
        with self.thread_lock:
            if not ctxid:
                first = AgentContext.first()
                if first:
                    return first
                return AgentContext(config=initialize())
            got = AgentContext.get(ctxid)
            if got:
                return got
            return AgentContext(config=initialize(), id=ctxid)



================================================
File: python/helpers/attachment_manager.py
================================================
import os
import io
import base64
from PIL import Image
from typing import Dict, List, Optional, Tuple
from werkzeug.utils import secure_filename

from python.helpers.print_style import PrintStyle

class AttachmentManager:
  ALLOWED_EXTENSIONS = {
      'image': {'jpg', 'jpeg', 'png', 'bmp'},
      'code': {'py', 'js', 'sh', 'html', 'css'},
      'document': {'md', 'pdf', 'txt', 'csv', 'json'}
  }
  
  def __init__(self, work_dir: str):
      self.work_dir = work_dir
      os.makedirs(work_dir, exist_ok=True)

  def is_allowed_file(self, filename: str) -> bool:
      ext = self.get_file_extension(filename)
      all_allowed = set().union(*self.ALLOWED_EXTENSIONS.values())
      return ext in all_allowed

  def get_file_type(self, filename: str) -> str:
      ext = self.get_file_extension(filename)
      for file_type, extensions in self.ALLOWED_EXTENSIONS.items():
          if ext in extensions:
              return file_type
      return 'unknown'

  @staticmethod
  def get_file_extension(filename: str) -> str:
      return filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
  
  def validate_mime_type(self, file) -> bool:
      try:
          mime_type = file.content_type
          return mime_type.split('/')[0] in ['image', 'text', 'application']
      except AttributeError:
          return False

  def save_file(self, file, filename: str) -> Tuple[str, Dict]:
      """Save file and return path and metadata"""
      try:
          filename = secure_filename(filename)
          if not filename:
              raise ValueError("Invalid filename")
              
          file_path = os.path.join(self.work_dir, filename)
          
          file_type = self.get_file_type(filename)
          metadata = {
              'filename': filename,
              'type': file_type,
              'extension': self.get_file_extension(filename),
              'preview': None
          }
  
          # Save file
          file.save(file_path)
  
          # Generate preview for images
          if file_type == 'image':
              metadata['preview'] = self.generate_image_preview(file_path)
  
          return file_path, metadata
        
      except Exception as e:
          PrintStyle.error(f"Error saving file {filename}: {e}")
          return None, {} # type: ignore

  def generate_image_preview(self, image_path: str, max_size: int = 800) -> Optional[str]:
      try:
          with Image.open(image_path) as img:
              # Convert image if needed
              if img.mode in ('RGBA', 'P'):
                  img = img.convert('RGB')
              
              # Resize for preview
              img.thumbnail((max_size, max_size))
              
              # Save to buffer
              buffer = io.BytesIO()
              img.save(buffer, format="JPEG", quality=70, optimize=True)
              
              # Convert to base64
              return base64.b64encode(buffer.getvalue()).decode('utf-8')
      except Exception as e:
          PrintStyle.error(f"Error generating preview for {image_path}: {e}")
          return None
      


================================================
File: python/helpers/browser.py
================================================
import asyncio
import re
from bs4 import BeautifulSoup
from playwright.async_api import (
    async_playwright,
    Browser as PlaywrightBrowser,
    Page,
    Frame,
    BrowserContext,
)

from python.helpers import files


class NoPageError(Exception):
    pass


class Browser:

    load_timeout = 10000
    interact_timeout = 3000
    selector_name = "data-a0sel3ct0r"

    def __init__(self, headless=True):
        self.browser: PlaywrightBrowser = None  # type: ignore
        self.context: BrowserContext = None  # type: ignore
        self.page: Page = None  # type: ignore
        self._playwright = None
        self.headless = headless
        self.contexts = {}
        self.last_selector = ""
        self.page_loaded = False
        self.navigation_count = 0

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def start(self):
        """Start browser session"""
        self._playwright = await async_playwright().start()
        if not self.browser:
            self.browser = await self._playwright.chromium.launch(
                headless=self.headless, args=["--disable-http2"]
            )
        if not self.context:
            self.context = await self.browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.6422.141 Safari/537.36"
            )

        self.page = await self.context.new_page()
        await self.page.set_viewport_size({"width": 1200, "height": 1200})

        # Inject the JavaScript to modify the attachShadow method
        js_override = files.read_file("lib/browser/init_override.js")
        await self.page.add_init_script(js_override)

        # Setup frame handling
        async def inject_script_into_frames(frame):
            try:
                await self.wait_tick()
                if not frame.is_detached():
                    async with asyncio.timeout(0.25):
                        await frame.evaluate(js_override)
                        print(f"Injected script into frame: {frame.url[:100]}")
            except Exception as e:
                # Frame might have been detached during injection, which is normal
                print(
                    f"Could not inject into frame (possibly detached): {str(e)[:100]}"
                )

        self.page.on(
            "frameattached",
            lambda frame: asyncio.ensure_future(inject_script_into_frames(frame)),
        )

        # Handle page navigation events
        async def handle_navigation(frame):
            if frame == self.page.main_frame:
                print(f"Page navigated to: {frame.url[:100]}")
                self.page_loaded = False
                self.navigation_count += 1

        async def handle_load(dummy):
            print("Page load completed")
            self.page_loaded = True

        async def handle_request(request):
            if (
                request.is_navigation_request()
                and request.frame == self.page.main_frame
            ):
                print(f"Navigation started to: {request.url[:100]}")
                self.page_loaded = False
                self.navigation_count += 1

        self.page.on("request", handle_request)
        self.page.on("framenavigated", handle_navigation)
        self.page.on("load", handle_load)

    async def close(self):
        """Close browser session"""
        if self.browser:
            await self.browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def open(self, url: str):
        """Open a URL in the browser"""
        self.last_selector = ""
        self.contexts = {}
        if self.page:
            await self.page.close()
        await self.start()
        try:
            await self.page.goto(
                url, wait_until="networkidle", timeout=Browser.load_timeout
            )
        except TimeoutError as e:
            pass
        except Exception as e:
            print(f"Error opening page: {e}")
            raise e
        await self.wait_tick()

    async def get_full_dom(self) -> str:
        """Get full DOM with unique selectors"""
        await self._check_page()
        js_code = files.read_file("lib/browser/extract_dom.js")

        # Get all frames
        self.contexts = {}
        frame_contents = {}

        # Extract content from each frame
        i = -1
        for frame in self.page.frames:
            try:
                if frame.url:  # and frame != self.page.main_frame:
                    i += 1
                    frame_mark = self._num_to_alpha(i)

                    # Check if frame is still valid
                    await self.wait_tick()
                    if not frame.is_detached():
                        try:
                            # short timeout to identify and skip unresponsive frames
                            async with asyncio.timeout(0.25):
                                await frame.evaluate("window.location.href")
                        except TimeoutError as e:
                            print(f"Skipping unresponsive frame: {frame.url}")
                            continue

                        await frame.wait_for_load_state(
                            "domcontentloaded", timeout=1000
                        )

                        async with asyncio.timeout(1):
                            content = await frame.evaluate(
                                js_code, [frame_mark, self.selector_name]
                            )
                            self.contexts[frame_mark] = frame
                            frame_contents[frame.url] = content
                    else:
                        print(f"Warning: Frame was detached: {frame.url}")
            except Exception as e:
                print(f"Error extracting from frame {frame.url}: {e}")

        # # Get main frame content
        # main_mark = self._num_to_alpha(0)
        # main_content = ""
        # try:
        #     async with asyncio.timeout(1):
        #         main_content = await self.page.evaluate(js_code, [main_mark, self.selector_name])
        #         self.contexts[main_mark] = self.page
        # except Exception as e:
        #     print(f"Error when extracting from main frame: {e}")

        # Replace iframe placeholders with actual content
        # for url, content in frame_contents.items():
        #     placeholder = f'<iframe src="{url}"'
        #     main_content = main_content.replace(placeholder, f'{placeholder}>\n<!-- IFrame Content Start -->\n{content}\n<!-- IFrame Content End -->\n</iframe')

        # return main_content + "".join(frame_contents.values())
        return "".join(frame_contents.values())

    def strip_html_dom(self, html_content: str) -> str:
        """Clean and strip HTML content"""
        if not html_content:
            return ""

        soup = BeautifulSoup(html_content, "html.parser")

        for tag in soup.find_all(
            ["br", "hr", "style", "script", "noscript", "meta", "link", "svg"]
        ):
            tag.decompose()

        for tag in soup.find_all(True):
            if tag.attrs and "invisible" in tag.attrs:
                tag.decompose()

        for tag in soup.find_all(True):
            allowed_attrs = [
                self.selector_name,
                "aria-label",
                "placeholder",
                "name",
                "value",
                "type",
            ]
            attrs = {
                "selector" if key == self.selector_name else key: tag.attrs[key]
                for key in allowed_attrs
                if key in tag.attrs and tag.attrs[key]
            }
            tag.attrs = attrs

        def remove_empty(tag_name: str) -> None:
            for tag in soup.find_all(tag_name):
                if not tag.attrs:
                    tag.unwrap()

        remove_empty("span")
        remove_empty("p")
        remove_empty("strong")

        return soup.prettify(formatter="minimal")

    def process_html_with_selectors(self, html_content: str) -> str:
        """Process HTML content and add selectors to interactive elements"""
        if not html_content:
            return ""

        html_content = re.sub(r"\s+", " ", html_content)
        soup = BeautifulSoup(html_content, "html.parser")

        structural_tags = [
            "html",
            "head",
            "body",
            "div",
            "span",
            "section",
            "main",
            "article",
            "header",
            "footer",
            "nav",
            "ul",
            "ol",
            "li",
            "tr",
            "td",
            "th",
        ]
        for tag in structural_tags:
            for element in soup.find_all(tag):
                element.unwrap()

        out = str(soup).strip()
        out = re.sub(r">\s*<", "><", out)
        out = re.sub(r'aria-label="', 'label="', out)

        # out = re.sub(r'selector="(\d+[a-zA-Z]+)"', r'selector=\1', out)
        return out

    async def get_clean_dom(self) -> str:
        """Get clean DOM with selectors"""
        full_dom = await self.get_full_dom()
        clean_dom = self.strip_html_dom(full_dom)
        return self.process_html_with_selectors(clean_dom)

    async def click(self, selector: str):
        await self._check_page()
        ctx, selector = self._parse_selector(selector)
        self.last_selector = selector
        # js_code = files.read_file("lib/browser/click.js")
        # result = await self.page.evaluate(js_code, [selector])
        # if not result:
        result = await ctx.hover(selector, force=True, timeout=Browser.interact_timeout)
        await self.wait_tick()
        result = await ctx.click(selector, force=True, timeout=Browser.interact_timeout)
        await self.wait_tick()

        # await self.page.wait_for_load_state("networkidle")
        return result

    async def press(self, key: str):
        await self._check_page()
        if self.last_selector:
            await self.page.press(
                self.last_selector, key, timeout=Browser.interact_timeout
            )
        else:
            await self.page.keyboard.press(key)

    async def fill(self, selector: str, text: str):
        await self._check_page()
        ctx, selector = self._parse_selector(selector)
        self.last_selector = selector
        try:
            await self.click(selector)
        except Exception as e:
            pass
        await ctx.fill(selector, text, force=True, timeout=Browser.interact_timeout)
        await self.wait_tick()

    async def execute(self, js_code: str):
        await self._check_page()
        result = await self.page.evaluate(js_code)
        return result

    async def screenshot(self, path: str, full_page=False):
        await self._check_page()
        await self.page.screenshot(path=path, full_page=full_page)

    def _parse_selector(self, selector: str) -> tuple[Page | Frame, str]:
        try:
            ctx = self.page
            # Check if selector is our UID, return
            if re.match(r"^\d+[a-zA-Z]+$", selector):
                alpha_part = "".join(filter(str.isalpha, selector))
                ctx = self.contexts[alpha_part]
                selector = f"[{self.selector_name}='{selector}']"
            return (ctx, selector)
        except Exception as e:
            raise Exception(f"Error evaluating selector: {selector}")

    async def _check_page(self):
        for _ in range(2):
            try:
                await self.wait_tick()
                self.page = self.context.pages[0]
                if not self.page:
                    raise NoPageError(
                        "No page is open in the browser. Please open a URL first."
                    )
                # await self.page.wait_for_load_state("networkidle",)
                async with asyncio.timeout(self.load_timeout / 1000):
                    if not self.page_loaded:
                        while not self.page_loaded:
                            await asyncio.sleep(0.1)
                        await self.wait_tick()
                return
            except TimeoutError as e:
                self.page_loaded = True
                return
            except NoPageError as e:
                raise e
            except Exception as e:
                print(f"Error checking page: {e}")

    def _num_to_alpha(self, num: int) -> str:
        if num < 0:
            return ""

        result = ""
        while num >= 0:
            result = chr(num % 26 + 97) + result
            num = num // 26 - 1

        return result

    async def wait_tick(self):
        if self.page:
            await self.page.evaluate("window.location.href")

    async def wait(self, seconds: float = 1.0):
        await asyncio.sleep(seconds)
        await self.wait_tick()

    async def wait_for_action(self):
        nav_count = self.navigation_count
        for _ in range(5):
            await self._check_page()
            if nav_count != self.navigation_count:
                print("Navigation detected")
                await asyncio.sleep(1)
                return
            await asyncio.sleep(0.1)



================================================
File: python/helpers/browser_use.py
================================================
from python.helpers import dotenv
dotenv.save_dotenv_value("ANONYMIZED_TELEMETRY", "false")
import browser_use
import browser_use.utils


================================================
File: python/helpers/call_llm.py
================================================
from typing import Callable, TypedDict
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

from langchain.schema import AIMessage
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM


class Example(TypedDict):
    input: str
    output: str

async def call_llm(
    system: str,
    model: BaseChatModel | BaseLLM,
    message: str,
    examples: list[Example] = [],
    callback: Callable[[str], None] | None = None
):

    example_prompt = ChatPromptTemplate.from_messages(
        [
            HumanMessage(content="{input}"),
            AIMessage(content="{output}"),
        ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,  # type: ignore
        input_variables=[],
    )

    few_shot_prompt.format()


    final_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system),
            few_shot_prompt,
            HumanMessage(content=message),
        ]
    )

    chain = final_prompt | model

    response = ""
    async for chunk in chain.astream({}):
        # await self.handle_intervention()  # wait for intervention and handle it, if paused

        if isinstance(chunk, str):
            content = chunk
        elif hasattr(chunk, "content"):
            content = str(chunk.content)
        else:
            content = str(chunk)

        if callback:
            callback(content)

        response += content

    return response




================================================
File: python/helpers/cloudflare_tunnel.py
================================================
import os
import platform
import requests
import subprocess
import threading
from python.helpers import files
from python.helpers.print_style import PrintStyle

class CloudflareTunnel:
    def __init__(self, port: int):
        self.port = port
        self.bin_dir = "tmp"  # Relative path
        self.cloudflared_path = None
        self.tunnel_process = None
        self.tunnel_url = None
        self._stop_event = threading.Event()
        
    def download_cloudflared(self):
        """Downloads the appropriate cloudflared binary for the current system"""
        # Create bin directory if it doesn't exist using files helper
        os.makedirs(files.get_abs_path(self.bin_dir), exist_ok=True)
        
        # Determine OS and architecture
        system = platform.system().lower()
        arch = platform.machine().lower()
        
        # Define executable name
        executable_name = "cloudflared.exe" if system == "windows" else "cloudflared"
        install_path = files.get_abs_path(self.bin_dir, executable_name)
        
        # Return if already exists
        if files.exists(self.bin_dir, executable_name):
            self.cloudflared_path = install_path
            return install_path
            
        # Map platform/arch to download URLs
        base_url = "https://github.com/cloudflare/cloudflared/releases/latest/download/"
        
        if system == "darwin":  # macOS
            # Download and extract .tgz for macOS
            download_file = "cloudflared-darwin-amd64.tgz" if arch == "x86_64" else "cloudflared-darwin-arm64.tgz"
            download_url = f"{base_url}{download_file}"
            download_path = files.get_abs_path(self.bin_dir, download_file)
            
            PrintStyle().print(f"\nDownloading cloudflared from: {download_url}")
            response = requests.get(download_url, stream=True)
            if response.status_code != 200:
                raise RuntimeError(f"Failed to download cloudflared: {response.status_code}")
                
            # Save the .tgz file
            with open(download_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            # Extract cloudflared binary from .tgz
            import tarfile
            with tarfile.open(download_path, "r:gz") as tar:
                tar.extract("cloudflared", files.get_abs_path(self.bin_dir))
                
            # Cleanup .tgz file
            os.remove(download_path)
            
        else:  # Linux and Windows
            if system == "linux":
                if arch in ["x86_64", "amd64"]:
                    download_file = "cloudflared-linux-amd64"
                elif arch == "arm64" or arch == "aarch64":
                    download_file = "cloudflared-linux-arm64"
                elif arch == "arm":
                    download_file = "cloudflared-linux-arm"
                else:
                    download_file = "cloudflared-linux-386"
            elif system == "windows":
                download_file = "cloudflared-windows-amd64.exe"
            else:
                raise RuntimeError(f"Unsupported platform: {system} {arch}")
                
            download_url = f"{base_url}{download_file}"
            download_path = files.get_abs_path(self.bin_dir, download_file)
            
            PrintStyle().print(f"\nDownloading cloudflared from: {download_url}")
            response = requests.get(download_url, stream=True)
            if response.status_code != 200:
                raise RuntimeError(f"Failed to download cloudflared: {response.status_code}")
                
            with open(download_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            
            # Rename and set permissions
            if os.path.exists(install_path):
                os.remove(install_path)
            os.rename(download_path, install_path)
        
        # Set executable permissions
        if system != "windows":
            os.chmod(install_path, 0o755)
            
        self.cloudflared_path = install_path
        return install_path

    def _extract_tunnel_url(self, process):
        """Extracts the tunnel URL from cloudflared output"""
        while not self._stop_event.is_set():
            line = process.stdout.readline()
            if not line:
                break
                
            if isinstance(line, bytes):
                line = line.decode('utf-8')
                
            if "trycloudflare.com" in line and "https://" in line:
                start = line.find("https://")
                end = line.find("trycloudflare.com") + len("trycloudflare.com")
                self.tunnel_url = line[start:end].strip()
                PrintStyle().print("\n=== Cloudflare Tunnel URL ===")
                PrintStyle().print(f"URL: {self.tunnel_url}")
                PrintStyle().print("============================\n")
                return

    def start(self):
        """Starts the cloudflare tunnel"""
        if not self.cloudflared_path:
            self.download_cloudflared()
            
        PrintStyle().print("\nStarting Cloudflare tunnel...")
        # Start tunnel process
        self.tunnel_process = subprocess.Popen(
            [
                str(self.cloudflared_path),
                "tunnel", 
                "--url",
                f"http://localhost:{self.port}"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True
        )
        
        # Extract tunnel URL in separate thread
        threading.Thread(
            target=self._extract_tunnel_url,
            args=(self.tunnel_process,),
            daemon=True
        ).start()

    def stop(self):
        """Stops the cloudflare tunnel"""
        self._stop_event.set()
        if self.tunnel_process:
            PrintStyle().print("\nStopping Cloudflare tunnel...")
            self.tunnel_process.terminate()
            self.tunnel_process.wait()
            self.tunnel_process = None
            self.tunnel_url = None


================================================
File: python/helpers/crypto.py
================================================
import hashlib
import hmac
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
import os


def hash_data(data: str, password: str):
    return hmac.new(password.encode(), data.encode(), hashlib.sha256).hexdigest()


def verify_data(data: str, hash: str, password: str):
    return hash_data(data, password) == hash


def _generate_private_key():
    return rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )


def _generate_public_key(private_key: rsa.RSAPrivateKey):
    return (
        private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .hex()
    )
    
def _decode_public_key(public_key: str) -> rsa.RSAPublicKey:
    # Decode hex string back to bytes
    pem_bytes = bytes.fromhex(public_key)
    # Load the PEM public key
    key = serialization.load_pem_public_key(pem_bytes)
    if not isinstance(key, rsa.RSAPublicKey):
        raise TypeError("The provided key is not an RSAPublicKey")
    return key

def encrypt_data(data: str, public_key_pem: str):
    return _encrypt_data(data.encode("utf-8"), _decode_public_key(public_key_pem))

def _encrypt_data(data: bytes, public_key: rsa.RSAPublicKey):
    b = public_key.encrypt(
        data,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return b.hex()

def decrypt_data(data: str, private_key: rsa.RSAPrivateKey):
    b = private_key.decrypt(
        bytes.fromhex(data),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return b.decode("utf-8")




================================================
File: python/helpers/defer.py
================================================
import asyncio
from dataclasses import dataclass
import threading
from concurrent.futures import Future
from typing import Any, Callable, Optional, Coroutine, TypeVar, Awaitable

T = TypeVar("T")

class EventLoopThread:
    _instances = {}
    _lock = threading.Lock()

    def __init__(self, thread_name: str = "default") -> None:
        """Initialize the event loop thread."""
        self.thread_name = thread_name
        self._start()

    def __new__(cls, thread_name: str = "default"):
        with cls._lock:
            if thread_name not in cls._instances:
                instance = super(EventLoopThread, cls).__new__(cls)
                cls._instances[thread_name] = instance
            return cls._instances[thread_name]

    def _start(self):
        if not hasattr(self, "loop") or not self.loop:
            self.loop = asyncio.new_event_loop()
        if not hasattr(self, "thread") or not self.thread:
            self.thread = threading.Thread(
                target=self._run_event_loop, daemon=True, name=self.thread_name
            )
            self.thread.start()

    def _run_event_loop(self):
        if not self.loop:
            raise RuntimeError("Event loop is not initialized")
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def terminate(self):
        if self.loop and self.loop.is_running():
            self.loop.stop()
        self.loop = None
        self.thread = None

    def run_coroutine(self, coro):
        self._start()
        if not self.loop:
            raise RuntimeError("Event loop is not initialized")
        return asyncio.run_coroutine_threadsafe(coro, self.loop)


@dataclass
class ChildTask:
    task: "DeferredTask"
    terminate_thread: bool


class DeferredTask:
    def __init__(
        self,
        thread_name: str = "default",
    ):
        self.event_loop_thread = EventLoopThread(thread_name)
        self._future: Optional[Future] = None
        self.children: list[ChildTask] = []

    def start_task(
        self, func: Callable[..., Coroutine[Any, Any, Any]], *args: Any, **kwargs: Any
    ):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._start_task()

    def __del__(self):
        self.kill()

    def _start_task(self):
        self._future = self.event_loop_thread.run_coroutine(self._run())

    async def _run(self):
        return await self.func(*self.args, **self.kwargs)

    def is_ready(self) -> bool:
        return self._future.done() if self._future else False

    def result_sync(self, timeout: Optional[float] = None) -> Any:
        if not self._future:
            raise RuntimeError("Task hasn't been started")
        try:
            return self._future.result(timeout)
        except TimeoutError:
            raise TimeoutError(
                "The task did not complete within the specified timeout."
            )

    async def result(self, timeout: Optional[float] = None) -> Any:
        if not self._future:
            raise RuntimeError("Task hasn't been started")

        loop = asyncio.get_running_loop()

        def _get_result():
            try:
                result = self._future.result(timeout)  # type: ignore
                # self.kill()
                return result
            except TimeoutError:
                raise TimeoutError(
                    "The task did not complete within the specified timeout."
                )

        return await loop.run_in_executor(None, _get_result)

    def kill(self, terminate_thread: bool = False) -> None:
        """Kill the task and optionally terminate its thread."""
        self.kill_children()
        if self._future and not self._future.done():
            self._future.cancel()

        if (
            terminate_thread
            and self.event_loop_thread.loop
            and self.event_loop_thread.loop.is_running()
        ):

            def cleanup():
                tasks = [
                    t
                    for t in asyncio.all_tasks(self.event_loop_thread.loop)
                    if t is not asyncio.current_task(self.event_loop_thread.loop)
                ]
                for task in tasks:
                    task.cancel()
                    try:
                        # Give tasks a chance to cleanup
                        if self.event_loop_thread.loop:
                            self.event_loop_thread.loop.run_until_complete(
                                asyncio.gather(task, return_exceptions=True)
                            )
                    except Exception:
                        pass  # Ignore cleanup errors

            self.event_loop_thread.loop.call_soon_threadsafe(cleanup)
            self.event_loop_thread.terminate()

    def kill_children(self) -> None:
        for child in self.children:
            child.task.kill(terminate_thread=child.terminate_thread)
        self.children = []

    def is_alive(self) -> bool:
        return self._future and not self._future.done()  # type: ignore

    def restart(self, terminate_thread: bool = False) -> None:
        self.kill(terminate_thread=terminate_thread)
        self._start_task()

    def add_child_task(
        self, task: "DeferredTask", terminate_thread: bool = False
    ) -> None:
        self.children.append(ChildTask(task, terminate_thread))

    async def _execute_in_task_context(
        self, func: Callable[..., T], *args, **kwargs
    ) -> T:
        """Execute a function in the task's context and return its result."""
        result = func(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    def execute_inside(self, func: Callable[..., T], *args, **kwargs) -> Awaitable[T]:
        if not self.event_loop_thread.loop:
            raise RuntimeError("Event loop is not initialized")

        future: Future = Future()

        async def wrapped():
            if not self.event_loop_thread.loop:
                raise RuntimeError("Event loop is not initialized")
            try:
                result = await self._execute_in_task_context(func, *args, **kwargs)
                # Keep awaiting until we get a concrete value
                while isinstance(result, Awaitable):
                    result = await result
                self.event_loop_thread.loop.call_soon_threadsafe(
                    future.set_result, result
                )
            except Exception as e:
                self.event_loop_thread.loop.call_soon_threadsafe(
                    future.set_exception, e
                )

        asyncio.run_coroutine_threadsafe(wrapped(), self.event_loop_thread.loop)
        return asyncio.wrap_future(future)



================================================
File: python/helpers/dirty_json.py
================================================
import json

def try_parse(json_string: str):
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        return DirtyJson.parse_string(json_string)


def parse(json_string: str):
    return DirtyJson.parse_string(json_string)


def stringify(obj, **kwargs):
    return json.dumps(obj, ensure_ascii=False, **kwargs)


class DirtyJson:
    def __init__(self):
        self._reset()

    def _reset(self):
        self.json_string = ""
        self.index = 0
        self.current_char = None
        self.result = None
        self.stack = []

    @staticmethod
    def parse_string(json_string):
        parser = DirtyJson()
        return parser.parse(json_string)

    def parse(self, json_string):
        self._reset()
        self.json_string = json_string
        self.index = self.get_start_pos(
            self.json_string
        )  # skip any text up to the first brace
        self.current_char = self.json_string[self.index]
        self._parse()
        return self.result

    def feed(self, chunk):
        self.json_string += chunk
        if not self.current_char and self.json_string:
            self.current_char = self.json_string[0]
        self._parse()
        return self.result

    def _advance(self, count=1):
        self.index += count
        if self.index < len(self.json_string):
            self.current_char = self.json_string[self.index]
        else:
            self.current_char = None

    def _skip_whitespace(self):
        while self.current_char is not None:
            if self.current_char.isspace():
                self._advance()
            elif (
                self.current_char == "/" and self._peek(1) == "/"
            ):  # Single-line comment
                self._skip_single_line_comment()
            elif (
                self.current_char == "/" and self._peek(1) == "*"
            ):  # Multi-line comment
                self._skip_multi_line_comment()
            else:
                break

    def _skip_single_line_comment(self):
        while self.current_char is not None and self.current_char != "\n":
            self._advance()
        if self.current_char == "\n":
            self._advance()

    def _skip_multi_line_comment(self):
        self._advance(2)  # Skip /*
        while self.current_char is not None:
            if self.current_char == "*" and self._peek(1) == "/":
                self._advance(2)  # Skip */
                break
            self._advance()

    def _parse(self):
        if self.result is None:
            self.result = self._parse_value()
        else:
            self._continue_parsing()

    def _continue_parsing(self):
        while self.current_char is not None:
            if isinstance(self.result, dict):
                self._parse_object_content()
            elif isinstance(self.result, list):
                self._parse_array_content()
            elif isinstance(self.result, str):
                self.result = self._parse_string()
            else:
                break

    def _parse_value(self):
        self._skip_whitespace()
        if self.current_char == "{":
            if self._peek(1) == "{":  # Handle {{
                self._advance(2)
            return self._parse_object()
        elif self.current_char == "[":
            return self._parse_array()
        elif self.current_char in ['"', "'", "`"]:
            if self._peek(2) == self.current_char * 2:  # type: ignore
                return self._parse_multiline_string()
            return self._parse_string()
        elif self.current_char and (
            self.current_char.isdigit() or self.current_char in ["-", "+"]
        ):
            return self._parse_number()
        elif self._match("true"):
            return True
        elif self._match("false"):
            return False
        elif self._match("null") or self._match("undefined"):
            return None
        elif self.current_char:
            return self._parse_unquoted_string()
        return None

    def _match(self, text: str) -> bool:
        # first char should match current char
        if not self.current_char or self.current_char.lower() != text[0].lower():
            return False

        # peek remaining chars
        remaining = len(text) - 1
        if self._peek(remaining).lower() == text[1:].lower():
            self._advance(len(text))
            return True
        return False

    def _parse_object(self):
        obj = {}
        self._advance()  # Skip opening brace
        self.stack.append(obj)
        self._parse_object_content()
        return obj

    def _parse_object_content(self):
        while self.current_char is not None:
            self._skip_whitespace()
            if self.current_char == "}":
                if self._peek(1) == "}":  # Handle }}
                    self._advance(2)
                else:
                    self._advance()
                self.stack.pop()
                return
            if self.current_char is None:
                self.stack.pop()
                return  # End of input reached while parsing object

            key = self._parse_key()
            value = None
            self._skip_whitespace()

            if self.current_char == ":":
                self._advance()
                value = self._parse_value()
            elif self.current_char is None:
                value = None  # End of input reached after key
            else:
                value = self._parse_value()

            self.stack[-1][key] = value

            self._skip_whitespace()
            if self.current_char == ",":
                self._advance()
                continue
            elif self.current_char != "}":
                if self.current_char is None:
                    self.stack.pop()
                    return  # End of input reached after value
                continue

    def _parse_key(self):
        self._skip_whitespace()
        if self.current_char in ['"', "'"]:
            return self._parse_string()
        else:
            return self._parse_unquoted_key()

    def _parse_unquoted_key(self):
        result = ""
        while (
            self.current_char is not None
            and not self.current_char.isspace()
            and self.current_char not in [":", ",", "}", "]"]
        ):
            result += self.current_char
            self._advance()
        return result

    def _parse_array(self):
        arr = []
        self._advance()  # Skip opening bracket
        self.stack.append(arr)
        self._parse_array_content()
        return arr

    def _parse_array_content(self):
        while self.current_char is not None:
            self._skip_whitespace()
            if self.current_char == "]":
                self._advance()
                self.stack.pop()
                return
            value = self._parse_value()
            self.stack[-1].append(value)
            self._skip_whitespace()
            if self.current_char == ",":
                self._advance()
                # handle trailing commas, end of array
                self._skip_whitespace()
                if self.current_char is None or self.current_char == "]":
                    if self.current_char == "]":
                        self._advance()
                    self.stack.pop()
                    return
            elif self.current_char != "]":
                self.stack.pop()
                return

    def _parse_string(self):
        result = ""
        quote_char = self.current_char
        self._advance()  # Skip opening quote
        while self.current_char is not None and self.current_char != quote_char:
            if self.current_char == "\\":
                self._advance()
                if self.current_char in ['"', "'", "\\", "/", "b", "f", "n", "r", "t"]:
                    result += {
                        "b": "\b",
                        "f": "\f",
                        "n": "\n",
                        "r": "\r",
                        "t": "\t",
                    }.get(self.current_char, self.current_char)
                elif self.current_char == "u":
                    self._advance()  # Skip 'u'
                    unicode_char = ""
                    # Try to collect exactly 4 hex digits
                    for _ in range(4):
                        if self.current_char is None or not self.current_char.isalnum():
                            # If we can't get 4 hex digits, treat it as a literal '\u' followed by whatever we got
                            return result + "\\u" + unicode_char
                        unicode_char += self.current_char
                        self._advance()
                    try:
                        result += chr(int(unicode_char, 16))
                    except ValueError:
                        # If invalid hex value, treat as literal
                        result += "\\u" + unicode_char
                    continue
            else:
                result += self.current_char
            self._advance()
        if self.current_char == quote_char:
            self._advance()  # Skip closing quote
        return result

    def _parse_multiline_string(self):
        result = ""
        quote_char = self.current_char
        self._advance(3)  # Skip first quote
        while self.current_char is not None:
            if self.current_char == quote_char and self._peek(2) == quote_char * 2:  # type: ignore
                self._advance(3)  # Skip first quote
                break
            result += self.current_char
            self._advance()
        return result.strip()

    def _parse_number(self):
        number_str = ""
        while self.current_char is not None and (
            self.current_char.isdigit()
            or self.current_char in ["-", "+", ".", "e", "E"]
        ):
            number_str += self.current_char
            self._advance()
        try:
            return int(number_str)
        except ValueError:
            return float(number_str)

    def _parse_unquoted_string(self):
        result = ""
        while self.current_char is not None and self.current_char not in [
            ":",
            ",",
            "}",
            "]",
        ]:
            result += self.current_char
            self._advance()
        self._advance()
        return result.strip()

    def _peek(self, n):
        peek_index = self.index + 1
        result = ""
        for _ in range(n):
            if peek_index < len(self.json_string):
                result += self.json_string[peek_index]
                peek_index += 1
            else:
                break
        return result

    def get_start_pos(self, input_str: str) -> int:
        chars = ["{", "[", '"']
        indices = [input_str.find(char) for char in chars if input_str.find(char) != -1]
        return min(indices) if indices else 0



================================================
File: python/helpers/docker.py
================================================
import time
import docker
import atexit
from typing import Optional
from python.helpers.files import get_abs_path
from python.helpers.errors import format_error
from python.helpers.print_style import PrintStyle
from python.helpers.log import Log

class DockerContainerManager:
    def __init__(self, image: str, name: str, ports: Optional[dict[str, int]] = None, volumes: Optional[dict[str, dict[str, str]]] = None,logger: Log|None=None):
        self.logger = logger
        self.image = image
        self.name = name
        self.ports = ports
        self.volumes = volumes
        self.init_docker()
                
    def init_docker(self):
        self.client = None
        while not self.client:
            try:
                self.client = docker.from_env()
                self.container = None
            except Exception as e:
                err = format_error(e)
                if ("ConnectionRefusedError(61," in err or "Error while fetching server API version" in err):
                    PrintStyle.hint("Connection to Docker failed. Is docker or Docker Desktop running?") # hint for user
                    if self.logger:self.logger.log(type="hint", content="Connection to Docker failed. Is docker or Docker Desktop running?")
                    PrintStyle.error(err)
                    if self.logger:self.logger.log(type="error", content=err)
                    time.sleep(5) # try again in 5 seconds
                else: raise
        return self.client
                            
    def cleanup_container(self) -> None:
        if self.container:
            try:
                self.container.stop()
                self.container.remove()
                PrintStyle.standard(f"Stopped and removed the container: {self.container.id}")
                if self.logger: self.logger.log(type="info", content=f"Stopped and removed the container: {self.container.id}")
            except Exception as e:
                PrintStyle.error(f"Failed to stop and remove the container: {e}")
                if self.logger: self.logger.log(type="error", content=f"Failed to stop and remove the container: {e}")

    def get_image_containers(self):
        if not self.client: self.client = self.init_docker()
        containers = self.client.containers.list(all=True, filters={"ancestor": self.image})
        infos = []
        for container in containers:
            infos.append({                
                "id": container.id,
                "name": container.name,
                "status": container.status,
                "image": container.image,
                "ports": container.ports,
                "web_port": (container.ports.get("80/tcp") or [{}])[0].get("HostPort"),
                "ssh_port": (container.ports.get("22/tcp") or [{}])[0].get("HostPort"),
                # "volumes": container.volumes,
                # "data_folder": container.volumes["/a0"],
            })
        return infos

    def start_container(self) -> None:
        if not self.client: self.client = self.init_docker()
        existing_container = None
        for container in self.client.containers.list(all=True):
            if container.name == self.name:
                existing_container = container
                break

        if existing_container:
            if existing_container.status != 'running':
                PrintStyle.standard(f"Starting existing container: {self.name} for safe code execution...")
                if self.logger: self.logger.log(type="info", content=f"Starting existing container: {self.name} for safe code execution...", temp=True)
                
                existing_container.start()
                self.container = existing_container
                time.sleep(2) # this helps to get SSH ready
                
            else:
                self.container = existing_container
                # PrintStyle.standard(f"Container with name '{self.name}' is already running with ID: {existing_container.id}")
        else:
            PrintStyle.standard(f"Initializing docker container {self.name} for safe code execution...")
            if self.logger: self.logger.log(type="info", content=f"Initializing docker container {self.name} for safe code execution...", temp=True)

            self.container = self.client.containers.run(
                self.image,
                detach=True,
                ports=self.ports, # type: ignore
                name=self.name,
                volumes=self.volumes, # type: ignore
            ) 
            # atexit.register(self.cleanup_container)
            PrintStyle.standard(f"Started container with ID: {self.container.id}")
            if self.logger: self.logger.log(type="info", content=f"Started container with ID: {self.container.id}")
            time.sleep(5) # this helps to get SSH ready



================================================
File: python/helpers/dotenv.py
================================================
import os
import re
from typing import Any

from .files import get_abs_path
from dotenv import load_dotenv as _load_dotenv

KEY_AUTH_LOGIN = "AUTH_LOGIN"
KEY_AUTH_PASSWORD = "AUTH_PASSWORD"
KEY_RFC_PASSWORD = "RFC_PASSWORD"
KEY_ROOT_PASSWORD = "ROOT_PASSWORD"

def load_dotenv():
    _load_dotenv(get_dotenv_file_path(), override=True)


def get_dotenv_file_path():
    return get_abs_path(".env")

def get_dotenv_value(key: str, default: Any = None):
    # load_dotenv()       
    return os.getenv(key, default)

def save_dotenv_value(key: str, value: str):
    if value is None:
        value = ""
    dotenv_path = get_dotenv_file_path()
    if not os.path.isfile(dotenv_path):
        with open(dotenv_path, "w") as f:
            f.write("")
    with open(dotenv_path, "r+") as f:
        lines = f.readlines()
        found = False
        for i, line in enumerate(lines):
            if re.match(rf"^\s*{key}\s*=", line):
                lines[i] = f"{key}={value}\n"
                found = True
        if not found:
            lines.append(f"\n{key}={value}\n")
        f.seek(0)
        f.writelines(lines)
        f.truncate()
    load_dotenv()



================================================
File: python/helpers/duckduckgo_search.py
================================================
# from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# def search(query: str, results = 5, region = "wt-wt", time="y") -> str:
#     # Create an instance with custom parameters
#     api = DuckDuckGoSearchAPIWrapper(
#         region=region,  # Set the region for search results
#         safesearch="off",  # Set safesearch level (options: strict, moderate, off)
#         time=time,  # Set time range (options: d, w, m, y)
#         max_results=results  # Set maximum number of results to return
#     )
#     # Perform a search
#     result = api.run(query)
#     return result

from duckduckgo_search import DDGS

def search(query: str, results = 5, region = "wt-wt", time="y") -> list[str]:

    ddgs = DDGS()
    src = ddgs.text(
        query,
        region=region,  # Specify region 
        safesearch="off",  # SafeSearch setting
        timelimit=time,  # Time limit (y = past year)
        max_results=results  # Number of results to return
    )
    results = []
    for s in src:
        results.append(str(s))
    return results


================================================
File: python/helpers/errors.py
================================================
import re
import traceback
import asyncio


def handle_error(e: Exception):
    # if asyncio.CancelledError, re-raise
    if isinstance(e, asyncio.CancelledError):
        raise e


def error_text(e: Exception):
    return str(e)


def format_error(e: Exception, start_entries=6, end_entries=4):
    traceback_text = traceback.format_exc()
    # Split the traceback into lines
    lines = traceback_text.split("\n")

    # Find all "File" lines
    file_indices = [
        i for i, line in enumerate(lines) if line.strip().startswith("File ")
    ]

    # If we found at least one "File" line, trim the middle if there are more than start_entries+end_entries lines
    if len(file_indices) > start_entries + end_entries:
        start_index = max(0, len(file_indices) - start_entries - end_entries)
        trimmed_lines = (
            lines[: file_indices[start_index]]
            + [
                f"\n>>>  {len(file_indices) - start_entries - end_entries} stack lines skipped <<<\n"
            ]
            + lines[file_indices[start_index + end_entries] :]
        )
    else:
        # If no "File" lines found, or not enough to trim, just return the original traceback
        trimmed_lines = lines

    # Find the error message at the end
    error_message = ""
    for line in reversed(trimmed_lines):
        if re.match(r"\w+Error:", line):
            error_message = line
            break

    # Combine the trimmed traceback with the error message
    result = "Traceback (most recent call last):\n" + "\n".join(trimmed_lines)
    if error_message:
        result += f"\n\n{error_message}"

    return result



================================================
File: python/helpers/extension.py
================================================
from abc import abstractmethod
from typing import Any
from agent import Agent
    
class Extension:

    def __init__(self, agent: Agent, *args, **kwargs):
        self.agent = agent
        self.kwargs = kwargs

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        pass


================================================
File: python/helpers/extract_tools.py
================================================
import re, os, importlib, inspect
from typing import Any, Type, TypeVar
from .dirty_json import DirtyJson
from .files import get_abs_path
import regex
from fnmatch import fnmatch

def json_parse_dirty(json:str) -> dict[str,Any] | None:
    ext_json = extract_json_object_string(json)
    if ext_json:
        data = DirtyJson.parse_string(ext_json)
        if isinstance(data,dict): return data
    return None

def extract_json_object_string(content):
    start = content.find('{')
    if start == -1:
        return ""

    # Find the first '{'
    end = content.rfind('}')
    if end == -1:
        # If there's no closing '}', return from start to the end
        return content[start:]
    else:
        # If there's a closing '}', return the substring from start to end
        return content[start:end+1]

def extract_json_string(content):
    # Regular expression pattern to match a JSON object
    pattern = r'\{(?:[^{}]|(?R))*\}|\[(?:[^\[\]]|(?R))*\]|"(?:\\.|[^"\\])*"|true|false|null|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?'
    
    # Search for the pattern in the content
    match = regex.search(pattern, content)
    
    if match:
        # Return the matched JSON string
        return match.group(0)
    else:
        return ""

def fix_json_string(json_string):
    # Function to replace unescaped line breaks within JSON string values
    def replace_unescaped_newlines(match):
        return match.group(0).replace('\n', '\\n')

    # Use regex to find string values and apply the replacement function
    fixed_string = re.sub(r'(?<=: ")(.*?)(?=")', replace_unescaped_newlines, json_string, flags=re.DOTALL)
    return fixed_string


T = TypeVar('T')  # Define a generic type variable

def load_classes_from_folder(folder: str, name_pattern: str, base_class: Type[T], one_per_file: bool = True) -> list[Type[T]]:
    classes = []
    abs_folder = get_abs_path(folder)

    # Get all .py files in the folder that match the pattern, sorted alphabetically
    py_files = sorted(
        [file_name for file_name in os.listdir(abs_folder) if fnmatch(file_name, name_pattern) and file_name.endswith(".py")]
    )

    # Iterate through the sorted list of files
    for file_name in py_files:
        module_name = file_name[:-3]  # remove .py extension
        module_path = folder.replace("/", ".") + "." + module_name
        module = importlib.import_module(module_path)

        # Get all classes in the module
        class_list = inspect.getmembers(module, inspect.isclass)

        # Filter for classes that are subclasses of the given base_class
        # iterate backwards to skip imported superclasses
        for cls in reversed(class_list):
            if cls[1] is not base_class and issubclass(cls[1], base_class):
                classes.append(cls[1])
                if one_per_file:
                    break

    return classes


================================================
File: python/helpers/file_browser.py
================================================
import os
from pathlib import Path
import shutil
import tempfile
import base64
from typing import Dict, List, Tuple, Optional, Any
import zipfile
from werkzeug.utils import secure_filename
from datetime import datetime

from python.helpers import files, runtime
from python.helpers.print_style import PrintStyle

class FileBrowser:
    ALLOWED_EXTENSIONS = {
        'image': {'jpg', 'jpeg', 'png', 'bmp'},
        'code': {'py', 'js', 'sh', 'html', 'css'},
        'document': {'md', 'pdf', 'txt', 'csv', 'json'}
    }

    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

    def __init__(self):
        # if runtime.is_development():
        #     base_dir = files.get_base_dir()
        # else:
        #     base_dir = "/"
        base_dir = "/"
        self.base_dir = Path(base_dir)
      
    def _check_file_size(self, file) -> bool:
        try:
            file.seek(0, os.SEEK_END)
            size = file.tell()
            file.seek(0)
            return size <= self.MAX_FILE_SIZE
        except (AttributeError, IOError):
            return False

    def save_file_b64(self, current_path: str, filename:str, base64_content: str):
        try:
            # Resolve the target directory path
            target_file = (self.base_dir / current_path / filename).resolve()
            if not str(target_file).startswith(str(self.base_dir)):
                raise ValueError("Invalid target directory")

            os.makedirs(target_file.parent, exist_ok=True)
            # Save file
            with open(target_file, "wb") as file:
                file.write(base64.b64decode(base64_content))
            return True
        except Exception as e:
            PrintStyle.error(f"Error saving file {filename}: {e}")
            return False

    def save_files(self, files: List, current_path: str = "") -> Tuple[List[str], List[str]]:
        """Save uploaded files and return successful and failed filenames"""
        successful = []
        failed = []
        
        try:
            # Resolve the target directory path
            target_dir = (self.base_dir / current_path).resolve()
            if not str(target_dir).startswith(str(self.base_dir)):
                raise ValueError("Invalid target directory")
                
            os.makedirs(target_dir, exist_ok=True)
            
            for file in files:
                try:
                    if file and self._is_allowed_file(file.filename, file):
                        filename = secure_filename(file.filename)
                        file_path = target_dir / filename

                        file.save(str(file_path))
                        successful.append(filename)
                    else:
                        failed.append(file.filename)
                except Exception as e:
                    PrintStyle.error(f"Error saving file {file.filename}: {e}")
                    failed.append(file.filename)
                    
            return successful, failed
            
        except Exception as e:
            PrintStyle.error(f"Error in save_files: {e}")
            return successful, failed
        
    def delete_file(self, file_path: str) -> bool:
        """Delete a file or empty directory"""
        try:
            # Resolve the full path while preventing directory traversal
            full_path = (self.base_dir / file_path).resolve()
            if not str(full_path).startswith(str(self.base_dir)):
                raise ValueError("Invalid path")
                
            if os.path.exists(full_path):
                if os.path.isfile(full_path):
                    os.remove(full_path)
                elif os.path.isdir(full_path):
                    shutil.rmtree(full_path)
                return True
                
            return False
            
        except Exception as e:
            PrintStyle.error(f"Error deleting {file_path}: {e}")
            return False

    def _is_allowed_file(self, filename: str, file) -> bool:
        # allow any file to be uploaded in file browser
        
        # if not filename:
        #     return False
        # ext = self._get_file_extension(filename)
        # all_allowed = set().union(*self.ALLOWED_EXTENSIONS.values())
        # if ext not in all_allowed:
        #     return False
        
        return True  # Allow the file if it passes the checks

    def _get_file_extension(self, filename: str) -> str:
        return filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        
    def get_files(self, current_path: str = "") -> Dict:
        try:
            # Resolve the full path while preventing directory traversal
            full_path = (self.base_dir / current_path).resolve()
            if not str(full_path).startswith(str(self.base_dir)):
                raise ValueError("Invalid path")

            files = []
            folders = []

            # List all entries in the current directory
            for entry in os.scandir(full_path):
                entry_data: Dict[str, Any] = {
                    "name": entry.name,
                    "path": str(Path(entry.path).relative_to(self.base_dir)),
                    "modified": datetime.fromtimestamp(entry.stat().st_mtime).isoformat()
                }

                if entry.is_file():
                    entry_data.update({
                        "type": self._get_file_type(entry.name),
                        "size": entry.stat().st_size,
                        "is_dir": False
                    })
                    files.append(entry_data)
                else:
                    entry_data.update({
                        "type": "folder",
                        "size": 0,  # Directories show as 0 bytes
                        "is_dir": True
                    })
                    folders.append(entry_data)

            # Combine folders and files, folders first
            all_entries = folders + files

            # Get parent directory path if not at root
            parent_path = ""
            if current_path:
                try:
                    # Get the absolute path of current directory
                    current_abs = (self.base_dir / current_path).resolve()

                    # parent_path is empty only if we're already at root
                    if str(current_abs) != str(self.base_dir):
                        parent_path = str(Path(current_path).parent)
                    
                except Exception as e:
                    parent_path = ""

            return {
                "entries": all_entries,
                "current_path": current_path,
                "parent_path": parent_path
            }

        except Exception as e:
            PrintStyle.error(f"Error reading directory: {e}")
            return {"entries": [], "current_path": "", "parent_path": ""}
        
    def get_full_path(self, file_path: str, allow_dir: bool = False) -> str:
        """Get full file path if it exists and is within base_dir"""
        full_path = files.get_abs_path(self.base_dir,file_path)
        if not files.exists(full_path):
            raise ValueError(f"File {file_path} not found")        
        return full_path
        
    def _get_file_type(self, filename: str) -> str:
        ext = self._get_file_extension(filename)
        for file_type, extensions in self.ALLOWED_EXTENSIONS.items():
            if ext in extensions:
                return file_type
        return 'unknown'


================================================
File: python/helpers/files.py
================================================
from fnmatch import fnmatch
import json
import os, re
import base64

import re
import shutil
import tempfile
import zipfile


def parse_file(_relative_path, _backup_dirs=None, _encoding="utf-8", **kwargs):
    content = read_file(_relative_path, _backup_dirs, _encoding)
    is_json = is_full_json_template(content)
    content = remove_code_fences(content)
    if is_json:
        content = replace_placeholders_json(content, **kwargs)
        obj = json.loads(content)
        # obj = replace_placeholders_dict(obj, **kwargs)
        return obj
    else:
        content = replace_placeholders_text(content, **kwargs)
        return content


def read_file(_relative_path, _backup_dirs=None, _encoding="utf-8", **kwargs):
    if _backup_dirs is None:
        _backup_dirs = []

    # Try to get the absolute path for the file from the original directory or backup directories
    absolute_path = find_file_in_dirs(_relative_path, _backup_dirs)

    # Read the file content
    with open(absolute_path, "r", encoding=_encoding) as f:
        # content = remove_code_fences(f.read())
        content = f.read()

    # Replace placeholders with values from kwargs
    content = replace_placeholders_text(content, **kwargs)

    # Process include statements
    content = process_includes(
        content, os.path.dirname(_relative_path), _backup_dirs, **kwargs
    )

    return content


def read_file_bin(_relative_path, _backup_dirs=None):
    # init backup dirs
    if _backup_dirs is None:
        _backup_dirs = []

    # get absolute path
    absolute_path = find_file_in_dirs(_relative_path, _backup_dirs)

    # read binary content
    with open(absolute_path, "rb") as f:
        return f.read()


def read_file_base64(_relative_path, _backup_dirs=None):
    # init backup dirs
    if _backup_dirs is None:
        _backup_dirs = []

    # get absolute path
    absolute_path = find_file_in_dirs(_relative_path, _backup_dirs)

    # read binary content and encode to base64
    with open(absolute_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def replace_placeholders_text(_content: str, **kwargs):
    # Replace placeholders with values from kwargs
    for key, value in kwargs.items():
        placeholder = "{{" + key + "}}"
        strval = str(value)
        _content = _content.replace(placeholder, strval)
    return _content


def replace_placeholders_json(_content: str, **kwargs):
    # Replace placeholders with values from kwargs
    for key, value in kwargs.items():
        placeholder = "{{" + key + "}}"
        strval = json.dumps(value)
        _content = _content.replace(placeholder, strval)
    return _content


def replace_placeholders_dict(_content: dict, **kwargs):
    def replace_value(value):
        if isinstance(value, str):
            placeholders = re.findall(r"{{(\w+)}}", value)
            if placeholders:
                for placeholder in placeholders:
                    if placeholder in kwargs:
                        replacement = kwargs[placeholder]
                        if value == f"{{{{{placeholder}}}}}":
                            return replacement
                        elif isinstance(replacement, (dict, list)):
                            value = value.replace(
                                f"{{{{{placeholder}}}}}", json.dumps(replacement)
                            )
                        else:
                            value = value.replace(
                                f"{{{{{placeholder}}}}}", str(replacement)
                            )
            return value
        elif isinstance(value, dict):
            return {k: replace_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [replace_value(item) for item in value]
        else:
            return value

    return replace_value(_content)


def process_includes(_content, _base_path, _backup_dirs, **kwargs):
    # Regex to find {{ include 'path' }} or {{include'path'}}
    include_pattern = re.compile(r"{{\s*include\s*['\"](.*?)['\"]\s*}}")

    def replace_include(match):
        include_path = match.group(1)
        # First attempt to resolve the include relative to the base path
        full_include_path = find_file_in_dirs(
            os.path.join(_base_path, include_path), _backup_dirs
        )

        # Recursively read the included file content, keeping the original base path
        included_content = read_file(full_include_path, _backup_dirs, **kwargs)
        return included_content

    # Replace all includes with the file content
    return re.sub(include_pattern, replace_include, _content)


def find_file_in_dirs(file_path, backup_dirs):
    """
    This function tries to find the file first in the given file_path,
    and then in the backup_dirs if not found in the original location.
    Returns the absolute path of the found file.
    """
    # Try the original path first
    if os.path.isfile(get_abs_path(file_path)):
        return get_abs_path(file_path)

    # Loop through the backup directories
    for backup_dir in backup_dirs:
        backup_path = os.path.join(backup_dir, os.path.basename(file_path))
        if os.path.isfile(get_abs_path(backup_path)):
            return get_abs_path(backup_path)

    # If the file is not found, let it raise the FileNotFoundError
    raise FileNotFoundError(
        f"File '{file_path}' not found in the original path or backup directories."
    )


import re


def remove_code_fences(text):
    # Pattern to match code fences with optional language specifier
    pattern = r"(```|~~~)(.*?\n)(.*?)(\1)"

    # Function to replace the code fences
    def replacer(match):
        return match.group(3)  # Return the code without fences

    # Use re.DOTALL to make '.' match newlines
    result = re.sub(pattern, replacer, text, flags=re.DOTALL)

    return result


import re


def is_full_json_template(text):
    # Pattern to match the entire text enclosed in ```json or ~~~json fences
    pattern = r"^\s*(```|~~~)\s*json\s*\n(.*?)\n\1\s*$"
    # Use re.DOTALL to make '.' match newlines
    match = re.fullmatch(pattern, text.strip(), flags=re.DOTALL)
    return bool(match)


def write_file(relative_path: str, content: str, encoding: str = "utf-8"):
    abs_path = get_abs_path(relative_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w", encoding=encoding) as f:
        f.write(content)


def write_file_bin(relative_path: str, content: bytes):
    abs_path = get_abs_path(relative_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "wb") as f:
        f.write(content)


def write_file_base64(relative_path: str, content: str):
    # decode base64 string to bytes
    data = base64.b64decode(content)
    abs_path = get_abs_path(relative_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "wb") as f:
        f.write(data)


def delete_file(relative_path: str):
    abs_path = get_abs_path(relative_path)
    if os.path.exists(abs_path):
        os.remove(abs_path)


def delete_dir(relative_path: str):
    abs_path = get_abs_path(relative_path)
    if os.path.exists(abs_path):
        shutil.rmtree(abs_path)


def list_files(relative_path: str, filter: str = "*"):
    abs_path = get_abs_path(relative_path)
    if not os.path.exists(abs_path):
        return []
    return [file for file in os.listdir(abs_path) if fnmatch(file, filter)]


def make_dirs(relative_path: str):
    abs_path = get_abs_path(relative_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)


def get_abs_path(*relative_paths):
    return os.path.join(get_base_dir(), *relative_paths)


def exists(*relative_paths):
    path = get_abs_path(*relative_paths)
    return os.path.exists(path)


def get_base_dir():
    # Get the base directory from the current file path
    base_dir = os.path.dirname(os.path.abspath(os.path.join(__file__, "../../")))
    return base_dir

def is_in_base_dir(path: str):
    # check if the given path is within the base directory
    base_dir = get_base_dir()
    # normalize paths to handle relative paths and symlinks
    abs_path = os.path.abspath(path)
    # check if the absolute path starts with the base directory
    return os.path.commonpath([abs_path, base_dir]) == base_dir


def get_subdirectories(relative_path: str, include: str | list[str] = "*", exclude: str | list[str] | None = None):
    abs_path = get_abs_path(relative_path)
    if not os.path.exists(abs_path):
        return []
    if isinstance(include, str):
        include = [include]
    if isinstance(exclude, str):
        exclude = [exclude]
    return [
        subdir
        for subdir in os.listdir(abs_path)
        if os.path.isdir(os.path.join(abs_path, subdir))
        and any(fnmatch(subdir, inc) for inc in include)
        and (exclude is None or not any(fnmatch(subdir, exc) for exc in exclude))
    ]


def zip_dir(dir_path: str):
    full_path = get_abs_path(dir_path)
    zip_file_path = tempfile.NamedTemporaryFile(suffix=".zip", delete=False).name
    base_name = os.path.basename(full_path)
    with zipfile.ZipFile(zip_file_path, "w", compression=zipfile.ZIP_DEFLATED) as zip:
        for root, _, files in os.walk(full_path):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, full_path)
                zip.write(file_path, os.path.join(base_name, rel_path))
    return zip_file_path


def move_file(relative_path: str, new_path: str):
    abs_path = get_abs_path(relative_path)
    new_abs_path = get_abs_path(new_path)
    os.makedirs(os.path.dirname(new_abs_path), exist_ok=True)
    os.rename(abs_path, new_abs_path)

def safe_file_name(filename:str)-> str:
    # Replace any character that's not alphanumeric, dash, underscore, or dot with underscore
    import re
    return re.sub(r'[^a-zA-Z0-9-._]', '_', filename)


================================================
File: python/helpers/git.py
================================================
from git import Repo
from datetime import datetime
import os
from python.helpers import files

def get_git_info():
    # Get the current working directory (assuming the repo is in the same folder as the script)
    repo_path = files.get_base_dir()
    
    # Open the Git repository
    repo = Repo(repo_path)

    # Ensure the repository is not bare
    if repo.bare:
        raise ValueError(f"Repository at {repo_path} is bare and cannot be used.")

    # Get the current branch name
    branch = repo.active_branch.name if repo.head.is_detached is False else ""

    # Get the latest commit hash
    commit_hash = repo.head.commit.hexsha

    # Get the commit date (ISO 8601 format)
    commit_time = datetime.fromtimestamp(repo.head.commit.committed_date).strftime('%y-%m-%d %H:%M')

    # Get the latest tag description (if available)
    short_tag = ""
    try:
        tag = repo.git.describe(tags=True)
        tag_split = tag.split('-')
        if len(tag_split) >= 3:
            short_tag = "-".join(tag_split[:-1])
        else:
            short_tag = tag
    except:
        tag = ""

    version = branch[0].upper() + " " + ( short_tag or commit_hash[:7] )

    # Create the dictionary with collected information
    git_info = {
        "branch": branch,
        "commit_hash": commit_hash,
        "commit_time": commit_time,
        "tag": tag,
        "short_tag": short_tag,
        "version": version
    }

    return git_info


================================================
File: python/helpers/history.py
================================================
from abc import abstractmethod
import asyncio
from collections import OrderedDict
from collections.abc import Mapping
import json
import math
from typing import Coroutine, Literal, TypedDict, cast, Union, Dict, List, Any
from python.helpers import messages, tokens, settings, call_llm
from enum import Enum
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage

BULK_MERGE_COUNT = 3
TOPICS_KEEP_COUNT = 3
CURRENT_TOPIC_RATIO = 0.5
HISTORY_TOPIC_RATIO = 0.3
HISTORY_BULK_RATIO = 0.2
TOPIC_COMPRESS_RATIO = 0.65
LARGE_MESSAGE_TO_TOPIC_RATIO = 0.25
RAW_MESSAGE_OUTPUT_TEXT_TRIM = 100


class RawMessage(TypedDict):
    raw_content: "MessageContent"
    preview: str | None


MessageContent = Union[
    List["MessageContent"],
    Dict[str, "MessageContent"],
    List[Dict[str, "MessageContent"]],
    str,
    List[str],
    RawMessage,
]


class OutputMessage(TypedDict):
    ai: bool
    content: MessageContent


class Record:
    def __init__(self):
        pass

    @abstractmethod
    def get_tokens(self) -> int:
        pass

    @abstractmethod
    async def compress(self) -> bool:
        pass

    @abstractmethod
    def output(self) -> list[OutputMessage]:
        pass

    @abstractmethod
    async def summarize(self) -> str:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass

    @staticmethod
    def from_dict(data: dict, history: "History"):
        cls = data["_cls"]
        return globals()[cls].from_dict(data, history=history)

    def output_langchain(self):
        return output_langchain(self.output())

    def output_text(self, human_label="user", ai_label="ai"):
        return output_text(self.output(), ai_label, human_label)


class Message(Record):
    def __init__(self, ai: bool, content: MessageContent, tokens: int = 0):
        self.ai = ai
        self.content = content
        self.summary: str = ""
        self.tokens: int = tokens or self.calculate_tokens()

    def get_tokens(self) -> int:
        if not self.tokens:
            self.tokens = self.calculate_tokens()
        return self.tokens

    def calculate_tokens(self):
        text = self.output_text()
        return tokens.approximate_tokens(text)

    def set_summary(self, summary: str):
        self.summary = summary
        self.tokens = self.calculate_tokens()

    async def compress(self):
        return False

    def output(self):
        return [OutputMessage(ai=self.ai, content=self.summary or self.content)]

    def output_langchain(self):
        return output_langchain(self.output())

    def output_text(self, human_label="user", ai_label="ai"):
        return output_text(self.output(), ai_label, human_label)

    def to_dict(self):
        return {
            "_cls": "Message",
            "ai": self.ai,
            "content": self.content,
            "summary": self.summary,
            "tokens": self.tokens,
        }

    @staticmethod
    def from_dict(data: dict, history: "History"):
        content = data.get("content", "Content lost")
        msg = Message(ai=data["ai"], content=content)
        msg.summary = data.get("summary", "")
        msg.tokens = data.get("tokens", 0)
        return msg


class Topic(Record):
    def __init__(self, history: "History"):
        self.history = history
        self.summary: str = ""
        self.messages: list[Message] = []

    def get_tokens(self):
        if self.summary:
            return tokens.approximate_tokens(self.summary)
        else:
            return sum(msg.get_tokens() for msg in self.messages)

    def add_message(
        self, ai: bool, content: MessageContent, tokens: int = 0
    ) -> Message:
        msg = Message(ai=ai, content=content, tokens=tokens)
        self.messages.append(msg)
        return msg

    def output(self) -> list[OutputMessage]:
        if self.summary:
            return [OutputMessage(ai=False, content=self.summary)]
        else:
            msgs = [m for r in self.messages for m in r.output()]
            return msgs

    async def summarize(self):
        self.summary = await self.summarize_messages(self.messages)
        return self.summary

    async def compress_large_messages(self) -> bool:
        set = settings.get_settings()
        msg_max_size = (
            set["chat_model_ctx_length"]
            * set["chat_model_ctx_history"]
            * CURRENT_TOPIC_RATIO
            * LARGE_MESSAGE_TO_TOPIC_RATIO
        )
        large_msgs = []
        for m in (m for m in self.messages if not m.summary):
            # TODO refactor this
            out = m.output()
            text = output_text(out)
            tok = m.get_tokens()
            leng = len(text)
            if tok > msg_max_size:
                large_msgs.append((m, tok, leng, out))
        large_msgs.sort(key=lambda x: x[1], reverse=True)
        for msg, tok, leng, out in large_msgs:
            trim_to_chars = leng * (msg_max_size / tok)
            # raw messages will be replaced as a whole, they would become invalid when truncated
            if _is_raw_message(out[0]["content"]):
                msg.set_summary(
                    "Message content replaced to save space in context window"
                )

            # regular messages will be truncated
            else:
                trunc = messages.truncate_dict_by_ratio(
                    self.history.agent,
                    out[0]["content"],
                    trim_to_chars * 1.15,
                    trim_to_chars * 0.85,
                )
                msg.set_summary(_json_dumps(trunc))

            return True
        return False

    async def compress(self) -> bool:
        compress = await self.compress_large_messages()
        if not compress:
            compress = await self.compress_attention()
        return compress

    async def compress_attention(self) -> bool:

        if len(self.messages) > 2:
            cnt_to_sum = math.ceil((len(self.messages) - 2) * TOPIC_COMPRESS_RATIO)
            msg_to_sum = self.messages[1 : cnt_to_sum + 1]
            summary = await self.summarize_messages(msg_to_sum)
            sum_msg_content = self.history.agent.parse_prompt(
                "fw.msg_summary.md", summary=summary
            )
            sum_msg = Message(False, sum_msg_content)
            self.messages[1 : cnt_to_sum + 1] = [sum_msg]
            return True
        return False

    async def summarize_messages(self, messages: list[Message]):
        # FIXME: vision bytes are sent to utility LLM, send summary instead
        msg_txt = [m.output_text() for m in messages]
        summary = await self.history.agent.call_utility_model(
            system=self.history.agent.read_prompt("fw.topic_summary.sys.md"),
            message=self.history.agent.read_prompt(
                "fw.topic_summary.msg.md", content=msg_txt
            ),
        )
        return summary

    def to_dict(self):
        return {
            "_cls": "Topic",
            "summary": self.summary,
            "messages": [m.to_dict() for m in self.messages],
        }

    @staticmethod
    def from_dict(data: dict, history: "History"):
        topic = Topic(history=history)
        topic.summary = data.get("summary", "")
        topic.messages = [
            Message.from_dict(m, history=history) for m in data.get("messages", [])
        ]
        return topic


class Bulk(Record):
    def __init__(self, history: "History"):
        self.history = history
        self.summary: str = ""
        self.records: list[Record] = []

    def get_tokens(self):
        if self.summary:
            return tokens.approximate_tokens(self.summary)
        else:
            return sum([r.get_tokens() for r in self.records])

    def output(
        self, human_label: str = "user", ai_label: str = "ai"
    ) -> list[OutputMessage]:
        if self.summary:
            return [OutputMessage(ai=False, content=self.summary)]
        else:
            msgs = [m for r in self.records for m in r.output()]
            return msgs

    async def compress(self):
        return False

    async def summarize(self):
        self.summary = await self.history.agent.call_utility_model(
            system=self.history.agent.read_prompt("fw.topic_summary.sys.md"),
            message=self.history.agent.read_prompt(
                "fw.topic_summary.msg.md", content=self.output_text()
            ),
        )
        return self.summary

    def to_dict(self):
        return {
            "_cls": "Bulk",
            "summary": self.summary,
            "records": [r.to_dict() for r in self.records],
        }

    @staticmethod
    def from_dict(data: dict, history: "History"):
        bulk = Bulk(history=history)
        bulk.summary = data["summary"]
        cls = data["_cls"]
        bulk.records = [Record.from_dict(r, history=history) for r in data["records"]]
        return bulk


class History(Record):
    def __init__(self, agent):
        from agent import Agent

        self.bulks: list[Bulk] = []
        self.topics: list[Topic] = []
        self.current = Topic(history=self)
        self.agent: Agent = agent

    def get_tokens(self) -> int:
        return (
            self.get_bulks_tokens()
            + self.get_topics_tokens()
            + self.get_current_topic_tokens()
        )

    def is_over_limit(self):
        limit = _get_ctx_size_for_history()
        total = self.get_tokens()
        return total > limit

    def get_bulks_tokens(self) -> int:
        return sum(record.get_tokens() for record in self.bulks)

    def get_topics_tokens(self) -> int:
        return sum(record.get_tokens() for record in self.topics)

    def get_current_topic_tokens(self) -> int:
        return self.current.get_tokens()

    def add_message(
        self, ai: bool, content: MessageContent, tokens: int = 0
    ) -> Message:
        return self.current.add_message(ai, content=content, tokens=tokens)

    def new_topic(self):
        if self.current.messages:
            self.topics.append(self.current)
            self.current = Topic(history=self)

    def output(self) -> list[OutputMessage]:
        result: list[OutputMessage] = []
        result += [m for b in self.bulks for m in b.output()]
        result += [m for t in self.topics for m in t.output()]
        result += self.current.output()
        return result

    @staticmethod
    def from_dict(data: dict, history: "History"):
        history.bulks = [Bulk.from_dict(b, history=history) for b in data["bulks"]]
        history.topics = [Topic.from_dict(t, history=history) for t in data["topics"]]
        history.current = Topic.from_dict(data["current"], history=history)
        return history

    def to_dict(self):
        return {
            "_cls": "History",
            "bulks": [b.to_dict() for b in self.bulks],
            "topics": [t.to_dict() for t in self.topics],
            "current": self.current.to_dict(),
        }

    def serialize(self):
        data = self.to_dict()
        return _json_dumps(data)

    async def compress(self):
        compressed = False
        while True:
            curr, hist, bulk = (
                self.get_current_topic_tokens(),
                self.get_topics_tokens(),
                self.get_bulks_tokens(),
            )
            total = _get_ctx_size_for_history()
            ratios = [
                (curr, CURRENT_TOPIC_RATIO, "current_topic"),
                (hist, HISTORY_TOPIC_RATIO, "history_topic"),
                (bulk, HISTORY_BULK_RATIO, "history_bulk"),
            ]
            ratios = sorted(ratios, key=lambda x: (x[0] / total) / x[1], reverse=True)
            compressed_part = False
            for ratio in ratios:
                if ratio[0] > ratio[1] * total:
                    over_part = ratio[2]
                    if over_part == "current_topic":
                        compressed_part = await self.current.compress()
                    elif over_part == "history_topic":
                        compressed_part = await self.compress_topics()
                    else:
                        compressed_part = await self.compress_bulks()
                    if compressed_part:
                        break

            if compressed_part:
                compressed = True
                continue
            else:
                return compressed

    async def compress_topics(self) -> bool:
        # summarize topics one by one
        for topic in self.topics:
            if not topic.summary:
                await topic.summarize()
                return True

        # move oldest topic to bulks and summarize
        for topic in self.topics:
            bulk = Bulk(history=self)
            bulk.records.append(topic)
            if topic.summary:
                bulk.summary = topic.summary
            else:
                await bulk.summarize()
            self.bulks.append(bulk)
            self.topics.remove(topic)
            return True
        return False

    async def compress_bulks(self):
        # merge bulks if possible
        compressed = await self.merge_bulks_by(BULK_MERGE_COUNT)
        # remove oldest bulk if necessary
        if not compressed:
            self.bulks.pop(0)
            return True
        return compressed

    async def merge_bulks_by(self, count: int):
        # if bulks is empty, return False
        if len(self.bulks) == 0:
            return False
        # merge bulks in groups of count, even if there are fewer than count
        bulks = await asyncio.gather(
            *[
                self.merge_bulks(self.bulks[i : i + count])
                for i in range(0, len(self.bulks), count)
            ]
        )
        self.bulks = bulks
        return True

    async def merge_bulks(self, bulks: list[Bulk]) -> Bulk:
        bulk = Bulk(history=self)
        bulk.records = cast(list[Record], bulks)
        await bulk.summarize()
        return bulk


def deserialize_history(json_data: str, agent) -> History:
    history = History(agent=agent)
    if json_data:
        data = _json_loads(json_data)
        history = History.from_dict(data, history=history)
    return history


def _get_ctx_size_for_history() -> int:
    set = settings.get_settings()
    return int(set["chat_model_ctx_length"] * set["chat_model_ctx_history"])


def _stringify_output(output: OutputMessage, ai_label="ai", human_label="human"):
    return f'{ai_label if output["ai"] else human_label}: {_stringify_content(output["content"])}'


def _stringify_content(content: MessageContent) -> str:
    # already a string
    if isinstance(content, str):
        return content
    
    # raw messages return preview or trimmed json
    if _is_raw_message(content):
        preview: str = content.get("preview", "") # type: ignore
        if preview:
            return preview
        text = _json_dumps(content)
        if len(text) > RAW_MESSAGE_OUTPUT_TEXT_TRIM:
            return text[:RAW_MESSAGE_OUTPUT_TEXT_TRIM] + "... TRIMMED"
        return text
    
    # regular messages of non-string are dumped as json
    return _json_dumps(content)


def _output_content_langchain(content: MessageContent):
    if isinstance(content, str):
        return content
    if _is_raw_message(content):
        return content["raw_content"]  # type: ignore
    try:
        return _json_dumps(content)
    except Exception as e:
        raise e


def group_outputs_abab(outputs: list[OutputMessage]) -> list[OutputMessage]:
    result = []
    for out in outputs:
        if result and result[-1]["ai"] == out["ai"]:
            result[-1] = OutputMessage(
                ai=result[-1]["ai"],
                content=_merge_outputs(result[-1]["content"], out["content"]),
            )
        else:
            result.append(out)
    return result


def group_messages_abab(messages: list[BaseMessage]) -> list[BaseMessage]:
    result = []
    for msg in messages:
        if result and isinstance(result[-1], type(msg)):
            # create new instance of the same type with merged content
            result[-1] = type(result[-1])(content=_merge_outputs(result[-1].content, msg.content))  # type: ignore
        else:
            result.append(msg)
    return result


def output_langchain(messages: list[OutputMessage]):
    result = []
    for m in messages:
        if m["ai"]:
            # result.append(AIMessage(content=serialize_content(m["content"])))
            result.append(AIMessage(_output_content_langchain(content=m["content"])))  # type: ignore
        else:
            # result.append(HumanMessage(content=serialize_content(m["content"])))
            result.append(HumanMessage(_output_content_langchain(content=m["content"])))  # type: ignore
    # ensure message type alternation
    result = group_messages_abab(result)
    return result


def output_text(messages: list[OutputMessage], ai_label="ai", human_label="human"):
    return "\n".join(_stringify_output(o, ai_label, human_label) for o in messages)


def _merge_outputs(a: MessageContent, b: MessageContent) -> MessageContent:
    if isinstance(a, str) and isinstance(b, str):
        return a + "\n" + b

    if not isinstance(a, list):
        a = [a]
    if not isinstance(b, list):
        b = [b]

    return cast(MessageContent, a + b)


def _merge_properties(
    a: Dict[str, MessageContent], b: Dict[str, MessageContent]
) -> Dict[str, MessageContent]:
    result = a.copy()
    for k, v in b.items():
        if k in result:
            result[k] = _merge_outputs(result[k], v)
        else:
            result[k] = v
    return result


def _is_raw_message(obj: object) -> bool:
    return isinstance(obj, Mapping) and "raw_content" in obj


def _json_dumps(obj):
    return json.dumps(obj, ensure_ascii=False)


def _json_loads(obj):
    return json.loads(obj)



================================================
File: python/helpers/images.py
================================================
from PIL import Image
import io
import math


def compress_image(image_data: bytes, *, max_pixels: int = 256_000, quality: int = 50) -> bytes:
    """Compress an image by scaling it down and converting to JPEG with quality settings.
    
    Args:
        image_data: Raw image bytes
        max_pixels: Maximum number of pixels in the output image (width * height)
        quality: JPEG quality setting (1-100)
    
    Returns:
        Compressed image as bytes
    """
    # load image from bytes
    img = Image.open(io.BytesIO(image_data))
    
    # calculate scaling factor to get to max_pixels
    current_pixels = img.width * img.height
    if current_pixels > max_pixels:
        scale = math.sqrt(max_pixels / current_pixels)
        new_width = int(img.width * scale)
        new_height = int(img.height * scale)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # convert to RGB if needed (for JPEG)
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    
    # save as JPEG with compression
    output = io.BytesIO()
    img.save(output, format='JPEG', quality=quality, optimize=True)
    return output.getvalue()



================================================
File: python/helpers/job_loop.py
================================================
import asyncio
from python.helpers.task_scheduler import TaskScheduler
from python.helpers.print_style import PrintStyle
from python.helpers import errors


async def run_loop():
    while True:
        try:
            await scheduler_tick()
        except Exception as e:
            PrintStyle().error(errors.format_error(e))
        await asyncio.sleep(60) # TODO! - if we lower it under 1min, it can run a 5min job multiple times in it's target minute


async def scheduler_tick():
    # Get the task scheduler instance and print detailed debug info
    scheduler = TaskScheduler.get()
    # Run the scheduler tick
    await scheduler.tick()


================================================
File: python/helpers/knowledge_import.py
================================================
import glob
import os
import hashlib
import json
from typing import Any, Dict, Literal, TypedDict
from langchain_community.document_loaders import (
    CSVLoader,
    JSONLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)
from python.helpers import files
from python.helpers.log import LogItem
from python.helpers.print_style import PrintStyle

text_loader_kwargs = {"autodetect_encoding": True}


class KnowledgeImport(TypedDict):
    file: str
    checksum: str
    ids: list[str]
    state: Literal["changed", "original", "removed"]
    documents: list[Any]


def calculate_checksum(file_path: str) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def load_knowledge(
    log_item: LogItem | None,
    knowledge_dir: str,
    index: Dict[str, KnowledgeImport],
    metadata: dict[str, Any] = {},
    filename_pattern: str = "**/*",
) -> Dict[str, KnowledgeImport]:

    # from python.helpers.memory import Memory

    # Mapping file extensions to corresponding loader classes
    file_types_loaders = {
        "txt": TextLoader,
        "pdf": PyPDFLoader,
        "csv": CSVLoader,
        "html": UnstructuredHTMLLoader,
        # "json": JSONLoader,
        "json": TextLoader,
        # "md": UnstructuredMarkdownLoader,
        "md": TextLoader,
    }

    cnt_files = 0
    cnt_docs = 0

    # for area in Memory.Area:
    #     subdir = files.get_abs_path(knowledge_dir, area.value)

    # if not os.path.exists(knowledge_dir):
    #     os.makedirs(knowledge_dir)
    #     continue

    # Fetch all files in the directory with specified extensions
    kn_files = glob.glob(knowledge_dir + "/" + filename_pattern, recursive=True)
    kn_files = [f for f in kn_files if os.path.isfile(f)]

    if kn_files:
        PrintStyle.standard(
            f"Found {len(kn_files)} knowledge files in {knowledge_dir}, processing..."
        )
        if log_item:
            log_item.stream(
                progress=f"\nFound {len(kn_files)} knowledge files in {knowledge_dir}, processing...",
            )

    for file_path in kn_files:
        ext = file_path.split(".")[-1].lower()
        if ext in file_types_loaders:
            checksum = calculate_checksum(file_path)
            file_key = file_path  # os.path.relpath(file_path, knowledge_dir)

            # Load existing data from the index or create a new entry
            file_data = index.get(file_key, {})

            if file_data.get("checksum") == checksum:
                file_data["state"] = "original"
            else:
                file_data["state"] = "changed"

            if file_data["state"] == "changed":
                file_data["checksum"] = checksum
                loader_cls = file_types_loaders[ext]
                loader = loader_cls(
                    file_path,
                    **(
                        text_loader_kwargs
                        if ext in ["txt", "csv", "html", "md"]
                        else {}
                    ),
                )
                file_data["documents"] = loader.load_and_split()
                for doc in file_data["documents"]:
                    doc.metadata = {**doc.metadata, **metadata}
                cnt_files += 1
                cnt_docs += len(file_data["documents"])
                # PrintStyle.standard(f"Imported {len(file_data['documents'])} documents from {file_path}")

            # Update the index
            index[file_key] = file_data  # type: ignore

    # loop index where state is not set and mark it as removed
    for file_key, file_data in index.items():
        if not file_data.get("state", ""):
            index[file_key]["state"] = "removed"

    PrintStyle.standard(f"Processed {cnt_docs} documents from {cnt_files} files.")
    if log_item:
        log_item.stream(
            progress=f"\nProcessed {cnt_docs} documents from {cnt_files} files."
        )
    return index



================================================
File: python/helpers/localization.py
================================================
from datetime import datetime
import pytz  # type: ignore

from python.helpers.print_style import PrintStyle
from python.helpers.dotenv import get_dotenv_value, save_dotenv_value


class Localization:
    """
    Localization class for handling timezone conversions between UTC and local time.
    """

    # singleton
    _instance = None

    @classmethod
    def get(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = cls(*args, **kwargs)
        return cls._instance

    def __init__(self, timezone: str | None = None):
        if timezone is not None:
            self.set_timezone(timezone)  # Use the setter to validate
        else:
            timezone = str(get_dotenv_value("DEFAULT_USER_TIMEZONE", "UTC"))
            self.set_timezone(timezone)

    def get_timezone(self) -> str:
        return self.timezone

    def set_timezone(self, timezone: str) -> None:
        """Set the timezone, with validation."""
        # Validate timezone
        try:
            pytz.timezone(timezone)
            if timezone != getattr(self, 'timezone', None):
                PrintStyle.debug(f"Changing timezone from {getattr(self, 'timezone', 'None')} to {timezone}")
                self.timezone = timezone
                save_dotenv_value("DEFAULT_USER_TIMEZONE", timezone)
        except pytz.exceptions.UnknownTimeZoneError:
            PrintStyle.error(f"Unknown timezone: {timezone}, defaulting to UTC")
            self.timezone = "UTC"
            # save the default timezone to the environment variable to avoid future errors on startup
            save_dotenv_value("DEFAULT_USER_TIMEZONE", "UTC")

    def localtime_str_to_utc_dt(self, localtime_str: str | None) -> datetime | None:
        """
        Convert a local time ISO string to a UTC datetime object.
        Returns None if input is None or invalid.
        """
        if not localtime_str:
            return None

        try:
            # Handle both with and without timezone info
            try:
                # Try parsing with timezone info first
                local_datetime_obj = datetime.fromisoformat(localtime_str)
                if local_datetime_obj.tzinfo is None:
                    # If no timezone info, assume it's in the configured timezone
                    local_datetime_obj = pytz.timezone(self.timezone).localize(local_datetime_obj)
            except ValueError:
                # If timezone parsing fails, try without timezone
                local_datetime_obj = datetime.fromisoformat(localtime_str.split('Z')[0].split('+')[0])
                local_datetime_obj = pytz.timezone(self.timezone).localize(local_datetime_obj)

            # Convert to UTC
            return local_datetime_obj.astimezone(pytz.utc)
        except Exception as e:
            PrintStyle.error(f"Error converting localtime string to UTC: {e}")
            return None

    def utc_dt_to_localtime_str(self, utc_dt: datetime | None, sep: str = "T", timespec: str = "auto") -> str | None:
        """
        Convert a UTC datetime object to a local time ISO string.
        Returns None if input is None.
        """
        if utc_dt is None:
            return None

        # At this point, utc_dt is definitely not None
        assert utc_dt is not None

        try:
            # Ensure datetime is timezone aware
            if utc_dt.tzinfo is None:
                utc_dt = pytz.utc.localize(utc_dt)
            elif utc_dt.tzinfo != pytz.utc:
                utc_dt = utc_dt.astimezone(pytz.utc)

            # Convert to local time
            local_datetime_obj = utc_dt.astimezone(pytz.timezone(self.timezone))
            # Return the local time string
            return local_datetime_obj.isoformat(sep=sep, timespec=timespec)
        except Exception as e:
            PrintStyle.error(f"Error converting UTC datetime to localtime string: {e}")
            return None

    def serialize_datetime(self, dt: datetime | None) -> str | None:
        """
        Serialize a datetime object to ISO format string in the user's timezone.
        This ensures the frontend receives dates in the correct timezone for display.
        """
        if dt is None:
            return None

        # At this point, dt is definitely not None
        assert dt is not None

        try:
            # Ensure datetime is timezone aware (if not, assume UTC)
            if dt.tzinfo is None:
                dt = pytz.utc.localize(dt)

            # Convert to the user's timezone
            local_timezone = pytz.timezone(self.timezone)
            local_dt = dt.astimezone(local_timezone)

            return local_dt.isoformat()
        except Exception as e:
            PrintStyle.error(f"Error serializing datetime: {e}")
            return None



================================================
File: python/helpers/log.py
================================================
from dataclasses import dataclass, field
import json
from typing import Any, Literal, Optional, Dict
import uuid
from collections import OrderedDict  # Import OrderedDict

Type = Literal[
    "agent",
    "browser",
    "code_exe",
    "error",
    "hint",
    "info",
    "progress",
    "response",
    "tool",
    "input",
    "user",
    "util",
    "warning",
]

ProgressUpdate = Literal["persistent", "temporary", "none"]


@dataclass
class LogItem:
    log: "Log"
    no: int
    type: str
    heading: str
    content: str
    temp: bool
    update_progress: Optional[ProgressUpdate] = "persistent"
    kvps: Optional[OrderedDict] = None  # Use OrderedDict for kvps
    id: Optional[str] = None  # Add id field
    guid: str = ""

    def __post_init__(self):
        self.guid = self.log.guid

    def update(
        self,
        type: Type | None = None,
        heading: str | None = None,
        content: str | None = None,
        kvps: dict | None = None,
        temp: bool | None = None,
        update_progress: ProgressUpdate | None = None,
        **kwargs,
    ):
        if self.guid == self.log.guid:
            self.log._update_item(
                self.no,
                type=type,
                heading=heading,
                content=content,
                kvps=kvps,
                temp=temp,
                update_progress=update_progress,
                **kwargs,
            )

    def stream(
        self,
        heading: str | None = None,
        content: str | None = None,
        **kwargs,
    ):
        if heading is not None:
            self.update(heading=self.heading + heading)
        if content is not None:
            self.update(content=self.content + content)

        for k, v in kwargs.items():
            prev = self.kvps.get(k, "") if self.kvps else ""
            self.update(**{k: prev + v})

    def output(self):
        return {
            "no": self.no,
            "id": self.id,  # Include id in output
            "type": self.type,
            "heading": self.heading,
            "content": self.content,
            "temp": self.temp,
            "kvps": self.kvps,
        }


class Log:

    def __init__(self):
        self.guid: str = str(uuid.uuid4())
        self.updates: list[int] = []
        self.logs: list[LogItem] = []
        self.set_initial_progress()

    def log(
        self,
        type: Type,
        heading: str | None = None,
        content: str | None = None,
        kvps: dict | None = None,
        temp: bool | None = None,
        update_progress: ProgressUpdate | None = None,
        id: Optional[str] = None,  # Add id parameter
        **kwargs,
    ) -> LogItem:
        # Use OrderedDict if kvps is provided
        if kvps is not None:
            kvps = OrderedDict(kvps)
        item = LogItem(
            log=self,
            no=len(self.logs),
            type=type,
            heading=heading or "",
            content=content or "",
            kvps=OrderedDict({**(kvps or {}), **(kwargs or {})}),
            update_progress=(
                update_progress if update_progress is not None else "persistent"
            ),
            temp=temp if temp is not None else False,
            id=id,  # Pass id to LogItem
        )
        self.logs.append(item)
        self.updates += [item.no]
        self._update_progress_from_item(item)
        return item

    def _update_item(
        self,
        no: int,
        type: str | None = None,
        heading: str | None = None,
        content: str | None = None,
        kvps: dict | None = None,
        temp: bool | None = None,
        update_progress: ProgressUpdate | None = None,
        **kwargs,
    ):
        item = self.logs[no]
        if type is not None:
            item.type = type
        if update_progress is not None:
            item.update_progress = update_progress
        if heading is not None:
            item.heading = heading
        if content is not None:
            item.content = content
        if kvps is not None:
            item.kvps = OrderedDict(kvps)  # Use OrderedDict to keep the order

        if temp is not None:
            item.temp = temp

        if kwargs:
            if item.kvps is None:
                item.kvps = OrderedDict()  # Ensure kvps is an OrderedDict
            for k, v in kwargs.items():
                item.kvps[k] = v

        self.updates += [item.no]
        self._update_progress_from_item(item)

    def set_progress(self, progress: str, no: int = 0, active: bool = True):
        self.progress = progress
        if not no:
            no = len(self.logs)
        self.progress_no = no
        self.progress_active = active

    def set_initial_progress(self):
        self.set_progress("Waiting for input", 0, False)

    def output(self, start=None, end=None):
        if start is None:
            start = 0
        if end is None:
            end = len(self.updates)

        out = []
        seen = set()
        for update in self.updates[start:end]:
            if update not in seen:
                out.append(self.logs[update].output())
                seen.add(update)

        return out

    def reset(self):
        self.guid = str(uuid.uuid4())
        self.updates = []
        self.logs = []
        self.set_initial_progress()

    def _update_progress_from_item(self, item: LogItem):
        if item.heading and item.update_progress != "none":
            if item.no >= self.progress_no:
                self.set_progress(
                    item.heading,
                    (item.no if item.update_progress == "persistent" else -1),
                )
            



================================================
File: python/helpers/memory.py
================================================
from datetime import datetime
from typing import Any, List, Sequence
from langchain.storage import InMemoryByteStore, LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings

# from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.utils import (
    DistanceStrategy,
)
from langchain_core.embeddings import Embeddings

import os, json

import numpy as np

from python.helpers.print_style import PrintStyle
from . import files
from langchain_core.documents import Document
import uuid
from python.helpers import knowledge_import
from python.helpers.log import Log, LogItem
from enum import Enum
from agent import Agent, ModelConfig
import models


class MyFaiss(FAISS):
    # override aget_by_ids
    def get_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        # return all self.docstore._dict[id] in ids
        return [self.docstore._dict[id] for id in (ids if isinstance(ids, list) else [ids]) if id in self.docstore._dict]  # type: ignore

    async def aget_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        return self.get_by_ids(ids)

    def get_all_docs(self):
        return self.docstore._dict  # type: ignore


class Memory:

    class Area(Enum):
        MAIN = "main"
        FRAGMENTS = "fragments"
        SOLUTIONS = "solutions"
        INSTRUMENTS = "instruments"

    index: dict[str, "MyFaiss"] = {}

    @staticmethod
    async def get(agent: Agent):
        memory_subdir = agent.config.memory_subdir or "default"
        if Memory.index.get(memory_subdir) is None:
            log_item = agent.context.log.log(
                type="util",
                heading=f"Initializing VectorDB in '/{memory_subdir}'",
            )
            db, created = Memory.initialize(
                log_item,
                agent.config.embeddings_model,
                memory_subdir,
                False,
            )
            Memory.index[memory_subdir] = db
            wrap = Memory(agent, db, memory_subdir=memory_subdir)
            if agent.config.knowledge_subdirs:
                await wrap.preload_knowledge(
                    log_item, agent.config.knowledge_subdirs, memory_subdir
                )
            return wrap
        else:
            return Memory(
                agent=agent,
                db=Memory.index[memory_subdir],
                memory_subdir=memory_subdir,
            )

    @staticmethod
    async def reload(agent: Agent):
        memory_subdir = agent.config.memory_subdir or "default"
        if Memory.index.get(memory_subdir):
            del Memory.index[memory_subdir]
        return await Memory.get(agent)

    @staticmethod
    def initialize(
        log_item: LogItem | None,
        model_config: ModelConfig,
        memory_subdir: str,
        in_memory=False,
    ) -> tuple[MyFaiss, bool]:

        PrintStyle.standard("Initializing VectorDB...")

        if log_item:
            log_item.stream(progress="\nInitializing VectorDB")

        em_dir = files.get_abs_path(
            "memory/embeddings"
        )  # just caching, no need to parameterize
        db_dir = Memory._abs_db_dir(memory_subdir)

        # make sure embeddings and database directories exist
        os.makedirs(db_dir, exist_ok=True)

        if in_memory:
            store = InMemoryByteStore()
        else:
            os.makedirs(em_dir, exist_ok=True)
            store = LocalFileStore(em_dir)

        embeddings_model = models.get_model(
            models.ModelType.EMBEDDING,
            model_config.provider,
            model_config.name,
            **model_config.kwargs,
        )
        embeddings_model_id = files.safe_file_name(
            model_config.provider.name + "_" + model_config.name
        )

        # here we setup the embeddings model with the chosen cache storage
        embedder = CacheBackedEmbeddings.from_bytes_store(
            embeddings_model, store, namespace=embeddings_model_id
        )

        # initial DB and docs variables
        db: MyFaiss | None = None
        docs: dict[str, Document] | None = None

        created = False

        # if db folder exists and is not empty:
        if os.path.exists(db_dir) and files.exists(db_dir, "index.faiss"):
            db = MyFaiss.load_local(
                folder_path=db_dir,
                embeddings=embedder,
                allow_dangerous_deserialization=True,
                distance_strategy=DistanceStrategy.COSINE,
                # normalize_L2=True,
                relevance_score_fn=Memory._cosine_normalizer,
            )  # type: ignore

            # if there is a mismatch in embeddings used, re-index the whole DB
            emb_ok = False
            emb_set_file = files.get_abs_path(db_dir, "embedding.json")
            if files.exists(emb_set_file):
                embedding_set = json.loads(files.read_file(emb_set_file))
                if (
                    embedding_set["model_provider"] == model_config.provider.name
                    and embedding_set["model_name"] == model_config.name
                ):
                    # model matches
                    emb_ok = True

            # re-index -  create new DB and insert existing docs
            if db and not emb_ok:
                docs = db.get_all_docs()
                db = None

        # DB not loaded, create one
        if not db:
            index = faiss.IndexFlatIP(len(embedder.embed_query("example")))

            db = MyFaiss(
                embedding_function=embedder,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
                distance_strategy=DistanceStrategy.COSINE,
                # normalize_L2=True,
                relevance_score_fn=Memory._cosine_normalizer,
            )

            # insert docs if reindexing
            if docs:
                PrintStyle.standard("Indexing memories...")
                if log_item:
                    log_item.stream(progress="\nIndexing memories")
                db.add_documents(documents=list(docs.values()), ids=list(docs.keys()))

            # save DB
            Memory._save_db_file(db, memory_subdir)
            # save meta file
            meta_file_path = files.get_abs_path(db_dir, "embedding.json")
            files.write_file(
                meta_file_path,
                json.dumps(
                    {
                        "model_provider": model_config.provider.name,
                        "model_name": model_config.name,
                    }
                ),
            )

            created = True

        return db, created

    def __init__(
        self,
        agent: Agent,
        db: MyFaiss,
        memory_subdir: str,
    ):
        self.agent = agent
        self.db = db
        self.memory_subdir = memory_subdir

    async def preload_knowledge(
        self, log_item: LogItem | None, kn_dirs: list[str], memory_subdir: str
    ):
        if log_item:
            log_item.update(heading="Preloading knowledge...")

        # db abs path
        db_dir = Memory._abs_db_dir(memory_subdir)

        # Load the index file if it exists
        index_path = files.get_abs_path(db_dir, "knowledge_import.json")

        # make sure directory exists
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)

        index: dict[str, knowledge_import.KnowledgeImport] = {}
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                index = json.load(f)

        # preload knowledge folders
        index = self._preload_knowledge_folders(log_item, kn_dirs, index)

        for file in index:
            if index[file]["state"] in ["changed", "removed"] and index[file].get(
                "ids", []
            ):  # for knowledge files that have been changed or removed and have IDs
                await self.delete_documents_by_ids(
                    index[file]["ids"]
                )  # remove original version
            if index[file]["state"] == "changed":
                index[file]["ids"] = await self.insert_documents(
                    index[file]["documents"]
                )  # insert new version

        # remove index where state="removed"
        index = {k: v for k, v in index.items() if v["state"] != "removed"}

        # strip state and documents from index and save it
        for file in index:
            if "documents" in index[file]:
                del index[file]["documents"]  # type: ignore
            if "state" in index[file]:
                del index[file]["state"]  # type: ignore
        with open(index_path, "w") as f:
            json.dump(index, f)

    def _preload_knowledge_folders(
        self,
        log_item: LogItem | None,
        kn_dirs: list[str],
        index: dict[str, knowledge_import.KnowledgeImport],
    ):
        # load knowledge folders, subfolders by area
        for kn_dir in kn_dirs:
            for area in Memory.Area:
                index = knowledge_import.load_knowledge(
                    log_item,
                    files.get_abs_path("knowledge", kn_dir, area.value),
                    index,
                    {"area": area.value},
                )

        # load instruments descriptions
        index = knowledge_import.load_knowledge(
            log_item,
            files.get_abs_path("instruments"),
            index,
            {"area": Memory.Area.INSTRUMENTS.value},
            filename_pattern="**/*.md",
        )

        return index

    async def search_similarity_threshold(
        self, query: str, limit: int, threshold: float, filter: str = ""
    ):
        comparator = Memory._get_comparator(filter) if filter else None

        # rate limiter
        await self.agent.rate_limiter(
            model_config=self.agent.config.embeddings_model, input=query
        )

        return await self.db.asearch(
            query,
            search_type="similarity_score_threshold",
            k=limit,
            score_threshold=threshold,
            filter=comparator,
        )

    async def delete_documents_by_query(
        self, query: str, threshold: float, filter: str = ""
    ):
        k = 100
        tot = 0
        removed = []

        while True:
            # Perform similarity search with score
            docs = await self.search_similarity_threshold(
                query, limit=k, threshold=threshold, filter=filter
            )
            removed += docs

            # Extract document IDs and filter based on score
            # document_ids = [result[0].metadata["id"] for result in docs if result[1] < score_limit]
            document_ids = [result.metadata["id"] for result in docs]

            # Delete documents with IDs over the threshold score
            if document_ids:
                # fnd = self.db.get(where={"id": {"$in": document_ids}})
                # if fnd["ids"]: self.db.delete(ids=fnd["ids"])
                # tot += len(fnd["ids"])
                self.db.delete(ids=document_ids)
                tot += len(document_ids)

            # If fewer than K document IDs, break the loop
            if len(document_ids) < k:
                break

        if tot:
            self._save_db()  # persist
        return removed

    async def delete_documents_by_ids(self, ids: list[str]):
        # aget_by_ids is not yet implemented in faiss, need to do a workaround
        rem_docs = self.db.get_by_ids(ids)  # existing docs to remove (prevents error)
        if rem_docs:
            rem_ids = [doc.metadata["id"] for doc in rem_docs]  # ids to remove
            await self.db.adelete(ids=rem_ids)

        if rem_docs:
            self._save_db()  # persist
        return rem_docs

    async def insert_text(self, text, metadata: dict = {}):
        doc = Document(text, metadata=metadata)
        ids = await self.insert_documents([doc])
        return ids[0]

    async def insert_documents(self, docs: list[Document]):
        ids = [str(uuid.uuid4()) for _ in range(len(docs))]
        timestamp = self.get_timestamp()

        if ids:
            for doc, id in zip(docs, ids):
                doc.metadata["id"] = id  # add ids to documents metadata
                doc.metadata["timestamp"] = timestamp  # add timestamp
                if not doc.metadata.get("area", ""):
                    doc.metadata["area"] = Memory.Area.MAIN.value

            # rate limiter
            docs_txt = "".join(self.format_docs_plain(docs))
            await self.agent.rate_limiter(
                model_config=self.agent.config.embeddings_model, input=docs_txt
            )

            self.db.add_documents(documents=docs, ids=ids)
            self._save_db()  # persist
        return ids

    def _save_db(self):
        Memory._save_db_file(self.db, self.memory_subdir)

    @staticmethod
    def _save_db_file(db: MyFaiss, memory_subdir: str):
        abs_dir = Memory._abs_db_dir(memory_subdir)
        db.save_local(folder_path=abs_dir)

    @staticmethod
    def _get_comparator(condition: str):
        def comparator(data: dict[str, Any]):
            try:
                return eval(condition, {}, data)
            except Exception as e:
                # PrintStyle.error(f"Error evaluating condition: {e}")
                return False

        return comparator

    @staticmethod
    def _score_normalizer(val: float) -> float:
        res = 1 - 1 / (1 + np.exp(val))
        return res

    @staticmethod
    def _cosine_normalizer(val: float) -> float:
        res = (1 + val) / 2
        res = max(
            0, min(1, res)
        )  # float precision can cause values like 1.0000000596046448
        return res

    @staticmethod
    def _abs_db_dir(memory_subdir: str) -> str:
        return files.get_abs_path("memory", memory_subdir)

    @staticmethod
    def format_docs_plain(docs: list[Document]) -> list[str]:
        result = []
        for doc in docs:
            text = ""
            for k, v in doc.metadata.items():
                text += f"{k}: {v}\n"
            text += f"Content: {doc.page_content}"
            result.append(text)
        return result

    @staticmethod
    def get_timestamp():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_memory_subdir_abs(agent: Agent) -> str:
    return files.get_abs_path("memory", agent.config.memory_subdir or "default")


def get_custom_knowledge_subdir_abs(agent: Agent) -> str:
    for dir in agent.config.knowledge_subdirs:
        if dir != "default":
            return files.get_abs_path("knowledge", dir)
    raise Exception("No custom knowledge subdir set")


def reload():
    # clear the memory index, this will force all DBs to reload
    Memory.index = {}



================================================
File: python/helpers/messages.py
================================================
# from . import files

import json


def truncate_text(agent, output, threshold=1000):
    threshold = int(threshold)
    if not threshold or len(output) <= threshold:
        return output

    # Adjust the file path as needed
    placeholder = agent.read_prompt(
        "fw.msg_truncated.md", length=(len(output) - threshold)
    )
    # placeholder = files.read_file("./prompts/default/fw.msg_truncated.md", length=(len(output) - threshold))

    start_len = (threshold - len(placeholder)) // 2
    end_len = threshold - len(placeholder) - start_len

    truncated_output = output[:start_len] + placeholder + output[-end_len:]
    return truncated_output


def truncate_dict_by_ratio(agent, data: dict|list|str, threshold_chars: int, truncate_to: int):
    threshold_chars = int(threshold_chars)
    truncate_to = int(truncate_to)
    
    def process_item(item):
        if isinstance(item, dict):
            truncated_dict = {}
            cumulative_size = 0

            for key, value in item.items():
                processed_value = process_item(value)
                serialized_value = json.dumps(processed_value, ensure_ascii=False)
                size = len(serialized_value)

                if cumulative_size + size > threshold_chars:
                    truncated_dict[key] = truncate_text(
                        agent, serialized_value, truncate_to
                    )
                else:
                    cumulative_size += size
                    truncated_dict[key] = processed_value

            return truncated_dict

        elif isinstance(item, list):
            truncated_list = []
            cumulative_size = 0

            for value in item:
                processed_value = process_item(value)
                serialized_value = json.dumps(processed_value, ensure_ascii=False)
                size = len(serialized_value)

                if cumulative_size + size > threshold_chars:
                    truncated_list.append(
                        truncate_text(agent, serialized_value, truncate_to)
                    )
                else:
                    cumulative_size += size
                    truncated_list.append(processed_value)

            return truncated_list

        elif isinstance(item, str):
            if len(item) > threshold_chars:
                return truncate_text(agent, item, truncate_to)
            return item

        else:
            return item

    return process_item(data)



================================================
File: python/helpers/perplexity_search.py
================================================

from openai import OpenAI
import models

def perplexity_search(query:str, model_name="llama-3.1-sonar-large-128k-online",api_key=None,base_url="https://api.perplexity.ai"):    
    api_key = api_key or models.get_api_key("perplexity")

    client = OpenAI(api_key=api_key, base_url=base_url)
        
    messages = [
    #It is recommended to use only single-turn conversations and avoid system prompts for the online LLMs (sonar-small-online and sonar-medium-online).
    
    # {
    #     "role": "system",
    #     "content": (
    #         "You are an artificial intelligence assistant and you need to "
    #         "engage in a helpful, detailed, polite conversation with a user."
    #     ),
    # },
    {
        "role": "user",
        "content": (
            query
        ),
    },
    ]
    
    response = client.chat.completions.create(
        model=model_name,
        messages=messages, # type: ignore
    )
    result = response.choices[0].message.content #only the text is returned
    return result


================================================
File: python/helpers/persist_chat.py
================================================
from collections import OrderedDict
from datetime import datetime
from typing import Any
import uuid
from agent import Agent, AgentConfig, AgentContext
from python.helpers import files, history
import json
from initialize import initialize

from python.helpers.log import Log, LogItem

CHATS_FOLDER = "tmp/chats"
LOG_SIZE = 1000
CHAT_FILE_NAME = "chat.json"


def get_chat_folder_path(ctxid: str):
    """
    Get the folder path for any context (chat or task).

    Args:
        ctxid: The context ID

    Returns:
        The absolute path to the context folder
    """
    return files.get_abs_path(CHATS_FOLDER, ctxid)


def save_tmp_chat(context: AgentContext):
    """Save context to the chats folder"""
    path = _get_chat_file_path(context.id)
    files.make_dirs(path)
    data = _serialize_context(context)
    js = _safe_json_serialize(data, ensure_ascii=False)
    files.write_file(path, js)


def load_tmp_chats():
    """Load all contexts from the chats folder"""
    _convert_v080_chats()
    folders = files.list_files(CHATS_FOLDER, "*")
    json_files = []
    for folder_name in folders:
        json_files.append(_get_chat_file_path(folder_name))

    ctxids = []
    for file in json_files:
        try:
            js = files.read_file(file)
            data = json.loads(js)
            ctx = _deserialize_context(data)
            ctxids.append(ctx.id)
        except Exception as e:
            print(f"Error loading chat {file}: {e}")
    return ctxids


def _get_chat_file_path(ctxid: str):
    return files.get_abs_path(CHATS_FOLDER, ctxid, CHAT_FILE_NAME)


def _convert_v080_chats():
    json_files = files.list_files(CHATS_FOLDER, "*.json")
    for file in json_files:
        path = files.get_abs_path(CHATS_FOLDER, file)
        name = file.rstrip(".json")
        new = _get_chat_file_path(name)
        files.move_file(path, new)


def load_json_chats(jsons: list[str]):
    """Load contexts from JSON strings"""
    ctxids = []
    for js in jsons:
        data = json.loads(js)
        if "id" in data:
            del data["id"]  # remove id to get new
        ctx = _deserialize_context(data)
        ctxids.append(ctx.id)
    return ctxids


def export_json_chat(context: AgentContext):
    """Export context as JSON string"""
    data = _serialize_context(context)
    js = _safe_json_serialize(data, ensure_ascii=False)
    return js


def remove_chat(ctxid):
    """Remove a chat or task context"""
    path = get_chat_folder_path(ctxid)
    files.delete_dir(path)


def _serialize_context(context: AgentContext):
    # serialize agents
    agents = []
    agent = context.agent0
    while agent:
        agents.append(_serialize_agent(agent))
        agent = agent.data.get(Agent.DATA_NAME_SUBORDINATE, None)

    return {
        "id": context.id,
        "name": context.name,
        "created_at": (
            context.created_at.isoformat() if context.created_at
            else datetime.fromtimestamp(0).isoformat()
        ),
        "agents": agents,
        "streaming_agent": (
            context.streaming_agent.number if context.streaming_agent else 0
        ),
        "log": _serialize_log(context.log),
    }


def _serialize_agent(agent: Agent):
    data = {k: v for k, v in agent.data.items() if not k.startswith("_")}

    history = agent.history.serialize()

    return {
        "number": agent.number,
        "data": data,
        "history": history,
    }


def _serialize_log(log: Log):
    return {
        "guid": log.guid,
        "logs": [
            item.output() for item in log.logs[-LOG_SIZE:]
        ],  # serialize LogItem objects
        "progress": log.progress,
        "progress_no": log.progress_no,
    }


def _deserialize_context(data):
    config = initialize()
    log = _deserialize_log(data.get("log", None))

    context = AgentContext(
        config=config,
        id=data.get("id", None),  # get new id
        name=data.get("name", None),
        created_at=(
            datetime.fromisoformat(
                # older chats may not have created_at - backcompat
                data.get("created_at", datetime.fromtimestamp(0).isoformat())
            )
        ),
        log=log,
        paused=False,
        # agent0=agent0,
        # streaming_agent=straming_agent,
    )

    agents = data.get("agents", [])
    agent0 = _deserialize_agents(agents, config, context)
    streaming_agent = agent0
    while streaming_agent.number != data.get("streaming_agent", 0):
        streaming_agent = streaming_agent.data.get(Agent.DATA_NAME_SUBORDINATE, None)

    context.agent0 = agent0
    context.streaming_agent = streaming_agent

    return context


def _deserialize_agents(
    agents: list[dict[str, Any]], config: AgentConfig, context: AgentContext
) -> Agent:
    prev: Agent | None = None
    zero: Agent | None = None

    for ag in agents:
        current = Agent(
            number=ag["number"],
            config=config,
            context=context,
        )
        current.data = ag.get("data", {})
        current.history = history.deserialize_history(
            ag.get("history", ""), agent=current
        )
        if not zero:
            zero = current

        if prev:
            prev.set_data(Agent.DATA_NAME_SUBORDINATE, current)
            current.set_data(Agent.DATA_NAME_SUPERIOR, prev)
        prev = current

    return zero or Agent(0, config, context)


# def _deserialize_history(history: list[dict[str, Any]]):
#     result = []
#     for hist in history:
#         content = hist.get("content", "")
#         msg = (
#             HumanMessage(content=content)
#             if hist.get("type") == "human"
#             else AIMessage(content=content)
#         )
#         result.append(msg)
#     return result


def _deserialize_log(data: dict[str, Any]) -> "Log":
    log = Log()
    log.guid = data.get("guid", str(uuid.uuid4()))
    log.set_initial_progress()

    # Deserialize the list of LogItem objects
    i = 0
    for item_data in data.get("logs", []):
        log.logs.append(
            LogItem(
                log=log,  # restore the log reference
                no=i,  # item_data["no"],
                type=item_data["type"],
                heading=item_data.get("heading", ""),
                content=item_data.get("content", ""),
                kvps=OrderedDict(item_data["kvps"]) if item_data["kvps"] else None,
                temp=item_data.get("temp", False),
            )
        )
        log.updates.append(i)
        i += 1

    return log


def _safe_json_serialize(obj, **kwargs):
    def serializer(o):
        if isinstance(o, dict):
            return {k: v for k, v in o.items() if is_json_serializable(v)}
        elif isinstance(o, (list, tuple)):
            return [item for item in o if is_json_serializable(item)]
        elif is_json_serializable(o):
            return o
        else:
            return None  # Skip this property

    def is_json_serializable(item):
        try:
            json.dumps(item)
            return True
        except (TypeError, OverflowError):
            return False

    return json.dumps(obj, default=serializer, **kwargs)



================================================
File: python/helpers/print_catch.py
================================================
import asyncio
import io
import sys
from typing import Callable, Any, Awaitable, Tuple

def capture_prints_async(
    func: Callable[..., Awaitable[Any]],
    *args,
    **kwargs
) -> Tuple[Awaitable[Any], Callable[[], str]]:
    # Create a StringIO object to capture the output
    captured_output = io.StringIO()
    original_stdout = sys.stdout

    # Define a function to get the current captured output
    def get_current_output() -> str:
        return captured_output.getvalue()

    async def wrapped_func() -> Any:
        nonlocal captured_output, original_stdout
        try:
            # Redirect sys.stdout to the StringIO object
            sys.stdout = captured_output
            # Await the provided function
            return await func(*args, **kwargs)
        finally:
            # Restore the original sys.stdout
            sys.stdout = original_stdout

    # Return the wrapped awaitable and the output retriever
    return asyncio.create_task(wrapped_func()), get_current_output


================================================
File: python/helpers/print_style.py
================================================
import os, webcolors, html
import sys
from datetime import datetime
from . import files

class PrintStyle:
    last_endline = True
    log_file_path = None

    def __init__(self, bold=False, italic=False, underline=False, font_color="default", background_color="default", padding=False, log_only=False):
        self.bold = bold
        self.italic = italic
        self.underline = underline
        self.font_color = font_color
        self.background_color = background_color
        self.padding = padding
        self.padding_added = False  # Flag to track if padding was added
        self.log_only = log_only

        if PrintStyle.log_file_path is None:
            logs_dir = files.get_abs_path("logs")
            os.makedirs(logs_dir, exist_ok=True)
            log_filename = datetime.now().strftime("log_%Y%m%d_%H%M%S.html")
            PrintStyle.log_file_path = os.path.join(logs_dir, log_filename)
            with open(PrintStyle.log_file_path, "w") as f:
                f.write("<html><body style='background-color:black;font-family: Arial, Helvetica, sans-serif;'><pre>\n")

    def _get_rgb_color_code(self, color, is_background=False):
        try:
            if color.startswith("#") and len(color) == 7:
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
            else:
                rgb_color = webcolors.name_to_rgb(color)
                r, g, b = rgb_color.red, rgb_color.green, rgb_color.blue

            if is_background:
                return f"\033[48;2;{r};{g};{b}m", f"background-color: rgb({r}, {g}, {b});"
            else:
                return f"\033[38;2;{r};{g};{b}m", f"color: rgb({r}, {g}, {b});"
        except ValueError:
            return "", ""

    def _get_styled_text(self, text):
        start = ""
        end = "\033[0m"  # Reset ANSI code
        if self.bold:
            start += "\033[1m"
        if self.italic:
            start += "\033[3m"
        if self.underline:
            start += "\033[4m"
        font_color_code, _ = self._get_rgb_color_code(self.font_color)
        background_color_code, _ = self._get_rgb_color_code(self.background_color, True)
        start += font_color_code
        start += background_color_code
        return start + text + end

    def _get_html_styled_text(self, text):
        styles = []
        if self.bold:
            styles.append("font-weight: bold;")
        if self.italic:
            styles.append("font-style: italic;")
        if self.underline:
            styles.append("text-decoration: underline;")
        _, font_color_code = self._get_rgb_color_code(self.font_color)
        _, background_color_code = self._get_rgb_color_code(self.background_color, True)
        styles.append(font_color_code)
        styles.append(background_color_code)
        style_attr = " ".join(styles)
        escaped_text = html.escape(text).replace("\n", "<br>")  # Escape HTML special characters
        return f'<span style="{style_attr}">{escaped_text}</span>'

    def _add_padding_if_needed(self):
        if self.padding and not self.padding_added:
            if not self.log_only:
                print()  # Print an empty line for padding
            self._log_html("<br>")
            self.padding_added = True

    def _log_html(self, html):
        with open(PrintStyle.log_file_path, "a", encoding='utf-8') as f: # type: ignore # add encoding='utf-8'
            f.write(html)

    @staticmethod
    def _close_html_log():
        if PrintStyle.log_file_path:
            with open(PrintStyle.log_file_path, "a") as f:
                f.write("</pre></body></html>")

    def get(self, *args, sep=' ', **kwargs):
        text = sep.join(map(str, args))
        return text, self._get_styled_text(text), self._get_html_styled_text(text)

    def print(self, *args, sep=' ', **kwargs):
        self._add_padding_if_needed()
        if not PrintStyle.last_endline:
            print()
            self._log_html("<br>")
        plain_text, styled_text, html_text = self.get(*args, sep=sep, **kwargs)
        if not self.log_only:
            print(styled_text, end='\n', flush=True)
        self._log_html(html_text+"<br>\n")
        PrintStyle.last_endline = True

    def stream(self, *args, sep=' ', **kwargs):
        self._add_padding_if_needed()
        plain_text, styled_text, html_text = self.get(*args, sep=sep, **kwargs)
        if not self.log_only:
            print(styled_text, end='', flush=True)
        self._log_html(html_text)
        PrintStyle.last_endline = False

    def is_last_line_empty(self):
        lines = sys.stdin.readlines()
        return bool(lines) and not lines[-1].strip()

    @staticmethod
    def standard(text: str):
        PrintStyle().print(text)

    @staticmethod
    def hint(text: str):
        PrintStyle(font_color="#6C3483", padding=True).print("Hint: "+text)

    @staticmethod
    def info(text: str):
        PrintStyle(font_color="#0000FF", padding=True).print("Info: "+text)

    @staticmethod
    def success(text: str):
        PrintStyle(font_color="#008000", padding=True).print("Success: "+text)

    @staticmethod
    def warning(text: str):
        PrintStyle(font_color="#FFA500", padding=True).print("Warning: "+text)

    @staticmethod
    def debug(text: str):
        PrintStyle(font_color="#808080", padding=True).print("Debug: "+text)

    @staticmethod
    def error(text: str):
        PrintStyle(font_color="red", padding=True).print("Error: "+text)

# Ensure HTML file is closed properly when the program exits
import atexit
atexit.register(PrintStyle._close_html_log)



================================================
File: python/helpers/process.py
================================================
import os
import sys
from python.helpers import runtime
from python.helpers.print_style import PrintStyle

_server = None

def set_server(server):
    global _server
    _server = server

def get_server(server):
    global _server
    return _server

def stop_server():
    global _server
    if _server:
        _server.shutdown()
        _server = None

def reload():
    stop_server()
    if runtime.is_dockerized():
        exit_process()
    else:
        restart_process()

def restart_process():
    PrintStyle.standard("Restarting process...")
    python = sys.executable
    os.execv(python, [python] + sys.argv)

def exit_process():
    PrintStyle.standard("Exiting process...")
    sys.exit(0)


================================================
File: python/helpers/rag.py
================================================
from typing import List

from langchain_core.documents import Document
from python.helpers import files

from langchain_community.document_loaders import (
    CSVLoader,
    JSONLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)

# def extract_file(path: str) -> List[Document]:
#     pass  # TODO finish implementing

def extract_text(content: bytes, chunk_size: int = 128) -> List[str]:
    result = []

    def is_binary_chunk(chunk: bytes) -> bool:
        # Check for high concentration of control chars
        try:
            text = chunk.decode("utf-8", errors="ignore")
            control_chars = sum(1 for c in text if ord(c) < 32 and c not in "\n\r\t")
            return control_chars / len(text) > 0.3
        except UnicodeDecodeError:
            return True

    # Process the content in overlapping chunks to handle boundaries
    pos = 0
    while pos < len(content):
        # Get current chunk with overlap
        chunk_end = min(pos + chunk_size, len(content))

        # Add overlap to catch word boundaries, unless at end of content
        if chunk_end < len(content):
            # Look ahead for next newline or space to avoid splitting words
            for i in range(chunk_end, min(chunk_end + 100, len(content))):
                if content[i : i + 1] in [b" ", b"\n", b"\r"]:
                    chunk_end = i + 1
                    break

        chunk = content[pos:chunk_end]

        if is_binary_chunk(chunk):
            if not result or result[-1] != "[BINARY]":
                result.append("[BINARY]")
        else:
            try:
                text = chunk.decode("utf-8", errors="ignore").strip()
                if text:  # Only add non-empty text chunks
                    result.append(text)
            except UnicodeDecodeError:
                if not result or result[-1] != "[BINARY]":
                    result.append("[BINARY]")

        pos = chunk_end

    return result



================================================
File: python/helpers/rate_limiter.py
================================================
import asyncio
import time
from typing import Callable, Awaitable


class RateLimiter:
    def __init__(self, seconds: int = 60, **limits: int):
        self.timeframe = seconds
        self.limits = {key: value if isinstance(value, (int, float)) else 0 for key, value in (limits or {}).items()}
        self.values = {key: [] for key in self.limits.keys()}
        self._lock = asyncio.Lock()

    def add(self, **kwargs: int):
        now = time.time()
        for key, value in kwargs.items():
            if not key in self.values:
                self.values[key] = []
            self.values[key].append((now, value))

    async def cleanup(self):
        async with self._lock:
            now = time.time()
            cutoff = now - self.timeframe
            for key in self.values:
                self.values[key] = [(t, v) for t, v in self.values[key] if t > cutoff]

    async def get_total(self, key: str) -> int:
        async with self._lock:
            if not key in self.values:
                return 0
            return sum(value for _, value in self.values[key])

    async def wait(
        self,
        callback: Callable[[str, str, int, int], Awaitable[None]] | None = None,
    ):
        while True:
            await self.cleanup()
            should_wait = False

            for key, limit in self.limits.items():
                if limit <= 0:  # Skip if no limit set
                    continue

                total = await self.get_total(key)
                if total > limit:
                    if callback:
                        msg = f"Rate limit exceeded for {key} ({total}/{limit}), waiting..."
                        await callback(msg, key, total, limit)
                    should_wait = True
                    break

            if not should_wait:
                break

            await asyncio.sleep(1)



================================================
File: python/helpers/rfc.py
================================================
import importlib
import inspect
import json
from typing import Any, TypedDict
import aiohttp
from python.helpers import crypto

from python.helpers import dotenv


# Remote Function Call library
# Call function via http request
# Secured by pre-shared key


class RFCInput(TypedDict):
    module: str
    function_name: str
    args: list[Any]
    kwargs: dict[str, Any]


class RFCCall(TypedDict):
    rfc_input: str
    hash: str


async def call_rfc(
    url: str, password: str, module: str, function_name: str, args: list, kwargs: dict
):
    input = RFCInput(
        module=module,
        function_name=function_name,
        args=args,
        kwargs=kwargs,
    )
    call = RFCCall(
        rfc_input=json.dumps(input), hash=crypto.hash_data(json.dumps(input), password)
    )
    result = await _send_json_data(url, call)
    return result


async def handle_rfc(rfc_call: RFCCall, password: str):
    if not crypto.verify_data(rfc_call["rfc_input"], rfc_call["hash"], password):
        raise Exception("Invalid RFC hash")

    input: RFCInput = json.loads(rfc_call["rfc_input"])
    return await _call_function(
        input["module"], input["function_name"], *input["args"], **input["kwargs"]
    )


async def _call_function(module: str, function_name: str, *args, **kwargs):
    func = _get_function(module, function_name)
    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    else:
        return func(*args, **kwargs)


def _get_function(module: str, function_name: str):
    # import module
    imp = importlib.import_module(module)
    # get function by the name
    func = getattr(imp, function_name)
    return func


async def _send_json_data(url: str, data):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            json=data,
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result
            else:
                error = await response.text()
                raise Exception(error)



================================================
File: python/helpers/rfc_exchange.py
================================================
from python.helpers import runtime, crypto, dotenv

async def get_root_password():
    if runtime.is_dockerized():
        pswd = _get_root_password()
    else:
        priv = crypto._generate_private_key()
        pub = crypto._generate_public_key(priv)
        enc = await runtime.call_development_function(_provide_root_password, pub)
        pswd = crypto.decrypt_data(enc, priv)
    return pswd
    
def _provide_root_password(public_key_pem: str):
    pswd = _get_root_password()
    enc = crypto.encrypt_data(pswd, public_key_pem)
    return enc

def _get_root_password():
    return dotenv.get_dotenv_value(dotenv.KEY_ROOT_PASSWORD) or ""


================================================
File: python/helpers/runtime.py
================================================
import argparse
import inspect
from typing import TypeVar, Callable, Awaitable, Union, overload, cast
from python.helpers import dotenv, rfc, settings
import asyncio
import threading
import queue

T = TypeVar('T')
R = TypeVar('R')

parser = argparse.ArgumentParser()
args = {}
dockerman = None


def initialize():
    global args
    if args:
        return
    parser.add_argument("--port", type=int, default=None, help="Web UI port")
    parser.add_argument("--host", type=str, default=None, help="Web UI host")
    parser.add_argument(
        "--cloudflare_tunnel",
        type=bool,
        default=False,
        help="Use cloudflare tunnel for public URL",
    )
    parser.add_argument(
        "--development", type=bool, default=False, help="Development mode"
    )

    known, unknown = parser.parse_known_args()
    args = vars(known)
    for arg in unknown:
        if "=" in arg:
            key, value = arg.split("=", 1)
            key = key.lstrip("-")
            args[key] = value


def get_arg(name: str):
    global args
    return args.get(name, None)

def has_arg(name: str):
    global args
    return name in args

def is_dockerized() -> bool:
    return get_arg("dockerized")

def is_development() -> bool:
    return not is_dockerized()

def get_local_url():
    if is_dockerized():
        return "host.docker.internal"
    return "127.0.0.1"

@overload
async def call_development_function(func: Callable[..., Awaitable[T]], *args, **kwargs) -> T: ...

@overload
async def call_development_function(func: Callable[..., T], *args, **kwargs) -> T: ...

async def call_development_function(func: Union[Callable[..., T], Callable[..., Awaitable[T]]], *args, **kwargs) -> T:
    if is_development():
        url = _get_rfc_url()
        password = _get_rfc_password()
        result = await rfc.call_rfc(
            url=url,
            password=password,
            module=func.__module__,
            function_name=func.__name__,
            args=list(args),
            kwargs=kwargs,
        )
        return cast(T, result)
    else:
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs) # type: ignore


async def handle_rfc(rfc_call: rfc.RFCCall):
    return await rfc.handle_rfc(rfc_call=rfc_call, password=_get_rfc_password())


def _get_rfc_password() -> str:
    password = dotenv.get_dotenv_value(dotenv.KEY_RFC_PASSWORD)
    if not password:
        raise Exception("No RFC password, cannot handle RFC calls.")
    return password


def _get_rfc_url() -> str:
    set = settings.get_settings()
    url = set["rfc_url"]
    if not "://" in url:
        url = "http://"+url
    if url.endswith("/"):
        url = url[:-1]
    url = url+":"+str(set["rfc_port_http"])
    url += "/rfc"
    return url


def call_development_function_sync(func: Union[Callable[..., T], Callable[..., Awaitable[T]]], *args, **kwargs) -> T:
    # run async function in sync manner
    result_queue = queue.Queue()
    
    def run_in_thread():
        result = asyncio.run(call_development_function(func, *args, **kwargs))
        result_queue.put(result)
    
    thread = threading.Thread(target=run_in_thread)
    thread.start()
    thread.join(timeout=30)  # wait for thread with timeout
    
    if thread.is_alive():
        raise TimeoutError("Function call timed out after 30 seconds")
    
    result = result_queue.get_nowait()
    return cast(T, result)


def get_web_ui_port():
    web_ui_port = (
        get_arg("port")
        or int(dotenv.get_dotenv_value("WEB_UI_PORT", 0))
        or 5000
    )
    return web_ui_port

def get_tunnel_api_port():
    tunnel_api_port = (
        get_arg("tunnel_api_port")
        or int(dotenv.get_dotenv_value("TUNNEL_API_PORT", 0))
        or 5070
    )
    return tunnel_api_port


================================================
File: python/helpers/searxng.py
================================================
import aiohttp
from python.helpers import runtime

URL = "http://localhost:8888/search"

async def search(query:str):
    return await runtime.call_development_function(_search, query=query)

async def _search(query:str):
    async with aiohttp.ClientSession() as session:
        async with session.post(URL, data={"q": query, "format": "json"}) as response:
            return await response.json()



================================================
File: python/helpers/settings.py
================================================
import json
import os
import re
import subprocess
from typing import Any, Literal, TypedDict

import models
from python.helpers import runtime, whisper, defer
from . import files, dotenv


class Settings(TypedDict):
    chat_model_provider: str
    chat_model_name: str
    chat_model_kwargs: dict[str, str]
    chat_model_ctx_length: int
    chat_model_ctx_history: float
    chat_model_vision: bool
    chat_model_rl_requests: int
    chat_model_rl_input: int
    chat_model_rl_output: int

    util_model_provider: str
    util_model_name: str
    util_model_kwargs: dict[str, str]
    util_model_ctx_length: int
    util_model_ctx_input: float
    util_model_rl_requests: int
    util_model_rl_input: int
    util_model_rl_output: int

    embed_model_provider: str
    embed_model_name: str
    embed_model_kwargs: dict[str, str]
    embed_model_rl_requests: int
    embed_model_rl_input: int

    browser_model_provider: str
    browser_model_name: str
    browser_model_vision: bool
    browser_model_kwargs: dict[str, str]

    agent_prompts_subdir: str
    agent_memory_subdir: str
    agent_knowledge_subdir: str

    api_keys: dict[str, str]

    auth_login: str
    auth_password: str
    root_password: str

    rfc_auto_docker: bool
    rfc_url: str
    rfc_password: str
    rfc_port_http: int
    rfc_port_ssh: int

    stt_model_size: str
    stt_language: str
    stt_silence_threshold: float
    stt_silence_duration: int
    stt_waiting_timeout: int


class PartialSettings(Settings, total=False):
    pass


class FieldOption(TypedDict):
    value: str
    label: str


class SettingsField(TypedDict, total=False):
    id: str
    title: str
    description: str
    type: Literal["text", "number", "select", "range", "textarea", "password", "switch"]
    value: Any
    min: float
    max: float
    step: float
    options: list[FieldOption]


class SettingsSection(TypedDict, total=False):
    id: str
    title: str
    description: str
    fields: list[SettingsField]
    tab: str  # Indicates which tab this section belongs to


class SettingsOutput(TypedDict):
    sections: list[SettingsSection]


PASSWORD_PLACEHOLDER = "****PSWD****"

SETTINGS_FILE = files.get_abs_path("tmp/settings.json")
_settings: Settings | None = None


def convert_out(settings: Settings) -> SettingsOutput:
    from models import ModelProvider

    # main model section
    chat_model_fields: list[SettingsField] = []
    chat_model_fields.append(
        {
            "id": "chat_model_provider",
            "title": "Chat model provider",
            "description": "Select provider for main chat model used by Agent Zero",
            "type": "select",
            "value": settings["chat_model_provider"],
            "options": [{"value": p.name, "label": p.value} for p in ModelProvider],
        }
    )
    chat_model_fields.append(
        {
            "id": "chat_model_name",
            "title": "Chat model name",
            "description": "Exact name of model from selected provider",
            "type": "text",
            "value": settings["chat_model_name"],
        }
    )

    chat_model_fields.append(
        {
            "id": "chat_model_ctx_length",
            "title": "Chat model context length",
            "description": "Maximum number of tokens in the context window for LLM. System prompt, chat history, RAG and response all count towards this limit.",
            "type": "number",
            "value": settings["chat_model_ctx_length"],
        }
    )

    chat_model_fields.append(
        {
            "id": "chat_model_ctx_history",
            "title": "Context window space for chat history",
            "description": "Portion of context window dedicated to chat history visible to the agent. Chat history will automatically be optimized to fit. Smaller size will result in shorter and more summarized history. The remaining space will be used for system prompt, RAG and response.",
            "type": "range",
            "min": 0.01,
            "max": 1,
            "step": 0.01,
            "value": settings["chat_model_ctx_history"],
        }
    )

    chat_model_fields.append(
        {
            "id": "chat_model_vision",
            "title": "Supports Vision",
            "description": "Models capable of Vision can for example natively see the content of image attachments.",
            "type": "switch",
            "value": settings["chat_model_vision"],
        }
    )

    chat_model_fields.append(
        {
            "id": "chat_model_rl_requests",
            "title": "Requests per minute limit",
            "description": "Limits the number of requests per minute to the chat model. Waits if the limit is exceeded. Set to 0 to disable rate limiting.",
            "type": "number",
            "value": settings["chat_model_rl_requests"],
        }
    )

    chat_model_fields.append(
        {
            "id": "chat_model_rl_input",
            "title": "Input tokens per minute limit",
            "description": "Limits the number of input tokens per minute to the chat model. Waits if the limit is exceeded. Set to 0 to disable rate limiting.",
            "type": "number",
            "value": settings["chat_model_rl_input"],
        }
    )

    chat_model_fields.append(
        {
            "id": "chat_model_rl_output",
            "title": "Output tokens per minute limit",
            "description": "Limits the number of output tokens per minute to the chat model. Waits if the limit is exceeded. Set to 0 to disable rate limiting.",
            "type": "number",
            "value": settings["chat_model_rl_output"],
        }
    )

    chat_model_fields.append(
        {
            "id": "chat_model_kwargs",
            "title": "Chat model additional parameters",
            "description": "Any other parameters supported by the model. Format is KEY=VALUE on individual lines, just like .env file.",
            "type": "textarea",
            "value": _dict_to_env(settings["chat_model_kwargs"]),
        }
    )

    chat_model_section: SettingsSection = {
        "id": "chat_model",
        "title": "Chat Model",
        "description": "Selection and settings for main chat model used by Agent Zero",
        "fields": chat_model_fields,
        "tab": "agent",
    }

    # main model section
    util_model_fields: list[SettingsField] = []
    util_model_fields.append(
        {
            "id": "util_model_provider",
            "title": "Utility model provider",
            "description": "Select provider for utility model used by the framework",
            "type": "select",
            "value": settings["util_model_provider"],
            "options": [{"value": p.name, "label": p.value} for p in ModelProvider],
        }
    )
    util_model_fields.append(
        {
            "id": "util_model_name",
            "title": "Utility model name",
            "description": "Exact name of model from selected provider",
            "type": "text",
            "value": settings["util_model_name"],
        }
    )

    util_model_fields.append(
        {
            "id": "util_model_rl_requests",
            "title": "Requests per minute limit",
            "description": "Limits the number of requests per minute to the utility model. Waits if the limit is exceeded. Set to 0 to disable rate limiting.",
            "type": "number",
            "value": settings["util_model_rl_requests"],
        }
    )

    util_model_fields.append(
        {
            "id": "util_model_rl_input",
            "title": "Input tokens per minute limit",
            "description": "Limits the number of input tokens per minute to the utility model. Waits if the limit is exceeded. Set to 0 to disable rate limiting.",
            "type": "number",
            "value": settings["util_model_rl_input"],
        }
    )

    util_model_fields.append(
        {
            "id": "util_model_rl_output",
            "title": "Output tokens per minute limit",
            "description": "Limits the number of output tokens per minute to the utility model. Waits if the limit is exceeded. Set to 0 to disable rate limiting.",
            "type": "number",
            "value": settings["util_model_rl_output"],
        }
    )

    util_model_fields.append(
        {
            "id": "util_model_kwargs",
            "title": "Utility model additional parameters",
            "description": "Any other parameters supported by the model. Format is KEY=VALUE on individual lines, just like .env file.",
            "type": "textarea",
            "value": _dict_to_env(settings["util_model_kwargs"]),
        }
    )

    util_model_section: SettingsSection = {
        "id": "util_model",
        "title": "Utility model",
        "description": "Smaller, cheaper, faster model for handling utility tasks like organizing memory, preparing prompts, summarizing.",
        "fields": util_model_fields,
        "tab": "agent",
    }

    # embedding model section
    embed_model_fields: list[SettingsField] = []
    embed_model_fields.append(
        {
            "id": "embed_model_provider",
            "title": "Embedding model provider",
            "description": "Select provider for embedding model used by the framework",
            "type": "select",
            "value": settings["embed_model_provider"],
            "options": [{"value": p.name, "label": p.value} for p in ModelProvider],
        }
    )
    embed_model_fields.append(
        {
            "id": "embed_model_name",
            "title": "Embedding model name",
            "description": "Exact name of model from selected provider",
            "type": "text",
            "value": settings["embed_model_name"],
        }
    )

    embed_model_fields.append(
        {
            "id": "embed_model_rl_requests",
            "title": "Requests per minute limit",
            "description": "Limits the number of requests per minute to the embedding model. Waits if the limit is exceeded. Set to 0 to disable rate limiting.",
            "type": "number",
            "value": settings["embed_model_rl_requests"],
        }
    )

    embed_model_fields.append(
        {
            "id": "embed_model_rl_input",
            "title": "Input tokens per minute limit",
            "description": "Limits the number of input tokens per minute to the embedding model. Waits if the limit is exceeded. Set to 0 to disable rate limiting.",
            "type": "number",
            "value": settings["embed_model_rl_input"],
        }
    )

    embed_model_fields.append(
        {
            "id": "embed_model_kwargs",
            "title": "Embedding model additional parameters",
            "description": "Any other parameters supported by the model. Format is KEY=VALUE on individual lines, just like .env file.",
            "type": "textarea",
            "value": _dict_to_env(settings["embed_model_kwargs"]),
        }
    )

    embed_model_section: SettingsSection = {
        "id": "embed_model",
        "title": "Embedding Model",
        "description": "Settings for the embedding model used by Agent Zero.",
        "fields": embed_model_fields,
        "tab": "agent",
    }

    # embedding model section
    browser_model_fields: list[SettingsField] = []
    browser_model_fields.append(
        {
            "id": "browser_model_provider",
            "title": "Web Browser model provider",
            "description": "Select provider for web browser model used by <a href='https://github.com/browser-use/browser-use' target='_blank'>browser-use</a> framework",
            "type": "select",
            "value": settings["browser_model_provider"],
            "options": [{"value": p.name, "label": p.value} for p in ModelProvider],
        }
    )
    browser_model_fields.append(
        {
            "id": "browser_model_name",
            "title": "Web Browser model name",
            "description": "Exact name of model from selected provider",
            "type": "text",
            "value": settings["browser_model_name"],
        }
    )

    browser_model_fields.append(
        {
            "id": "browser_model_vision",
            "title": "Use Vision",
            "description": "Models capable of Vision can use it to analyze web pages from screenshots. Increases quality but also token usage.",
            "type": "switch",
            "value": settings["browser_model_vision"],
        }
    )

    browser_model_fields.append(
        {
            "id": "browser_model_kwargs",
            "title": "Web Browser model additional parameters",
            "description": "Any other parameters supported by the model. Format is KEY=VALUE on individual lines, just like .env file.",
            "type": "textarea",
            "value": _dict_to_env(settings["browser_model_kwargs"]),
        }
    )

    browser_model_section: SettingsSection = {
        "id": "browser_model",
        "title": "Web Browser Model",
        "description": "Settings for the web browser model. Agent Zero uses <a href='https://github.com/browser-use/browser-use' target='_blank'>browser-use</a> agentic framework to handle web interactions.",
        "fields": browser_model_fields,
        "tab": "agent",
    }

    # # Memory settings section
    # memory_fields: list[SettingsField] = []
    # memory_fields.append(
    #     {
    #         "id": "memory_settings",
    #         "title": "Memory Settings",
    #         "description": "<settings for memory>",
    #         "type": "text",
    #         "value": "",
    #     }
    # )

    # memory_section: SettingsSection = {
    #     "id": "memory",
    #     "title": "Memory Settings",
    #     "description": "<settings for memory management here>",
    #     "fields": memory_fields,
    # }

    # basic auth section
    auth_fields: list[SettingsField] = []

    auth_fields.append(
        {
            "id": "auth_login",
            "title": "UI Login",
            "description": "Set user name for web UI",
            "type": "text",
            "value": dotenv.get_dotenv_value(dotenv.KEY_AUTH_LOGIN) or "",
        }
    )

    auth_fields.append(
        {
            "id": "auth_password",
            "title": "UI Password",
            "description": "Set user password for web UI",
            "type": "password",
            "value": (
                PASSWORD_PLACEHOLDER
                if dotenv.get_dotenv_value(dotenv.KEY_AUTH_PASSWORD)
                else ""
            ),
        }
    )

    if runtime.is_dockerized():
        auth_fields.append(
            {
                "id": "root_password",
                "title": "root Password",
                "description": "Change linux root password in docker container. This password can be used for SSH access. Original password was randomly generated during setup.",
                "type": "password",
                "value": "",
            }
        )

    auth_section: SettingsSection = {
        "id": "auth",
        "title": "Authentication",
        "description": "Settings for authentication to use Agent Zero Web UI.",
        "fields": auth_fields,
        "tab": "external",
    }

    # api keys model section
    api_keys_fields: list[SettingsField] = []
    api_keys_fields.append(_get_api_key_field(settings, "openai", "OpenAI API Key"))
    api_keys_fields.append(
        _get_api_key_field(settings, "anthropic", "Anthropic API Key")
    )
    api_keys_fields.append(
        _get_api_key_field(settings, "chutes", "Chutes API Key")
    )
    api_keys_fields.append(_get_api_key_field(settings, "deepseek", "DeepSeek API Key"))
    api_keys_fields.append(_get_api_key_field(settings, "google", "Google API Key"))
    api_keys_fields.append(_get_api_key_field(settings, "groq", "Groq API Key"))
    api_keys_fields.append(
        _get_api_key_field(settings, "huggingface", "HuggingFace API Key")
    )
    api_keys_fields.append(
        _get_api_key_field(settings, "mistralai", "MistralAI API Key")
    )
    api_keys_fields.append(
        _get_api_key_field(settings, "openrouter", "OpenRouter API Key")
    )
    api_keys_fields.append(
        _get_api_key_field(settings, "sambanova", "Sambanova API Key")
    )

    api_keys_section: SettingsSection = {
        "id": "api_keys",
        "title": "API Keys",
        "description": "API keys for model providers and services used by Agent Zero.",
        "fields": api_keys_fields,
        "tab": "external",
    }

    # Agent config section
    agent_fields: list[SettingsField] = []

    agent_fields.append(
        {
            "id": "agent_prompts_subdir",
            "title": "Prompts Subdirectory",
            "description": "Subdirectory of /prompts folder to use for agent prompts. Used to adjust agent behaviour.",
            "type": "select",
            "value": settings["agent_prompts_subdir"],
            "options": [
                {"value": subdir, "label": subdir}
                for subdir in files.get_subdirectories("prompts")
            ],
        }
    )

    agent_fields.append(
        {
            "id": "agent_memory_subdir",
            "title": "Memory Subdirectory",
            "description": "Subdirectory of /memory folder to use for agent memory storage. Used to separate memory storage between different instances.",
            "type": "text",
            "value": settings["agent_memory_subdir"],
            # "options": [
            #     {"value": subdir, "label": subdir}
            #     for subdir in files.get_subdirectories("memory", exclude="embeddings")
            # ],
        }
    )

    agent_fields.append(
        {
            "id": "agent_knowledge_subdir",
            "title": "Knowledge subdirectory",
            "description": "Subdirectory of /knowledge folder to use for agent knowledge import. 'default' subfolder is always imported and contains framework knowledge.",
            "type": "select",
            "value": settings["agent_knowledge_subdir"],
            "options": [
                {"value": subdir, "label": subdir}
                for subdir in files.get_subdirectories("knowledge", exclude="default")
            ],
        }
    )

    agent_section: SettingsSection = {
        "id": "agent",
        "title": "Agent Config",
        "description": "Agent parameters.",
        "fields": agent_fields,
        "tab": "agent",
    }

    dev_fields: list[SettingsField] = []

    if runtime.is_development():
        # dev_fields.append(
        #     {
        #         "id": "rfc_auto_docker",
        #         "title": "RFC Auto Docker Management",
        #         "description": "Automatically create dockerized instance of A0 for RFCs using this instance's code base and, settings and .env.",
        #         "type": "text",
        #         "value": settings["rfc_auto_docker"],
        #     }
        # )

        dev_fields.append(
            {
                "id": "rfc_url",
                "title": "RFC Destination URL",
                "description": "URL of dockerized A0 instance for remote function calls. Do not specify port here.",
                "type": "text",
                "value": settings["rfc_url"],
            }
        )

    dev_fields.append(
        {
            "id": "rfc_password",
            "title": "RFC Password",
            "description": "Password for remote function calls. Passwords must match on both instances. RFCs can not be used with empty password.",
            "type": "password",
            "value": (
                PASSWORD_PLACEHOLDER
                if dotenv.get_dotenv_value(dotenv.KEY_RFC_PASSWORD)
                else ""
            ),
        }
    )

    if runtime.is_development():
        dev_fields.append(
            {
                "id": "rfc_port_http",
                "title": "RFC HTTP port",
                "description": "HTTP port for dockerized instance of A0.",
                "type": "text",
                "value": settings["rfc_port_http"],
            }
        )

        dev_fields.append(
            {
                "id": "rfc_port_ssh",
                "title": "RFC SSH port",
                "description": "SSH port for dockerized instance of A0.",
                "type": "text",
                "value": settings["rfc_port_ssh"],
            }
        )

    dev_section: SettingsSection = {
        "id": "dev",
        "title": "Development",
        "description": "Parameters for A0 framework development. RFCs (remote function calls) are used to call functions on another A0 instance. You can develop and debug A0 natively on your local system while redirecting some functions to A0 instance in docker. This is crucial for development as A0 needs to run in standardized environment to support all features.",
        "fields": dev_fields,
        "tab": "developer",
    }

    # Speech to text section
    stt_fields: list[SettingsField] = []

    stt_fields.append(
        {
            "id": "stt_model_size",
            "title": "Model Size",
            "description": "Select the speech recognition model size",
            "type": "select",
            "value": settings["stt_model_size"],
            "options": [
                {"value": "tiny", "label": "Tiny (39M, English)"},
                {"value": "base", "label": "Base (74M, English)"},
                {"value": "small", "label": "Small (244M, English)"},
                {"value": "medium", "label": "Medium (769M, English)"},
                {"value": "large", "label": "Large (1.5B, Multilingual)"},
                {"value": "turbo", "label": "Turbo (Multilingual)"},
            ],
        }
    )

    stt_fields.append(
        {
            "id": "stt_language",
            "title": "Language Code",
            "description": "Language code (e.g. en, fr, it)",
            "type": "text",
            "value": settings["stt_language"],
        }
    )

    stt_fields.append(
        {
            "id": "stt_silence_threshold",
            "title": "Silence threshold",
            "description": "Silence detection threshold. Lower values are more sensitive.",
            "type": "range",
            "min": 0,
            "max": 1,
            "step": 0.01,
            "value": settings["stt_silence_threshold"],
        }
    )

    stt_fields.append(
        {
            "id": "stt_silence_duration",
            "title": "Silence duration (ms)",
            "description": "Duration of silence before the server considers speaking to have ended.",
            "type": "text",
            "value": settings["stt_silence_duration"],
        }
    )

    stt_fields.append(
        {
            "id": "stt_waiting_timeout",
            "title": "Waiting timeout (ms)",
            "description": "Duration before the server closes the microphone.",
            "type": "text",
            "value": settings["stt_waiting_timeout"],
        }
    )

    stt_section: SettingsSection = {
        "id": "stt",
        "title": "Speech to Text",
        "description": "Voice transcription preferences and server turn detection settings.",
        "fields": stt_fields,
        "tab": "agent",
    }

    # Add the section to the result
    result: SettingsOutput = {
        "sections": [
            agent_section,
            chat_model_section,
            util_model_section,
            embed_model_section,
            browser_model_section,
            # memory_section,
            stt_section,
            api_keys_section,
            auth_section,
            dev_section,
        ]
    }
    return result


def _get_api_key_field(settings: Settings, provider: str, title: str) -> SettingsField:
    key = settings["api_keys"].get(provider, models.get_api_key(provider))
    return {
        "id": f"api_key_{provider}",
        "title": title,
        "type": "password",
        "value": (PASSWORD_PLACEHOLDER if key and key != "None" else ""),
    }


def convert_in(settings: dict) -> Settings:
    current = get_settings()
    for section in settings["sections"]:
        if "fields" in section:
            for field in section["fields"]:
                if field["value"] != PASSWORD_PLACEHOLDER:
                    if field["id"].endswith("_kwargs"):
                        current[field["id"]] = _env_to_dict(field["value"])
                    elif field["id"].startswith("api_key_"):
                        current["api_keys"][field["id"]] = field["value"]
                    else:
                        current[field["id"]] = field["value"]
    return current


def get_settings() -> Settings:
    global _settings
    if not _settings:
        _settings = _read_settings_file()
    if not _settings:
        _settings = get_default_settings()
    norm = normalize_settings(_settings)
    return norm


def set_settings(settings: Settings):
    global _settings
    previous = _settings
    _settings = normalize_settings(settings)
    _write_settings_file(_settings)
    _apply_settings(previous)


def normalize_settings(settings: Settings) -> Settings:
    copy = settings.copy()
    default = get_default_settings()
    for key, value in default.items():
        if key not in copy:
            copy[key] = value
        else:
            try:
                copy[key] = type(value)(copy[key])  # type: ignore
            except (ValueError, TypeError):
                copy[key] = value  # make default instead
    return copy


def _read_settings_file() -> Settings | None:
    if os.path.exists(SETTINGS_FILE):
        content = files.read_file(SETTINGS_FILE)
        parsed = json.loads(content)
        return normalize_settings(parsed)


def _write_settings_file(settings: Settings):
    _write_sensitive_settings(settings)
    _remove_sensitive_settings(settings)

    # write settings
    content = json.dumps(settings, indent=4)
    files.write_file(SETTINGS_FILE, content)


def _remove_sensitive_settings(settings: Settings):
    settings["api_keys"] = {}
    settings["auth_login"] = ""
    settings["auth_password"] = ""
    settings["rfc_password"] = ""
    settings["root_password"] = ""


def _write_sensitive_settings(settings: Settings):
    for key, val in settings["api_keys"].items():
        dotenv.save_dotenv_value(key.upper(), val)

    dotenv.save_dotenv_value(dotenv.KEY_AUTH_LOGIN, settings["auth_login"])
    if settings["auth_password"]:
        dotenv.save_dotenv_value(dotenv.KEY_AUTH_PASSWORD, settings["auth_password"])
    if settings["rfc_password"]:
        dotenv.save_dotenv_value(dotenv.KEY_RFC_PASSWORD, settings["rfc_password"])

    if settings["root_password"]:
        dotenv.save_dotenv_value(dotenv.KEY_ROOT_PASSWORD, settings["root_password"])
    if settings["root_password"]:
        set_root_password(settings["root_password"])


def get_default_settings() -> Settings:
    from models import ModelProvider

    return Settings(
        chat_model_provider=ModelProvider.OPENAI.name,
        chat_model_name="gpt-4.1",
        chat_model_kwargs={"temperature": "0"},
        chat_model_ctx_length=100000,
        chat_model_ctx_history=0.7,
        chat_model_vision=True,
        chat_model_rl_requests=0,
        chat_model_rl_input=0,
        chat_model_rl_output=0,
        util_model_provider=ModelProvider.OPENAI.name,
        util_model_name="gpt-4.1-nano",
        util_model_ctx_length=100000,
        util_model_ctx_input=0.7,
        util_model_kwargs={"temperature": "0"},
        util_model_rl_requests=0,
        util_model_rl_input=0,
        util_model_rl_output=0,
        embed_model_provider=ModelProvider.HUGGINGFACE.name,
        embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
        embed_model_kwargs={},
        embed_model_rl_requests=0,
        embed_model_rl_input=0,
        browser_model_provider=ModelProvider.OPENAI.name,
        browser_model_name="gpt-4.1",
        browser_model_vision=True,
        browser_model_kwargs={"temperature": "0"},
        api_keys={},
        auth_login="",
        auth_password="",
        root_password="",
        agent_prompts_subdir="default",
        agent_memory_subdir="default",
        agent_knowledge_subdir="custom",
        rfc_auto_docker=True,
        rfc_url="localhost",
        rfc_password="",
        rfc_port_http=55080,
        rfc_port_ssh=55022,
        stt_model_size="base",
        stt_language="en",
        stt_silence_threshold=0.3,
        stt_silence_duration=1000,
        stt_waiting_timeout=2000,
    )


def _apply_settings(previous: Settings | None):
    global _settings
    if _settings:
        from agent import AgentContext
        from initialize import initialize

        for ctx in AgentContext._contexts.values():
            ctx.config = initialize()  # reinitialize context config with new settings
            # apply config to agents
            agent = ctx.agent0
            while agent:
                agent.config = ctx.config
                agent = agent.get_data(agent.DATA_NAME_SUBORDINATE)

        # reload whisper model if necessary
        task = defer.DeferredTask().start_task(
            whisper.preload, _settings["stt_model_size"]
        )  # TODO overkill, replace with background task

        # force memory reload on embedding model change
        if previous and (
            _settings["embed_model_name"] != previous["embed_model_name"]
            or _settings["embed_model_provider"] != previous["embed_model_provider"]
            or _settings["embed_model_kwargs"] != previous["embed_model_kwargs"]
        ):
            from python.helpers.memory import reload as memory_reload
            memory_reload()


def _env_to_dict(data: str):
    env_dict = {}
    line_pattern = re.compile(r"\s*([^#][^=]*)\s*=\s*(.*)")
    for line in data.splitlines():
        match = line_pattern.match(line)
        if match:
            key, value = match.groups()
            # Remove optional surrounding quotes (single or double)
            value = value.strip().strip('"').strip("'")
            env_dict[key.strip()] = value
    return env_dict


def _dict_to_env(data_dict):
    lines = []
    for key, value in data_dict.items():
        if "\n" in value:
            value = f"'{value}'"
        elif " " in value or value == "" or any(c in value for c in "\"'"):
            value = f'"{value}"'
        lines.append(f"{key}={value}")
    return "\n".join(lines)


def set_root_password(password: str):
    if not runtime.is_dockerized():
        raise Exception("root password can only be set in dockerized environments")
    subprocess.run(f"echo 'root:{password}' | chpasswd", shell=True, check=True)
    dotenv.save_dotenv_value(dotenv.KEY_ROOT_PASSWORD, password)


def get_runtime_config(set: Settings):
    if runtime.is_dockerized():
        return {
            "code_exec_ssh_addr": "localhost",
            "code_exec_ssh_port": 22,
            "code_exec_http_port": 80,
            "code_exec_ssh_user": "root",
        }
    else:
        host = set["rfc_url"]
        if "//" in host:
            host = host.split("//")[1]
        if ":" in host:
            host, port = host.split(":")
        if host.endswith("/"):
            host = host[:-1]
        return {
            "code_exec_ssh_addr": host,
            "code_exec_ssh_port": set["rfc_port_ssh"],
            "code_exec_http_port": set["rfc_port_http"],
            "code_exec_ssh_user": "root",
        }



================================================
File: python/helpers/shell_local.py
================================================
import select
import subprocess
import time
import sys
from typing import Optional, Tuple

class LocalInteractiveSession:
    def __init__(self):
        self.process = None
        self.full_output = ''

    async def connect(self):
        # Start a new subprocess with the appropriate shell for the OS
        if sys.platform.startswith('win'):
            # Windows
            self.process = subprocess.Popen(
                ['cmd.exe'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
        else:
            # macOS and Linux
            self.process = subprocess.Popen(
                ['/bin/bash'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

    def close(self):
        if self.process:
            self.process.terminate()
            self.process.wait()

    def send_command(self, command: str):
        if not self.process:
            raise Exception("Shell not connected")
        self.full_output = ""
        self.process.stdin.write(command + '\n') # type: ignore
        self.process.stdin.flush() # type: ignore
 
    async def read_output(self, timeout: float = 0, reset_full_output: bool = False) -> Tuple[str, Optional[str]]:
        if not self.process:
            raise Exception("Shell not connected")

        if reset_full_output:
            self.full_output = ""
        partial_output = ''
        start_time = time.time()
        
        while (timeout <= 0 or time.time() - start_time < timeout):
            rlist, _, _ = select.select([self.process.stdout], [], [], 0.1)
            if rlist:
                line = self.process.stdout.readline()  # type: ignore
                if line:
                    partial_output += line
                    self.full_output += line
                    time.sleep(0.1)
                else:
                    break  # No more output
            else:
                break  # No data available

        if not partial_output:
            return self.full_output, None
        
        return self.full_output, partial_output


================================================
File: python/helpers/shell_ssh.py
================================================
import asyncio
import paramiko
import time
import re
from typing import Tuple
from python.helpers.log import Log
from python.helpers.print_style import PrintStyle
from python.helpers.strings import calculate_valid_match_lengths


class SSHInteractiveSession:

    # end_comment = "# @@==>> SSHInteractiveSession End-of-Command  <<==@@"
    # ps1_label = "SSHInteractiveSession CLI>"

    def __init__(
        self, logger: Log, hostname: str, port: int, username: str, password: str
    ):
        self.logger = logger
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.shell = None
        self.full_output = b""
        self.last_command = b""
        self.trimmed_command_length = 0  # Initialize trimmed_command_length

    async def connect(self):
        # try 3 times with wait and then except
        errors = 0
        while True:
            try:
                self.client.connect(
                    self.hostname,
                    self.port,
                    self.username,
                    self.password,
                    allow_agent=False,
                    look_for_keys=False,
                )
                self.shell = self.client.invoke_shell(width=160, height=48)
                # self.shell.send(f'PS1="{SSHInteractiveSession.ps1_label}"'.encode())
                # return
                while True:  # wait for end of initial output
                    full, part = await self.read_output()
                    if full and not part:
                        return
                    time.sleep(0.1)
            except Exception as e:
                errors += 1
                if errors < 3:
                    PrintStyle.standard(f"SSH Connection attempt {errors}...")
                    self.logger.log(
                        type="info",
                        content=f"SSH Connection attempt {errors}...",
                        temp=True,
                    )

                    time.sleep(5)
                else:
                    raise e

    def close(self):
        if self.shell:
            self.shell.close()
        if self.client:
            self.client.close()

    def send_command(self, command: str):
        if not self.shell:
            raise Exception("Shell not connected")
        self.full_output = b""
        # if len(command) > 10: # if command is long, add end_comment to split output
        #     command = (command + " \\\n" +SSHInteractiveSession.end_comment + "\n")
        # else:
        command = command + "\n"
        self.last_command = command.encode()
        self.trimmed_command_length = 0
        self.shell.send(self.last_command)

    async def read_output(
        self, timeout: float = 0, reset_full_output: bool = False
    ) -> Tuple[str, str]:
        if not self.shell:
            raise Exception("Shell not connected")

        if reset_full_output:
            self.full_output = b""
        partial_output = b""
        leftover = b""
        start_time = time.time()

        while self.shell.recv_ready() and (
            timeout <= 0 or time.time() - start_time < timeout
        ):

            # data = self.shell.recv(1024)
            data = self.receive_bytes()

            # Trim own command from output
            if (
                self.last_command
                and len(self.last_command) > self.trimmed_command_length
            ):
                command_to_trim = self.last_command[self.trimmed_command_length :]
                data_to_trim = leftover + data

                trim_com, trim_out = calculate_valid_match_lengths(
                    command_to_trim,
                    data_to_trim,
                    deviation_threshold=8,
                    deviation_reset=2,
                    ignore_patterns=[
                        rb"\x1b\[\?\d{4}[a-zA-Z](?:> )?",  # ANSI escape sequences
                        rb"\r",  # Carriage return
                        rb">\s",  # Greater-than symbol
                    ],
                    debug=False,
                )

                leftover = b""
                if trim_com > 0 and trim_out > 0:
                    data = data_to_trim[trim_out:]
                    leftover = data
                    self.trimmed_command_length += trim_com

            partial_output += data
            self.full_output += data
            await asyncio.sleep(0.1)  # Prevent busy waiting

        # Decode once at the end
        decoded_partial_output = partial_output.decode("utf-8", errors="replace")
        decoded_full_output = self.full_output.decode("utf-8", errors="replace")

        decoded_partial_output = self.clean_string(decoded_partial_output)
        decoded_full_output = self.clean_string(decoded_full_output)

        return decoded_full_output, decoded_partial_output

    def receive_bytes(self, num_bytes=1024):
        if not self.shell:
            raise Exception("Shell not connected")
        # Receive initial chunk of data
        shell = self.shell
        data = self.shell.recv(num_bytes)

        # Helper function to ensure that we receive exactly `num_bytes`
        def recv_all(num_bytes):
            data = b""
            while len(data) < num_bytes:
                chunk = shell.recv(num_bytes - len(data))
                if not chunk:
                    break  # Connection might be closed or no more data
                data += chunk
            return data

        # Check if the last byte(s) form an incomplete multi-byte UTF-8 sequence
        if len(data) > 0:
            last_byte = data[-1]

            # Check if the last byte is part of a multi-byte UTF-8 sequence (continuation byte)
            if (last_byte & 0b11000000) == 0b10000000:  # It's a continuation byte
                # Now, find the start of this sequence by checking earlier bytes
                for i in range(
                    2, 5
                ):  # Look back up to 4 bytes (since UTF-8 is up to 4 bytes long)
                    if len(data) - i < 0:
                        break
                    byte = data[-i]

                    # Detect the leading byte of a multi-byte sequence
                    if (byte & 0b11100000) == 0b11000000:  # 2-byte sequence (110xxxxx)
                        data += recv_all(1)  # Need 1 more byte to complete
                        break
                    elif (
                        byte & 0b11110000
                    ) == 0b11100000:  # 3-byte sequence (1110xxxx)
                        data += recv_all(2)  # Need 2 more bytes to complete
                        break
                    elif (
                        byte & 0b11111000
                    ) == 0b11110000:  # 4-byte sequence (11110xxx)
                        data += recv_all(3)  # Need 3 more bytes to complete
                        break

        return data

    def clean_string(self, input_string):
        # Remove ANSI escape codes
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        cleaned = ansi_escape.sub("", input_string)

        # Replace '\r\n' with '\n'
        cleaned = cleaned.replace("\r\n", "\n")

        # remove leading \r
        cleaned = cleaned.lstrip("\r")

        # Split the string by newline characters to process each segment separately
        lines = cleaned.split("\n")

        for i in range(len(lines)):
            # Handle carriage returns '\r' by splitting and taking the last part
            parts = [part for part in lines[i].split("\r") if part.strip()]
            if parts:
                lines[i] = parts[
                    -1
                ].rstrip()  # Overwrite with the last part after the last '\r'

        return "\n".join(lines)



================================================
File: python/helpers/strings.py
================================================
import re
import sys
import time

from python.helpers import files

def calculate_valid_match_lengths(first: bytes | str, second: bytes | str, 
                                  deviation_threshold: int = 5, 
                                  deviation_reset: int = 5, 
                                  ignore_patterns: list[bytes|str] = [],
                                  debug: bool = False) -> tuple[int, int]:
    
    first_length = len(first)
    second_length = len(second)

    i, j = 0, 0
    deviations = 0
    matched_since_deviation = 0
    last_matched_i, last_matched_j = 0, 0  # Track the last matched index

    def skip_ignored_patterns(s, index):
        """Skip characters in `s` that match any pattern in `ignore_patterns` starting from `index`."""
        while index < len(s):
            for pattern in ignore_patterns:
                match = re.match(pattern, s[index:])
                if match:
                    index += len(match.group(0))
                    break
            else:
                break
        return index

    while i < first_length and j < second_length:
        # Skip ignored patterns
        i = skip_ignored_patterns(first, i)
        j = skip_ignored_patterns(second, j)

        if i < first_length and j < second_length and first[i] == second[j]:
            last_matched_i, last_matched_j = i + 1, j + 1  # Update last matched position
            i += 1
            j += 1
            matched_since_deviation += 1

            # Reset the deviation counter if we've matched enough characters since the last deviation
            if matched_since_deviation >= deviation_reset:
                deviations = 0
                matched_since_deviation = 0
        else:
            # Determine the look-ahead based on the remaining deviation threshold
            look_ahead = deviation_threshold - deviations

            # Look ahead to find the best match within the remaining deviation allowance
            best_match = None
            for k in range(1, look_ahead + 1):
                if i + k < first_length and j < second_length and first[i + k] == second[j]:
                    best_match = ('i', k)
                    break
                if j + k < second_length and i < first_length and first[i] == second[j + k]:
                    best_match = ('j', k)
                    break

            if best_match:
                if best_match[0] == 'i':
                    i += best_match[1]
                elif best_match[0] == 'j':
                    j += best_match[1]
            else:
                i += 1
                j += 1

            deviations += 1
            matched_since_deviation = 0

            if deviations > deviation_threshold:
                break

        if debug:
            output = (
                f"First (up to {last_matched_i}): {first[:last_matched_i]!r}\n"
                "\n"
                f"Second (up to {last_matched_j}): {second[:last_matched_j]!r}\n"
                "\n"
                f"Current deviation: {deviations}\n"
                f"Matched since last deviation: {matched_since_deviation}\n"
                + "-" * 40 + "\n"
            )
            sys.stdout.write("\r" + output)
            sys.stdout.flush()
            time.sleep(0.01)  # Add a short delay for readability (optional)

    # Return the last matched positions instead of the current indices
    return last_matched_i, last_matched_j

def format_key(key: str) -> str:
    """Format a key string to be more readable.
    Converts camelCase and snake_case to Title Case with spaces."""
    # First replace non-alphanumeric with spaces
    result = ''.join(' ' if not c.isalnum() else c for c in key)
    
    # Handle camelCase
    formatted = ''
    for i, c in enumerate(result):
        if i > 0 and c.isupper() and result[i-1].islower():
            formatted += ' ' + c
        else:
            formatted += c
            
    # Split on spaces and capitalize each word
    return ' '.join(word.capitalize() for word in formatted.split())

def dict_to_text(d: dict) -> str:
    parts = []
    for key, value in d.items():
        parts.append(f"{format_key(str(key))}:")
        parts.append(f"{value}")
        parts.append("")  # Add empty line between entries
    
    return "\n".join(parts).rstrip()  # rstrip to remove trailing newline


================================================
File: python/helpers/task_scheduler.py
================================================
import asyncio
from datetime import datetime, timezone, timedelta
import os
import random
import threading
from urllib.parse import urlparse
import uuid
from enum import Enum
from os.path import exists
from typing import Any, Callable, Dict, Literal, Optional, Type, TypeVar, Union, cast, ClassVar

import nest_asyncio
nest_asyncio.apply()

from crontab import CronTab
from pydantic import BaseModel, Field, PrivateAttr

from agent import Agent, AgentContext, UserMessage
from initialize import initialize
from python.helpers.persist_chat import save_tmp_chat
from python.helpers.print_style import PrintStyle
from python.helpers.defer import DeferredTask
from python.helpers.files import get_abs_path, make_dirs, read_file, write_file
from python.helpers.localization import Localization
import pytz
from typing import Annotated

SCHEDULER_FOLDER = "tmp/scheduler"

# ----------------------
# Task Models
# ----------------------


class TaskState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    DISABLED = "disabled"
    ERROR = "error"


class TaskType(str, Enum):
    AD_HOC = "adhoc"
    SCHEDULED = "scheduled"
    PLANNED = "planned"


class TaskSchedule(BaseModel):
    minute: str
    hour: str
    day: str
    month: str
    weekday: str
    timezone: str = Field(default_factory=lambda: Localization.get().get_timezone())

    def to_crontab(self) -> str:
        return f"{self.minute} {self.hour} {self.day} {self.month} {self.weekday}"


class TaskPlan(BaseModel):
    todo: list[datetime] = Field(default_factory=list)
    in_progress: datetime | None = None
    done: list[datetime] = Field(default_factory=list)

    @classmethod
    def create(cls, todo: list[datetime] = list(), in_progress: datetime | None = None, done: list[datetime] = list()):
        if todo:
            for idx, dt in enumerate(todo):
                if dt.tzinfo is None:
                    todo[idx] = pytz.timezone("UTC").localize(dt)
        if in_progress:
            if in_progress.tzinfo is None:
                in_progress = pytz.timezone("UTC").localize(in_progress)
        if done:
            for idx, dt in enumerate(done):
                if dt.tzinfo is None:
                    done[idx] = pytz.timezone("UTC").localize(dt)
        return cls(todo=todo, in_progress=in_progress, done=done)

    def add_todo(self, launch_time: datetime):
        if launch_time.tzinfo is None:
            launch_time = pytz.timezone("UTC").localize(launch_time)
        self.todo.append(launch_time)
        self.todo = sorted(self.todo)

    def set_in_progress(self, launch_time: datetime):
        if launch_time.tzinfo is None:
            launch_time = pytz.timezone("UTC").localize(launch_time)
        if launch_time not in self.todo:
            raise ValueError(f"Launch time {launch_time} not in todo list")
        self.todo.remove(launch_time)
        self.todo = sorted(self.todo)
        self.in_progress = launch_time

    def set_done(self, launch_time: datetime):
        if launch_time.tzinfo is None:
            launch_time = pytz.timezone("UTC").localize(launch_time)
        if launch_time != self.in_progress:
            raise ValueError(f"Launch time {launch_time} is not the same as in progress time {self.in_progress}")
        if launch_time in self.done:
            raise ValueError(f"Launch time {launch_time} already in done list")
        self.in_progress = None
        self.done.append(launch_time)
        self.done = sorted(self.done)

    def get_next_launch_time(self) -> datetime | None:
        return self.todo[0] if self.todo else None

    def should_launch(self) -> datetime | None:
        next_launch_time = self.get_next_launch_time()
        if next_launch_time is None:
            return None
        # return next launch time if current datetime utc is later than next launch time
        if datetime.now(timezone.utc) > next_launch_time:
            return next_launch_time
        return None


class BaseTask(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    context_id: Optional[str] = Field(default=None)
    state: TaskState = Field(default=TaskState.IDLE)
    name: str = Field()
    system_prompt: str
    prompt: str
    attachments: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_run: datetime | None = None
    last_result: str | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.context_id:
            self.context_id = self.uuid
        self._lock = threading.RLock()

    def update(self,
               name: str | None = None,
               state: TaskState | None = None,
               system_prompt: str | None = None,
               prompt: str | None = None,
               attachments: list[str] | None = None,
               last_run: datetime | None = None,
               last_result: str | None = None,
               context_id: str | None = None,
               **kwargs):
        with self._lock:
            if name is not None:
                self.name = name
                self.updated_at = datetime.now(timezone.utc)
            if state is not None:
                self.state = state
                self.updated_at = datetime.now(timezone.utc)
            if system_prompt is not None:
                self.system_prompt = system_prompt
                self.updated_at = datetime.now(timezone.utc)
            if prompt is not None:
                self.prompt = prompt
                self.updated_at = datetime.now(timezone.utc)
            if attachments is not None:
                self.attachments = attachments
                self.updated_at = datetime.now(timezone.utc)
            if last_run is not None:
                self.last_run = last_run
                self.updated_at = datetime.now(timezone.utc)
            if last_result is not None:
                self.last_result = last_result
                self.updated_at = datetime.now(timezone.utc)
            if context_id is not None:
                self.context_id = context_id
                self.updated_at = datetime.now(timezone.utc)
            for key, value in kwargs.items():
                if value is not None:
                    setattr(self, key, value)
                    self.updated_at = datetime.now(timezone.utc)

    def check_schedule(self, frequency_seconds: float = 60.0) -> bool:
        return False

    def get_next_run(self) -> datetime | None:
        return None

    def get_next_run_minutes(self) -> int | None:
        next_run = self.get_next_run()
        if next_run is None:
            return None
        return int((next_run - datetime.now(timezone.utc)).total_seconds() / 60)

    async def on_run(self):
        pass

    async def on_finish(self):
        # Ensure that updated_at is refreshed to reflect completion time
        # This helps track when the task actually finished, regardless of success/error
        await TaskScheduler.get().update_task(
            self.uuid,
            updated_at=datetime.now(timezone.utc)
        )

    async def on_error(self, error: str):
        # Update task state to ERROR and set last result
        scheduler = TaskScheduler.get()
        await scheduler.reload()  # Ensure we have the latest state
        updated_task = await scheduler.update_task(
            self.uuid,
            state=TaskState.ERROR,
            last_run=datetime.now(timezone.utc),
            last_result=f"ERROR: {error}"
        )
        if not updated_task:
            PrintStyle(italic=True, font_color="red", padding=False).print(
                f"Failed to update task {self.uuid} state to ERROR after error: {error}"
            )
        await scheduler.save()  # Force save after update

    async def on_success(self, result: str):
        # Update task state to IDLE and set last result
        scheduler = TaskScheduler.get()
        await scheduler.reload()  # Ensure we have the latest state
        updated_task = await scheduler.update_task(
            self.uuid,
            state=TaskState.IDLE,
            last_run=datetime.now(timezone.utc),
            last_result=result
        )
        if not updated_task:
            PrintStyle(italic=True, font_color="red", padding=False).print(
                f"Failed to update task {self.uuid} state to IDLE after success"
            )
        await scheduler.save()  # Force save after update


class AdHocTask(BaseTask):
    type: Literal[TaskType.AD_HOC] = TaskType.AD_HOC
    token: str = Field(default_factory=lambda: str(random.randint(1000000000000000000, 9999999999999999999)))

    @classmethod
    def create(
        cls,
        name: str,
        system_prompt: str,
        prompt: str,
        token: str,
        attachments: list[str] = list(),
        context_id: str | None = None
    ):
        return cls(name=name,
                   system_prompt=system_prompt,
                   prompt=prompt,
                   attachments=attachments,
                   token=token,
                   context_id=context_id)

    def update(self,
               name: str | None = None,
               state: TaskState | None = None,
               system_prompt: str | None = None,
               prompt: str | None = None,
               attachments: list[str] | None = None,
               last_run: datetime | None = None,
               last_result: str | None = None,
               context_id: str | None = None,
               token: str | None = None,
               **kwargs):
        super().update(name=name,
                       state=state,
                       system_prompt=system_prompt,
                       prompt=prompt,
                       attachments=attachments,
                       last_run=last_run,
                       last_result=last_result,
                       context_id=context_id,
                       token=token,
                       **kwargs)


class ScheduledTask(BaseTask):
    type: Literal[TaskType.SCHEDULED] = TaskType.SCHEDULED
    schedule: TaskSchedule

    @classmethod
    def create(
        cls,
        name: str,
        system_prompt: str,
        prompt: str,
        schedule: TaskSchedule,
        attachments: list[str] = list(),
        context_id: str | None = None,
        timezone: str | None = None
    ):
        # Set timezone in schedule if provided
        if timezone is not None:
            schedule.timezone = timezone
        else:
            schedule.timezone = Localization.get().get_timezone()

        return cls(name=name,
                   system_prompt=system_prompt,
                   prompt=prompt,
                   attachments=attachments,
                   schedule=schedule,
                   context_id=context_id)

    def update(self,
               name: str | None = None,
               state: TaskState | None = None,
               system_prompt: str | None = None,
               prompt: str | None = None,
               attachments: list[str] | None = None,
               last_run: datetime | None = None,
               last_result: str | None = None,
               context_id: str | None = None,
               schedule: TaskSchedule | None = None,
               **kwargs):
        super().update(name=name,
                       state=state,
                       system_prompt=system_prompt,
                       prompt=prompt,
                       attachments=attachments,
                       last_run=last_run,
                       last_result=last_result,
                       context_id=context_id,
                       schedule=schedule,
                       **kwargs)

    def check_schedule(self, frequency_seconds: float = 60.0) -> bool:
        with self._lock:
            crontab = CronTab(crontab=self.schedule.to_crontab())  # type: ignore

            # Get the timezone from the schedule or use UTC as fallback
            task_timezone = pytz.timezone(self.schedule.timezone or Localization.get().get_timezone())

            # Get reference time in task's timezone (by default now - frequency_seconds)
            reference_time = datetime.now(timezone.utc) - timedelta(seconds=frequency_seconds)
            reference_time = reference_time.astimezone(task_timezone)

            # Get next run time as seconds until next execution
            next_run_seconds: Optional[float] = crontab.next(  # type: ignore
                now=reference_time,
                return_datetime=False
            )  # type: ignore

            if next_run_seconds is None:
                return False

            return next_run_seconds < frequency_seconds

    def get_next_run(self) -> datetime | None:
        with self._lock:
            crontab = CronTab(crontab=self.schedule.to_crontab())  # type: ignore
            return crontab.next(now=datetime.now(timezone.utc), return_datetime=True)  # type: ignore


class PlannedTask(BaseTask):
    type: Literal[TaskType.PLANNED] = TaskType.PLANNED
    plan: TaskPlan

    @classmethod
    def create(
        cls,
        name: str,
        system_prompt: str,
        prompt: str,
        plan: TaskPlan,
        attachments: list[str] = list(),
        context_id: str | None = None
    ):
        return cls(name=name,
                   system_prompt=system_prompt,
                   prompt=prompt,
                   plan=plan,
                   attachments=attachments,
                   context_id=context_id)

    def update(self,
               name: str | None = None,
               state: TaskState | None = None,
               system_prompt: str | None = None,
               prompt: str | None = None,
               attachments: list[str] | None = None,
               last_run: datetime | None = None,
               last_result: str | None = None,
               context_id: str | None = None,
               plan: TaskPlan | None = None,
               **kwargs):
        super().update(name=name,
                       state=state,
                       system_prompt=system_prompt,
                       prompt=prompt,
                       attachments=attachments,
                       last_run=last_run,
                       last_result=last_result,
                       context_id=context_id,
                       plan=plan,
                       **kwargs)

    def check_schedule(self, frequency_seconds: float = 60.0) -> bool:
        with self._lock:
            return self.plan.should_launch() is not None

    def get_next_run(self) -> datetime | None:
        with self._lock:
            return self.plan.get_next_launch_time()

    async def on_run(self):
        with self._lock:
            # Get the next launch time and set it as in_progress
            next_launch_time = self.plan.should_launch()
            if next_launch_time is not None:
                self.plan.set_in_progress(next_launch_time)
        await super().on_run()

    async def on_finish(self):
        # Handle plan item progression regardless of success or error
        plan_updated = False

        with self._lock:
            # If there's an in_progress time, mark it as done
            if self.plan.in_progress is not None:
                self.plan.set_done(self.plan.in_progress)
                plan_updated = True

        # If we updated the plan, make sure to persist it
        if plan_updated:
            scheduler = TaskScheduler.get()
            await scheduler.reload()
            await scheduler.update_task(self.uuid, plan=self.plan)
            await scheduler.save()  # Force save

        # Call the parent implementation for any additional cleanup
        await super().on_finish()

    async def on_success(self, result: str):
        # Call parent implementation to update state, etc.
        await super().on_success(result)

    async def on_error(self, error: str):
        # Call parent implementation to update state, etc.
        await super().on_error(error)


class SchedulerTaskList(BaseModel):
    tasks: list[Annotated[Union[ScheduledTask, AdHocTask, PlannedTask], Field(discriminator="type")]] = Field(default_factory=list)
    # Singleton instance
    __instance: ClassVar[Optional["SchedulerTaskList"]] = PrivateAttr(default=None)

    # lock: threading.Lock = Field(exclude=True, default=threading.Lock())

    @classmethod
    def get(cls) -> "SchedulerTaskList":
        path = get_abs_path(SCHEDULER_FOLDER, "tasks.json")
        if cls.__instance is None:
            if not exists(path):
                make_dirs(path)
                cls.__instance = asyncio.run(cls(tasks=[]).save())
            else:
                cls.__instance = cls.model_validate_json(read_file(path))
        else:
            asyncio.run(cls.__instance.reload())
        return cls.__instance

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = threading.RLock()

    async def reload(self) -> "SchedulerTaskList":
        path = get_abs_path(SCHEDULER_FOLDER, "tasks.json")
        if exists(path):
            with self._lock:
                data = self.__class__.model_validate_json(read_file(path))
                self.tasks.clear()
                self.tasks.extend(data.tasks)
        return self

    async def add_task(self, task: Union[ScheduledTask, AdHocTask, PlannedTask]) -> "SchedulerTaskList":
        with self._lock:
            self.tasks.append(task)
            await self.save()
        return self

    async def save(self) -> "SchedulerTaskList":
        with self._lock:
            # Debug: check for AdHocTasks with null tokens before saving
            for task in self.tasks:
                if isinstance(task, AdHocTask):
                    if task.token is None or task.token == "":
                        PrintStyle(italic=True, font_color="red", padding=False).print(
                            f"WARNING: AdHocTask {task.name} ({task.uuid}) has a null or empty token before saving: '{task.token}'"
                        )
                        # Generate a new token to prevent errors
                        task.token = str(random.randint(1000000000000000000, 9999999999999999999))
                        PrintStyle(italic=True, font_color="red", padding=False).print(
                            f"Fixed: Generated new token '{task.token}' for task {task.name}"
                        )

            path = get_abs_path(SCHEDULER_FOLDER, "tasks.json")
            if not exists(path):
                make_dirs(path)

            # Get the JSON string before writing
            json_data = self.model_dump_json()

            # Debug: check if 'null' appears as token value in JSON
            if '"type": "adhoc"' in json_data and '"token": null' in json_data:
                PrintStyle(italic=True, font_color="red", padding=False).print(
                    "ERROR: Found null token in JSON output for an adhoc task"
                )

            write_file(path, json_data)

            # Debug: Verify after saving
            if exists(path):
                loaded_json = read_file(path)
                if '"type": "adhoc"' in loaded_json and '"token": null' in loaded_json:
                    PrintStyle(italic=True, font_color="red", padding=False).print(
                        "ERROR: Null token persisted in JSON file for an adhoc task"
                    )

        return self

    async def update_task_by_uuid(
        self,
        task_uuid: str,
        updater_func: Callable[[Union[ScheduledTask, AdHocTask, PlannedTask]], None],
        verify_func: Callable[[Union[ScheduledTask, AdHocTask, PlannedTask]], bool] = lambda task: True
    ) -> Union[ScheduledTask, AdHocTask, PlannedTask] | None:
        """
        Atomically update a task by UUID using the provided updater function.

        The updater_func should take the task as an argument and perform any necessary updates.
        This method ensures that the task is updated and saved atomically, preventing race conditions.

        Returns the updated task or None if not found.
        """
        with self._lock:
            # Reload to ensure we have the latest state
            await self.reload()

            # Find the task
            task = next((task for task in self.tasks if task.uuid == task_uuid and verify_func(task)), None)
            if task is None:
                return None

            # Apply the updates via the provided function
            updater_func(task)

            # Save the changes
            await self.save()

            return task

    def get_tasks(self) -> list[Union[ScheduledTask, AdHocTask, PlannedTask]]:
        with self._lock:
            return self.tasks

    def get_tasks_by_context_id(self, context_id: str, only_running: bool = False) -> list[Union[ScheduledTask, AdHocTask, PlannedTask]]:
        with self._lock:
            return [
                task for task in self.tasks
                if task.context_id == context_id
                and (not only_running or task.state == TaskState.RUNNING)
            ]

    async def get_due_tasks(self) -> list[Union[ScheduledTask, AdHocTask, PlannedTask]]:
        with self._lock:
            await self.reload()
            return [
                task for task in self.tasks
                if task.check_schedule() and task.state == TaskState.IDLE
            ]

    def get_task_by_uuid(self, task_uuid: str) -> Union[ScheduledTask, AdHocTask, PlannedTask] | None:
        with self._lock:
            return next((task for task in self.tasks if task.uuid == task_uuid), None)

    def get_task_by_name(self, name: str) -> Union[ScheduledTask, AdHocTask, PlannedTask] | None:
        with self._lock:
            return next((task for task in self.tasks if task.name == name), None)

    def find_task_by_name(self, name: str) -> list[Union[ScheduledTask, AdHocTask, PlannedTask]]:
        with self._lock:
            return [task for task in self.tasks if name.lower() in task.name.lower()]

    async def remove_task_by_uuid(self, task_uuid: str) -> "SchedulerTaskList":
        with self._lock:
            self.tasks = [task for task in self.tasks if task.uuid != task_uuid]
            await self.save()
        return self

    async def remove_task_by_name(self, name: str) -> "SchedulerTaskList":
        with self._lock:
            self.tasks = [task for task in self.tasks if task.name != name]
            await self.save()
        return self


class TaskScheduler:

    _tasks: SchedulerTaskList
    _printer: PrintStyle
    _instance = None

    @classmethod
    def get(cls) -> "TaskScheduler":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # Only initialize if this is a new instance
        if not hasattr(self, '_initialized'):
            self._tasks = SchedulerTaskList.get()
            self._printer = PrintStyle(italic=True, font_color="green", padding=False)
            self._initialized = True

    async def reload(self):
        await self._tasks.reload()

    def get_tasks(self) -> list[Union[ScheduledTask, AdHocTask, PlannedTask]]:
        return self._tasks.get_tasks()

    def get_tasks_by_context_id(self, context_id: str, only_running: bool = False) -> list[Union[ScheduledTask, AdHocTask, PlannedTask]]:
        return self._tasks.get_tasks_by_context_id(context_id, only_running)

    async def add_task(self, task: Union[ScheduledTask, AdHocTask, PlannedTask]) -> "TaskScheduler":
        await self._tasks.add_task(task)
        ctx = await self._get_chat_context(task)  # invoke context creation
        return self

    async def remove_task_by_uuid(self, task_uuid: str) -> "TaskScheduler":
        await self._tasks.remove_task_by_uuid(task_uuid)
        return self

    async def remove_task_by_name(self, name: str) -> "TaskScheduler":
        await self._tasks.remove_task_by_name(name)
        return self

    def get_task_by_uuid(self, task_uuid: str) -> Union[ScheduledTask, AdHocTask, PlannedTask] | None:
        return self._tasks.get_task_by_uuid(task_uuid)

    def get_task_by_name(self, name: str) -> Union[ScheduledTask, AdHocTask, PlannedTask] | None:
        return self._tasks.get_task_by_name(name)

    def find_task_by_name(self, name: str) -> list[Union[ScheduledTask, AdHocTask, PlannedTask]]:
        return self._tasks.find_task_by_name(name)

    async def tick(self):
        for task in await self._tasks.get_due_tasks():
            await self._run_task(task)

    async def run_task_by_uuid(self, task_uuid: str, task_context: str | None = None):
        # First reload tasks to ensure we have the latest state
        await self._tasks.reload()

        # Get the task to run
        task = self.get_task_by_uuid(task_uuid)
        if not task:
            raise ValueError(f"Task with UUID '{task_uuid}' not found")

        # If the task is already running, raise an error
        if task.state == TaskState.RUNNING:
            raise ValueError(f"Task '{task.name}' is already running")

        # If the task is disabled, raise an error
        if task.state == TaskState.DISABLED:
            raise ValueError(f"Task '{task.name}' is disabled")

        # If the task is in error state, reset it to IDLE first
        if task.state == TaskState.ERROR:
            self._printer.print(f"Resetting task '{task.name}' from ERROR to IDLE state before running")
            await self.update_task(task_uuid, state=TaskState.IDLE)
            # Force a reload to ensure we have the updated state
            await self._tasks.reload()
            task = self.get_task_by_uuid(task_uuid)
            if not task:
                raise ValueError(f"Task with UUID '{task_uuid}' not found after state reset")

        # Run the task
        await self._run_task(task, task_context)

    async def run_task_by_name(self, name: str, task_context: str | None = None):
        task = self._tasks.get_task_by_name(name)
        if task is None:
            raise ValueError(f"Task with name {name} not found")
        await self._run_task(task, task_context)

    async def save(self):
        await self._tasks.save()

    async def update_task_checked(
        self,
        task_uuid: str,
        verify_func: Callable[[Union[ScheduledTask, AdHocTask, PlannedTask]], bool] = lambda task: True,
        **update_params
    ) -> Union[ScheduledTask, AdHocTask, PlannedTask] | None:
        """
        Atomically update a task by UUID with the provided parameters.
        This prevents race conditions when multiple processes update tasks concurrently.

        Returns the updated task or None if not found.
        """
        def _update_task(task):
            task.update(**update_params)

        return await self._tasks.update_task_by_uuid(task_uuid, _update_task, verify_func)

    async def update_task(self, task_uuid: str, **update_params) -> Union[ScheduledTask, AdHocTask, PlannedTask] | None:
        return await self.update_task_checked(task_uuid, lambda task: True, **update_params)

    async def __new_context(self, task: Union[ScheduledTask, AdHocTask, PlannedTask]) -> AgentContext:
        if not task.context_id:
            raise ValueError(f"Task {task.name} has no context ID")

        config = initialize()
        context: AgentContext = AgentContext(config, id=task.context_id, name=task.name)
        # context.id = task.context_id
        # initial name before renaming is same as task name
        # context.name = task.name

        # Save the context
        save_tmp_chat(context)
        return context

    async def _get_chat_context(self, task: Union[ScheduledTask, AdHocTask, PlannedTask]) -> AgentContext:
        context = AgentContext.get(task.context_id) if task.context_id else None

        if context:
            assert isinstance(context, AgentContext)
            self._printer.print(
                f"Scheduler Task {task.name} loaded from task {task.uuid}, context ok"
            )
            save_tmp_chat(context)
            return context
        else:
            self._printer.print(
                f"Scheduler Task {task.name} loaded from task {task.uuid} but context not found"
            )
            return await self.__new_context(task)

    async def _persist_chat(self, task: Union[ScheduledTask, AdHocTask, PlannedTask], context: AgentContext):
        if context.id != task.context_id:
            raise ValueError(f"Context ID mismatch for task {task.name}: context {context.id} != task {task.context_id}")
        save_tmp_chat(context)

    async def _run_task(self, task: Union[ScheduledTask, AdHocTask, PlannedTask], task_context: str | None = None):

        async def _run_task_wrapper(task_uuid: str, task_context: str | None = None):

            # preflight checks with a snapshot of the task
            task_snapshot: Union[ScheduledTask, AdHocTask, PlannedTask] | None = self.get_task_by_uuid(task_uuid)
            if task_snapshot is None:
                self._printer.print(f"Scheduler Task with UUID '{task_uuid}' not found")
                return
            if task_snapshot.state == TaskState.RUNNING:
                self._printer.print(f"Scheduler Task '{task_snapshot.name}' already running, skipping")
                return

            # Atomically fetch and check the task's current state
            current_task = await self.update_task_checked(task_uuid, lambda task: task.state != TaskState.RUNNING, state=TaskState.RUNNING)
            if not current_task:
                self._printer.print(f"Scheduler Task with UUID '{task_uuid}' not found or updated by another process")
                return
            if current_task.state != TaskState.RUNNING:
                # This means the update failed due to state conflict
                self._printer.print(f"Scheduler Task '{current_task.name}' state is '{current_task.state}', skipping")
                return

            await current_task.on_run()

            # the agent instance - init in try block
            agent = None

            try:
                self._printer.print(f"Scheduler Task '{current_task.name}' started")

                context = await self._get_chat_context(current_task)

                # Ensure the context is properly registered in the AgentContext._contexts
                # This is critical for the polling mechanism to find and stream logs
                # Dict operations are atomic
                # AgentContext._contexts[context.id] = context
                agent = context.streaming_agent or context.agent0

                # Prepare attachment filenames for logging
                attachment_filenames = []
                if current_task.attachments:
                    for attachment in current_task.attachments:
                        if os.path.exists(attachment):
                            attachment_filenames.append(attachment)
                        else:
                            try:
                                url = urlparse(attachment)
                                if url.scheme in ["http", "https", "ftp", "ftps", "sftp"]:
                                    attachment_filenames.append(attachment)
                                else:
                                    self._printer.print(f"Skipping attachment: [{attachment}]")
                            except Exception:
                                self._printer.print(f"Skipping attachment: [{attachment}]")

                self._printer.print("User message:")
                self._printer.print(f"> {current_task.prompt}")
                if attachment_filenames:
                    self._printer.print("Attachments:")
                    for filename in attachment_filenames:
                        self._printer.print(f"- {filename}")

                task_prompt = f"# Starting scheduler task '{current_task.name}' ({current_task.uuid})"
                if task_context:
                    task_prompt = f"## Context:\n{task_context}\n\n## Task:\n{current_task.prompt}"
                else:
                    task_prompt = f"## Task:\n{current_task.prompt}"

                # Log the message with message_id and attachments
                context.log.log(
                    type="user",
                    heading="User message",
                    content=task_prompt,
                    kvps={"attachments": attachment_filenames},
                    id=str(uuid.uuid4()),
                )

                agent.hist_add_user_message(
                    UserMessage(
                        message=task_prompt,
                        system_message=[current_task.system_prompt],
                        attachments=attachment_filenames))

                # Persist after setting up the context but before running the agent
                # This ensures the task context is saved and can be found by polling
                await self._persist_chat(current_task, context)

                result = await agent.monologue()

                # Success
                self._printer.print(f"Scheduler Task '{current_task.name}' completed: {result}")
                await self._persist_chat(current_task, context)
                await current_task.on_success(result)

                # Explicitly verify task was updated in storage after success
                await self._tasks.reload()
                updated_task = self.get_task_by_uuid(task_uuid)
                if updated_task and updated_task.state != TaskState.IDLE:
                    self._printer.print(f"Fixing task state consistency: '{current_task.name}' state is not IDLE after success")
                    await self.update_task(task_uuid, state=TaskState.IDLE)

            except Exception as e:
                # Error
                self._printer.print(f"Scheduler Task '{current_task.name}' failed: {e}")
                await current_task.on_error(str(e))

                # Explicitly verify task was updated in storage after error
                await self._tasks.reload()
                updated_task = self.get_task_by_uuid(task_uuid)
                if updated_task and updated_task.state != TaskState.ERROR:
                    self._printer.print(f"Fixing task state consistency: '{current_task.name}' state is not ERROR after failure")
                    await self.update_task(task_uuid, state=TaskState.ERROR)

                if agent:
                    agent.handle_critical_exception(e)
            finally:
                # Call on_finish for task-specific cleanup
                await current_task.on_finish()

                # Make one final save to ensure all states are persisted
                await self._tasks.save()

        deferred_task = DeferredTask(thread_name=self.__class__.__name__)
        deferred_task.start_task(_run_task_wrapper, task.uuid, task_context)

        # Ensure background execution doesn't exit immediately on async await, especially in script contexts
        # This helps prevent premature exits when running from non-event-loop contexts
        asyncio.create_task(asyncio.sleep(0.1))

    def serialize_all_tasks(self) -> list[Dict[str, Any]]:
        """
        Serialize all tasks in the scheduler to a list of dictionaries.
        """
        return serialize_tasks(self.get_tasks())

    def serialize_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Serialize a specific task in the scheduler by UUID.
        Returns None if task is not found.
        """
        # Get task without locking, as get_task_by_uuid() is already thread-safe
        task = self.get_task_by_uuid(task_id)
        if task:
            return serialize_task(task)
        return None


# ----------------------
# Task Serialization Helpers
# ----------------------

def serialize_datetime(dt: Optional[datetime]) -> Optional[str]:
    """
    Serialize a datetime object to ISO format string in the user's timezone.

    This uses the Localization singleton to convert the datetime to the user's timezone
    before serializing it to an ISO format string for frontend display.

    Returns None if the input is None.
    """
    # Use the Localization singleton for timezone conversion and serialization
    return Localization.get().serialize_datetime(dt)


def parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    """
    Parse ISO format datetime string with timezone awareness.

    This converts from the localized ISO format returned by serialize_datetime
    back to a datetime object with proper timezone handling.

    Returns None if dt_str is None.
    """
    if not dt_str:
        return None

    try:
        # Use the Localization singleton for consistent timezone handling
        return Localization.get().localtime_str_to_utc_dt(dt_str)
    except ValueError as e:
        raise ValueError(f"Invalid datetime format: {dt_str}. Expected ISO format. Error: {e}")


def serialize_task_schedule(schedule: TaskSchedule) -> Dict[str, str]:
    """Convert TaskSchedule to a standardized dictionary format."""
    return {
        'minute': schedule.minute,
        'hour': schedule.hour,
        'day': schedule.day,
        'month': schedule.month,
        'weekday': schedule.weekday,
        'timezone': schedule.timezone
    }


def parse_task_schedule(schedule_data: Dict[str, str]) -> TaskSchedule:
    """Parse dictionary into TaskSchedule with validation."""
    try:
        return TaskSchedule(
            minute=schedule_data.get('minute', '*'),
            hour=schedule_data.get('hour', '*'),
            day=schedule_data.get('day', '*'),
            month=schedule_data.get('month', '*'),
            weekday=schedule_data.get('weekday', '*'),
            timezone=schedule_data.get('timezone', Localization.get().get_timezone())
        )
    except Exception as e:
        raise ValueError(f"Invalid schedule format: {e}") from e


def serialize_task_plan(plan: TaskPlan) -> Dict[str, Any]:
    """Convert TaskPlan to a standardized dictionary format."""
    return {
        'todo': [serialize_datetime(dt) for dt in plan.todo],
        'in_progress': serialize_datetime(plan.in_progress) if plan.in_progress else None,
        'done': [serialize_datetime(dt) for dt in plan.done]
    }


def parse_task_plan(plan_data: Dict[str, Any]) -> TaskPlan:
    """Parse dictionary into TaskPlan with validation."""
    try:
        # Handle case where plan_data might be None or empty
        if not plan_data:
            return TaskPlan(todo=[], in_progress=None, done=[])

        # Parse todo items with careful validation
        todo_dates = []
        for dt_str in plan_data.get('todo', []):
            if dt_str:
                parsed_dt = parse_datetime(dt_str)
                if parsed_dt:
                    # Ensure datetime is timezone-aware (use UTC if not specified)
                    if parsed_dt.tzinfo is None:
                        parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
                    todo_dates.append(parsed_dt)

        # Parse in_progress with validation
        in_progress = None
        if plan_data.get('in_progress'):
            in_progress = parse_datetime(plan_data.get('in_progress'))
            # Ensure datetime is timezone-aware
            if in_progress and in_progress.tzinfo is None:
                in_progress = in_progress.replace(tzinfo=timezone.utc)

        # Parse done items with validation
        done_dates = []
        for dt_str in plan_data.get('done', []):
            if dt_str:
                parsed_dt = parse_datetime(dt_str)
                if parsed_dt:
                    # Ensure datetime is timezone-aware
                    if parsed_dt.tzinfo is None:
                        parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
                    done_dates.append(parsed_dt)

        # Sort dates for better usability
        todo_dates.sort()
        done_dates.sort(reverse=True)  # Most recent first for done items

        # Cast to ensure type safety
        todo_dates_cast: list[datetime] = cast(list[datetime], todo_dates)
        done_dates_cast: list[datetime] = cast(list[datetime], done_dates)

        return TaskPlan.create(
            todo=todo_dates_cast,
            in_progress=in_progress,
            done=done_dates_cast
        )
    except Exception as e:
        PrintStyle(italic=True, font_color="red", padding=False).print(
            f"Error parsing task plan: {e}"
        )
        # Return empty plan instead of failing
        return TaskPlan(todo=[], in_progress=None, done=[])


T = TypeVar('T', bound=Union[ScheduledTask, AdHocTask, PlannedTask])


def serialize_task(task: Union[ScheduledTask, AdHocTask, PlannedTask]) -> Dict[str, Any]:
    """
    Standardized serialization for task objects with proper handling of all complex types.
    """
    # Start with a basic dictionary
    task_dict = {
        "uuid": task.uuid,
        "name": task.name,
        "state": task.state,
        "system_prompt": task.system_prompt,
        "prompt": task.prompt,
        "attachments": task.attachments,
        "created_at": serialize_datetime(task.created_at),
        "updated_at": serialize_datetime(task.updated_at),
        "last_run": serialize_datetime(task.last_run),
        "next_run": serialize_datetime(task.get_next_run()),
        "last_result": task.last_result,
        "context_id": task.context_id
    }

    # Add type-specific fields
    if isinstance(task, ScheduledTask):
        task_dict['type'] = 'scheduled'
        task_dict['schedule'] = serialize_task_schedule(task.schedule)  # type: ignore
    elif isinstance(task, AdHocTask):
        task_dict['type'] = 'adhoc'
        adhoc_task = cast(AdHocTask, task)
        task_dict['token'] = adhoc_task.token
    else:
        task_dict['type'] = 'planned'
        planned_task = cast(PlannedTask, task)
        task_dict['plan'] = serialize_task_plan(planned_task.plan)  # type: ignore

    return task_dict


def serialize_tasks(tasks: list[Union[ScheduledTask, AdHocTask, PlannedTask]]) -> list[Dict[str, Any]]:
    """
    Serialize a list of tasks to a list of dictionaries.
    """
    return [serialize_task(task) for task in tasks]


def deserialize_task(task_data: Dict[str, Any], task_class: Optional[Type[T]] = None) -> T:
    """
    Deserialize dictionary into appropriate task object with validation.
    If task_class is provided, uses that type. Otherwise determines type from data.
    """
    task_type_str = task_data.get('type', '')
    determined_class = None

    if not task_class:
        # Determine task class from data
        if task_type_str == 'scheduled':
            determined_class = cast(Type[T], ScheduledTask)
        elif task_type_str == 'adhoc':
            determined_class = cast(Type[T], AdHocTask)
            # Ensure token is a valid non-empty string
            if not task_data.get('token'):
                task_data['token'] = str(random.randint(1000000000000000000, 9999999999999999999))
        elif task_type_str == 'planned':
            determined_class = cast(Type[T], PlannedTask)
        else:
            raise ValueError(f"Unknown task type: {task_type_str}")
    else:
        determined_class = task_class
        # If this is an AdHocTask, ensure token is valid
        if determined_class == AdHocTask and not task_data.get('token'):  # type: ignore
            task_data['token'] = str(random.randint(1000000000000000000, 9999999999999999999))

    common_args = {
        "uuid": task_data.get("uuid"),
        "name": task_data.get("name"),
        "state": TaskState(task_data.get("state", TaskState.IDLE)),
        "system_prompt": task_data.get("system_prompt", ""),
        "prompt": task_data.get("prompt", ""),
        "attachments": task_data.get("attachments", []),
        "created_at": parse_datetime(task_data.get("created_at")),
        "updated_at": parse_datetime(task_data.get("updated_at")),
        "last_run": parse_datetime(task_data.get("last_run")),
        "last_result": task_data.get("last_result"),
        "context_id": task_data.get("context_id")
    }

    # Add type-specific fields
    if determined_class == ScheduledTask:  # type: ignore
        schedule_data = task_data.get("schedule", {})
        common_args["schedule"] = parse_task_schedule(schedule_data)
        return ScheduledTask(**common_args)  # type: ignore
    elif determined_class == AdHocTask:  # type: ignore
        common_args["token"] = task_data.get("token", "")
        return AdHocTask(**common_args)  # type: ignore
    else:
        plan_data = task_data.get("plan", {})
        common_args["plan"] = parse_task_plan(plan_data)
        return PlannedTask(**common_args)  # type: ignore



================================================
File: python/helpers/timed_input.py
================================================
import sys
from inputimeout import inputimeout, TimeoutOccurred

def timeout_input(prompt, timeout=10):
    try:
        if sys.platform != "win32": import readline
        user_input = inputimeout(prompt=prompt, timeout=timeout)
        return user_input
    except TimeoutOccurred:
        return ""


================================================
File: python/helpers/tokens.py
================================================
from typing import Literal
import tiktoken

APPROX_BUFFER = 1.1
TRIM_BUFFER = 0.8


def count_tokens(text: str, encoding_name="cl100k_base") -> int:
    if not text:
        return 0

    # Get the encoding
    encoding = tiktoken.get_encoding(encoding_name)

    # Encode the text and count the tokens
    tokens = encoding.encode(text)
    token_count = len(tokens)

    return token_count


def approximate_tokens(
    text: str,
) -> int:
    return int(count_tokens(text) * APPROX_BUFFER)


def trim_to_tokens(
    text: str,
    max_tokens: int,
    direction: Literal["start", "end"],
    ellipsis: str = "...",
) -> str:
    chars = len(text)
    tokens = count_tokens(text)

    if tokens <= max_tokens:
        return text

    approx_chars = int(chars * (max_tokens / tokens) * TRIM_BUFFER)

    if direction == "start":
        return text[:approx_chars] + ellipsis
    return ellipsis + text[chars - approx_chars : chars]



================================================
File: python/helpers/tool.py
================================================
from abc import abstractmethod
from dataclasses import dataclass

from agent import Agent
from python.helpers.print_style import PrintStyle


@dataclass
class Response:
    message:str
    break_loop: bool

class Tool:

    def __init__(self, agent: Agent, name: str, method: str | None, args: dict[str,str], message: str, **kwargs) -> None:
        self.agent = agent
        self.name = name
        self.method = method
        self.args = args
        self.message = message

    @abstractmethod
    async def execute(self,**kwargs) -> Response:
        pass

    async def before_execution(self, **kwargs):
        PrintStyle(font_color="#1B4F72", padding=True, background_color="white", bold=True).print(f"{self.agent.agent_name}: Using tool '{self.name}'")
        self.log = self.get_log_object()
        if self.args and isinstance(self.args, dict):
            for key, value in self.args.items():
                PrintStyle(font_color="#85C1E9", bold=True).stream(self.nice_key(key)+": ")
                PrintStyle(font_color="#85C1E9", padding=isinstance(value,str) and "\n" in value).stream(value)
                PrintStyle().print()

    async def after_execution(self, response: Response, **kwargs):
        text = response.message.strip()
        self.agent.hist_add_tool_result(self.name, text)
        PrintStyle(font_color="#1B4F72", background_color="white", padding=True, bold=True).print(f"{self.agent.agent_name}: Response from tool '{self.name}'")
        PrintStyle(font_color="#85C1E9").print(response.message)
        self.log.update(content=response.message)

    def get_log_object(self):
        if self.method:
            heading = f"{self.agent.agent_name}: Using tool '{self.name}:{self.method}'"
        else:
            heading = f"{self.agent.agent_name}: Using tool '{self.name}'"
        return self.agent.context.log.log(type="tool", heading=heading, content="", kvps=self.args)

    def nice_key(self, key:str):
        words = key.split('_')
        words = [words[0].capitalize()] + [word.lower() for word in words[1:]]
        result = ' '.join(words)
        return result



================================================
File: python/helpers/tunnel_manager.py
================================================
from flaredantic import FlareTunnel, FlareConfig
import threading


# Singleton to manage the tunnel instance
class TunnelManager:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        self.tunnel = None
        self.tunnel_url = None
        self.is_running = False

    def start_tunnel(self, port=80):
        """Start a new tunnel or return the existing one's URL"""
        if self.is_running and self.tunnel_url:
            return self.tunnel_url

        # Create and start a new tunnel
        config = FlareConfig(
            port=port,
            verbose=True,
            timeout=60,  # Increase timeout from default 30 to 60 seconds
        )

        try:
            # Start tunnel in a separate thread to avoid blocking
            def run_tunnel():
                try:
                    self.tunnel = FlareTunnel(config)
                    self.tunnel.start()
                    self.tunnel_url = self.tunnel.tunnel_url
                    self.is_running = True
                except Exception as e:
                    print(f"Error in tunnel thread: {str(e)}")

            tunnel_thread = threading.Thread(target=run_tunnel)
            tunnel_thread.daemon = True
            tunnel_thread.start()

            # Wait for tunnel to start (max 15 seconds instead of 5)
            for _ in range(150):  # Increased from 50 to 150 iterations
                if self.tunnel_url:
                    break
                import time

                time.sleep(0.1)

            return self.tunnel_url
        except Exception as e:
            print(f"Error starting tunnel: {str(e)}")
            return None

    def stop_tunnel(self):
        """Stop the running tunnel"""
        if self.tunnel and self.is_running:
            try:
                self.tunnel.stop()
                self.is_running = False
                self.tunnel_url = None
                return True
            except Exception:
                return False
        return False

    def get_tunnel_url(self):
        """Get the current tunnel URL if available"""
        return self.tunnel_url if self.is_running else None



================================================
File: python/helpers/vector_db.py
================================================
from typing import Any, List, Sequence
import uuid
from langchain_community.vectorstores import FAISS
import faiss
from langchain_core.documents import Document
from langchain.storage import InMemoryByteStore
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.utils import (
    DistanceStrategy,
)
from langchain.embeddings import CacheBackedEmbeddings

from agent import Agent


class MyFaiss(FAISS):
    # override aget_by_ids
    def get_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        # return all self.docstore._dict[id] in ids
        return [self.docstore._dict[id] for id in (ids if isinstance(ids, list) else [ids]) if id in self.docstore._dict]  # type: ignore

    async def aget_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        return self.get_by_ids(ids)


class VectorDB:
    def __init__(self, agent: Agent):
        self.agent = agent
        self.store = InMemoryByteStore()
        self.model = agent.get_embedding_model()

        self.embedder = CacheBackedEmbeddings.from_bytes_store(
            self.model,
            self.store,
            namespace=getattr(
                self.model,
                "model",
                getattr(self.model, "model_name", "default"),
            ),
        )

        self.index = faiss.IndexFlatIP(len(self.embedder.embed_query("example")))

        self.db = MyFaiss(
            embedding_function=self.embedder,
            index=self.index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            distance_strategy=DistanceStrategy.COSINE,
            # normalize_L2=True,
            relevance_score_fn=cosine_normalizer,
        )

    async def search_similarity_threshold(
        self, query: str, limit: int, threshold: float, filter: str = ""
    ):
        comparator = get_comparator(filter) if filter else None

        # rate limiter
        await self.agent.rate_limiter(
            model_config=self.agent.config.embeddings_model, input=query
        )

        return await self.db.asearch(
            query,
            search_type="similarity_score_threshold",
            k=limit,
            score_threshold=threshold,
            filter=comparator,
        )

    async def insert_documents(self, docs: list[Document]):
        ids = [str(uuid.uuid4()) for _ in range(len(docs))]

        if ids:
            for doc, id in zip(docs, ids):
                doc.metadata["id"] = id  # add ids to documents metadata

            # rate limiter
            docs_txt = "".join(format_docs_plain(docs))
            await self.agent.rate_limiter(
                model_config=self.agent.config.embeddings_model, input=docs_txt
            )

            self.db.add_documents(documents=docs, ids=ids)
        return ids


def format_docs_plain(docs: list[Document]) -> list[str]:
    result = []
    for doc in docs:
        text = ""
        for k, v in doc.metadata.items():
            text += f"{k}: {v}\n"
        text += f"Content: {doc.page_content}"
        result.append(text)
    return result


def cosine_normalizer(val: float) -> float:
    res = (1 + val) / 2
    res = max(
        0, min(1, res)
    )  # float precision can cause values like 1.0000000596046448
    return res


def get_comparator(condition: str):
    def comparator(data: dict[str, Any]):
        try:
            return eval(condition, {}, data)
        except Exception as e:
            # PrintStyle.error(f"Error evaluating condition: {e}")
            return False

    return comparator



================================================
File: python/helpers/whisper.py
================================================
import base64
import warnings
import whisper
import tempfile
import asyncio
from python.helpers import runtime, rfc, settings
from python.helpers.print_style import PrintStyle

# Suppress FutureWarning from torch.load
warnings.filterwarnings("ignore", category=FutureWarning)

_model = None
_model_name = ""
is_updating_model = False  # Tracks whether the model is currently updating

async def preload(model_name:str):
    try:
        return await runtime.call_development_function(_preload, model_name)
    except Exception as e:
        if not runtime.is_development():
            raise e
        
async def _preload(model_name:str):
    global _model, _model_name, is_updating_model

    while is_updating_model:
        await asyncio.sleep(0.1)

    try:
        is_updating_model = True
        if not _model or _model_name != model_name:
                PrintStyle.standard(f"Loading Whisper model: {model_name}")
                _model = whisper.load_model(name=model_name) # type: ignore
                _model_name = model_name
    finally:
        is_updating_model = False

async def is_downloading():
    return await runtime.call_development_function(_is_downloading)

def _is_downloading():
    return is_updating_model

async def transcribe(model_name:str, audio_bytes_b64: str):
    return await runtime.call_development_function(_transcribe, model_name, audio_bytes_b64)


async def _transcribe(model_name:str, audio_bytes_b64: str):
    await _preload(model_name)
    
    # Decode audio bytes if encoded as a base64 string
    audio_bytes = base64.b64decode(audio_bytes_b64)

    # Create temp audio file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
        audio_file.write(audio_bytes)

    # Transcribe the audio file
    result = _model.transcribe(audio_file.name, fp16=False) # type: ignore
    return result



================================================
File: python/tools/behaviour_adjustment.py
================================================
from python.helpers import files, memory
from python.helpers.tool import Tool, Response
from agent import Agent
from python.helpers.log import LogItem


class UpdateBehaviour(Tool):

    async def execute(self, adjustments="", **kwargs):

        # stringify adjustments if needed
        if not isinstance(adjustments, str):
            adjustments = str(adjustments)

        await update_behaviour(self.agent, self.log, adjustments)
        return Response(
            message=self.agent.read_prompt("behaviour.updated.md"), break_loop=False
        )

    # async def before_execution(self, **kwargs):
    #     pass

    # async def after_execution(self, response, **kwargs):
    #     pass


async def update_behaviour(agent: Agent, log_item: LogItem, adjustments: str):

    # get system message and current ruleset
    system = agent.read_prompt("behaviour.merge.sys.md")
    current_rules = read_rules(agent)

    # log query streamed by LLM
    async def log_callback(content):
        log_item.stream(ruleset=content)

    msg = agent.read_prompt(
        "behaviour.merge.msg.md", current_rules=current_rules, adjustments=adjustments
    )

    # call util llm to find solutions in history
    adjustments_merge = await agent.call_utility_model(
        system=system,
        message=msg,
        callback=log_callback,
    )

    # update rules file
    rules_file = get_custom_rules_file(agent)
    files.write_file(rules_file, adjustments_merge)
    log_item.update(result="Behaviour updated")


def get_custom_rules_file(agent: Agent):
    return memory.get_memory_subdir_abs(agent) + f"/behaviour.md"


def read_rules(agent: Agent):
    rules_file = get_custom_rules_file(agent)
    if files.exists(rules_file):
        rules = files.read_file(rules_file)
        return agent.read_prompt("agent.system.behaviour.md", rules=rules)
    else:
        rules = agent.read_prompt("agent.system.behaviour_default.md")
        return agent.read_prompt("agent.system.behaviour.md", rules=rules)



================================================
File: python/tools/browser.py
================================================
import asyncio
from dataclasses import dataclass
import time
from python.helpers.tool import Tool, Response
from python.helpers import files, rfc_exchange
from python.helpers.print_style import PrintStyle
from python.helpers.browser import Browser as BrowserManager
import uuid


@dataclass
class State:
    browser: BrowserManager


class Browser(Tool):

    async def execute(self, **kwargs):
        raise NotImplementedError

    def get_log_object(self):
        return self.agent.context.log.log(
            type="browser",
            heading=f"{self.agent.agent_name}: Using tool '{self.name}'",
            content="",
            kvps=self.args,
        )

    # async def after_execution(self, response, **kwargs):
    #     await self.agent.hist_add_tool_result(self.name, response.message)

    async def save_screenshot(self):
        await self.prepare_state()
        path = files.get_abs_path("tmp/browser", f"{uuid.uuid4()}.png")
        await self.state.browser.screenshot(path, True)
        return "img://" + path

    async def prepare_state(self, reset=False):
        self.state = self.agent.get_data("_browser_state")
        if not self.state or reset:
            self.state = State(browser=BrowserManager())
        self.agent.set_data("_browser_state", self.state)

    def update_progress(self, text):
        progress = f"Browser: {text}"
        self.log.update(progress=text)
        self.agent.context.log.set_progress(progress)

    def cleanup_history(self):
        def cleanup_message(msg):
            if not msg.ai and isinstance(msg.content, dict) and "tool_name" in msg.content and str(msg.content["tool_name"]).startswith("browser_"):
                if not msg.summary:
                    msg.summary = "browser content removed to save space"

        for msg in self.agent.history.current.messages:
            cleanup_message(msg)
        
        for prev in self.agent.history.topics:
            if not prev.summary:
                for msg in prev.messages:
                    cleanup_message(msg)



================================================
File: python/tools/browser_agent.py
================================================
import asyncio
import json
import time
from agent import Agent, InterventionException

import models
from python.helpers.tool import Tool, Response
from python.helpers import files, defer, persist_chat
from python.helpers.browser_use import browser_use
from python.extensions.message_loop_start._10_iteration_no import get_iter_no
from pydantic import BaseModel
import uuid
from python.helpers.dirty_json import DirtyJson
from langchain_core.messages import SystemMessage

class State:
    @staticmethod
    async def create(agent: Agent):
        state = State(agent)
        return state

    def __init__(self, agent: Agent):
        self.agent = agent
        self.context = None
        self.task = None
        self.use_agent = None
        self.browser = None
        self.iter_no = 0


    def __del__(self):
        self.kill_task()

    async def _initialize(self):
        if self.context:
            return

        self.browser = browser_use.Browser(
            config=browser_use.BrowserConfig(
                headless=True,
                disable_security=True,
            )
        )

        # Await the coroutine to get the browser context
        self.context = await self.browser.new_context()

        # override async methods to create hooks
        self.override_hooks()

        # Add init script to the context - this will be applied to all new pages
        await self.context._initialize_session()
        pw_context = self.context.session.context  # type: ignore
        js_override = files.get_abs_path("lib/browser/init_override.js")
        await pw_context.add_init_script(path=js_override)  # type: ignore

    def start_task(self, task: str):
        if self.task and self.task.is_alive():
            self.kill_task()

        if not self.task:
            self.task = defer.DeferredTask(
                thread_name="BrowserAgent" + self.agent.context.id
            )
            if self.agent.context.task:
                self.agent.context.task.add_child_task(self.task, terminate_thread=True)
        self.task.start_task(self._run_task, task)
        return self.task

    def kill_task(self):
        if self.task:
            self.task.kill(terminate_thread=True)
            self.task = None
            self.context = None
            self.use_agent = None
            self.browser = None
            self.iter_no = 0

    async def _run_task(self, task: str):

        agent = self.agent

        await self._initialize()

        class CustomSystemPrompt(browser_use.SystemPrompt):
            def get_system_message(self) -> SystemMessage:
                existing_rules = super().get_system_message().text()
                new_rules = agent.read_prompt("prompts/browser_agent.system.md")
                return SystemMessage(content=f"{existing_rules}\n{new_rules}".strip())

        # Model of task result
        class DoneResult(BaseModel):
            title: str
            response: str
            page_summary: str

        # Initialize controller
        controller = browser_use.Controller()

        # we overwrite done() in this example to demonstrate the validator
        @controller.registry.action("Done with task", param_model=DoneResult)
        async def done(params: DoneResult):
            result = browser_use.ActionResult(
                is_done=True, extracted_content=params.model_dump_json()
            )
            return result

        # @controller.action("Ask user for information")
        # def ask_user(question: str) -> str:
        #     return "..."

        model = models.get_model(
            type=models.ModelType.CHAT,
            provider=self.agent.config.browser_model.provider,
            name=self.agent.config.browser_model.name,
            **self.agent.config.browser_model.kwargs,
        )

        self.use_agent = browser_use.Agent(
            task=task,
            browser_context=self.context,
            llm=model,
            use_vision=self.agent.config.browser_model.vision,
            system_prompt_class=CustomSystemPrompt,
            controller=controller,
        )

        self.iter_no = get_iter_no(self.agent)

        # orig_err_hnd = self.use_agent._handle_step_error
        # def new_err_hnd(*args, **kwargs):
        #     if isinstance(args[0], InterventionException):
        #         raise args[0]
        #     return orig_err_hnd(*args, **kwargs)
        # self.use_agent._handle_step_error = new_err_hnd

        result = await self.use_agent.run()
        return result

    def override_hooks(self):
        # override async function to create a hook
        def override_hook(func):
            async def wrapper(*args, **kwargs):
                await self.agent.wait_if_paused()
                if self.iter_no != get_iter_no(self.agent):
                    raise InterventionException("Task cancelled")
                return await func(*args, **kwargs)
            return wrapper

        if self.context:
            self.context.get_state = override_hook(self.context.get_state)
            self.context.get_session = override_hook(self.context.get_session)
            self.context.remove_highlights = override_hook(self.context.remove_highlights)

    async def get_page(self):
        if self.use_agent:
            return await self.use_agent.browser_context.get_current_page()


class BrowserAgent(Tool):

    async def execute(self, message="", reset="", **kwargs):
        self.guid = str(uuid.uuid4())
        reset = str(reset).lower().strip() == "true"
        await self.prepare_state(reset=reset)
        task = self.state.start_task(message)

        # wait for browser agent to finish and update progress
        while not task.is_ready():
            await self.agent.handle_intervention()
            await asyncio.sleep(1)
            try:
                update = await self.get_update()
                log = update.get("log")
                if log:
                    self.update_progress("\n".join(log))
                screenshot = update.get("screenshot", None)
                if screenshot:
                    self.log.update(screenshot=screenshot)
            except Exception as e:
                pass

        # collect result
        result = await task.result()
        answer = result.final_result()
        try:
            answer_data = DirtyJson.parse_string(answer)
            answer_text = strings.dict_to_text(answer_data)  # type: ignore
        except Exception as e:
            answer_text = answer
        self.log.update(answer=answer_text)
        return Response(message=answer, break_loop=False)

    def get_log_object(self):
        return self.agent.context.log.log(
            type="browser",
            heading=f"{self.agent.agent_name}: Using tool '{self.name}'",
            content="",
            kvps=self.args,
        )

    # async def after_execution(self, response, **kwargs):
    #     await self.agent.hist_add_tool_result(self.name, response.message)

    async def get_update(self):
        await self.prepare_state()

        result = {}
        agent = self.agent
        ua = self.state.use_agent
        page = await self.state.get_page()
        ctx = self.state.context

        if ua and page:
            try:

                async def _get_update():

                    await agent.wait_if_paused()

                    log = []

                    # dom_service = browser_use.DomService(page)
                    # dom_state = await browser_use.utils.time_execution_sync('get_clickable_elements')(
                    #     dom_service.get_clickable_elements
                    # )()
                    # elements = dom_state.element_tree
                    # selector_map = dom_state.selector_map
                    # el_text = elements.clickable_elements_to_string()

                    for message in ua.message_manager.get_messages():
                        if message.type == "system":
                            continue
                        if message.type == "ai":
                            try:
                                data = json.loads(message.content)  # type: ignore
                                cs = data.get("current_state")
                                if cs:
                                    log.append("AI:" + cs["memory"])
                                    log.append("AI:" + cs["next_goal"])
                            except Exception:
                                pass
                        if message.type == "human":
                            content = str(message.content).strip()
                            part = content.split("\n", 1)[0].split(",", 1)[0]
                            if part:
                                if len(part) > 150:
                                    part = part[:150] + "..."
                                log.append("FW:" + part)
                    result["log"] = log

                    path = files.get_abs_path(
                        persist_chat.get_chat_folder_path(agent.context.id),
                        "browser",
                        "screenshots",
                        f"{self.guid}.png",
                    )
                    files.make_dirs(path)
                    await page.screenshot(path=path, full_page=False, timeout=3000)
                    result["screenshot"] = f"img://{path}&t={str(time.time())}"

                if self.state.task:
                    await self.state.task.execute_inside(_get_update)

            except Exception as e:
                pass

        return result

    async def prepare_state(self, reset=False):
        self.state = self.agent.get_data("_browser_agent_state")
        if not self.state or reset:
            self.state = await State.create(self.agent)
        self.agent.set_data("_browser_agent_state", self.state)

    def update_progress(self, text):
        short = text.split("\n")[-1]
        if len(short) > 50:
            short = short[:50] + "..."
        progress = f"Browser: {short}"

        self.log.update(progress=text)
        self.agent.context.log.set_progress(progress)

    # def __del__(self):
    #     if self.state:
    #         self.state.kill_task()



================================================
File: python/tools/browser_do.py
================================================
import asyncio
from python.helpers.tool import Tool, Response
from python.tools.browser import Browser
from python.helpers.browser import NoPageError
import asyncio


class BrowserDo(Browser):

    async def execute(self, fill=[], press=[], click=[], execute="", **kwargs):
        await self.prepare_state()
        result = ""
        try:
            if fill:
                self.update_progress("Filling fields...")
                for f in fill:
                    await self.state.browser.fill(f["selector"], f["text"])
                    await self.state.browser.wait(0.5)
            if press:
                self.update_progress("Pressing keys...")
                if fill:
                    await self.state.browser.wait(1)
                for p in press:
                    await self.state.browser.press(p)
                    await self.state.browser.wait(0.5)
            if click:
                self.update_progress("Clicking...")
                if fill:
                    await self.state.browser.wait(1)
                for c in click:
                    await self.state.browser.click(c)
                    await self.state.browser.wait(0.5)
            if execute:
                if fill or press or click:
                    await self.state.browser.wait(1)
                self.update_progress("Executing...")
                result = await self.state.browser.execute(execute)
                self.log.update(result=result)

            self.update_progress("Retrieving...")
            await self.state.browser.wait_for_action()
            dom = await self.state.browser.get_clean_dom()
            if result:
                response = f"Result:\n{result}\n\nDOM:\n{dom}"
            else:
                response = dom
            self.update_progress("Taking screenshot...")
            screenshot = await self.save_screenshot()
            self.log.update(screenshot=screenshot)
        except Exception as e:
            response = str(e)
            self.log.update(error=response)
            
            try:
                screenshot = await self.save_screenshot()
                dom = await self.state.browser.get_clean_dom()
                response = f"Error:\n{response}\n\nDOM:\n{dom}"
                self.log.update(screenshot=screenshot)
            except Exception:
                pass

        self.cleanup_history()
        self.update_progress("Done")
        return Response(message=response, break_loop=False)



================================================
File: python/tools/browser_open.py
================================================
import asyncio
from python.helpers.tool import Tool, Response
from python.tools import browser
from python.tools.browser import Browser


class BrowserOpen(Browser):

    async def execute(self, url="", **kwargs):
        self.update_progress("Initializing...")
        await self.prepare_state()

        try:
            if url:
                self.update_progress("Opening page...")
                await self.state.browser.open(url)
            
            self.update_progress("Retrieving...")
            await self.state.browser.wait_for_action()
            response = await self.state.browser.get_clean_dom()
            self.update_progress("Taking screenshot...")
            screenshot = await self.save_screenshot()
            self.log.update(screenshot=screenshot)
        except Exception as e:
            response = str(e)
            self.log.update(error=response)

        self.cleanup_history()
        self.update_progress("Done")
        return Response(message=response, break_loop=False)



================================================
File: python/tools/call_subordinate.py
================================================
from agent import Agent, UserMessage
from python.helpers.tool import Tool, Response


class Delegation(Tool):

    async def execute(self, message="", reset="", **kwargs):
        # create subordinate agent using the data object on this agent and set superior agent to his data object
        if (
            self.agent.get_data(Agent.DATA_NAME_SUBORDINATE) is None
            or str(reset).lower().strip() == "true"
        ):
            sub = Agent(
                self.agent.number + 1, self.agent.config, self.agent.context
            )
            sub.set_data(Agent.DATA_NAME_SUPERIOR, self.agent)
            self.agent.set_data(Agent.DATA_NAME_SUBORDINATE, sub)

        # add user message to subordinate agent
        subordinate: Agent = self.agent.get_data(Agent.DATA_NAME_SUBORDINATE)
        subordinate.hist_add_user_message(UserMessage(message=message, attachments=[]))
        # run subordinate monologue
        result = await subordinate.monologue()
        # result
        return Response(message=result, break_loop=False)



================================================
File: python/tools/code_execution_tool.py
================================================
import asyncio
from dataclasses import dataclass
import shlex
import time
from python.helpers.tool import Tool, Response
from python.helpers import files, rfc_exchange
from python.helpers.print_style import PrintStyle
from python.helpers.shell_local import LocalInteractiveSession
from python.helpers.shell_ssh import SSHInteractiveSession
from python.helpers.docker import DockerContainerManager
from python.helpers.messages import truncate_text
import re


@dataclass
class State:
    shells: dict[int, LocalInteractiveSession | SSHInteractiveSession]
    docker: DockerContainerManager | None


class CodeExecution(Tool):

    async def execute(self, **kwargs):

        await self.agent.handle_intervention()  # wait for intervention and handle it, if paused

        await self.prepare_state()

        # os.chdir(files.get_abs_path("./work_dir")) #change CWD to work_dir

        runtime = self.args.get("runtime", "").lower().strip()
        session = int(self.args.get("session", 0))

        if runtime == "python":
            response = await self.execute_python_code(
                code=self.args["code"], session=session
            )
        elif runtime == "nodejs":
            response = await self.execute_nodejs_code(
                code=self.args["code"], session=session
            )
        elif runtime == "terminal":
            response = await self.execute_terminal_command(
                command=self.args["code"], session=session
            )
        elif runtime == "output":
            response = await self.get_terminal_output(
                session=session, first_output_timeout=60, between_output_timeout=5
            )
        elif runtime == "reset":
            response = await self.reset_terminal(session=session)
        else:
            response = self.agent.read_prompt(
                "fw.code.runtime_wrong.md", runtime=runtime
            )

        if not response:
            response = self.agent.read_prompt(
                "fw.code.info.md", info=self.agent.read_prompt("fw.code.no_output.md")
            )
        return Response(message=response, break_loop=False)

    def get_log_object(self):
        return self.agent.context.log.log(
            type="code_exe",
            heading=f"{self.agent.agent_name}: Using tool '{self.name}'",
            content="",
            kvps=self.args,
        )

    async def after_execution(self, response, **kwargs):
        self.agent.hist_add_tool_result(self.name, response.message)

    async def prepare_state(self, reset=False, session=None):
        self.state = self.agent.get_data("_cet_state")
        if not self.state or reset:

            # initialize docker container if execution in docker is configured
            if not self.state and self.agent.config.code_exec_docker_enabled:
                docker = DockerContainerManager(
                    logger=self.agent.context.log,
                    name=self.agent.config.code_exec_docker_name,
                    image=self.agent.config.code_exec_docker_image,
                    ports=self.agent.config.code_exec_docker_ports,
                    volumes=self.agent.config.code_exec_docker_volumes,
                )
                docker.start_container()
            else:
                docker = self.state.docker if self.state else None

            # initialize shells dictionary if not exists
            shells = {} if not self.state else self.state.shells.copy()

            # Only reset the specified session if provided
            if session is not None and session in shells:
                shells[session].close()
                del shells[session]
            elif reset and not session:
                # Close all sessions if full reset requested
                for s in list(shells.keys()):
                    shells[s].close()
                shells = {}

            # initialize local or remote interactive shell interface for session 0 if needed
            if 0 not in shells:
                if self.agent.config.code_exec_ssh_enabled:
                    pswd = (
                        self.agent.config.code_exec_ssh_pass
                        if self.agent.config.code_exec_ssh_pass
                        else await rfc_exchange.get_root_password()
                    )
                    shell = SSHInteractiveSession(
                        self.agent.context.log,
                        self.agent.config.code_exec_ssh_addr,
                        self.agent.config.code_exec_ssh_port,
                        self.agent.config.code_exec_ssh_user,
                        pswd,
                    )
                else:
                    shell = LocalInteractiveSession()

                shells[0] = shell
                await shell.connect()

            self.state = State(shells=shells, docker=docker)
        self.agent.set_data("_cet_state", self.state)

    async def execute_python_code(self, session: int, code: str, reset: bool = False):
        escaped_code = shlex.quote(code)
        command = f"ipython -c {escaped_code}"
        return await self.terminal_session(session, command, reset)

    async def execute_nodejs_code(self, session: int, code: str, reset: bool = False):
        escaped_code = shlex.quote(code)
        command = f"node /exe/node_eval.js {escaped_code}"
        return await self.terminal_session(session, command, reset)

    async def execute_terminal_command(
        self, session: int, command: str, reset: bool = False
    ):
        return await self.terminal_session(session, command, reset)

    async def terminal_session(self, session: int, command: str, reset: bool = False):

        await self.agent.handle_intervention()  # wait for intervention and handle it, if paused
        # try again on lost connection
        for i in range(2):
            try:

                if reset:
                    await self.reset_terminal()

                if session not in self.state.shells:
                    if self.agent.config.code_exec_ssh_enabled:
                        pswd = (
                            self.agent.config.code_exec_ssh_pass
                            if self.agent.config.code_exec_ssh_pass
                            else await rfc_exchange.get_root_password()
                        )
                        shell = SSHInteractiveSession(
                            self.agent.context.log,
                            self.agent.config.code_exec_ssh_addr,
                            self.agent.config.code_exec_ssh_port,
                            self.agent.config.code_exec_ssh_user,
                            pswd,
                        )
                    else:
                        shell = LocalInteractiveSession()
                    self.state.shells[session] = shell
                    await shell.connect()

                self.state.shells[session].send_command(command)

                PrintStyle(
                    background_color="white", font_color="#1B4F72", bold=True
                ).print(f"{self.agent.agent_name} code execution output")
                return await self.get_terminal_output(session)

            except Exception as e:
                if i == 1:
                    # try again on lost connection
                    PrintStyle.error(str(e))
                    await self.prepare_state(reset=True)
                    continue
                else:
                    raise e

    async def get_terminal_output(
        self,
        session=0,
        reset_full_output=True,
        first_output_timeout=30,  # Wait up to x seconds for first output
        between_output_timeout=15,  # Wait up to x seconds between outputs
        max_exec_timeout=180,  #hard cap on total runtime
        sleep_time=0.1,
    ):
        # Common shell prompt regex patterns (add more as needed)
        prompt_patterns = [
            re.compile(r"\\(venv\\).+[$#] ?$"),  # (venv) ...$ or (venv) ...#
            re.compile(r"root@[^:]+:[^#]+# ?$"),  # root@container:~#
            re.compile(r"[a-zA-Z0-9_.-]+@[^:]+:[^$#]+[$#] ?$"),  # user@host:~$
        ]

        start_time = time.time()
        last_output_time = start_time
        full_output = ""
        truncated_output = ""
        got_output = False

        while True:
            await asyncio.sleep(sleep_time)
            full_output, partial_output = await self.state.shells[session].read_output(
                timeout=3, reset_full_output=reset_full_output
            )
            reset_full_output = False  # only reset once

            await self.agent.handle_intervention()

            now = time.time()
            if partial_output:
                PrintStyle(font_color="#85C1E9").stream(partial_output)
                # full_output += partial_output # Append new output
                truncated_output = truncate_text(
                    agent=self.agent, output=full_output, threshold=10000
                )
                self.log.update(content=truncated_output)
                last_output_time = now
                got_output = True

                # Check for shell prompt at the end of output
                last_lines = truncated_output.splitlines()[-3:] if truncated_output else []
                for line in last_lines:
                    for pat in prompt_patterns:
                        if pat.search(line.strip()):
                            PrintStyle.info(
                                "Detected shell prompt, returning output early."
                            )
                            return truncated_output

            # Check for max execution time
            if now - start_time > max_exec_timeout:
                sysinfo = self.agent.read_prompt(
                    "fw.code.max_time.md", timeout=max_exec_timeout
                )
                response = self.agent.read_prompt("fw.code.info.md", info=sysinfo)
                if truncated_output:
                    response = truncated_output + "\n\n" + response
                PrintStyle.warning(sysinfo)
                self.log.update(content=response)
                return response

            # Waiting for first output
            if not got_output:
                if now - start_time > first_output_timeout:
                    sysinfo = self.agent.read_prompt(
                        "fw.code.no_out_time.md", timeout=first_output_timeout
                    )
                    response = self.agent.read_prompt("fw.code.info.md", info=sysinfo)
                    PrintStyle.warning(sysinfo)
                    self.log.update(content=response)
                    return response
            else:
                # Waiting for more output after first output
                if now - last_output_time > between_output_timeout:
                    sysinfo = self.agent.read_prompt(
                        "fw.code.pause_time.md", timeout=between_output_timeout
                    )
                    response = self.agent.read_prompt("fw.code.info.md", info=sysinfo)
                    if truncated_output:
                        response = truncated_output + "\n\n" + response
                    PrintStyle.warning(sysinfo)
                    self.log.update(content=response)
                    return response

    async def reset_terminal(self, session=0, reason: str | None = None):
        # Print the reason for the reset to the console if provided
        if reason:
            PrintStyle(font_color="#FFA500", bold=True).print(
                f"Resetting terminal session {session}... Reason: {reason}"
            )
        else:
            PrintStyle(font_color="#FFA500", bold=True).print(
                f"Resetting terminal session {session}..."
            )

        # Only reset the specified session while preserving others
        await self.prepare_state(reset=True, session=session)
        response = self.agent.read_prompt(
            "fw.code.info.md", info=self.agent.read_prompt("fw.code.reset.md")
        )
        self.log.update(content=response)
        return response



================================================
File: python/tools/input.py
================================================
from agent import Agent, UserMessage
from python.helpers.tool import Tool, Response
from python.tools.code_execution_tool import CodeExecution


class Input(Tool):

    async def execute(self, keyboard="", **kwargs):
        # normalize keyboard input
        keyboard = keyboard.rstrip()
        keyboard += "\n"
        
        # terminal session number
        session = int(self.args.get("session", 0))

        # forward keyboard input to code execution tool
        args = {"runtime": "terminal", "code": keyboard, "session": session}
        cet = CodeExecution(self.agent, "code_execution_tool", "", args, self.message)
        cet.log = self.log
        return await cet.execute(**args)

    def get_log_object(self):
        return self.agent.context.log.log(type="code_exe", heading=f"{self.agent.agent_name}: Using tool '{self.name}'", content="", kvps=self.args)

    async def after_execution(self, response, **kwargs):
        self.agent.hist_add_tool_result(self.name, response.message)


================================================
File: python/tools/knowledge_tool.py
================================================
import os
import asyncio
from python.helpers import dotenv, memory, perplexity_search, duckduckgo_search
from python.helpers.tool import Tool, Response
from python.helpers.print_style import PrintStyle
from python.helpers.errors import handle_error
from python.helpers.searxng import search as searxng
from python.tools.memory_load import DEFAULT_THRESHOLD as DEFAULT_MEMORY_THRESHOLD

SEARCH_ENGINE_RESULTS = 10


class Knowledge(Tool):
    async def execute(self, question="", **kwargs):
        # Create tasks for all three search methods
        tasks = [
            self.searxng_search(question),
            # self.perplexity_search(question),
            # self.duckduckgo_search(question),
            self.mem_search(question),
        ]

        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # perplexity_result, duckduckgo_result, memory_result = results
        searxng_result, memory_result = results

        # Handle exceptions and format results
        # perplexity_result = self.format_result(perplexity_result, "Perplexity")
        # duckduckgo_result = self.format_result(duckduckgo_result, "DuckDuckGo")
        searxng_result = self.format_result_searxng(searxng_result, "Search Engine")
        memory_result = self.format_result(memory_result, "Memory")

        msg = self.agent.read_prompt(
            "tool.knowledge.response.md",
            #   online_sources = ((perplexity_result + "\n\n") if perplexity_result else "") + str(duckduckgo_result),
            online_sources=((searxng_result + "\n\n") if searxng_result else ""),
            memory=memory_result,
        )

        await self.agent.handle_intervention(
            msg
        )  # wait for intervention and handle it, if paused

        return Response(message=msg, break_loop=False)

    async def perplexity_search(self, question):
        if dotenv.get_dotenv_value("API_KEY_PERPLEXITY"):
            return await asyncio.to_thread(
                perplexity_search.perplexity_search, question
            )
        else:
            PrintStyle.hint(
                "No API key provided for Perplexity. Skipping Perplexity search."
            )
            self.agent.context.log.log(
                type="hint",
                content="No API key provided for Perplexity. Skipping Perplexity search.",
            )
            return None

    async def duckduckgo_search(self, question):
        return await asyncio.to_thread(duckduckgo_search.search, question)

    async def searxng_search(self, question):
        return await searxng(question)

    async def mem_search(self, question: str):
        db = await memory.Memory.get(self.agent)
        docs = await db.search_similarity_threshold(
            query=question, limit=5, threshold=DEFAULT_MEMORY_THRESHOLD
        )
        text = memory.Memory.format_docs_plain(docs)
        return "\n\n".join(text)

    def format_result(self, result, source):
        if isinstance(result, Exception):
            handle_error(result)
            return f"{source} search failed: {str(result)}"
        return result if result else ""

    def format_result_searxng(self, result, source):
        if isinstance(result, Exception):
            handle_error(result)
            return f"{source} search failed: {str(result)}"

        outputs = []
        for item in result["results"]:
            outputs.append(f"{item['title']}\n{item['url']}\n{item['content']}")

        return "\n\n".join(outputs[:SEARCH_ENGINE_RESULTS]).strip()



================================================
File: python/tools/memory_delete.py
================================================
from python.helpers.memory import Memory
from python.helpers.tool import Tool, Response


class MemoryDelete(Tool):

    async def execute(self, ids="", **kwargs):
        db = await Memory.get(self.agent)
        ids = [id.strip() for id in ids.split(",") if id.strip()]
        dels = await db.delete_documents_by_ids(ids=ids)

        result = self.agent.read_prompt("fw.memories_deleted.md", memory_count=len(dels))
        return Response(message=result, break_loop=False)



================================================
File: python/tools/memory_forget.py
================================================
from python.helpers.memory import Memory
from python.helpers.tool import Tool, Response
from python.tools.memory_load import DEFAULT_THRESHOLD


class MemoryForget(Tool):

    async def execute(self, query="", threshold=DEFAULT_THRESHOLD, filter="", **kwargs):
        db = await Memory.get(self.agent)
        dels = await db.delete_documents_by_query(query=query, threshold=threshold, filter=filter)

        result = self.agent.read_prompt("fw.memories_deleted.md", memory_count=len(dels))
        return Response(message=result, break_loop=False)



================================================
File: python/tools/memory_load.py
================================================
from python.helpers.memory import Memory
from python.helpers.tool import Tool, Response

DEFAULT_THRESHOLD = 0.7
DEFAULT_LIMIT = 10


class MemoryLoad(Tool):

    async def execute(self, query="", threshold=DEFAULT_THRESHOLD, limit=DEFAULT_LIMIT, filter="", **kwargs):
        db = await Memory.get(self.agent)
        docs = await db.search_similarity_threshold(query=query, limit=limit, threshold=threshold, filter=filter)

        if len(docs) == 0:
            result = self.agent.read_prompt("fw.memories_not_found.md", query=query)
        else:
            text = "\n\n".join(Memory.format_docs_plain(docs))
            result = str(text)

        return Response(message=result, break_loop=False)



================================================
File: python/tools/memory_save.py
================================================
from python.helpers.memory import Memory
from python.helpers.tool import Tool, Response


class MemorySave(Tool):

    async def execute(self, text="", area="", **kwargs):

        if not area:
            area = Memory.Area.MAIN.value

        metadata = {"area": area, **kwargs}

        db = await Memory.get(self.agent)
        id = await db.insert_text(text, metadata)

        result = self.agent.read_prompt("fw.memory_saved.md", memory_id=id)
        return Response(message=result, break_loop=False)



================================================
File: python/tools/response.py
================================================
from python.helpers.tool import Tool, Response

class ResponseTool(Tool):

    async def execute(self,**kwargs):
        return Response(message=self.args["text"], break_loop=True)

    async def before_execution(self, **kwargs):
        self.log = self.agent.context.log.log(type="response", heading=f"{self.agent.agent_name}: Responding", content=self.args.get("text", ""))

    
    async def after_execution(self, response, **kwargs):
        pass # do not add anything to the history or output


================================================
File: python/tools/scheduler.py
================================================
import asyncio
from datetime import datetime
import json
import random
import re
from python.helpers.tool import Tool, Response
from python.helpers.task_scheduler import (
    TaskScheduler, ScheduledTask, AdHocTask, PlannedTask,
    serialize_task, TaskState, TaskSchedule, TaskPlan, parse_datetime, serialize_datetime
)
from agent import AgentContext
from python.helpers import persist_chat

DEFAULT_WAIT_TIMEOUT = 300


class SchedulerTool(Tool):

    async def execute(self, **kwargs):
        if self.method == "list_tasks":
            return await self.list_tasks(**kwargs)
        elif self.method == "find_task_by_name":
            return await self.find_task_by_name(**kwargs)
        elif self.method == "show_task":
            return await self.show_task(**kwargs)
        elif self.method == "run_task":
            return await self.run_task(**kwargs)
        elif self.method == "delete_task":
            return await self.delete_task(**kwargs)
        elif self.method == "create_scheduled_task":
            return await self.create_scheduled_task(**kwargs)
        elif self.method == "create_adhoc_task":
            return await self.create_adhoc_task(**kwargs)
        elif self.method == "create_planned_task":
            return await self.create_planned_task(**kwargs)
        elif self.method == "wait_for_task":
            return await self.wait_for_task(**kwargs)
        else:
            return Response(message=f"Unknown method '{self.name}:{self.method}'", break_loop=False)

    async def list_tasks(self, **kwargs) -> Response:
        state_filter: list[str] | None = kwargs.get("state", None)
        type_filter: list[str] | None = kwargs.get("type", None)
        next_run_within_filter: int | None = kwargs.get("next_run_within", None)
        next_run_after_filter: int | None = kwargs.get("next_run_after", None)

        tasks: list[ScheduledTask | AdHocTask | PlannedTask] = TaskScheduler.get().get_tasks()
        filtered_tasks = []
        for task in tasks:
            if state_filter and task.state not in state_filter:
                continue
            if type_filter and task.type not in type_filter:
                continue
            if next_run_within_filter and task.get_next_run_minutes() is not None and task.get_next_run_minutes() > next_run_within_filter:  # type: ignore
                continue
            if next_run_after_filter and task.get_next_run_minutes() is not None and task.get_next_run_minutes() < next_run_after_filter:  # type: ignore
                continue
            filtered_tasks.append(serialize_task(task))

        return Response(message=json.dumps(filtered_tasks, indent=4), break_loop=False)

    async def find_task_by_name(self, **kwargs) -> Response:
        name: str = kwargs.get("name", None)
        if not name:
            return Response(message="Task name is required", break_loop=False)
        tasks: list[ScheduledTask | AdHocTask | PlannedTask] = TaskScheduler.get().find_task_by_name(name)
        if not tasks:
            return Response(message=f"Task not found: {name}", break_loop=False)
        return Response(message=json.dumps([serialize_task(task) for task in tasks], indent=4), break_loop=False)

    async def show_task(self, **kwargs) -> Response:
        task_uuid: str = kwargs.get("uuid", None)
        if not task_uuid:
            return Response(message="Task UUID is required", break_loop=False)
        task: ScheduledTask | AdHocTask | PlannedTask | None = TaskScheduler.get().get_task_by_uuid(task_uuid)
        if not task:
            return Response(message=f"Task not found: {task_uuid}", break_loop=False)
        return Response(message=json.dumps(serialize_task(task), indent=4), break_loop=False)

    async def run_task(self, **kwargs) -> Response:
        task_uuid: str = kwargs.get("uuid", None)
        if not task_uuid:
            return Response(message="Task UUID is required", break_loop=False)
        task_context: str | None = kwargs.get("context", None)
        task: ScheduledTask | AdHocTask | PlannedTask | None = TaskScheduler.get().get_task_by_uuid(task_uuid)
        if not task:
            return Response(message=f"Task not found: {task_uuid}", break_loop=False)
        await TaskScheduler.get().run_task_by_uuid(task_uuid, task_context)
        if task.context_id == self.agent.context.id:
            break_loop = True  # break loop if task is running in the same context, otherwise it would start two conversations in one window
        else:
            break_loop = False
        return Response(message=f"Task started: {task_uuid}", break_loop=break_loop)

    async def delete_task(self, **kwargs) -> Response:
        task_uuid: str = kwargs.get("uuid", None)
        if not task_uuid:
            return Response(message="Task UUID is required", break_loop=False)

        task: ScheduledTask | AdHocTask | PlannedTask | None = TaskScheduler.get().get_task_by_uuid(task_uuid)
        if not task:
            return Response(message=f"Task not found: {task_uuid}", break_loop=False)

        context = None
        if task.context_id:
            context = AgentContext.get(task.context_id)

        if task.state == TaskState.RUNNING:
            if context:
                context.reset()
            await TaskScheduler.get().update_task(task_uuid, state=TaskState.IDLE)
            await TaskScheduler.get().save()

        if context and context.id == task.uuid:
            AgentContext.remove(context.id)
            persist_chat.remove_chat(context.id)

        await TaskScheduler.get().remove_task_by_uuid(task_uuid)
        if TaskScheduler.get().get_task_by_uuid(task_uuid) is None:
            return Response(message=f"Task deleted: {task_uuid}", break_loop=False)
        else:
            return Response(message=f"Task failed to delete: {task_uuid}", break_loop=False)

    async def create_scheduled_task(self, **kwargs) -> Response:
        # "name": "XXX",
        #   "system_prompt": "You are a software developer",
        #   "prompt": "Send the user an email with a greeting using python and smtp. The user's address is: xxx@yyy.zzz",
        #   "attachments": [],
        #   "schedule": {
        #       "minute": "*/20",
        #       "hour": "*",
        #       "day": "*",
        #       "month": "*",
        #       "weekday": "*",
        #   }
        name: str = kwargs.get("name", None)
        system_prompt: str = kwargs.get("system_prompt", None)
        prompt: str = kwargs.get("prompt", None)
        attachments: list[str] = kwargs.get("attachments", [])
        schedule: dict[str, str] = kwargs.get("schedule", {})
        dedicated_context: bool = kwargs.get("dedicated_context", False)

        task_schedule = TaskSchedule(
            minute=schedule.get("minute", "*"),
            hour=schedule.get("hour", "*"),
            day=schedule.get("day", "*"),
            month=schedule.get("month", "*"),
            weekday=schedule.get("weekday", "*"),
        )

        # Validate cron expression, agent might hallucinate
        cron_regex = "^((((\d+,)+\d+|(\d+(\/|-|#)\d+)|\d+L?|\*(\/\d+)?|L(-\d+)?|\?|[A-Z]{3}(-[A-Z]{3})?) ?){5,7})$"
        if not re.match(cron_regex, task_schedule.to_crontab()):
            return Response(message="Invalid cron expression: " + task_schedule.to_crontab(), break_loop=False)

        task = ScheduledTask.create(
            name=name,
            system_prompt=system_prompt,
            prompt=prompt,
            attachments=attachments,
            schedule=task_schedule,
            context_id=None if dedicated_context else self.agent.context.id
        )
        await TaskScheduler.get().add_task(task)
        return Response(message=f"Scheduled task '{name}' created: {task.uuid}", break_loop=False)

    async def create_adhoc_task(self, **kwargs) -> Response:
        name: str = kwargs.get("name", None)
        system_prompt: str = kwargs.get("system_prompt", None)
        prompt: str = kwargs.get("prompt", None)
        attachments: list[str] = kwargs.get("attachments", [])
        token: str = str(random.randint(1000000000000000000, 9999999999999999999))
        dedicated_context: bool = kwargs.get("dedicated_context", False)

        task = AdHocTask.create(
            name=name,
            system_prompt=system_prompt,
            prompt=prompt,
            attachments=attachments,
            token=token,
            context_id=None if dedicated_context else self.agent.context.id
        )
        await TaskScheduler.get().add_task(task)
        return Response(message=f"Adhoc task '{name}' created: {task.uuid}", break_loop=False)

    async def create_planned_task(self, **kwargs) -> Response:  # TODO: Implement
        name: str = kwargs.get("name", None)
        system_prompt: str = kwargs.get("system_prompt", None)
        prompt: str = kwargs.get("prompt", None)
        attachments: list[str] = kwargs.get("attachments", [])
        plan: list[str] = kwargs.get("plan", [])
        dedicated_context: bool = kwargs.get("dedicated_context", False)

        # Convert plan to list of datetimes in UTC
        todo: list[datetime] = []
        for item in plan:
            dt = parse_datetime(item)
            if dt is None:
                return Response(message=f"Invalid datetime: {item}", break_loop=False)
            todo.append(dt)

        # Create task plan with todo list
        task_plan = TaskPlan.create(
            todo=todo,
            in_progress=None,
            done=[]
        )

        # Create planned task with task plan
        task = PlannedTask.create(
            name=name,
            system_prompt=system_prompt,
            prompt=prompt,
            attachments=attachments,
            plan=task_plan,
            context_id=None if dedicated_context else self.agent.context.id
        )
        await TaskScheduler.get().add_task(task)
        return Response(message=f"Planned task '{name}' created: {task.uuid}", break_loop=False)

    async def wait_for_task(self, **kwargs) -> Response:
        task_uuid: str = kwargs.get("uuid", None)
        if not task_uuid:
            return Response(message="Task UUID is required", break_loop=False)

        scheduler = TaskScheduler.get()
        task: ScheduledTask | AdHocTask | PlannedTask | None = scheduler.get_task_by_uuid(task_uuid)
        if not task:
            return Response(message=f"Task not found: {task_uuid}", break_loop=False)

        if task.context_id == self.agent.context.id:
            return Response(message="You can only wait for tasks running in a different chat context (dedicated_context=True).", break_loop=False)

        done = False
        elapsed = 0
        while not done:
            await scheduler.reload()
            task = scheduler.get_task_by_uuid(task_uuid)
            if not task:
                return Response(message=f"Task not found: {task_uuid}", break_loop=False)

            if task.state == TaskState.RUNNING:
                await asyncio.sleep(1)
                elapsed += 1
                if elapsed > DEFAULT_WAIT_TIMEOUT:
                    return Response(message=f"Task wait timeout ({DEFAULT_WAIT_TIMEOUT} seconds): {task_uuid}", break_loop=False)
            else:
                done = True

        return Response(
            message=f"*Task*: {task_uuid}\n*State*: {task.state}\n*Last run*: {serialize_datetime(task.last_run)}\n*Result*:\n{task.last_result}",
            break_loop=False
        )



================================================
File: python/tools/search_engine.py
================================================
import os
import asyncio
from python.helpers import dotenv, memory, perplexity_search, duckduckgo_search
from python.helpers.tool import Tool, Response
from python.helpers.print_style import PrintStyle
from python.helpers.errors import handle_error
from python.helpers.searxng import search as searxng

SEARCH_ENGINE_RESULTS = 10


class SearchEngine(Tool):
    async def execute(self, query="", **kwargs):


        searxng_result = await self.searxng_search(query)

        await self.agent.handle_intervention(
            searxng_result
        )  # wait for intervention and handle it, if paused

        return Response(message=searxng_result, break_loop=False)


    async def searxng_search(self, question):
        results = await searxng(question)
        return self.format_result_searxng(results, "Search Engine")

    def format_result_searxng(self, result, source):
        if isinstance(result, Exception):
            handle_error(result)
            return f"{source} search failed: {str(result)}"

        outputs = []
        for item in result["results"]:
            outputs.append(f"{item['title']}\n{item['url']}\n{item['content']}")

        return "\n\n".join(outputs[:SEARCH_ENGINE_RESULTS]).strip()



================================================
File: python/tools/task_done.py
================================================
from python.helpers.tool import Tool, Response

class TaskDone(Tool):

    async def execute(self,**kwargs):
        self.agent.set_data("timeout", 0)
        return Response(message=self.args["text"], break_loop=True)

    async def before_execution(self, **kwargs):
        self.log = self.agent.context.log.log(type="response", heading=f"{self.agent.agent_name}: Task done", content=self.args.get("text", ""))
    
    async def after_execution(self, response, **kwargs):
        pass # do add anything to the history or output


================================================
File: python/tools/unknown.py
================================================
from python.helpers.tool import Tool, Response
from python.extensions.system_prompt._10_system_prompt import (
    get_tools_prompt,
)


class Unknown(Tool):
    async def execute(self, **kwargs):
        tools = get_tools_prompt(self.agent)
        return Response(
            message=self.agent.read_prompt(
                "fw.tool_not_found.md", tool_name=self.name, tools_prompt=tools
            ),
            break_loop=False,
        )



================================================
File: python/tools/vision_load.py
================================================
import base64
from python.helpers.print_style import PrintStyle
from python.helpers.tool import Tool, Response
from python.helpers import runtime, files, images
from mimetypes import guess_type
from python.helpers import history

# image optimization and token estimation for context window
MAX_PIXELS = 768_000
QUALITY = 75
TOKENS_ESTIMATE = 1500


class VisionLoad(Tool):
    async def execute(self, paths: list[str] = [], **kwargs) -> Response:

        self.images_dict = {}
        template: list[dict[str, str]] = []  # type: ignore

        for path in paths:
            if not await runtime.call_development_function(files.exists, str(path)):
                continue

            if path not in self.images_dict:
                mime_type, _ = guess_type(str(path))
                if mime_type and mime_type.startswith("image/"):
                    try:
                        # Read binary file
                        file_content = await runtime.call_development_function(
                            files.read_file_base64, str(path)
                        )
                        file_content = base64.b64decode(file_content)
                        # Compress and convert to JPEG
                        compressed = images.compress_image(
                            file_content, max_pixels=MAX_PIXELS, quality=QUALITY
                        )
                        # Encode as base64
                        file_content_b64 = base64.b64encode(compressed).decode("utf-8")

                        # DEBUG: Save compressed image
                        # await runtime.call_development_function(
                        #     files.write_file_base64, str(path), file_content_b64
                        # )

                        # Construct the data URL (always JPEG after compression)
                        self.images_dict[path] = file_content_b64
                    except Exception as e:
                        self.images_dict[path] = None
                        PrintStyle().error(f"Error processing image {path}: {e}")
                        self.agent.context.log.log("warning", f"Error processing image {path}: {e}")

        return Response(message="dummy", break_loop=False)

    async def after_execution(self, response: Response, **kwargs):

        # build image data messages for LLMs, or error message
        content = []
        if self.images_dict:
            for path, image in self.images_dict.items():
                if image:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                        }
                    )
                else:
                    content.append(
                        {
                            "type": "text",
                            "text": "Error processing image " + path,
                        }
                    )
            # append as raw message content for LLMs with vision tokens estimate
            msg = history.RawMessage(raw_content=content, preview="<Base64 encoded image data>")
            self.agent.hist_add_message(
                False, content=msg, tokens=TOKENS_ESTIMATE * len(content)
            )
        else:
            self.agent.hist_add_tool_result(self.name, "No images processed")

        # print and log short version
        message = (
            "No images processed"
            if not self.images_dict
            else f"{len(self.images_dict)} images processed"
        )
        PrintStyle(
            font_color="#1B4F72", background_color="white", padding=True, bold=True
        ).print(f"{self.agent.agent_name}: Response from tool '{self.name}'")
        PrintStyle(font_color="#85C1E9").print(message)
        self.log.update(result=message)



================================================
File: python/tools/webpage_content_tool.py
================================================
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from newspaper import Article
from python.helpers.tool import Tool, Response
from python.helpers.errors import handle_error


class WebpageContentTool(Tool):
    async def execute(self, url="", **kwargs):
        if not url:
            return Response(message="Error: No URL provided.", break_loop=False)

        try:
            # Validate URL
            parsed_url = urlparse(url)
            if not all([parsed_url.scheme, parsed_url.netloc]):
                return Response(message="Error: Invalid URL format.", break_loop=False)

            # Fetch webpage content
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            # Use newspaper3k for article extraction
            article = Article(url)
            article.download()
            article.parse()

            # If it's not an article, fall back to BeautifulSoup
            if not article.text:
                soup = BeautifulSoup(response.content, 'html.parser')
                text_content = ' '.join(soup.stripped_strings)
            else:
                text_content = article.text

            return Response(message=f"Webpage content:\n\n{text_content}", break_loop=False)

        except requests.RequestException as e:
            return Response(message=f"Error fetching webpage: {str(e)}", break_loop=False)
        except Exception as e:
            handle_error(e)
            return Response(message=f"An error occurred: {str(e)}", break_loop=False)







```

