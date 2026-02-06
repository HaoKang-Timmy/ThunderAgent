"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation."""

import concurrent.futures
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

from jinja2 import StrictUndefined, Template

from minisweagent import Environment, Model


@dataclass
class AgentConfig:
    # The default settings are the bare minimum to run the agent. Take a look at the config files for improved settings.
    system_template: str = "You are a helpful assistant that can do anything."
    instance_template: str = (
        "Your task: {{task}}. Please reply with a single shell command in triple backticks. "
        "To finish, the first line of the output of the shell command must be 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'."
    )
    timeout_template: str = (
        "The last command <command>{{action['action']}}</command> timed out and has been killed.\n"
        "The output of the command was:\n <output>\n{{output}}\n</output>\n"
        "Please try another command and make sure to avoid those requiring interactive input."
    )
    format_error_template: str = "Please always provide EXACTLY ONE action in triple backticks."
    action_observation_template: str = "Observation: {{output}}"
    step_limit: int = 0
    cost_limit: float = 3.0
    job_timeout: float = 0  


class NonTerminatingException(Exception):
    """Raised for conditions that can be handled by the agent."""


class FormatError(NonTerminatingException):
    """Raised when the LM's output is not in the expected format."""


class ExecutionTimeoutError(NonTerminatingException):
    """Raised when the action execution timed out."""


class TerminatingException(Exception):
    """Raised for conditions that terminate the agent."""


class Submitted(TerminatingException):
    """Raised when the LM declares that the agent has finished its task."""


class LimitsExceeded(TerminatingException):
    """Raised when the agent has reached its cost or step limit."""


class ContextLengthExceeded(TerminatingException):
    """Raised when the context length exceeds the model's maximum."""


class JobTimeoutError(TerminatingException):
    """Raised when the job execution exceeds the maximum allowed time."""


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: Callable = AgentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.job_start_time: float | None = None
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self.step_timings: list[dict] = []
        self._active_step_timing: dict | None = None

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = asdict(self.config) | self.env.get_template_vars() | self.model.get_template_vars()
        return Template(template, undefined=StrictUndefined).render(
            **kwargs, **template_vars, **self.extra_template_vars
        )

    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append({"role": role, "content": content, **kwargs})

    def _get_remaining_time(self) -> float | None:
        """Get remaining time for job timeout, or None if no timeout configured."""
        if self.config.job_timeout <= 0 or self.job_start_time is None:
            return None
        elapsed = time.time() - self.job_start_time
        remaining = self.config.job_timeout - elapsed
        return max(0, remaining)

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run step() until agent is finished. Return exit status & message"""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.step_timings = []
        self._active_step_timing = None
        self.job_start_time = time.time()

        # Create thread pool executor for async timeout handling
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        try:
            self.add_message("system", self.render_template(self.config.system_template))
            self.add_message("user", self.render_template(self.config.instance_template))
            while True:
                try:
                    self.step()
                except NonTerminatingException as e:
                    self.add_message("user", str(e))
                except TerminatingException as e:
                    self.add_message("user", str(e))
                    return type(e).__name__, str(e)
        finally:
            # Shutdown executor when run completes or exits
            if self._executor is not None:
                self._executor.shutdown(wait=False)

    def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        step_timing: dict = {
            "step_index": len(self.step_timings) + 1,
            "query": {},
            "tool": {},
        }
        self.step_timings.append(step_timing)
        self._active_step_timing = step_timing
        try:
            response = self.query()
            return self.get_observation(response)
        finally:
            # Ensure we never accidentally write the next step's timings into this one.
            self._active_step_timing = None

    def _query_model(self):
        """Internal method to query the model (runs in thread pool)."""
        # Prefer a streaming+timed query if the model provides it.
        query_stream_timed = getattr(self.model, "query_stream_timed", None)
        if callable(query_stream_timed):
            return query_stream_timed(self.messages)
        return self.model.query(self.messages)

    def query(self) -> dict:
        """Query the model and return the response."""
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded()

        query_start_ts = time.time()
        if self._active_step_timing is not None:
            self._active_step_timing["query"].update({
                "start_ts": query_start_ts,
            })

        # Get remaining time for timeout
        remaining_time = self._get_remaining_time()

        # Submit query to thread pool for async execution with timeout
        if self._executor is not None and remaining_time is not None:
            future = self._executor.submit(self._query_model)
            try:
                response = future.result(timeout=remaining_time)
            except concurrent.futures.TimeoutError:
                elapsed_time = time.time() - self.job_start_time if self.job_start_time else 0
                raise JobTimeoutError(
                    f"Job execution exceeded the maximum allowed time of {self.config.job_timeout:.0f} seconds "
                    f"({self.config.job_timeout/60:.1f} minutes). Elapsed time: {elapsed_time:.1f} seconds. "
                    f"Timeout occurred during model query."
                )
            except Exception as e:
                # Check if this is a ContextLengthExceededError from vLLM model
                if type(e).__name__ == "ContextLengthExceededError":
                    raise ContextLengthExceeded(str(e)) from e
                # Re-raise other exceptions
                raise
        else:
            # No timeout configured, execute synchronously
            try:
                response = self._query_model()
            except Exception as e:
                # Check if this is a ContextLengthExceededError from vLLM model
                if type(e).__name__ == "ContextLengthExceededError":
                    raise ContextLengthExceeded(str(e)) from e
                # Re-raise other exceptions
                raise

        # If model returned streaming timing info, normalize it.
        query_first_token_ts: float | None = None
        query_last_token_ts: float | None = None

        if isinstance(response, tuple) and len(response) == 2 and isinstance(response[0], dict) and isinstance(response[1], dict):
            response, timing = response
            query_first_token_ts = timing.get("first_token_ts")
            query_last_token_ts = timing.get("last_token_ts")

        if query_last_token_ts is None:
            query_last_token_ts = time.time()
        if query_first_token_ts is None:
            query_first_token_ts = query_last_token_ts

        if self._active_step_timing is not None:
            self._active_step_timing["query"].update({
                "first_token_ts": query_first_token_ts,
                "last_token_ts": query_last_token_ts,
                "prefill_s": max(0.0, query_first_token_ts - query_start_ts),
                "decode_s": max(0.0, query_last_token_ts - query_first_token_ts),
                "total_s": max(0.0, query_last_token_ts - query_start_ts),
            })

        self.add_message("assistant", **response)
        return response

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation."""
        output = self.execute_action(self.parse_action(response))
        observation = self.render_template(self.config.action_observation_template, output=output)
        self.add_message("user", observation)
        return output

    def parse_action(self, response: dict) -> dict:
        """Parse the action from the message. Returns the action."""
        actions = re.findall(r"```bash\s*\n(.*?)\n```", response["content"], re.DOTALL)
        if len(actions) == 1:
            return {"action": actions[0].strip(), **response}
        raise FormatError(self.render_template(self.config.format_error_template, actions=actions))

    def execute_action(self, action: dict) -> dict:
        tool_start_ts = time.time()
        if self._active_step_timing is not None:
            self._active_step_timing["tool"].update({
                "start_ts": tool_start_ts,
                "action": action.get("action", ""),
            })
        try:
            output = self.env.execute(action["action"])
        except subprocess.TimeoutExpired as e:
            tool_end_ts = time.time()
            if self._active_step_timing is not None:
                self._active_step_timing["tool"].update({
                    "end_ts": tool_end_ts,
                    "total_s": max(0.0, tool_end_ts - tool_start_ts),
                    "exception": "TimeoutExpired",
                })
            output_text = e.output.decode("utf-8", errors="replace") if e.output else ""
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output=output_text)
            )
        except TimeoutError:
            tool_end_ts = time.time()
            if self._active_step_timing is not None:
                self._active_step_timing["tool"].update({
                    "end_ts": tool_end_ts,
                    "total_s": max(0.0, tool_end_ts - tool_start_ts),
                    "exception": "TimeoutError",
                })
            raise ExecutionTimeoutError(self.render_template(self.config.timeout_template, action=action, output=""))
        else:
            tool_end_ts = time.time()
            if self._active_step_timing is not None:
                self._active_step_timing["tool"].update({
                    "end_ts": tool_end_ts,
                    "total_s": max(0.0, tool_end_ts - tool_start_ts),
                    "returncode": output.get("returncode"),
                })
        self.has_finished(output)
        return output

    def has_finished(self, output: dict[str, str]):
        """Raises Submitted exception with final output if the agent has finished its task."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() in ["MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]:
            raise Submitted("".join(lines[1:]))
