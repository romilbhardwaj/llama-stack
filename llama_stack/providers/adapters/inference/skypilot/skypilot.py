# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import os
import subprocess
from typing import AsyncGenerator, List, Optional

from openai import OpenAI

from llama_stack.apis.inference import *
from llama_stack.providers.utils.inference.augment_messages import augment_messages_for_tools
from llama_stack.providers.utils.inference.routable import RoutableProviderForModels

from llama_models.llama3.api.chat_format import ChatFormat
from llama_models.llama3.api.datatypes import Message, StopReason
from llama_models.llama3.api.tokenizer import Tokenizer

from .config import SkyPilotImplConfig

SKYPILOT_SUPPORTED_MODELS = {
    "Meta-Llama3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct"
}

yaml_dir = os.path.dirname(__file__) + "/yamls"
print("Using YAML directory: ", yaml_dir)

SKYPILOT_YAML_MAP = {
    "Meta-Llama3.1-8B-Instruct": os.path.join(yaml_dir, "llama-3_1.yaml")
}

CLUSTER_NAME = "llama"

class SkyPilotInferenceAdapter(Inference, RoutableProviderForModels):
    """
    Inference adapter that uses SkyPilot to run inference.

    Uses vLLM under the hood to run inference on a remote Sky Cluster.
    """
    def __init__(self, config: SkyPilotImplConfig):
        RoutableProviderForModels.__init__(
            self, stack_to_provider_models_map=SKYPILOT_SUPPORTED_MODELS
        )
        self.config = config
        self.process = None
        self.endpoint_url = None    # Fetched dynamically from SkyPilot
        self.is_cluster_running = False
        tokenizer = Tokenizer.get_instance()
        self.formatter = ChatFormat(tokenizer)

    async def initialize(self) -> None:
        pass

    @property
    def client(self) -> OpenAI:
        return OpenAI(
            base_url=self.endpoint_url
        )
    
    async def shutdown(self) -> None:
        await self.down_cluster()

    async def down_cluster(self):
        """Terminates the SkyPilot cluster."""
        try:
            cmd = f"sky down -y {CLUSTER_NAME}"
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                raise RuntimeError(f"Cluster termination failed: {stderr.decode()}")
            
            print(f"Cluster {CLUSTER_NAME} terminated successfully.")
        except Exception as e:
            print(f"An error occurred while terminating the cluster: {str(e)}")

    async def setup_cluster(self, config: SkyPilotImplConfig):
        if not self.is_cluster_running:
            self.launch_cluster(config)
            self.is_cluster_running = True
        if self.endpoint_url is None:
            self.endpoint_url = self.get_endpoint_url()

    async def launch_cluster(self, config: SkyPilotImplConfig):
        # TODO: Remove this hardcoded model:
        model = "Llama3.1-8B-Instruct"
        yaml_file = SKYPILOT_YAML_MAP.get(model)
        if not yaml_file:
            raise ValueError(f"No YAML file defined for model: {model}")

        # TODO: Remove hardcoding of cluster name
        cmd = f"sky launch -c {CLUSTER_NAME} --env HF_TOKEN={self.config.hf_token} -y {yaml_file}"
        self.process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        try:
            stdout, stderr = await asyncio.wait_for(self.process.communicate(), timeout=self.config.timeout)
        except asyncio.TimeoutError:
            self.process.terminate()
            raise TimeoutError(f"Cluster launch timed out after {self.config.timeout} seconds")

        if self.process.returncode != 0:
            raise RuntimeError(f"Cluster launch failed: {stderr.decode()}")

    async def get_endpoint_url(self) -> str:
        # Use `sky status --endpoint` to get the endpoint URL
        try:
            process = await asyncio.create_subprocess_exec(
                'sky', 'status', '--endpoint', '8000', CLUSTER_NAME,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, 'sky status', stderr.decode())
            
            endpoint_url = stdout.decode().strip()
            if not endpoint_url:
                raise ValueError("Empty endpoint URL returned")
            return endpoint_url
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to get endpoint URL: {e.stderr}")
        except Exception as e:
            raise RuntimeError(f"An error occurred while getting endpoint URL: {str(e)}")


    def _messages_to_vllm_messages(self, messages: list[Message]) -> list:
        vllm_messages = []
        for message in messages:
            if message.role == "ipython":
                role = "tool"
            else:
                role = message.role
            vllm_messages.append({"role": role, "content": message.content})

        return vllm_messages

    def resolve_vllm_model(self, model_name: str) -> str:
        model = augment_messages_for_tools(model_name)
        assert (
            model is not None
            and model.descriptor(shorten_default_variant=True)
            in SKYPILOT_SUPPORTED_MODELS
        ), f"Unsupported model: {model_name}, use one of the supported models: {','.join(SKYPILOT_SUPPORTED_MODELS.keys())}"

        return SKYPILOT_SUPPORTED_MODELS.get(
            model.descriptor(shorten_default_variant=True)
        )

    async def chat_completion(
        self,
        model: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = ToolPromptFormat.json,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
        # wrapper request to make it easier to pass around (internal only, not exposed to API)
        request = ChatCompletionRequest(
            model=model,
            messages=messages,
            sampling_params=sampling_params,
            tools=tools or [],
            tool_choice=tool_choice,
            tool_prompt_format=tool_prompt_format,
            stream=stream,
            logprobs=logprobs,
        )

        vllm_model = self.resolve_vllm_model(request.model)
        messages = augment_messages_for_tools(request)
        model_input = self.formatter.encode_dialog_prompt(messages)

        input_tokens = len(model_input.tokens)
        max_new_tokens = min(
            request.sampling_params.max_tokens or (self.max_tokens - input_tokens),
            self.max_tokens - input_tokens - 1,
        )

        print(f"Calculated max_new_tokens: {max_new_tokens}")

        assert (
            request.model == self.model_name
        ), f"Model mismatch, expected {self.model_name}, got {request.model}"

        if not request.stream:
            r = self.client.chat.completions.create(
                model=vllm_model,
                messages=self._messages_to_vllm_messages(messages),
                max_tokens=max_new_tokens,
                stream=False
            )
            stop_reason = None
            if r.choices[0].finish_reason:
                if (
                    r.choices[0].finish_reason == "stop"
                    or r.choices[0].finish_reason == "eos"
                ):
                    stop_reason = StopReason.end_of_turn
                elif r.choices[0].finish_reason == "length":
                    stop_reason = StopReason.out_of_tokens

            completion_message = self.formatter.decode_assistant_message_from_content(
                r.choices[0].message.content, stop_reason
            )
            yield ChatCompletionResponse(
                completion_message=completion_message,
                logprobs=None,
            )
        else:
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=ChatCompletionResponseEventType.start,
                    delta="",
                )
            )

            buffer = ""
            ipython = False
            stop_reason = None

            for chunk in self.client.chat.completions.create(
                model=vllm_model,
                messages=self._messages_to_vllm_messages(messages),
                max_tokens=max_new_tokens,
                stream=True
            ):
                if chunk.choices[0].finish_reason:
                    if (
                        stop_reason is None and chunk.choices[0].finish_reason == "stop"
                    ) or (
                        stop_reason is None and chunk.choices[0].finish_reason == "eos"
                    ):
                        stop_reason = StopReason.end_of_turn
                    elif (
                        stop_reason is None
                        and chunk.choices[0].finish_reason == "length"
                    ):
                        stop_reason = StopReason.out_of_tokens
                    break

                text = chunk.choices[0].message.content
                if text is None:
                    continue

                # check if it's a tool call ( aka starts with <|python_tag|> )
                if not ipython and text.startswith("<|python_tag|>"):
                    ipython = True
                    yield ChatCompletionResponseStreamChunk(
                        event=ChatCompletionResponseEvent(
                            event_type=ChatCompletionResponseEventType.progress,
                            delta=ToolCallDelta(
                                content="",
                                parse_status=ToolCallParseStatus.started,
                            ),
                        )
                    )
                    buffer += text
                    continue

                if ipython:
                    if text == "<|eot_id|>":
                        stop_reason = StopReason.end_of_turn
                        text = ""
                        continue
                    elif text == "<|eom_id|>":
                        stop_reason = StopReason.end_of_message
                        text = ""
                        continue

                    buffer += text
                    delta = ToolCallDelta(
                        content=text,
                        parse_status=ToolCallParseStatus.in_progress,
                    )

                    yield ChatCompletionResponseStreamChunk(
                        event=ChatCompletionResponseEvent(
                            event_type=ChatCompletionResponseEventType.progress,
                            delta=delta,
                            stop_reason=stop_reason,
                        )
                    )
                else:
                    buffer += text
                    yield ChatCompletionResponseStreamChunk(
                        event=ChatCompletionResponseEvent(
                            event_type=ChatCompletionResponseEventType.progress,
                            delta=text,
                            stop_reason=stop_reason,
                        )
                    )

            # parse tool calls and report errors
            message = self.formatter.decode_assistant_message_from_content(
                buffer, stop_reason
            )
            parsed_tool_calls = len(message.tool_calls) > 0
            if ipython and not parsed_tool_calls:
                yield ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=ChatCompletionResponseEventType.progress,
                        delta=ToolCallDelta(
                            content="",
                            parse_status=ToolCallParseStatus.failure,
                        ),
                        stop_reason=stop_reason,
                    )
                )

            for tool_call in message.tool_calls:
                yield ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=ChatCompletionResponseEventType.progress,
                        delta=ToolCallDelta(
                            content=tool_call,
                            parse_status=ToolCallParseStatus.success,
                        ),
                        stop_reason=stop_reason,
                    )
                )

            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=ChatCompletionResponseEventType.complete,
                    delta="",
                    stop_reason=stop_reason,
                )
            )
