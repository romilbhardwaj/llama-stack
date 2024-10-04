# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel, Field


@json_schema_type
class SkyPilotImplConfig(BaseModel):
    hf_token: str = Field(
        default="",
        description="Huggingface token, required to download models on the remote machine",
    )
    gpus: str = Field(
        default="L4:1",
        description="GPU type and GPU count to use for inference. E.g., L4:1, A100:8",
    )
