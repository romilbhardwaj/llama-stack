# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import SkyPilotImplConfig

async def get_adapter_impl(config: SkyPilotImplConfig, _deps):
    from .skypilot import SkyPilotInferenceAdapter
    
    assert isinstance(config, SkyPilotImplConfig), f"Unexpected config type: {type(config)}"
    impl = SkyPilotInferenceAdapter(config)
    await impl.initialize()
    return impl