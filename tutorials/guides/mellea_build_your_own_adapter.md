# Build Your Own Adapter with Mellea

This guide explains how to configure your own adapter with Mellea to be used by Granite Switch model.

## Overview

Together, Mellea + Granite Switch + vLLM provide a production-ready inference stack for adapter-based AI applications that can utilize custom adapters.
- See [Mellea With Granite Switch](mellea_with_granite_switch.md) for a detailed explanation of how granite-switch and Mellea work together.
- See [Build Your Own Adapter](build_your_own_adapter.md) for info on how to train your own adapter.
- See Mellea's [Lora and aLoRA adapters](https://docs.mellea.ai/advanced/lora-and-alora-adapters) for info on how to train your own custom adapters using Mellea.

## Prerequisites

1. **A composed Granite Switch model** with the adapters you need
2. **A running vLLM server** serving that model

## Installation

```bash
pip install mellea
```

## Quick Example

By default, calling custom adapters requires utilizing lower level interfaces in Mellea:

```python
import json

from mellea.backends.model_options import ModelOption
from mellea.backends.openai import OpenAIBackend
from mellea.stdlib.context import ChatContext
from mellea.stdlib.components import Message, Intrinsic
import mellea.stdlib.functional as mfuncs

# 1. Initialize the Mellea Backend
backend = OpenAIBackend(
    model_id="path/to/your/granite-switch-model",  # Local files or Huggingface model id
    base_url="http://localhost:8000/v1",  # vLLM server
    api_key="unused",  # vLLM doesn't require auth by default
    load_embedded_adapters=True
)

# By default, load_embedded_adapters will autoload the adapters for the provided switch model.
# If you need to explicitly load adapters from another location (that are supported by the running vLLM
# server / model), you can use `backend.register_embedded_adapter_model` as shown in the "Mellea With Granite Switch"
# example.

# 2. Use Your Custom Adapter
# Your custom adapter likely requires certain inputs. Here, the example assumes a simple
# chat / conversation is enough.
context = ChatContext().add(Message("assistant", "Hello there, how can I help you?"))
action = Intrinsic("<your-custom-adapter-name>")

out, _ = mfuncs.act(
    action,
    context,
    backend,
    model_options={ModelOption.TEMPERATURE: 0.0},
    strategy=None,
)

# Adapter / Intrinsic processing in Mellea utilizes the io.yaml format forcing the output
# to be a json. See the "Build Your Own Adapter" linked example above.
result = json.loads(str(out))
print(result)
```

If you want to define helper functions so that your adapters operate similarly to the high-level
intrinsic wrapper functions in Mellea, you can do the following:

```python
# Continuing with imports / code from above.
from typing import Any

from mellea.backends.adapters import AdapterMixin
from mellea.stdlib.components.intrinsic._util import call_intrinsic
# Importing from a private API is fragile. You can imitate the same functionality below without it as well.

# Replace the return type with the actual type.
def your_custom_functionality(context: ChatContext, backend: AdapterMixin) -> Any:
    result_json = call_intrinsic("<your-custom-adapter-name>", context, backend)
    return result_json["<your-io-yaml-output-field(s)>"]
```
