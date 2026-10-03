# Summit documentation

Start with the root [README](../README.md) for installation and a no-spend check.
Agents should first read [AGENTS.md](../AGENTS.md).

| Task | Guide |
| --- | --- |
| Run the three workflows and evaluate outputs | [Recipes](recipes.md) |
| Prepare canonical inputs and teacher targets | [Data](data.md) |
| Understand checks and Phantora | [Validation](validation.md) |
| Handle credentials, GPUs, cost limits and HF retention | [Operations](operations.md) |
| Add an architecture or develop Summit | [Extending](extending.md) |
| Audit and package a release | [Releasing](releasing.md) |

Only the three YAMLs linked by the README are release entry points. Other local
YAMLs and historical reports are not required by these docs and are not shipped.
Maintained test fixtures exercise compatibility without becoming public recipes.
