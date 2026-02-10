This directory contains **example environment configurations** and **task JSON files** used by ACuRL.

# Directory Structure

- `config_examples/`
  - `environment_config.json`: example environment connection configuration.
- `tasks/`
  - `examples/`: task definitions, organized by software/application.
  - `task_index/`: lightweight task indices defining task pools per software.

# Environment Configuration

- `config_examples/environment_config.json` is an example configuration.
- Modify it to match your deployment setup (e.g., environment server address).

# Task Definitions (`tasks/examples/`)

Tasks are organized by software, for example:

- `libreoffice_calc/`
- `thunderbird/`
- `Celestia/`

Each software directory may include:

- `environment_exploration.json`: exploration tasks used to collect broad environment experience.
- `context_review/`: tasks with different user contexts (diverse initial states).

# Task Indices (`tasks/task_index/`)

Each index file defines a **task pool** as a list of task IDs:

```
<SOFTWARE>.<TASK_NAME>
```

These indices control which tasks are sampled during training or evaluation.

# Adding or Updating Tasks

- **Add a task**: place a new JSON file under  
  `tasks/examples/<SOFTWARE>/context_review/`.
- **Enable it**: add the corresponding task ID to  
  `tasks/task_index/<SOFTWARE>.context_review.json`.

# Notes

Due to **copyright and privacy** considerations, we cannot open-source all contexts. But, we provide some examples for `libreoffice_impress`.

You may create additional contexts using publicly available resources by placing the resulting task files under:

```
tasks/examples/<SOFTWARE>/context_review/
```

and registering them in the corresponding task index.
