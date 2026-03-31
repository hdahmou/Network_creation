Based on a previous frankenstein monster of a package: https://github.com/hdahmou/linking_game

The objective is to have a smaller/faster implementation by sacrificing the scope.

Repository layout:

- `network_creation/`: reusable simulation and analysis code
- `tasks/`: orchestration layers for distributed sweep jobs
- `configs/`: checked-in task presets and sampling specs
- `artifacts/`: generated run outputs, summaries, and copied dispatch inputs
