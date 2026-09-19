"""Configuration layer.

- `settings`: every tunable on Settings, plus the four-layer resolution
  (code defaults -> platform.yaml -> platform.local.yaml -> environment).
- `seed`: the agent definitions archived in config/agents.yaml.

Nothing here imports from domain/ or infra/; configuration is a leaf.
"""
