"""Infrastructure layer: everything that talks to the outside world.

Redis-backed stores (checkpoints, agent config, secrets, approval grants),
the DeepSeek HTTP client and the provider registry, and the AgentManager that
builds a domain AgentLoop from stored configuration.

Depends on domain/; is depended on by api/. The reverse would be a layering
violation.
"""
