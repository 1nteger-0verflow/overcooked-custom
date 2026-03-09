# Project: JAX-Flax Reinforcement Learning (IPPO)

## Environment
- Use `uv` for all dependencies. `uv run <script>` is the default.
- GPU acceleration (CUDA) is assumed. Check JAX device count before pmap.

## Core Rules (Strict)
- **Functional Purity**: All transition functions and update steps MUST be pure functions. No global state.
- **JIT Guidelines**: Always use `@jax.jit` for `update` and `act` functions. Avoid JIT-ing functions with side effects.
- **Type Hinting**: Use `jaxtyping` and `beartype` for tensor shapes. E.g., `Float[Array, "batch obs_dim"]`.
- **PRNG Management**: Never reuse `jax.random.PRNGKey`. Always use `jax.random.split` and pass keys explicitly.

## Python General
- **Dataclasses**: Group configs and hyperparameters into `@dataclass` or `chex.dataclass`. Never use plain dicts for structured config.
- **Type Annotations**: All function signatures must have type hints. Use `beartype` for runtime validation.
- **No Magic Numbers**: Never use unnamed constants inline. Define named constants or collect them in a config dataclass.
- **Import Order**: Use `ruff` + `isort` for import formatting (`uv run ruff check --fix`).
- **Explicit Errors**: Raise explicit exceptions for RL-specific errors (shape mismatches, device mismatches) rather than letting them propagate silently.

## Structure
<!-- - `models/`: Flax Linen modules. -->
- `src/environment/`: JAX-native or Gymnax-based environments.
- `ippo_rnn.py`: Main training entry point using `uv run`.
