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
- `src/ippo_rnn.py`: Main training entry point.
- `tests/`: Test files mirroring `src/` structure.

## uv & Tooling

- **Script execution**: Always use `uv run <script>` instead of `python` directly.
- **Add dependency**: Use `uv add <package>` (never `pip install`).
- **Add dev dependency**: Use `uv add --dev <package>`.
- **Sync environment**: Use `uv sync`.
- **Lint & fix**: `uv run --frozen ruff check --fix .`
- **Format**: `uv run --frozen ruff format .` (do not use `black`).
- **Run tests**: `uv run --frozen pytest`

## Ruff Rules (from pyproject.toml)

- **Line length**: 120 characters max.
- **Docstring style**: Google convention.
- **Max function arguments**: 10.
- **Max cyclomatic complexity**: 10.
- **`assert` is allowed**: `S101` is suppressed — use `assert` freely in tests and checks.
- **Full-width characters are allowed**: `RUF001-003` suppressed — Japanese/full-width characters OK in comments and strings.
- **Return type annotations are optional**: `ANN2` suppressed — omitting return types is acceptable.
- **Docstrings are optional**: `D1` suppressed — omitting docstrings is acceptable.
