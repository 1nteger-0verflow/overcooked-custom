# JAX/Flax Reinforcement Learning Rules

- **Shape Validation**: Always specify expected shapes in docstrings or `jaxtyping`.
- **IPPO Specifics**:
    - Ensure centralized value functions or local-observation-only policies as per IPPO spec.
    - Standardize advantages across the agent dimension.
- **NaN Prevention**:
    - Use `jax.nn.softplus` or epsilon offsets for log-probs.
    - Check for `inf` in gradients during update steps.
- **Flax State**: Use `TrainState` pattern to bundle params, opt_state, and steps.

## JAX

### Function Design
- **Static Arguments**: Explicitly declare Python control-flow values with `static_argnums` / `static_argnames` in JIT.
- **No `print` in JIT**: Use `jax.debug.print` for in-JIT debug output. Never use Python `print` inside JIT-ted functions.
- **Conditional Branching**: Use `jax.lax.cond` instead of `if/else` inside JIT.
- **Loop Rollouts**: Prefer `jax.lax.scan` over Python `for` loops for rollout collection.

### Device & Parallelism
- **`pmap` axis name**: Always specify `axis_name` (e.g., `pmap(f, axis_name="devices")`).
- **Device count check**: Call `jax.device_count()` before using `pmap` to verify GPU count.
- **PyTree operations**: Use `jax.tree.map` instead of manual loops over nested structures.

### Debugging & Numerical Stability
- **NaN callbacks**: Use `jax.debug.callback` to move NaN checks outside JIT.
- **`chex` assertions**: Use `chex.assert_shape`, `chex.assert_type`, etc. for shape and type assertions.
- **Gradient clipping**: Always clip gradients with `optax.clip_by_global_norm` to protect against exploding gradients.

## Flax Linen

### Module Design
- **Consistent init style**: Do not mix `setup()` and `@nn.compact` within the same module.
- **Explicit `dtype`**: Always specify `dtype` explicitly (e.g., `nn.Dense(features, dtype=jnp.float32)`).
- **`param_dtype` for mixed precision**: Set `param_dtype=jnp.float32` to maintain precision when using mixed-precision training.

### State Management
- **Extend `TrainState`**: Manage custom state (e.g., RNN hidden states) by subclassing `TrainState`.
- **Separate `variables`**: Clearly separate `params` and `batch_stats` when calling `model.apply`.

### RNN (IPPO with RNN)
- **`initialize_carry`**: Use `cell.initialize_carry` to initialize hidden states.
- **Sequence dimension convention**: Standardize on either `(time, batch, feature)` or `(batch, time, feature)` and document the choice.
- **Episode boundary reset**: Explicitly implement carry reset logic using `done` flags at episode boundaries.
