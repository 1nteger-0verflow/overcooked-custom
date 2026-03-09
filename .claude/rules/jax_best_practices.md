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
- **Multi-way Dispatch**: Use `jax.lax.switch` (not `jax.lax.cond`) for action-type or interaction-type dispatch with 3+ branches.
- **Loop Rollouts**: Prefer `jax.lax.scan` over Python `for` loops for rollout collection.
- **Conditional PyTree Update**: Use a `tree_select(cond, true_tree, false_tree)` helper (via `jax.tree.map` + `jnp.where`) for conditional updates on PyTree state instead of manual field-by-field `jnp.where`.

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
- **`nn.scan` for RNN unroll**: Use `flax.linen.scan` (not raw `jax.lax.scan`) to unroll RNN cells over sequences so that Flax variable scoping is handled correctly.

## Environment State Design

- **`struct.PyTreeNode` for state classes**: All state structs (`Agent`, `Customer`, `State`, etc.) must subclass `flax.struct.PyTreeNode`. Do not use plain `@dataclass` for JAX-traced state — it will not register as a PyTree automatically.
- **`chex.dataclass` for transitions**: Trajectory data (`Transition`, etc.) should use `chex.dataclass` instead of `collections.namedtuple`. It auto-registers as a PyTree and provides type safety.
- **Multi-channel observation layering**: Observations are stacked channel-wise (agents, static objects, customers, timers, context). When adding new information, add a new channel rather than modifying existing channels.
- **Partial observability via view box**: Agent field-of-view is computed from position and direction. Always use `agent.get_view_box()` to slice the global grid rather than passing the full state.

## Training Patterns

- **Multi-seed parallelism via `vmap`**: Parallel training across seeds uses `jax.vmap` over the seed axis (single-device). Reserve `pmap` for multi-GPU scenarios.
- **Learning rate and entropy annealing**: Use `optax.linear_schedule` for both learning rate decay and entropy coefficient annealing. Define schedules in the config dataclass.
- **Checkpointing with Orbax**: Save and restore model parameters using `orbax.checkpoint`. Never use `pickle` or `np.save` for model weights.

## Visualization

- **State → RGB as pure function**: Rendering logic in `visualize/` converts JAX state to NumPy-based RGB arrays. Keep rendering functions side-effect-free; do not call JAX-compiled functions from within renderers.
- **Action log format**: Save replay logs in NPZ format (`np.savez`) with keys `actions` and `rng_key` so that `ReplayLog` controller can load and replay them.
