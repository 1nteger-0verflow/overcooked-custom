# Testing Best Practices (chex + pytest)

## pytest General

- **Test layout**: Mirror `src/` structure under `tests/` (e.g., `src/foo.py` → `tests/test_foo.py`).
- **Fixtures**: Encapsulate common setup (models, PRNGKeys, environments) in `@pytest.fixture`.
- **Parametrize**: Use `@pytest.mark.parametrize` to cover multiple input patterns in a single test function.
- **Markers**: Define `@pytest.mark.slow` and `@pytest.mark.gpu` to categorize tests and enable selective CI execution.
- **`conftest.py`**: Collect project-wide fixtures and configuration in `tests/conftest.py`.

## chex

- **`chex.assert_shape`**: Always validate output tensor shapes.
- **`chex.assert_type`**: Validate dtype matches expected (e.g., `jnp.float32`).
- **`chex.assert_trees_all_close`**: Use for numerical PyTree comparison instead of `np.testing.assert_allclose`.
- **`chex.assert_trees_all_equal`**: Use for exact equality checks on deterministic outputs.
- **`chex.variants`**: Decorate tests with `@chex.variants(with_jit=True, without_jit=True)` to test both JIT and non-JIT paths.
- **`chex.assert_max_traces`**: Limit JIT retracing count to detect trace explosions.
- **`chex.assert_scalar`**: Verify scalar outputs (e.g., loss) are correctly reduced.

## JAX / RL Specific

- **Fixed PRNG**: Use `jax.random.PRNGKey(0)` in tests to guarantee reproducibility.
- **Disable JIT for debugging**: Wrap debug tests in `with jax.disable_jit():` for readable stack traces.
- **Separate shape and value tests**: Keep shape/type assertions in separate test functions from numerical correctness checks.
- **Gradient tests**: Assert that `jax.grad` outputs are not `None` and match expected shapes using `chex`.

## Commands & Project Config

- **Test directory**: `tests/` (configured via `testpaths = ["tests"]` in `pyproject.toml`).
- **Python path**: `src/` is automatically added to `sys.path` — no manual `sys.path` manipulation needed.
- **Run tests**: `uv run --frozen pytest` (`-v` is enabled by default).
