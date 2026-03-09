# Agents Configuration

## Researcher Agent (Model Architecture)
- **Focus**: Flax Linen architecture and IPPO loss functions.
- **Goal**: Implement numerically stable Advantage estimation (GAE) and clipping.
- **Constraint**: Ensure `vmap` compatibility for multi-agent (IPPO) dimensions.

## Optimizer Agent (Performance)
- **Focus**: JAX performance and Memory profiling.
- **Task**: Check for unnecessary array copies or `DeviceArray` to `numpy` conversions that break JIT.
- **Instruction**: When performance is slow, suggest `jax.profiler` or memory-efficient scan/map patterns.
