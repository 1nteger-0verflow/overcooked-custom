from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax
from omegaconf import DictConfig

from environment.actions import Actions
from environment.layouts import Layout
from environment.menus import MenuList
from environment.observation import Observer
from environment.process import Processor
from environment.reset_env import Initializer
from environment.reward import RewardType
from environment.state import State


class OvercookedCustom:
    def __init__(
        self,
        config: DictConfig,
        random_agent_position: bool = False,
    ):
        self.config = config
        self.layout = Layout.from_string(grid=config.layout)
        self.menu = MenuList.load(menus=config.menu)
        self.num_agents = self.layout.num_agents
        self.layout = self.layout
        self.height = self.layout.height
        self.width = self.layout.width

        self.initializer = Initializer(
            config, self.layout, self.menu, random_agent_position
        )
        self.processor = Processor(config, self.layout)
        self.observer = Observer(config, self.layout)

        self.action_set = Actions.declare_action_set(config.parameter.capacity)
        self.max_steps = int(config.schedule.terminal_time)

    def reset(self, key: jax.Array) -> Tuple[jax.Array, State]:
        state = self.initializer.initialize(key)
        obs, _ = self.observer.get_obs(state)

        return lax.stop_gradient(obs), lax.stop_gradient(state)

    @jax.jit(static_argnums=(0,))
    def step_env(
        self,
        state: State,
        actions: jax.Array,
        key: jax.Array,
    ) -> Tuple[jax.Array, State, jax.Array, jax.Array, RewardType, jax.Array]:
        """Perform single timestep state transition."""

        state, reward, shaped_rewards_value, reward_type = self.processor.step(
            key, state, actions
        )
        state, done = self.update_timestep(state)
        obs, _ = self.observer.get_obs(state)

        rewards = jnp.array([reward for _ in range(self.num_agents)])
        shaped_rewards = jnp.array(
            [shaped_reward for shaped_reward in shaped_rewards_value]
        )

        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(state),
            rewards,
            shaped_rewards,
            reward_type,
            done,
        )

    def update_timestep(self, state: State) -> Tuple[State, jax.Array]:
        state = state.replace(time=state.time + 1)
        is_done = state.time >= self.max_steps
        return state, is_done

    @property
    def name(self) -> str:
        """Environment name."""
        return "Overcooked custom"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(self.action_set)

    @property
    def obs_shape(self) -> Tuple[int, int, int, int]:
        return self.observer.obs_shape
