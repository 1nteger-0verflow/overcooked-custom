import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from environment.actions import Actions, ActionType
from environment.agent import Agent
from environment.dynamic_object import DynamicObject
from environment.interaction import process_interact
from environment.layouts import Layout
from environment.pick_place import pick_and_place
from environment.reward import RewardType
from environment.state import Channel, State
from environment.static_object import StaticObject
from environment.update_env import update_step


def tree_select(predicate, a, b):
    return jax.tree_util.tree_map(lambda x, y: jax.lax.select(predicate, x, y), a, b)


class Processor:
    def __init__(self, config: DictConfig, layout: Layout):
        self.config = config
        self.layout = layout
        self.width = layout.width
        self.height = layout.height

    @jax.jit(static_argnums=(0,))
    def step(self, key: jax.Array, state: State, actions: jax.Array) -> tuple[State, float, jax.Array, RewardType]:
        key, interact_key = jax.random.split(key)
        # Move action:
        prev_agents = state.agents
        state = self.update_positions(state, actions)
        # interaction between agent and environment
        (state, reward, shaped_rewards, reward_type) = self.execute_interaction(
            state, actions, prev_agents, interact_key
        )
        # 時間経過による変化(客の状態遷移、汚れの発生、調理・食事の進行)
        state = update_step(state, self.config, key)
        # 前回実行した行動を記憶
        state = state.replace(prev_actions=actions)

        return (state, reward, shaped_rewards, reward_type)

    def update_positions(self, state: State, actions: jax.Array) -> State:
        # 1. move agent to new position (if possible on the grid)
        new_agents = self.move_agents(state, actions)
        # 2. resolve collisions
        new_agents = self.resolve_collisions(state, new_agents)
        # 3. prevent swapping
        new_agents = self.prevent_swapping(state, new_agents)
        # 4. update view area
        new_agents = new_agents.update_observed_grid(state.time, self.height, self.width)
        # ここまでの処理で更新後のエージェント位置は確定するので状態に反映させる
        return state.replace(agents=new_agents)

    def move_agents(self, state: State, actions: jax.Array) -> Agent:
        # 各エージェントを環境内で移動させる(エージェント間の衝突は考えない)
        grid = state.grid

        def _move_wrapper(agent: Agent, action: jax.Array):
            direction = Actions.action_to_direction(action)

            def _move(agent: Agent, dir: jnp.ndarray):
                new_pos = agent.move_in_bounds(dir, self.height, self.width)

                new_pos = tree_select(
                    (grid[*new_pos, Channel.env] == StaticObject.EMPTY)
                    & (grid[*new_pos, Channel.obj] & DynamicObject.DIRT == 0),
                    new_pos,
                    agent.pos,
                )

                return agent.replace(pos=new_pos, dir=dir)

            return jax.lax.cond(jnp.all(direction == jnp.array([0, 0])), lambda a, _: a, _move, agent, direction)

        new_agents = jax.vmap(_move_wrapper)(state.agents, actions)
        return new_agents

    def resolve_collisions(self, state: State, new_agents: Agent) -> Agent:
        # エージェント同士が衝突しないようにする
        def _masked_positions(agent_mask: jax.Array):
            return jax.vmap(lambda is_collide, cur_pos, new_pos: jax.lax.select(is_collide, cur_pos, new_pos))(
                agent_mask, state.agents.pos, new_agents.pos
            )

        def _get_collisions(agent_mask: jax.Array):
            positions = _masked_positions(agent_mask)

            # 移動後の各マス上のエージェント数>1なら衝突あり
            collision_grid = jnp.zeros((self.height, self.width))
            collision_grid, _ = jax.lax.scan(lambda grid, pos: (grid.at[*pos].add(1), None), collision_grid, positions)

            collision_mask = collision_grid > 1
            collisions = jax.vmap(lambda p: collision_mask[*p])(positions)
            return collisions

        initial_agent_mask = jnp.zeros((state.agents.num_agents,), dtype=bool)
        agent_mask = jax.lax.while_loop(
            # 衝突なしになったら終了
            lambda agent_mask: jnp.any(_get_collisions(agent_mask)),
            # エージェント若い順に移動キャンセルするかを決定
            lambda agent_mask: agent_mask | _get_collisions(agent_mask),
            initial_agent_mask,
        )
        new_agents = new_agents.replace(pos=_masked_positions(agent_mask))
        return new_agents

    def prevent_swapping(self, state: State, new_agents: Agent) -> Agent:
        # エージェントがすり抜けないようにする
        def _masked_positions(agent_mask: jax.Array):
            return jax.vmap(lambda is_collide, cur_pos, new_pos: jax.lax.select(is_collide, cur_pos, new_pos))(
                agent_mask, state.agents.pos, new_agents.pos
            )

        def _compute_swapped_agents(original_positions, new_positions):
            original_pos_expanded = jnp.expand_dims(original_positions, axis=0)
            new_pos_expanded = jnp.expand_dims(new_positions, axis=1)

            # 各エージェントの現在位置と移動後位置が重なるか比較
            swap_mask = (original_pos_expanded == new_pos_expanded).all(axis=-1)
            # 自身の移動前後が同じなのは不問(対角線をFalseにする)
            swap_mask = jnp.fill_diagonal(swap_mask, False, inplace=False)

            swap_pairs = jnp.logical_and(swap_mask, swap_mask.T)

            swapped_agents = jnp.any(swap_pairs, axis=0)
            return swapped_agents

        swap_mask = _compute_swapped_agents(state.agents.pos, new_agents.pos)
        new_agents = new_agents.replace(pos=_masked_positions(swap_mask))

        return new_agents

    def execute_interaction(
        self, state: State, actions: jax.Array, prev_agents: Agent, interact_key: jax.Array
    ) -> tuple[State, float, jax.Array, RewardType]:
        penalty = self.config.reward.penalty

        # Interact action:
        def _interact_wrapper(carry, x):
            agent, action, prev_agent, prev_action = x
            action_type, storage_idx = Actions.action_type(agent.modify_action(action))

            def _move(carry: tuple[State, float], agent: Agent):
                # prev_agentsの位置と向きを考慮し、ペナルティを与える
                moved = jnp.any(agent.pos != prev_agent.pos)
                turned = jnp.any(agent.dir != prev_agent.dir)
                blocked_move_eff = (~moved) & (~turned)  # "悪い壁押し"
                move_penalty = -penalty.step_cost - penalty.block_cost * blocked_move_eff
                return carry, (agent, move_penalty, RewardType.MOVE)

            def _nop(carry: tuple[State, float], agent: Agent):
                return carry, (agent, 0.0, RewardType.STAY)

            def _interact(carry: tuple[State, float], agent: Agent):
                state, reward = carry

                (new_state, new_agent, interact_reward, shaped_reward, reward_type) = process_interact(
                    state, agent, interact_key, self.config, self.layout
                )

                carry = (new_state, reward + interact_reward)
                return carry, (new_agent, shaped_reward, reward_type)

            def _pick_place(carry: tuple[State, float], agent: Agent):
                state, reward = carry

                (new_state, new_agent, interact_reward, shaped_reward, reward_type) = pick_and_place(
                    state, agent, interact_key, storage_idx, self.config
                )

                carry = (new_state, reward + interact_reward)
                return carry, (new_agent, shaped_reward, reward_type)

            actiontype_branches = action_type == jnp.array([e.value for e in ActionType])
            branch_idx = jnp.argmax(actiontype_branches)
            return jax.lax.switch(branch_idx, [_move, _nop, _interact, _pick_place], carry, agent)

        # エージェント1体ずつinteractionの処理
        carry = (state, 0.0)
        xs = (state.agents, actions, prev_agents, state.prev_actions)
        # returnの1つ目はcarryが全ループを経たもの、2つ目はループごとの出力のstack
        ((new_state, reward), (new_agents, shaped_rewards, reward_type)) = jax.lax.scan(_interact_wrapper, carry, xs)

        return (new_state.replace(agents=new_agents), reward, shaped_rewards, reward_type)
