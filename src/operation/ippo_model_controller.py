from collections import abc
from pathlib import Path

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from omegaconf import DictConfig

from ippo_rnn import ActorCriticRNN, ScannedRNN
from operation.agent_controller import AgentController
from utils.schema import IppoConfig


class IPPOModelInput(AgentController):
    def __init__(self, agent_id: int, model_config: abc.Mapping, num_actions: int):
        self.agent_id = agent_id
        self.num_actions = num_actions
        chkpt_dir = Path(model_config.get("checkpoint"))
        step = model_config.get("step")
        # load checkpoint
        options = ocp.CheckpointManagerOptions()
        mngr = ocp.CheckpointManager(chkpt_dir.resolve(), options=options, item_names=["obs_shape", "params"])
        config: IppoConfig = DictConfig(mngr.metadata().custom_metadata)
        self.network = ActorCriticRNN(num_actions, config=config)
        # load obs shape
        abs_obs_shape = (0, 0, 0, 0)  # (num_agent, height, width, layer)
        restored_shape = mngr.restore(step, args=ocp.args.Composite(obs_shape=ocp.args.StandardRestore(abs_obs_shape)))
        obs_shape = restored_shape["obs_shape"]
        # load parameters
        num_actor = 1
        init_x = (jnp.zeros((1, num_actor, *obs_shape[1:])), jnp.zeros((1, num_actor)))
        hstate = jnp.zeros((1, config["GRU_HIDDEN_DIM"]))
        abs_params = self.network.init(jax.random.key(0), hstate, init_x)
        restored = mngr.restore(step, args=ocp.args.Composite(params=ocp.args.StandardRestore(abs_params)))
        self.params = restored["params"]
        self.hstate = ScannedRNN.initialize_carry(1, config["GRU_HIDDEN_DIM"])

    @jax.jit(static_argnums=(0,))
    def sample_action(self, obs):
        obs_batch = obs[jnp.newaxis, :]
        ac_in = (obs_batch[jnp.newaxis, :], jnp.array([[0]]))
        self.hstate, pi, val = self.network.apply(self.params, self.hstate[jnp.newaxis, :], ac_in)
        log_probs = jnp.array([pi.log_prob(i) for i in range(self.num_actions)])
        probs = jnp.exp(log_probs)
        jax.debug.print(
            "IPPO model for agent{}: {}\ncritic value = {}", self.agent_id, jnp.rint(probs.squeeze() * 100), val
        )
        return jnp.argmax(log_probs)

    def input_observation(self, obs):
        self.next_action = self.sample_action(obs["agents"][self.agent_id])
        jax.debug.print("selected action : {}", self.next_action)

    def get_action(self):
        return self.next_action
