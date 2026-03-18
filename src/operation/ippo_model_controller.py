import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from config import IPPOModelConfig, NetworkConfig
from ippo_rnn import ActorCriticRNN, ScannedRNN
from operation.agent_controller import AgentController


class IPPOModelInput(AgentController):
    def __init__(self, agent_id: int, model_config: IPPOModelConfig, num_actions: int, *, verbose: bool):
        self.agent_id = agent_id
        self.num_actions = num_actions
        self.verbose = verbose
        chkpt_dir = Path(model_config.checkpoint)
        step = model_config.step
        # load checkpoint
        options = ocp.CheckpointManagerOptions()
        mngr = ocp.CheckpointManager(chkpt_dir.resolve(), options=options, item_names=["obs_shape", "params"])
        metadata = mngr.metadata().custom_metadata or {}
        valid_fields = {f.name for f in dataclasses.fields(NetworkConfig)}
        network_config = NetworkConfig(**{k: v for k, v in metadata.items() if k in valid_fields})
        self.network = ActorCriticRNN(num_actions, config=network_config)
        # load obs shape
        abs_obs_shape = (0, 0, 0, 0)  # (num_agent, height, width, layer)
        restored_shape = mngr.restore(step, args=ocp.args.Composite(obs_shape=ocp.args.StandardRestore(abs_obs_shape)))
        obs_shape = restored_shape["obs_shape"]
        # load parameters
        num_actor = 1
        init_x = (jnp.zeros((1, num_actor, *obs_shape[1:])), jnp.zeros((1, num_actor)))
        hstate = jnp.zeros((1, network_config.GRU_HIDDEN_DIM))
        abs_params = self.network.init(jax.random.key(0), hstate, init_x)
        restored = mngr.restore(step, args=ocp.args.Composite(params=ocp.args.StandardRestore(abs_params)))
        self.params = restored["params"]
        self.hidden_dim = network_config.GRU_HIDDEN_DIM
        self.hstate = ScannedRNN.initialize_carry(1, self.hidden_dim)
        print(f"[Agent {agent_id}] IPPO agent model is loaded from {chkpt_dir}/{step}.")

    @jax.jit(static_argnums=(0,))
    def sample_action(self, obs: jnp.ndarray, hstate: jnp.ndarray):
        obs_batch = obs[jnp.newaxis, :]
        ac_in = (obs_batch[jnp.newaxis, :], jnp.array([[0]]))
        hstate, pi, val = self.network.apply(self.params, hstate, ac_in)
        log_probs = jnp.array([pi.log_prob(i) for i in range(self.num_actions)])
        jax.debug.callback(self.print_decision, log_probs, val)
        return jnp.argmax(log_probs), hstate

    def input_observation(self, obs: jnp.ndarray):
        self.next_action, self.hstate = self.sample_action(obs[self.agent_id])

    def get_action(self):
        return self.next_action

    def print_decision(self, log_probs: jnp.ndarray, val: jnp.ndarray):
        if self.verbose:
            probs = jnp.exp(log_probs)
            jax.debug.print(
                "[agent{}] action: {}, {}, value = {:.6f}",
                self.agent_id,
                jnp.argmax(log_probs),
                jnp.rint(probs.squeeze() * 100),
                val.squeeze(),
            )
