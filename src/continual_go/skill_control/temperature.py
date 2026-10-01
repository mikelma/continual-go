from jaxtyping import ScalarLike, Integer, Array, Float, PRNGKeyArray
from mctx import PolicyOutput
import jax
import jax.numpy as jnp
from . import SkillControl


class TemperatureSkillControl(SkillControl):
    def get_action(
        self,
        key: PRNGKeyArray,
        policy_output: PolicyOutput,
        legal_actions: Float[Array, " num_actions"],
        skill_level: ScalarLike,
    ) -> Integer[ScalarLike, ""]:
        """Returns the action to take according to the given skill level."""
        logits = jnp.log(policy_output.action_weights + 1e-8)
        scaled_logits = skill_level * logits
        scaled_logits = jnp.where(legal_actions, scaled_logits, -jnp.inf)
        action = jax.random.categorical(key, scaled_logits)
        return jnp.squeeze(action)
