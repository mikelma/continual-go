from jaxtyping import ScalarLike, Integer, Array, Float, PRNGKeyArray
from mctx import PolicyOutput
import jax
import jax.numpy as jnp
from . import SkillControl


class EpsilonSkillControl(SkillControl):
    """Returns samples a legal action uniformly with probability 1-skill_level, otherwise returning the original action."""

    def get_action(
        self,
        key: PRNGKeyArray,
        policy_output: PolicyOutput,
        legal_actions: Float[Array, " num_actions"],
        skill_level: ScalarLike,
    ) -> Integer[ScalarLike, ""]:
        """Returns the action to take according to the given skill level."""
        key_sample, key_choice = jax.random.split(key)

        # Sample a random but legal action
        legal_logits = jnp.where(legal_actions, 0.0, -jnp.inf)
        random_action = jax.random.categorical(key_sample, legal_logits)

        return jax.lax.select(
            jax.random.uniform(key_choice) >= skill_level,
            random_action,
            jnp.squeeze(policy_output.action),
        )
