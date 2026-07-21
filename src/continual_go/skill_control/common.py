from abc import ABC, abstractmethod
from jaxtyping import ScalarLike, Integer, Array, Float, PRNGKeyArray
from flax.struct import PyTreeNode
from mctx import PolicyOutput


class SkillControl(ABC, PyTreeNode):
    """Common interface for skill controlers."""

    @abstractmethod
    def get_action(
        self,
        key: PRNGKeyArray,
        policy_output: PolicyOutput,
        legal_actions: Float[Array, " num_actions"],
        skill_level: ScalarLike,
    ) -> Integer[ScalarLike, ""]:
        """Returns the action to take according to the given skill level."""
