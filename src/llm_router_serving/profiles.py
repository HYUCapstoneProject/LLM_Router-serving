"""Release objective guard shared by startup and model hot reload."""

from llm_router_serving.config import Settings
from llm_router_serving.errors import ModelContractError
from llm_router_serving.model_source import ResolvedModel


def validate_router_profile(settings: Settings, resolved: ResolvedModel) -> None:
    if settings.router_profile != "r0":
        return
    if (
        not settings.router_model_require_approved
        or settings.router_max_length != 8192
        or resolved.max_length != 8192
        or resolved.input_contract != "structure-preserving-v2"
        or resolved.training_max_length is None
        or resolved.manifest_sha256 is None
        or (resolved.routing_policy is not None and resolved.routing_policy.objective != "ordinal")
    ):
        raise ModelContractError("R0 profile requires a verified approved ordinal 8192 package")
