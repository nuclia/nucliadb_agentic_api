from hyperforge.api.models import InteractionRequest

from hyperforge_nucliadb_agentic.ask.model import AskRequest


def interaction_from_ask_request(ask_request: AskRequest) -> InteractionRequest:
    # Transform the AskRequest into an ARAG interaction
    # This is a placeholder implementation and should be adapted to the actual ARAG interaction format
    interaction = InteractionRequest(question=ask_request.query, streaming=True)
    interaction.arguments["ask_request"] = ask_request.model_dump_json()
    return interaction
