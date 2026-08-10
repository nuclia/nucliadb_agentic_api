from enum import Enum

from nucliadb_protos.utils_pb2 import RelationNode


class SendToPredictError(Exception):
    pass


class AnswerStatusCode(str, Enum):
    SUCCESS = "0"
    ERROR = "-1"
    NO_CONTEXT = "-2"
    NO_RETRIEVAL_DATA = "-3"

    def prettify(self) -> str:
        return {
            AnswerStatusCode.SUCCESS: "success",
            AnswerStatusCode.ERROR: "error",
            AnswerStatusCode.NO_CONTEXT: "no_context",
            AnswerStatusCode.NO_RETRIEVAL_DATA: "no_retrieval_data",
        }[self]


def convert_relations(data: dict[str, list[dict[str, str]]]) -> list[RelationNode]:
    return [
        RelationNode(
            value=token["text"],
            ntype=RelationNode.NodeType.ENTITY,
            subtype=token["ner"],
        )
        for token in data["tokens"]
    ]
