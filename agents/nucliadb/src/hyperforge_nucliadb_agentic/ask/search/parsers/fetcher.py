from google.protobuf.json_format import ParseDict
from hyperforge.manager import Manager
from nuclia.lib.nua import PredictQueryRequest
from nucliadb_models.internal.predict import QueryInfo
from nucliadb_protos import knowledgebox_pb2, utils_pb2
from nucliadb_sdk import NucliaDBAsync

from hyperforge_nucliadb_agentic.ask import logger
from hyperforge_nucliadb_agentic.ask.exceptions import (
    InvalidQueryError,
)
from hyperforge_nucliadb_agentic.ask.model import Image, MaxTokens
from hyperforge_nucliadb_agentic.ask.predict import (
    SendToPredictError,
    convert_relations,
)
from hyperforge_nucliadb_agentic.ask.search import rpc


class Fetcher:
    """This class is an encapsulation of data gathering across different parts of
    the system. Given the user query input, it aims to be as efficient as
    possible removing redundant expensive calls to other parts of the system. An
    instance of a fetcher caches it's results and it's thought to be used in the
    context of a single request.

    *DO NOT* use this as a global object!

    """

    def __init__(
        self,
        kbid: str,
        predict_manager: Manager,
        *,
        query: str,
        user_vector: list[float] | None,
        vectorset: str | None,
        rephrase: bool,
        rephrase_prompt: str | None,
        generative_model: str | None,
        query_image: Image | None,
    ):
        self.kbid = kbid
        self.predict_manager = predict_manager
        self.query = query
        self.user_vector = user_vector
        self.user_vectorset = vectorset
        self.user_vectorset_validated = False
        self.rephrase = rephrase
        self.rephrase_prompt = rephrase_prompt
        self.generative_model = generative_model
        self.query_image = query_image

        self._query_info: QueryInfo | None = None
        self._vectorset: str | None = None

    async def query_information(self) -> QueryInfo:
        if self._query_info is None:
            item = PredictQueryRequest(
                text=self.query,
                semantic_models=[self.user_vectorset] if self.user_vectorset else None,
                generative_model=self.generative_model,
                rephrase=self.rephrase,
                rephrase_prompt=self.rephrase_prompt,
                query_image=self.query_image,
            )
            try:
                self._query_info = await self.predict_manager.predict_query(
                    item, kbid=self.kbid
                )
            except TimeoutError as exc:
                raise SendToPredictError(
                    "timeout while requesting Predict API /query"
                ) from exc

        return self._query_info

    # Retrieval

    async def get_rephrased_query(self) -> str | None:
        query_info = await self.query_information()
        return query_info.rephrased_query

    def get_cached_rephrased_query(self) -> str | None:
        if self._query_info is None:
            return None
        return self._query_info.rephrased_query

    async def get_detected_entities(self) -> list[utils_pb2.RelationNode]:
        query_info = await self.query_information()
        if query_info.entities is not None:
            detected_entities = convert_relations(query_info.entities.model_dump())
        else:
            detected_entities = []
        return detected_entities

    async def get_semantic_min_score(self) -> float | None:
        query_info = await self.query_information()
        vectorset = await self.get_vectorset()
        return query_info.semantic_thresholds.get(vectorset, None)

    async def get_vectorset(self) -> str:
        if self._vectorset is None:
            if self.user_vectorset is not None:
                self._vectorset = self.user_vectorset
            else:
                # when it's not provided, we get the default from Predict API
                query_info = await self.query_information()
                if query_info.sentence is None or len(query_info.sentence.vectors) == 0:
                    logger.error(
                        "Asking for a vectorset but /query didn't return one",
                        extra={"kbid": self.kbid},
                    )
                    raise SendToPredictError(
                        "Predict API didn't return a sentence vectorset"
                    )
                # vectors field is enforced by the data model to have at least one key
                for vectorset in query_info.sentence.vectors.keys():
                    self._vectorset = vectorset
                    break
        assert self._vectorset is not None
        return self._vectorset

    async def get_query_vector(self) -> list[float]:
        if self.user_vector is not None:
            return self.user_vector

        query_info = await self.query_information()
        if query_info.sentence is None:
            logger.error(
                "Asking for a semantic query vector but /query didn't return a sentence",
                extra={"kbid": self.kbid},
            )
            raise SendToPredictError(
                "Predict API didn't return a sentence for semantic search"
            )

        vectorset = await self.get_vectorset()
        if vectorset not in query_info.sentence.vectors:
            logger.error(
                "Predict is not responding with a valid query nucliadb vectorset",
                extra={
                    "kbid": self.kbid,
                    "vectorset": vectorset,
                    "predict_vectorsets": ",".join(query_info.sentence.vectors.keys()),
                },
            )
            raise SendToPredictError(
                "Predict API didn't return the requested vectorset"
            )

        query_vector = query_info.sentence.vectors[vectorset]
        return query_vector

    async def get_classification_labels(
        self, reader_sdk: NucliaDBAsync
    ) -> knowledgebox_pb2.Labels:
        labelsets = await rpc.labelsets(reader_sdk=reader_sdk, kbid=self.kbid)

        # TODO(decoupled-ask): remove this conversion and refactor code to use API models instead of protobuf
        kb_labels = knowledgebox_pb2.Labels()
        for labelset, labels in labelsets.labelsets.items():
            ParseDict(labels.model_dump(), kb_labels.labelset[labelset])

        return kb_labels

    # Generative

    async def get_visual_llm_enabled(self) -> bool:
        query_info = await self.query_information()
        if query_info is None:
            raise SendToPredictError("Error while using predict's query endpoint")

        return query_info.visual_llm

    async def get_max_context_tokens(self, max_tokens: MaxTokens | None) -> int:
        query_info = await self.query_information()
        if query_info is None:
            raise SendToPredictError("Error while using predict's query endpoint")

        model_max = query_info.max_context
        if max_tokens is not None and max_tokens.context is not None:
            if max_tokens.context > model_max:
                raise InvalidQueryError(
                    "max_tokens.context",
                    f"Max context tokens is higher than the model's limit of {model_max}",
                )
            return max_tokens.context
        return model_max

    def get_max_answer_tokens(self, max_tokens: MaxTokens | None) -> int | None:
        if max_tokens is not None and max_tokens.answer is not None:
            return max_tokens.answer
        return None
