from typing import TYPE_CHECKING, Union, cast

from fastapi import Header, HTTPException, Request, Response
from nucliadb_models.resource import NucliaDBRoles
from nucliadb_utils.authentication import requires

from nucliadb_agentic_api import exceptions
from nucliadb_agentic_api.models import (
    GoogleSourceSchema,
    MCPSourceSchema,
    NucliaDBSourceSchema,
    PerplexitySourceSchema,
)
from nucliadb_agentic_api.v1.router import router

if TYPE_CHECKING:
    from nucliadb_agentic_api.app import HTTPApplication

# FastAPI / OpenAPI needs a concrete type for response_model — use the Union directly.
_AnySource = Union[
    NucliaDBSourceSchema,
    PerplexitySourceSchema,
    MCPSourceSchema,
    GoogleSourceSchema,
]


@router.post(
    "/api/v1/kb/{kbid}/sources/{source_id}",
    status_code=201,
    summary="Create a source",
    tags=["Sources"],
)
@requires(NucliaDBRoles.OWNER)
async def create_source_endpoint(
    request: Request,
    kbid: str,
    source_id: str,
    item: _AnySource,
    x_nucliadb_account: str = Header(default="", include_in_schema=False),
) -> Response:
    app = cast("HTTPApplication", request.app)
    try:
        await app.source_manager.create_source(
            account=x_nucliadb_account,
            kbid=kbid,
            source_id=source_id,
            source=item,
        )
    except exceptions.Conflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return Response(status_code=201)


@router.get(
    "/api/v1/kb/{kbid}/sources/{source_id}",
    status_code=200,
    summary="Get a source",
    tags=["Sources"],
    response_model=_AnySource,
    response_model_exclude_none=True,
)
@requires(NucliaDBRoles.READER)
async def get_source_endpoint(
    request: Request,
    kbid: str,
    source_id: str,
    x_nucliadb_account: str = Header(default="", include_in_schema=False),
) -> _AnySource:
    app = cast("HTTPApplication", request.app)
    try:
        return await app.source_manager.get_source(  # type: ignore[return-value]
            account=x_nucliadb_account,
            kbid=kbid,
            source_id=source_id,
        )
    except exceptions.NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/api/v1/kb/{kbid}/sources",
    status_code=200,
    summary="List sources",
    tags=["Sources"],
    response_model=dict[str, _AnySource],
    response_model_exclude_none=True,
)
@requires(NucliaDBRoles.READER)
async def list_sources_endpoint(
    request: Request,
    kbid: str,
    x_nucliadb_account: str = Header(default="", include_in_schema=False),
) -> dict[str, _AnySource]:
    app = cast("HTTPApplication", request.app)
    return await app.source_manager.list_sources(  # type: ignore[return-value]
        account=x_nucliadb_account,
        kbid=kbid,
    )


@router.patch(
    "/api/v1/kb/{kbid}/sources/{source_id}",
    status_code=204,
    summary="Update a source",
    tags=["Sources"],
)
@requires(NucliaDBRoles.OWNER)
async def patch_source_endpoint(
    request: Request,
    kbid: str,
    source_id: str,
    item: _AnySource,
    x_nucliadb_account: str = Header(default="", include_in_schema=False),
) -> Response:
    app = cast("HTTPApplication", request.app)
    try:
        await app.source_manager.patch_source(
            account=x_nucliadb_account,
            kbid=kbid,
            source_id=source_id,
            source=item,
        )
    except exceptions.NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(status_code=204)


@router.delete(
    "/api/v1/kb/{kbid}/sources/{source_id}",
    status_code=204,
    summary="Delete a source",
    tags=["Sources"],
)
@requires(NucliaDBRoles.OWNER)
async def delete_source_endpoint(
    request: Request,
    kbid: str,
    source_id: str,
    x_nucliadb_account: str = Header(default="", include_in_schema=False),
) -> Response:
    app = cast("HTTPApplication", request.app)
    try:
        configs = await app.agent_manager.configs_referencing_source(
            account=x_nucliadb_account,
            kbid=kbid,
            source_id=source_id,
        )
        if configs:
            raise exceptions.InUse(
                "Source is in use by agentic configuration(s): "
                + ", ".join(sorted(configs))
            )
        await app.source_manager.delete_source(
            account=x_nucliadb_account,
            kbid=kbid,
            source_id=source_id,
        )
    except exceptions.NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except exceptions.InUse as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return Response(status_code=204)
