"""ThunderReact FastAPI application entry point."""
import logging
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .config import VLLM_BACKENDS
from .scheduler import MultiBackendRouter
from .program import ProgramState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# FastAPI Application
# =============================================================================

router = MultiBackendRouter(VLLM_BACKENDS)
app = FastAPI(title="ThunderReact - Multi-Backend Prefix Cache Router")


@app.on_event("startup")
async def startup_event():
    await router.start()


@app.on_event("shutdown")
async def shutdown_event():
    await router.stop()


def get_program_id(payload: Dict[str, Any], _request: Request) -> str:
    """Extract program_id from the request."""
    if "job_id" in payload:
        return str(payload["job_id"])
    extra_body = payload.get("extra_body", {})
    if isinstance(extra_body, dict) and "job_id" in extra_body:
        return str(extra_body["job_id"])
    return "default"


@app.post("/v1/chat/completions")
async def route_chat_completions(request: Request):
    """Route chat completions request to the appropriate backend."""
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc

    program_id = get_program_id(payload, request)

    while True:
        backend = router.get_backend_for_program(program_id)
        backend = await router.apply_pending_transfer(program_id, backend)
        async with backend.lock:
            if program_id not in backend.programs:
                backend.programs[program_id] = ProgramState(
                    context_len=0, step_count=0
                )
            state = backend.programs[program_id]

            if not state.paused:
                state.inflight = True
                state.step_count += 1
                break

            wait_event = state.resume_event
            wait_state = state
            wait_state.waiting_on_resume = True

        try:
            await wait_event.wait()
        finally:
            wait_state.waiting_on_resume = False

    async def update_total_tokens(tokens: int) -> None:
        async with backend.lock:
            state = backend.programs.get(program_id)
            if state is not None:
                state.context_len = tokens

    try:
        return await router.proxy_request(backend, payload, on_total_tokens=update_total_tokens)
    finally:
        async with backend.lock:
            if program_id in backend.programs:
                state = backend.programs[program_id]
                state.inflight = False


@app.get("/programs")
async def list_programs():
    """List all programs across all backends."""
    result = {}
    for backend in router.backends.values():
        async with backend.lock:
            for pid, s in backend.programs.items():
                result[pid] = {
                    "backend": backend.url,
                    "context_len": s.context_len,
                    "step": s.step_count,
                    "inflight": s.inflight,
                    "paused": s.paused,
                }
    return JSONResponse(result)


@app.post("/programs/release")
async def release_program(request: Request):
    """Force-release a program from router state."""
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc

    program_id = payload.get("job_id") or payload.get("program_id")
    if not program_id:
        raise HTTPException(status_code=400, detail="Missing job_id")
    program_id = str(program_id)

    released = False
    for backend in router.backends.values():
        async with backend.lock:
            if program_id in backend.programs:
                del backend.programs[program_id]
                released = True

    if program_id in router.program_affinity:
        del router.program_affinity[program_id]
        released = True

    if released:
        logger.info(f"Released program {program_id}")
    return JSONResponse({"job_id": program_id, "released": released})


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    healthy_backends = [
        url for url, backend in router.backends.items() if backend.healthy
    ]
    return JSONResponse({
        "status": "ok" if healthy_backends else "degraded",
        "backends": {
            url: {
                "healthy": backend.healthy,
                "usage": backend.usage,
                "running_requests": backend.running_requests,
                "waiting_requests": backend.waiting_requests,
                "programs_count": len(backend.programs),
                "paused_count": backend.paused_count,
            }
            for url, backend in router.backends.items()
        }
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8300, log_level="info")
