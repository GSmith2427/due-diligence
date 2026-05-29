# ADR-0002: Runtime topology — native Ollama, containerised Qdrant

- **Status**: Accepted
- **Date**: 2026-05-29
- **Deciders**: Project author
- **Tags**: infrastructure, performance, apple-silicon, deployment

## Context

The system has two long-running runtime dependencies: an LLM inference server (Ollama) and a vector database (Qdrant). The obvious deployment choice — and the path most tutorials take — is to put both in a single `docker-compose.yml` file. One command brings the whole stack up, one command tears it down, and the same setup works on any developer's machine.

This project, however, is being developed primarily on Apple Silicon (M1 Max, 32-core GPU, 32 GB RAM), and the LLM workload is GPU-bound. The naive "everything in Docker" approach has a specific and significant performance problem on this platform, which forces a more deliberate topology decision.

### The Apple Silicon GPU passthrough problem

Docker Desktop on macOS does not run containers directly on the host. It spins up a lightweight Linux VM (currently `linuxkit` / `vz` framework) and runs containers inside that VM. This architecture is necessary because Docker is fundamentally a Linux technology — its primitives (namespaces, cgroups) do not exist on Darwin.

The consequence is that **no Docker container on macOS has access to the Apple Silicon GPU**. There is no equivalent of NVIDIA's container toolkit for Metal. The Linux VM sees a virtualised CPU and memory, and that is all. Any GPU-accelerated workload run inside a container falls back to CPU inference.

For Ollama specifically, this means a `qwen2.5:14b-instruct` model that achieves ~40 tokens/s natively on the M1 Max's GPU drops to ~5–8 tokens/s when run inside Docker. The user-facing impact is that report generation goes from ~30 seconds to several minutes per query. For an interactive due-diligence tool, that latency difference is not acceptable.

Qdrant has no such constraint. It is a CPU-bound service whose performance is dominated by memory bandwidth and disk I/O, both of which Docker on macOS passes through with negligible overhead.

## Decision

The runtime topology is **split**:

- **Ollama runs natively on the host** via the Homebrew-installed `ollama` service. It binds to `127.0.0.1:11434` on the host machine.
- **Qdrant runs in a Docker container** managed by `docker-compose.yml`. It binds to `127.0.0.1:6333` (REST) and `127.0.0.1:6334` (gRPC) on the host.
- **All future services that need GPU access** (e.g. a local reranker, OCR for scanned PDFs) will follow the same pattern: native on macOS, containerised on Linux.
- **Configuration is environment-driven.** The application reads `OLLAMA_HOST` and `QDRANT_HOST` from the environment, defaulting to `localhost` for both. This keeps the application code topology-agnostic.

For Linux deployments (CI, future production), Ollama can be containerised with the NVIDIA container toolkit and use the same Compose file pattern, since the GPU-passthrough problem is macOS-specific. The same application code runs in both environments because of the env-driven configuration.

## Consequences

### Positive

- **Full GPU utilisation on the development machine.** Inference runs at native speed, making the system genuinely interactive rather than a batch job.
- **Honest deployment story.** The project does not pretend to be "one-click reproducible" while quietly running 5× slower than it could. This matters because the README's quickstart will be tested by recruiters or curious developers.
- **Cleaner separation of concerns.** Stateful, slow-changing infrastructure (the vector store) lives in Docker where its lifecycle is managed declaratively. The LLM runtime, which the user may want to switch (different models, different backends like vLLM or llama.cpp), runs natively where it is easier to swap.
- **Lower memory pressure.** Docker Desktop's VM reserves whatever memory you allocate to it regardless of container usage. Keeping the 9 GB Ollama model out of the VM means a smaller Docker memory allocation, leaving more headroom for the host.

### Negative

- **Two install steps instead of one.** The README quickstart has to say "install Ollama via Homebrew, then run `docker compose up`." This is a real cost in friction for new contributors. It is mitigated by a short setup script (see Implementation notes below).
- **Platform-conditional documentation.** Linux developers can run everything in Docker; macOS developers cannot. The README and Compose file need to make this distinction clearly without becoming a wall of conditionals.
- **CI does not exactly mirror local development.** CI runs everything in Linux containers, including Ollama. This means a class of macOS-specific bugs (e.g. networking quirks between host and container) will only surface locally. Acceptable: those bugs are rare and obvious when they occur.
- **Loss of `depends_on` ordering between Ollama and Qdrant.** Within a single Compose file, services can declare startup dependencies. Across the host/container boundary, the application has to handle "Ollama not ready yet" itself via retries. This is good practice anyway — production systems should never assume their dependencies are ready — but it is a small amount of extra code in the LLM client.

## Alternatives considered

- **Everything in Docker, CPU-only inference.** Rejected on performance grounds detailed above. A 5× latency hit makes the tool unpleasant to use and would lead to a "this is too slow" first impression for anyone who clones the repo.
- **Everything native (Qdrant via Homebrew or binary install).** Possible, since Qdrant ships a macOS binary. Rejected because Qdrant benefits from container isolation (data volume management, easy reset, version pinning), and the native install would have to be re-documented for Linux users anyway. Containerising the stateful piece while leaving the stateless GPU consumer native is the more defensible split.
- **Ollama via `colima` with GPU support.** Colima is a Docker Desktop alternative that *has* attempted Apple Silicon GPU passthrough via experimental Apple Virtualization framework integration. As of this writing the GPU support is not reliable for sustained inference workloads, and it requires every developer to replace Docker Desktop. Too much imposed setup cost for a partial fix.
- **Remote Ollama on a Linux box with a GPU.** Out of scope — the project's constraint is that it must run fully self-hosted on a single developer machine.
- **`vLLM` or `llama.cpp` directly, no Ollama.** Both are higher-performance LLM servers. Ollama wraps `llama.cpp` and adds model management, an OpenAI-compatible API, and Homebrew packaging. The convenience is worth the marginal performance cost. If profiling later shows Ollama as a bottleneck, swapping the backend is contained behind the LLM client abstraction.

## Implementation notes

- A `scripts/bootstrap.sh` will codify the two-step setup so contributors do not have to read the ADR to get running. It installs Ollama if missing, starts the service, pulls the required models, and brings up the Compose stack.
- The `Makefile` (or `justfile`) target `dev` will run the bootstrap, run the API server, and tail the relevant logs.
- The LLM client module will implement bounded exponential backoff for connection errors against Ollama, capped at three retries. This handles the "host service not yet ready" case without masking real outages.
- Health endpoints on the FastAPI service will independently surface the reachability of both Ollama and Qdrant, so a misconfigured environment is diagnosable from one URL.

## Open questions

- Whether to add a `colima` profile alongside the default `docker compose` config for users who want a single-command setup and are willing to accept the GPU caveats. Likely deferred until someone actually asks for it.
- Whether to provide a Linux-flavoured `docker-compose.linux.yml` that includes the Ollama service with GPU passthrough, for users running the project on Linux workstations or servers. Worth doing once the project is otherwise stable.

## References

- Docker Desktop architecture on macOS: <https://docs.docker.com/desktop/mac/>
- Ollama on macOS uses the Metal Performance Shaders framework for GPU acceleration via `llama.cpp`: <https://github.com/ollama/ollama/blob/main/docs/gpu.md>
- Qdrant deployment guide: <https://qdrant.tech/documentation/guides/installation/>
- Discussion of GPU passthrough limitations on Docker Desktop for Mac: <https://github.com/docker/roadmap/issues/145>
