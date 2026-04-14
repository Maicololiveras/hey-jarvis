"""Maix AI Engine gRPC client for Jarvis fallback queries."""

from __future__ import annotations

import importlib
import logging
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType

log = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_GRPC_PORT = 50051
DEFAULT_PORT_FILE = Path(tempfile.gettempdir()) / "maix_ai_port.txt"
_PROTO_CACHE_DIR = Path(tempfile.gettempdir()) / "jarvis_maix_proto_cache"

_COMMON_PROTO = """syntax = \"proto3\";

package praxisgenai.v1;

option go_package = \"github.com/Maicololiveras/praxisgenai-motor-ai-sdk/gen/go/praxisgenai/v1\";
option java_package = \"com.praxisgenai.v1\";
option csharp_namespace = \"PraxisGenAI.V1\";

import \"google/protobuf/timestamp.proto\";
import \"google/protobuf/struct.proto\";

enum StatusCode {
  STATUS_CODE_UNSPECIFIED = 0;
  STATUS_CODE_OK = 1;
  STATUS_CODE_ERROR = 2;
  STATUS_CODE_NOT_FOUND = 3;
  STATUS_CODE_ALREADY_EXISTS = 4;
  STATUS_CODE_PERMISSION_DENIED = 5;
  STATUS_CODE_RATE_LIMITED = 6;
  STATUS_CODE_UNAVAILABLE = 7;
  STATUS_CODE_INTERNAL = 8;
}

message Status {
  StatusCode code = 1;
  string message = 2;
  repeated ErrorDetail details = 3;
}

message ErrorDetail {
  string field = 1;
  string description = 2;
  string code = 3;
}

message PaginationRequest {
  int32 page_size = 1;
  string page_token = 2;
}

message PaginationResponse {
  string next_page_token = 1;
  int32 total_count = 2;
}

message Metadata {
  map<string, string> labels = 1;
  map<string, string> annotations = 2;
  google.protobuf.Timestamp created_at = 3;
  google.protobuf.Timestamp updated_at = 4;
}

message TokenUsage {
  int32 prompt_tokens = 1;
  int32 completion_tokens = 2;
  int32 total_tokens = 3;
}

enum HealthStatus {
  HEALTH_STATUS_UNSPECIFIED = 0;
  HEALTH_STATUS_HEALTHY = 1;
  HEALTH_STATUS_DEGRADED = 2;
  HEALTH_STATUS_UNHEALTHY = 3;
}
"""

_MODEL_SERVICE_PROTO = """syntax = \"proto3\";

package praxisgenai.v1;

option go_package = \"github.com/Maicololiveras/praxisgenai-motor-ai-sdk/gen/go/praxisgenai/v1\";
option csharp_namespace = \"PraxisGenAI.V1\";

import \"praxisgenai/common.proto\";

service ModelService {
  rpc ListModels(ListModelsRequest) returns (ListModelsResponse);
  rpc GetModelInfo(GetModelInfoRequest) returns (GetModelInfoResponse);
  rpc HealthCheck(HealthCheckRequest) returns (HealthCheckResponse);
  rpc Generate(GenerateRequest) returns (GenerateResponse);
  rpc StreamGenerate(GenerateRequest) returns (stream StreamGenerateResponse);
}

message ListModelsRequest {
  string provider = 1;
  PaginationRequest pagination = 2;
}

message ModelInfo {
  string provider = 1;
  string model_id = 2;
  string display_name = 3;
  HealthStatus status = 4;
  int32 context_window = 5;
  bool supports_streaming = 6;
  bool supports_tools = 7;
  bool supports_embeddings = 8;
}

message ListModelsResponse {
  Status status = 1;
  repeated ModelInfo models = 2;
  PaginationResponse pagination = 3;
}

message GetModelInfoRequest {
  string provider = 1;
  string model_id = 2;
}

message GetModelInfoResponse {
  Status status = 1;
  ModelInfo model = 2;
}

message HealthCheckRequest {
  string provider = 1;
}

message ProviderHealth {
  string provider = 1;
  HealthStatus status = 2;
  double latency_ms = 3;
  string message = 4;
  repeated string available_models = 5;
}

message HealthCheckResponse {
  Status status = 1;
  repeated ProviderHealth providers = 2;
}

message GenerateRequest {
  string prompt = 1;
  repeated ChatMessage messages = 2;
  string model = 3;
  string provider = 4;
  float temperature = 5;
  int32 max_tokens = 6;
  float top_p = 7;
  repeated string stop = 8;
  string system = 9;
}

message ChatMessage {
  string role = 1;
  string content = 2;
}

message GenerateResponse {
  Status status = 1;
  string text = 2;
  string model = 3;
  string provider = 4;
  TokenUsage usage = 5;
  string finish_reason = 6;
}

message StreamGenerateResponse {
  string chunk = 1;
  bool done = 2;
  TokenUsage usage = 3;
  string model = 4;
  string provider = 5;
}
"""

_PROTO_MODULES: tuple[ModuleType, ModuleType, ModuleType] | None = None


def _ensure_proto_modules() -> tuple[ModuleType, ModuleType, ModuleType]:
    global _PROTO_MODULES
    if _PROTO_MODULES is not None:
        return _PROTO_MODULES

    cache_root = str(_PROTO_CACHE_DIR)
    if cache_root not in sys.path:
        sys.path.insert(0, cache_root)

    try:
        model_pb2 = importlib.import_module("praxisgenai.model_service_pb2")
        common_pb2 = importlib.import_module("praxisgenai.common_pb2")
        model_pb2_grpc = importlib.import_module("praxisgenai.model_service_pb2_grpc")
        _PROTO_MODULES = (common_pb2, model_pb2, model_pb2_grpc)
        return _PROTO_MODULES
    except ImportError:
        pass

    try:
        import grpc_tools  # noqa: WPS433 (lazy import)
        from grpc_tools import protoc  # noqa: WPS433 (lazy import)
    except ImportError as exc:
        raise RuntimeError(
            "Maix gRPC proto stubs unavailable; install grpcio-tools or ship generated protobuf modules"
        ) from exc

    proto_dir = _PROTO_CACHE_DIR / "praxisgenai"
    proto_dir.mkdir(parents=True, exist_ok=True)
    (_PROTO_CACHE_DIR / "praxisgenai" / "__init__.py").touch(exist_ok=True)
    (_PROTO_CACHE_DIR / "praxisgenai" / "common.proto").write_text(
        _COMMON_PROTO,
        encoding="utf-8",
    )
    (_PROTO_CACHE_DIR / "praxisgenai" / "model_service.proto").write_text(
        _MODEL_SERVICE_PROTO,
        encoding="utf-8",
    )

    grpc_include = Path(grpc_tools.__file__).resolve().parent / "_proto"
    result = protoc.main(
        [
            "grpc_tools.protoc",
            f"-I{_PROTO_CACHE_DIR}",
            f"-I{grpc_include}",
            f"--python_out={_PROTO_CACHE_DIR}",
            f"--grpc_python_out={_PROTO_CACHE_DIR}",
            str(proto_dir / "common.proto"),
            str(proto_dir / "model_service.proto"),
        ]
    )
    if result != 0:
        raise RuntimeError(
            f"Failed to generate Maix gRPC proto stubs (grpc_tools.protoc exit {result})"
        )

    model_pb2 = importlib.import_module("praxisgenai.model_service_pb2")
    common_pb2 = importlib.import_module("praxisgenai.common_pb2")
    model_pb2_grpc = importlib.import_module("praxisgenai.model_service_pb2_grpc")
    _PROTO_MODULES = (common_pb2, model_pb2, model_pb2_grpc)
    return _PROTO_MODULES


class MaixEngineClient:
    """Thin gRPC client for the Maix AI Engine ModelService."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port_file: str | os.PathLike[str] | None = None,
        provider: str = "",
        model: str = "",
    ) -> None:
        self.host = str(host or DEFAULT_HOST)
        resolved_port_file = os.path.expandvars(str(port_file or DEFAULT_PORT_FILE))
        self.port_file = Path(resolved_port_file).expanduser()
        self.provider = str(provider or "")
        self.model = str(model or "")
        self._last_port: int = DEFAULT_GRPC_PORT

    @property
    def url(self) -> str:
        return f"{self.host}:{self._last_port}"

    @property
    def is_running(self) -> bool:
        return False

    def _read_port(self) -> int:
        log.info("[MaixEngine] Reading gRPC port from %s", self.port_file)
        if not self.port_file.exists():
            raise RuntimeError(
                f"Maix AI Engine port file not found: {self.port_file} (expected engine gRPC port, default {DEFAULT_GRPC_PORT})"
            )

        raw_value = self.port_file.read_text(encoding="utf-8").strip()
        if not raw_value:
            raise RuntimeError(f"Maix AI Engine port file is empty: {self.port_file}")

        try:
            port = int(raw_value)
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid Maix AI Engine port '{raw_value}' in {self.port_file}"
            ) from exc

        if port <= 0 or port > 65535:
            raise RuntimeError(
                f"Invalid Maix AI Engine port {port} in {self.port_file}"
            )

        self._last_port = port
        return port

    def _load_runtime(self) -> tuple[ModuleType, ModuleType, ModuleType, ModuleType]:
        try:
            import grpc  # noqa: WPS433 (lazy import)
        except ImportError as exc:
            raise RuntimeError(
                "grpc runtime not installed; add package 'grpcio' to use Maix AI Engine"
            ) from exc

        try:
            common_pb2, model_pb2, model_pb2_grpc = _ensure_proto_modules()
        except RuntimeError:
            raise
        except Exception as exc:  # pragma: no cover - defensive import surface
            raise RuntimeError(
                f"Failed to load Maix gRPC protobuf modules: {exc}"
            ) from exc

        return grpc, common_pb2, model_pb2, model_pb2_grpc

    def _connect(self, timeout: float):
        grpc, common_pb2, model_pb2, model_pb2_grpc = self._load_runtime()
        endpoint = f"{self.host}:{self._read_port()}"
        log.info("[MaixEngine] Connecting to gRPC engine at %s", endpoint)
        channel = grpc.insecure_channel(endpoint)

        try:
            grpc.channel_ready_future(channel).result(timeout=timeout)
        except grpc.FutureTimeoutError as exc:
            channel.close()
            raise RuntimeError(
                f"Maix AI Engine gRPC unavailable at {endpoint} after {timeout:.1f}s"
            ) from exc

        stub = model_pb2_grpc.ModelServiceStub(channel)
        return channel, stub, common_pb2, model_pb2, endpoint, grpc

    def start(self, timeout: float = 5.0) -> bool:
        """Validate Maix AI Engine availability without spawning local processes."""
        try:
            channel, stub, common_pb2, model_pb2, endpoint, grpc = self._connect(
                timeout
            )
            try:
                response = stub.HealthCheck(
                    model_pb2.HealthCheckRequest(provider=self.provider),
                    timeout=timeout,
                )
            finally:
                channel.close()
        except RuntimeError as exc:
            log.error("[MaixEngine] Availability check failed: %s", exc)
            return False
        except grpc.RpcError as exc:  # pragma: no cover - network/runtime surface
            log.error("[MaixEngine] HealthCheck RPC failed at %s: %s", endpoint, exc)
            return False

        if response.status.code != common_pb2.STATUS_CODE_OK:
            message = response.status.message or "unknown engine health status"
            log.error("[MaixEngine] HealthCheck returned non-OK status: %s", message)
            return False

        log.info("[MaixEngine] Engine ready at %s", endpoint)
        return True

    def stop(self) -> None:
        """No-op for remote engine integration."""
        log.info("[MaixEngine] Remote engine client stop requested (no-op)")

    def generate(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
        timeout: float,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """Generate text through ModelService.Generate."""
        try:
            channel, stub, common_pb2, model_pb2, endpoint, grpc = self._connect(
                timeout
            )
        except RuntimeError:
            raise

        try:
            request = model_pb2.GenerateRequest(
                messages=[
                    model_pb2.ChatMessage(
                        role=str(message.get("role", "user")),
                        content=str(message.get("content", "")),
                    )
                    for message in messages
                ],
                model=self.model,
                provider=self.provider,
                temperature=temperature,
                max_tokens=max_tokens,
                system=system_prompt,
            )
            response = stub.Generate(request, timeout=timeout)
        except grpc.RpcError as exc:  # pragma: no cover - network/runtime surface
            log.error("[MaixEngine] Generate RPC failed at %s: %s", endpoint, exc)
            raise RuntimeError(f"Maix AI Engine Generate RPC failed: {exc}") from exc
        finally:
            channel.close()

        if response.status.code != common_pb2.STATUS_CODE_OK:
            message = response.status.message or "generation failed"
            log.error("[MaixEngine] Generate returned non-OK status: %s", message)
            raise RuntimeError(f"Maix AI Engine generation failed: {message}")

        text = response.text.strip()
        if not text:
            log.error("[MaixEngine] Generate returned empty text")
            raise RuntimeError("Maix AI Engine returned empty text")

        return text


def get_server_from_config(config: dict) -> MaixEngineClient | None:
    """Create a MaixEngineClient from Jarvis config."""
    backends = config.get("query", {}).get("backends", {})
    maix_engine = backends.get("maix-engine", {})
    if not maix_engine:
        log.warning("[MaixEngine] maix-engine backend not configured")
        return None

    return MaixEngineClient(
        host=maix_engine.get("host", DEFAULT_HOST),
        port_file=maix_engine.get("port_file") or DEFAULT_PORT_FILE,
        provider=maix_engine.get("provider", ""),
        model=maix_engine.get("model", ""),
    )
