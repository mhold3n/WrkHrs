group "default" {
  targets = [
    "gateway",
    "orchestrator",
    "rag",
    "tool_registry",
    "mcp",
    "asr"
  ]
}

target "gateway" {
  context    = ".."
  dockerfile = "docker/gateway/Dockerfile"
  tags       = ["ghcr.io/${GITHUB_REPOSITORY_OWNER}/ai-gateway:latest"]
}

target "orchestrator" {
  context    = ".."
  dockerfile = "docker/orchestrator/Dockerfile"
  tags       = ["ghcr.io/${GITHUB_REPOSITORY_OWNER}/ai-orchestrator:latest"]
}

target "rag" {
  context    = ".."
  dockerfile = "docker/rag/Dockerfile"
  tags       = ["ghcr.io/${GITHUB_REPOSITORY_OWNER}/ai-rag:latest"]
}

target "tool_registry" {
  context    = ".."
  dockerfile = "docker/tool-registry/Dockerfile"
  tags       = ["ghcr.io/${GITHUB_REPOSITORY_OWNER}/ai-tool-registry:latest"]
}

target "mcp" {
  context    = ".."
  dockerfile = "docker/mcp/Dockerfile"
  tags       = ["ghcr.io/${GITHUB_REPOSITORY_OWNER}/ai-mcp:latest"]
}

target "asr" {
  context    = ".."
  dockerfile = "docker/asr/Dockerfile"
  tags       = ["ghcr.io/${GITHUB_REPOSITORY_OWNER}/ai-asr:latest"]
}


