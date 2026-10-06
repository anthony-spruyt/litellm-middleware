FROM scratch

LABEL org.opencontainers.image.title="litellm-middleware" \
      org.opencontainers.image.description="LiteLLM proxy middleware package for a Kubernetes image volume" \
      org.opencontainers.image.source="https://github.com/anthony-spruyt/litellm-middleware" \
      org.opencontainers.image.licenses="MIT"

COPY src/litellm_middleware/ /litellm_middleware/

USER 65534:65534

HEALTHCHECK NONE
