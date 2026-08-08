# ADR 0003: Contracts are backend-neutral

Status: accepted.

Boundary schemas use Pydantic and content-addressed artifact references. Tensor frameworks and assay
containers remain adapters so a serialized belief does not depend on one training stack.
