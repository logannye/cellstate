# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e
ARG SOURCE_DATE_EPOCH=1786406400

FROM python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3 AS runtime-builder

ENV LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    VIRTUAL_ENV=/opt/runtime \
    PATH=/opt/runtime/bin:/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

ARG SOURCE_DATE_EPOCH

RUN --mount=type=bind,source=requirements.lock,target=/tmp/cellstate-requirements.lock,ro \
    python -m venv /opt/runtime \
    && /opt/runtime/bin/python -m pip install \
        --disable-pip-version-check \
        --no-cache-dir \
        --no-compile \
        --only-binary=:all: \
        --require-hashes \
        --requirement /tmp/cellstate-requirements.lock \
    && /opt/runtime/bin/python -c \
        "import h5py,numpy,pydantic,scipy; assert (h5py.__version__,numpy.__version__,pydantic.__version__,scipy.__version__)==('3.16.0','2.4.6','2.13.4','1.17.1')" \
    && mkdir -p \
        /cellstate-root/opt \
        /cellstate-root/workspace \
        /cellstate-root/run/cellstate/snapshot \
    && mv /opt/runtime /cellstate-root/opt/runtime \
    && chmod 0755 \
        /cellstate-root \
        /cellstate-root/opt \
        /cellstate-root/workspace \
        /cellstate-root/run \
        /cellstate-root/run/cellstate \
    && chmod 1777 /cellstate-root/run/cellstate/snapshot \
    && find /cellstate-root -exec touch -h -d "@${SOURCE_DATE_EPOCH}" {} +

FROM python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3

ENV LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    VIRTUAL_ENV=/opt/runtime \
    PATH=/opt/runtime/bin:/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

ARG SOURCE_DATE_EPOCH

COPY --from=runtime-builder /cellstate-root/ /

LABEL org.openai.cellstate.item12.family="sciplex3-v5-source-touching-runtime" \
      org.openai.cellstate.item12.base-image-digest="sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3" \
      org.openai.cellstate.item12.python-version="3.11.15" \
      org.openai.cellstate.item12.requirements-sha256="82f1d12b6ed3370ccd91c1e6e84bc04e64627c3daa8ce1d39c422a0f819c32d6"

WORKDIR /workspace
ENTRYPOINT ["/opt/runtime/bin/python"]
