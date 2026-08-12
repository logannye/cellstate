# sci-Plex3 v5 contained runtime

**Suspended.** This image was built and validated source-free in historical Item 12.2. It was the
source-touching worker runtime for a suspended Item 12.3 attempt; it is retained as frozen evidence,
not as a runtime to be invoked. The base image and every Python
wheel are content pinned. The authorized host wrapper and supervisor must invoke the built image by
the exact digest recorded in `runtime-image-lock.json`; tags are never execution authority.

The image itself supplies no data authority. The host supervisor accounts its 3,600-second budget
before public code/input staging and actively bounds every Docker command and wait; a returned
staging overrun fails before container creation. An independent 3,540-second in-container watchdog
begins before protected-source open and covers snapshot, fit, and close-reauthentication even if the
supervisor dies. Docker supplies the aggregate memory, total-memory-plus-swap, and PID cgroup
limits. The host launches the image with a read-only root filesystem, no network, no Linux
capabilities, `no-new-privileges`, a read-only source mount, and one isolated staging mount. The
canonical publication tree is never mounted into the worker.

Native-Linux execution is frozen as `host-effective-uid-gid`: Docker runs the worker as the host's
numeric effective UID/GID, and the bounded mode-`0700` tmpfs is mounted with that same UID/GID. The
anonymous snapshot volume initializes from the empty in-image `/run/cellstate/snapshot` directory,
whose mode is `1777` (`empty-image-directory-mode-1777`). This makes the declared `0400`/`0700` host
binds usable by the non-root worker without widening their permissions. The successful live path
also requires the parent to re-inventory and seal the exact worker stage.

The reproducibility authority is the checked-in runtime lock and the pinned builder setup in the CI
workflow: Buildx `v0.28.0` at commit `b1281b81bba797b21d9eaf256e6a13eb14419836`, BuildKit
`v0.24.0` from image
`sha256:6eceb8971ce4fceb3daca562832642706238b7eea72941fcf9896c93c3c4a53e`, and Dockerfile frontend
`sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e`.
After selecting that exact builder, build the Linux x86-64 image from this directory:

```console
docker buildx build --no-cache --provenance=false --platform linux/amd64 \
  --build-arg SOURCE_DATE_EPOCH=1786406400 --output type=oci,dest=runtime.oci.tar \
  --tag cellstate-sciplex3-v5-runtime:20260811-locked .
```

Under those frozen inputs and toolchain, three independent no-cache, provenance-disabled builds—
Docker Desktop `amd64` emulation, the separate local `tinyzkp` `docker-container` builder pinned to
the locked BuildKit image, and native Linux `amd64` GitHub Actions—produced the same byte-for-byte
OCI archive, SHA-256 `37c2fa5846acfbd8357476859bd7f8f0ac6591261d79c2f6f46f0aa22fb76454`.
Its index is
`sha256:e0f0afd6c66197a37d0ab7a05e7cccfe5990da1fd8497e175fdf3ab909a67812`, its runnable
`linux/amd64` child manifest is
`sha256:12c2faa6019fb60cdcabaa8f38f70e99be7998997b97ddb0ca59fbe2e82f1e25`, and its config is
`sha256:80ed48f278d7a46c0ae7811285efc69181ae59872a358cc9b176079aa09f3cc8`. The Dockerfile SHA-256 is
`a3a71c3d61c71235d9c1a99c16aa00568b398971adfc2da65388b0c7ea3987a0`.

The multi-stage Dockerfile installs dependencies only in the build stage, then copies the curated
`/opt`, `/run`, and `/workspace` runtime tree into a clean final base. Build-host artifacts such as
Docker Desktop's Rosetta cache under `/root/.cache/rosetta` cannot enter the final image. The lock
records the exact archive, index, child, config, ordered layers, builder identity, build options, and
source-free probe. Rebuilding with a different builder or changing any layer creates a new candidate
runtime version and requires a new lock; the lock must never be edited to bless an unreviewed image.

Load the exact OCI output through Docker's containerd image store before execution; Docker's classic
graphdriver image store does not import OCI-layout archives. CI enables the documented
`containerd-snapshotter` feature only after the two archives pass exact verification, then verifies
the isolated Docker 29.7.2 daemon before loading the archive and running the live probes. The image
is durably distributed as the sole asset of the immutable, attested
[`sciplex3-v5-runtime-20260811-locked`](https://github.com/logannye/cellstate/releases/tag/sciplex3-v5-runtime-20260811-locked)
GitHub Release. Consumers must verify that release, asset, archive, complete image closure, and exact
Docker 29.7.2 native-Linux runtime after one-use attempt consumption and before protected-source
work. This asset-bearing runtime release is distinct from the immutable, asset-free attempt release
that would consume one exact approved proposal. Neither publication grants source authority by
itself. No canonical pending Item 12.3 proposal exists, no protected execution has run, and this
source-free runtime work issued no candidate artifact, training record, or lifecycle result.
