# sci-Plex3 v5 contained runtime

This image is the source-touching worker runtime for roadmap Item 12.2. The base image and every
Python wheel are content pinned. The host supervisor must invoke the built image by the exact digest
recorded in `runtime-image-lock.json`; tags are never execution authority.

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

Build the Linux x86-64 image from this directory:

```console
docker buildx build --no-cache --provenance=false --platform linux/amd64 \
  --build-arg SOURCE_DATE_EPOCH=1786406400 --output type=oci,dest=runtime.oci.tar \
  --tag cellstate-sciplex3-v5-runtime:20260811-locked .
```

Two independent no-cache builds produced the same OCI index
`sha256:ababac344fae7f3d679cf9b3bbf4c46b8f3b169b358566d4abd6e3b0e7b8251e`,
whose runnable linux/amd64 child manifest is
`sha256:edd451f171161472c1a3bb6a1ae434cdedc5b776e228757ac732522c1035df18`.
Its config is `sha256:b9cdf1e179f149319b038f2f58bb80470c2a1b5bda8f1cf9d2ccbe17fe3b59e5`,
and the Dockerfile SHA-256 is
`ec21cc81a3b4d71f5de745adde74506d63da0d9b317996c8f97b067e90347e7a`.
Load that OCI output before execution; the image is not claimed to be published or remotely
pullable. The complete index, child, config, build, and probe identities are recorded in
`runtime-image-lock.json`. Rebuilding or changing any layer creates a new candidate runtime version
and requires a new lock; the lock must never be edited to bless an unreviewed image.
