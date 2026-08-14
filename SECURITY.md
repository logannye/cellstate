# Security

## Reporting

Report privately through GitHub's private vulnerability reporting for this repository, if it is
enabled:

<https://github.com/logannye/cellstate/security/advisories/new>

If that page is not available to you, open a public issue that says only that you have a report and
asks for a private channel. Do not put exploit detail in a public issue.

## Scope

This is a research library. Its runtime dependencies are `numpy`, `pydantic`, and `scipy`, with
`h5py` behind the `sciplex3` extra. The security-relevant surface is the trusted admission boundary:
byte-stream verification, externally loaded interface observation, and the authenticated execution
sources described in
[ADR 0010](docs/adr/0010-trusted-admission-verification.md). A defect that lets an unverified
artifact, code object, or validation result be accepted as verified is a security issue here, not
merely a correctness bug.

## Data handling

Do not commit donor-identifiable, clinical, genomic, proprietary assay, or model-weight artifacts.
This package stores references and digests for large artifacts so that access control can remain
with an external artifact store. Report any commit that violates this through the channel above
rather than in a public issue.
