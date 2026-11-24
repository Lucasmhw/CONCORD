# CONCORD 0.1.0 archival record

Title: CONCORD: Concept-Oriented Graph-Coupled Dynamics for Forecasting

Authors: Hongwei Ma; Junbin Gao; Dai Shi; Minh-Ngoc Tran.

Version: 0.1.0 (GitHub tag `v0.1.0`).

Version DOI: https://doi.org/10.5281/zenodo.24504658

Release date: 24 November 2026.

## Source identity

This release preserves the source code, configurations, preprocessing, training,
evaluation, tests and dependencies at commit
`6e79895c3cb93c307883835dd786c7abba967683`, the architectural reference identified
in the manuscript. It includes the baseline-provenance documentation introduced
at `6630366992cea1168f437f0fcdb2280bdd020456`; that commit did not change the model
or experiment files. Release preparation adds LICENSE, CITATION.cff and this
file, and appends citation/licensing information to README.md.

The GitHub release and Zenodo record identify the exact release commit. The
Zenodo ZIP is exported directly from that commit, excluding Git history and
untracked local files. Third-party benchmark files already tracked in the
repository are retained without relicensing them.

The default execution path uses a symmetric graph. Directed-graph controls in
the manuscript are specified separately and are not represented as that default
execution path. The provenance documentation distinguishes author-confirmed
reruns from published comparison tables and inspected upstream implementations.

## Archiving and rights

The archive is deposited manually with its custom license selected explicitly.
Automatic Zenodo ingestion of GitHub releases is disabled for this repository
to prevent a default license from replacing those terms. Future versions should
be deposited as new versions of the same Zenodo record, with the license checked
before publication.

See LICENSE for the permission for free noncommercial research, reproduction and
evaluation, and the separate written agreement required for commercial use.
Third-party datasets and dependencies remain under their respective terms.
