# Contributing to industrial_ts

Thank you for your interest in contributing to `industrial_ts`.

## License of contributions

By submitting a contribution to this repository, you agree that your contribution is provided under the same license as the project: the Apache License, Version 2.0.

Unless explicitly stated otherwise in writing, contributions intentionally submitted for inclusion in this project are licensed under Apache-2.0 without additional terms or conditions.

## Scope

This repository is intended for research and software development related to industrial time-series modeling, including recurrent models, ODE-based models, PatchTST-style models, imputation, forecasting, and future-time anomaly/event prediction.

Do not submit proprietary, confidential, export-controlled, or operationally sensitive industrial data.

## Data policy

Contributions must not include real industrial operating data unless the contributor has the legal right to disclose and license that data. Prefer synthetic, public, or properly anonymized datasets with an explicit license.

If datasets are added in the future, they should include a separate license file, preferably CC BY 4.0 or CC0 when appropriate.

## Academic context

This library is associated with the forthcoming article:

> Safety-Oriented ODEJump Adaptation for Future-Time Anomaly Prediction in Oil and Gas Compressor Systems

Please cite the project and related article when using this software in academic work. A `CITATION.cff` file is provided for citation metadata and may be updated after publication.

## Development guidelines

- Keep model wrappers compatible with the shared IndustrialTS API.
- Add or update tests when changing model behavior.
- Avoid breaking existing configuration names unless there is a clear migration path.
- Document new cost functions, heads, encoders, decoders, and prediction modes in the README.
- Keep examples free of confidential or proprietary data.
