# Changelog

All notable changes to this project are documented in this file.

## [0.2.0] - 2026-06-16

### Added
- hardened AL-1.0 SDK primitives:
  - typed models (`AttributionReceipt`, `SourceRegistryEntry`)
  - structured errors
  - tracing helpers (`SourceTagSidecar`, `DecodeStepLogger`)
  - registry + manifest utilities
  - receipt + manifest validators
- adapter suite:
  - `BaseAL10Adapter`
  - `PyTorchDecodeAdapter`
  - `HuggingFaceGenerateAdapter`
  - `VLLMDecodeAdapter`
- CLI expansion:
  - `init-registry`
  - `stamp-dataset`
  - `run-demo`
  - `validate-receipt`
  - `validate-manifests`
  - `init-plugin`
  - `make-receipt`
  - `serve-api`
  - `bench-smoke`
- local HTTP API sandbox with routes:
  - `GET /health`
  - `GET /v1/demo`
  - `POST /v1/receipt`
  - `POST /v1/validate-receipt`
- API hardening controls:
  - optional API key auth (`Authorization: Bearer` / `X-API-Key`)
  - optional per-IP in-memory rate limiting
- documentation additions:
  - quickstart
  - integrations
  - engineering guide
  - HTTP API guide
  - end-to-end app flow
  - quality/CI guide
- runnable examples:
  - PyTorch loop
  - Hugging Face wrapper
  - vLLM-style loop
  - performance smoke
  - end-to-end HTTP app
- CI and release automation:
  - Python matrix CI on PR/push
  - release workflow for build artifacts + tag releases
- testing expansion:
  - unit tests across adapters, validators, registry, CLI, server
  - golden receipt fixtures
  - randomized invariant/property tests
  - live HTTP route integration tests
