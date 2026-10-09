# Changelog

## [1.0.1](https://github.com/anthony-spruyt/litellm-middleware/compare/v1.0.0...v1.0.1) (2026-10-09)


### Code Refactoring

* prepare for ruff PLR rules ([#33](https://github.com/anthony-spruyt/litellm-middleware/issues/33)) ([03839be](https://github.com/anthony-spruyt/litellm-middleware/commit/03839be85822ccbd53b6825fea3c23c88633b0f7))

## 1.0.0 (2026-10-06)


### ⚠ BREAKING CHANGES

* self-contained src-layout package with package-only image ([#7](https://github.com/anthony-spruyt/litellm-middleware/issues/7))

### Features

* build and release the package image through shared CI ([#9](https://github.com/anthony-spruyt/litellm-middleware/issues/9)) ([d1b9e64](https://github.com/anthony-spruyt/litellm-middleware/commit/d1b9e6449dfb907c62ab021ecaee236cd59c4498))
* **devcontainer:** add claude state management script ([d90e15d](https://github.com/anthony-spruyt/litellm-middleware/commit/d90e15d163248c3fa5815e6b963bb980f570b567))
* **devcontainer:** initialize devcontainer configuration ([921aa4a](https://github.com/anthony-spruyt/litellm-middleware/commit/921aa4ac0f695a54cebc90387cfe9ecea6d5f032))
* **hindsight:** fix memory chunking + tune extraction for Claude Code ([a60fe4d](https://github.com/anthony-spruyt/litellm-middleware/commit/a60fe4dff2481bde56a2a5877dc32bfc5a9bf83a)), closes [#2270](https://github.com/anthony-spruyt/litellm-middleware/issues/2270)
* **litellm:** add callback middleware pipeline ([f59ae3d](https://github.com/anthony-spruyt/litellm-middleware/commit/f59ae3d63c850c5f89148fe9a493e5ca0f75931a))
* **litellm:** mask credentials sent to model providers ([#3215](https://github.com/anthony-spruyt/litellm-middleware/issues/3215)) ([5ca310f](https://github.com/anthony-spruyt/litellm-middleware/commit/5ca310ff34aac8dd5fb51a2cdd089c13c658d3ff))
* **litellm:** proxy-side Hindsight memory callback for Claude Code ([b016513](https://github.com/anthony-spruyt/litellm-middleware/commit/b016513583671fdf1c35369e573e0362cb26af44)), closes [#1890](https://github.com/anthony-spruyt/litellm-middleware/issues/1890)
* **litellm:** share secret-masking fake map across replicas via Valkey ([#3238](https://github.com/anthony-spruyt/litellm-middleware/issues/3238)) ([9dc1ada](https://github.com/anthony-spruyt/litellm-middleware/commit/9dc1ada927a116ee5e52799e31504c59f09b4f0c))
* self-contained src-layout package with package-only image ([#7](https://github.com/anthony-spruyt/litellm-middleware/issues/7)) ([0259ea4](https://github.com/anthony-spruyt/litellm-middleware/commit/0259ea4f26e446b5409deac9e90e6b3a77b27aaa))


### Bug Fixes

* **hindsight:** consolidate CNP into network-policies, trim comments ([8b79c1b](https://github.com/anthony-spruyt/litellm-middleware/commit/8b79c1b2db5a496084647ee510c4f6754d3ab05e)), closes [#2270](https://github.com/anthony-spruyt/litellm-middleware/issues/2270)
* **litellm:** add chatgpt middleware telemetry ([ed511e0](https://github.com/anthony-spruyt/litellm-middleware/commit/ed511e0126016fb7be997f4f0f7e251580088557)), closes [#1960](https://github.com/anthony-spruyt/litellm-middleware/issues/1960) [#1961](https://github.com/anthony-spruyt/litellm-middleware/issues/1961)
* **litellm:** correct hindsight retain MemoryItem body shape ([30059cd](https://github.com/anthony-spruyt/litellm-middleware/commit/30059cd8632d349abc18df44f6252bfe590d3033)), closes [#1890](https://github.com/anthony-spruyt/litellm-middleware/issues/1890)
* **litellm:** force chatgpt responses streaming ([c9f6160](https://github.com/anthony-spruyt/litellm-middleware/commit/c9f61607ae764a8b1d0167376bacf7c3d5e574be)), closes [#1960](https://github.com/anthony-spruyt/litellm-middleware/issues/1960) [#1961](https://github.com/anthony-spruyt/litellm-middleware/issues/1961)
* **litellm:** hydrate gpt-5.5 retry count ([fe8c7e8](https://github.com/anthony-spruyt/litellm-middleware/commit/fe8c7e85530a39a7003f49fa6dbd1278e429f44f)), closes [#1960](https://github.com/anthony-spruyt/litellm-middleware/issues/1960)
* **litellm:** isolate chatgpt responses patch ([6c1d239](https://github.com/anthony-spruyt/litellm-middleware/commit/6c1d23902d0a5cf68edb5426948245acd3b12a2f)), closes [#1961](https://github.com/anthony-spruyt/litellm-middleware/issues/1961)
* **litellm:** patch routed chatgpt responses provider ([51b2839](https://github.com/anthony-spruyt/litellm-middleware/commit/51b2839d8c42e31d65379afec89557ae203f172e)), closes [#1961](https://github.com/anthony-spruyt/litellm-middleware/issues/1961)
* **litellm:** recover forced responses output ([d354bcb](https://github.com/anthony-spruyt/litellm-middleware/commit/d354bcbead7fe44ba0184ff5a8ad289d7305f2b0)), closes [#1961](https://github.com/anthony-spruyt/litellm-middleware/issues/1961)
* **litellm:** resolve hindsight retain bank from nested litellm_params ([5d63072](https://github.com/anthony-spruyt/litellm-middleware/commit/5d63072bfd15aa1de03522794d5f950b580066fb)), closes [#1890](https://github.com/anthony-spruyt/litellm-middleware/issues/1890)
* **litellm:** restore anthropic-ratelimit-unified-* response headers ([#3253](https://github.com/anthony-spruyt/litellm-middleware/issues/3253)) ([37d49c7](https://github.com/anthony-spruyt/litellm-middleware/commit/37d49c7c01a03a01c0a54c15d245b1cc47ab6d93)), closes [#3252](https://github.com/anthony-spruyt/litellm-middleware/issues/3252)
* **litellm:** strip system-reminders from hindsight recall+retain ([9217976](https://github.com/anthony-spruyt/litellm-middleware/commit/92179762f988e453309bb002be3c4c60ef3c88af)), closes [#2270](https://github.com/anthony-spruyt/litellm-middleware/issues/2270)
* **litellm:** try responses path for opus 4.8 ([9a437f5](https://github.com/anthony-spruyt/litellm-middleware/commit/9a437f5f7a4c383cfce5a7ceed99c7d2e2138e1b)), closes [#1961](https://github.com/anthony-spruyt/litellm-middleware/issues/1961)
* **nut-system:** grant shutdown-orchestrator node patch and pod drain RBAC ([f05a5f2](https://github.com/anthony-spruyt/litellm-middleware/commit/f05a5f23ad16264bccedd47601c21ee327e74860)), closes [#3181](https://github.com/anthony-spruyt/litellm-middleware/issues/3181) [#3182](https://github.com/anthony-spruyt/litellm-middleware/issues/3182)


### Code Refactoring

* **litellm:** give each proxy middleware its own sub-folder ([c87f3a7](https://github.com/anthony-spruyt/litellm-middleware/commit/c87f3a71e64a0cd894eb75e28ab944975460dade)), closes [#3257](https://github.com/anthony-spruyt/litellm-middleware/issues/3257)
* **litellm:** move chatgpt and hindsight under middleware/ ([653cd97](https://github.com/anthony-spruyt/litellm-middleware/commit/653cd97f3b0df997e21eea0052d2609da9a2f974)), closes [#3257](https://github.com/anthony-spruyt/litellm-middleware/issues/3257)
