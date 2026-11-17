# jero-core

<p align="center">
  <a href="https://hook.lma1980.duckdns.org/jobs">
    <img src="https://hook.lma1980.duckdns.org/cards/LMA1980/jero-core/card.svg" alt="jero-core Factory Scorecard" width="520"/>
  </a>
</p>

<p align="center">
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-core/status.svg" alt="CI Status"/></a>
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-core/score.svg" alt="Quality Score"/></a>
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-core/coverage.svg" alt="Coverage"/></a>
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-core/benchmark.svg" alt="Benchmark"/></a>
</p>

______________________________________________________________________

Core traits, zero-copy tensor views, single-prompt execution context, and structured diagnostics for the SpacemiT K3 AI platform.

## Overview

`jero-core` defines the foundational abstractions shared across `jero-engine`, architecture recipe crates (`jero-arch-*`), and compute kernel crates (`jero-kernels-*`):

- **Quantization & Tensor Views**: `QuantType` (`F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`, `Q2_K`, `Q3_K`, `Q4_K_M`, `Q5_K`, `Q6_K`, `IQ4_NL`, `TQ1_0`, `TQ2_0`, `I2_S`, `MXFP4`), zero-copy `TensorView` with checked bounds calculation, and the `ModelWeights` trait.
- **Cycle-Exclusive Execution**: Single-prompt `ExecutionContext` and `ArchRecipe` (`prefill_step` / `decode_step`) enforcing strict prompt exclusivity and fixed context bounds.
- **Structured Diagnostics**: `K3Error` covering capability mismatch, insufficient lockable memory, context length exceeded, core affinity errors, and tensor bounds errors.

## Ecosystem & Documentation

- **Public Documentation Portal**: [`lma1980.github.io/model-runner`](https://github.com/LMA1980/lma1980.github.io/tree/main/model-runner) ([Reference Manual](https://github.com/LMA1980/lma1980.github.io/blob/main/model-runner/manual.md), [Crate Docs](https://github.com/LMA1980/lma1980.github.io/tree/main/model-runner/docs/crates))
- **Downstream Consumers**: [`jero-kernels-rvv`](https://github.com/LMA1980/jero-kernels-rvv), [`jero-tokenizer`](https://github.com/LMA1980/jero-tokenizer), [`jero-kv-cache`](https://github.com/LMA1980/jero-kv-cache), [`jero-engine`](https://github.com/LMA1980/jero-engine), [`jero-manager`](https://github.com/LMA1980/jero-manager), and the architecture recipe crates ([`jero-arch-synthetic`](https://github.com/LMA1980/jero-arch-synthetic), [`jero-arch-llama`](https://github.com/LMA1980/jero-arch-llama), [`jero-arch-mistral`](https://github.com/LMA1980/jero-arch-mistral), [`jero-arch-phi3`](https://github.com/LMA1980/jero-arch-phi3), [`jero-arch-bitnet`](https://github.com/LMA1980/jero-arch-bitnet), [`jero-arch-dory`](https://github.com/LMA1980/jero-arch-dory))

## License

Licensed under the GNU Lesser General Public License v3.0 with the LGPL-3.0 Linking Exception (`LGPL-3.0-only WITH LGPL-3.0-linking-exception`). See [LICENSE](LICENSE) for details.
