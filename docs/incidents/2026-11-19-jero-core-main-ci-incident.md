# Incident Report: `main` Branch CI Failure in `jero-core` (`RegressionDetected` & Rust `1.100.0` Clippy)

- **Date**: 2026-11-19
- **Repository**: `LMA1980/jero-core`
- **Affected Commit**: `3a55ddfb4a364afa44ac1883e8c9fcd7bbf04d76`
- **Tracking Issue**: [#81](https://github.com/LMA1980/jero-core/issues/81)
- **Resolution Branch**: `feat/fix-main-ci-regression-gate`

______________________________________________________________________

## 1. Summary & Impact

The `continuous integration` commit status on `LMA1980/jero-core` `main` (`3a55ddfb`) reported `failure` from `job-1794945907-3a55ddfb` (`2026-11-17T20:37:48Z`) with:

```text
RegressionDetected: verdict=Fail: 0 regressed benchmarks, 8 flaky/failed tests
```

All seven declarative CI stages (`fmt-check`, `clippy-check`, `audit-check`, `build-check`, `test-check`, `coverage-check`, `miri-check`) completed with `exit_code = 0`, including `miri-check` (`113 passed; 0 failed; 8 ignored`). Additionally, subsequent toolchain upgrades to Rust `1.100.0` introduced `clippy::needless_range_loop` diagnostics in `tests/empirical_challenge_us07_m1.rs`.

______________________________________________________________________

## 2. Root Cause Analysis

1. **Pre-Fix `mu-observable-hub` Regression Gate**:
   At `2026-11-17T20:37:48Z`, `mu-observable-hub` had not yet deployed the regression gate fix (`mu-observable-hub` PR #135) that excludes `#[cfg_attr(miri, ignore)]` `cargo-miri` rows (`status = 'ignored'`) from hard test failure counts and partitions `v_tests` by `test_binary`. Re-evaluating `3a55ddfb` against the updated live `mu-observable-hub` returns `verdict=Pass: 0 regressed benchmarks, 0 flaky/failed tests`.
1. **Toolchain Skew on `clippy::needless_range_loop` (Rust `1.100.0`)**:
   In `tests/empirical_challenge_us07_m1.rs` (lines 377 and 386), `for i in 0..10` and `for i in 10..20` indexed `deserialized.data[i]` directly, which triggers `clippy::needless_range_loop` under Rust `1.100.0` (`-D warnings`).
1. **How the Regression Reached `main`**:
   `3a55ddfb` merged during the window before `mu-observable-hub` PR #135 went live, leaving a stale post-stage `failure` status on `main` that could not be cleared without a new commit, while subsequent `x86_64` CI worker updates to Rust `1.100.0` surfaced the range-loop lint.

______________________________________________________________________

## 3. Remediation & Preventive Measures

### Remediation

1. Replaced the index range loops in `tests/empirical_challenge_us07_m1.rs` with slice iteration (`&deserialized.data[..10]` and `&deserialized.data[10..20]`) while preserving 100% of assertions.
1. Verified the full Pre-PR Local CI Gate Sequence (`cargo fmt`, `cargo +1.100.0 clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`, and `cargo +nightly miri test --workspace --lib`) prior to opening the PR.

### Preventive Measures

1. Keep `mu-observable-hub` regression-gate contract changes paired with immediate re-verification of any affected `main` pipelines after rollout.
1. Run `cargo clippy --workspace --all-targets -- -D warnings` against the active CI Rust toolchain (`1.100.0`) before merging remediation branches.
