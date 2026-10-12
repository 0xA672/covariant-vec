use std::process::Command;

fn main() {
    // `core::error::Error` is stable since Rust 1.81, but the crate's MSRV is
    // 1.80 (see `rust-version` in Cargo.toml). The impl is therefore gated on
    // a custom cfg set here, from the *building* compiler's version, instead
    // of bumping the MSRV.
    println!("cargo:rustc-check-cfg=cfg(has_error_trait)");
    println!("cargo:rerun-if-changed=build.rs");

    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let Ok(out) = Command::new(&rustc).arg("--version").output() else {
        // Do not fail the build over a version probe, but do not let the
        // missing `Error` impl be silent either.
        println!(
            "cargo:warning=build.rs could not run `{rustc} --version`; assuming a compiler \
             older than 1.81, so `TryReserveError` will not implement `core::error::Error`"
        );
        return;
    };
    // `rustc 1.101.0-nightly (…)/rustc 1.80.1 (…)/rustc 1.81.0 (…)`
    let version = String::from_utf8_lossy(&out.stdout);
    let numbers: String = version
        .strip_prefix("rustc ")
        .unwrap_or("")
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    let mut parts = numbers.split('.');
    let major: u64 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let minor: u64 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    if (major, minor) >= (1, 81) {
        println!("cargo:rustc-cfg=has_error_trait");
    } else if (major, minor) == (0, 0) {
        // The probe ran but produced nothing parseable (e.g. a wrapper that
        // prints extra output). Same conservative default, but say why.
        println!(
            "cargo:warning=build.rs could not parse a version out of `{rustc} --version` \
             output {version:?}; assuming a compiler older than 1.81, so `TryReserveError` \
             will not implement `core::error::Error`"
        );
    }
}
