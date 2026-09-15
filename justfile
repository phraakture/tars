# tars development commands
# Install just: cargo install just

# Run all tests
test:
    cargo test --workspace

# Run tests with logging
test-verbose:
    RUST_LOG=info cargo test --workspace

# Check formatting
fmt-check:
    cargo fmt --check

# Apply formatting
fmt:
    cargo fmt

# Run clippy
clippy:
    cargo clippy --workspace -- -D warnings

# Build everything
build:
    cargo build --workspace

# Build worker binary (needed for integration tests)
build-worker:
    cargo build -p tars-worker

# Build release
build-release:
    cargo build --release --workspace

# Run all CI checks
ci: fmt-check clippy test

# Clean build artifacts
clean:
    cargo clean

# Check workspace dependencies
check-deps:
    cargo outdated --workspace 2>/dev/null || echo "cargo-outdated not installed"

# Generate lockfile
lock:
    cargo generate-lockfile

# Check packaging
package-check:
    cargo package --workspace --allow-dirty 2>&1 | head -20

# Summary
summary:
    @echo "=== tars workspace ==="
    @echo "Crates:"
    @grep -r "^\[package\]" crates/*/Cargo.toml | sed 's/.*name = "\([^"]*\)".*/  \1/'
    @echo ""
    @echo "Test count:"
    @cargo test --workspace 2>&1 | grep "^test result:" | grep -v "0 passed" | wc -l
