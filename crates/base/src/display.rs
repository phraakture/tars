//! Human-readable formatting for stats and token counts.
//!
//! Deliberately kept **out** of the protocol module: these are presentation
//! concerns for the CLI/TUI, not part of the wire format.

use crate::protocol::{SessionStats, TokenStats};

/// Format a token count for display: `1234` → `"1.2K"`, `1234567` → `"1.2M"`.
pub fn format_tokens(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

/// Format session stats as a compact footer:
/// `↑12.0K ↓81.0K R18.0M W353.0K $13.434 18.4%/200.0K`
///
/// Only non-zero parts are emitted, so a fresh session renders as `""`.
#[allow(clippy::cast_precision_loss)]
pub fn format_stats(stats: &SessionStats) -> String {
    let mut parts = Vec::new();

    if stats.tokens.input > 0 {
        parts.push(format!("↑{}", format_tokens(stats.tokens.input)));
    }
    if stats.tokens.output > 0 {
        parts.push(format!("↓{}", format_tokens(stats.tokens.output)));
    }
    if stats.tokens.cache_read > 0 {
        parts.push(format!("R{}", format_tokens(stats.tokens.cache_read)));
    }
    if stats.tokens.cache_write > 0 {
        parts.push(format!("W{}", format_tokens(stats.tokens.cache_write)));
    }
    if stats.cost > 0.0 {
        parts.push(format!("${:.3}", stats.cost));
    }
    if stats.context_window > 0 {
        let ctx = match stats.context_tokens {
            Some(t) => {
                let pct = (t as f64 / stats.context_window as f64) * 100.0;
                format!("{:.1}%/{}", pct, format_tokens(stats.context_window))
            }
            None => format!("?/{}", format_tokens(stats.context_window)),
        };
        parts.push(ctx);
    }

    parts.join(" ")
}

/// Format a `TokenStats` for display: `"↑1.2K ↓900"`.
pub fn format_token_stats(tokens: &TokenStats) -> String {
    let mut parts = Vec::new();
    if tokens.input > 0 {
        parts.push(format!("↑{}", format_tokens(tokens.input)));
    }
    if tokens.output > 0 {
        parts.push(format!("↓{}", format_tokens(tokens.output)));
    }
    if tokens.cache_read > 0 {
        parts.push(format!("R{}", format_tokens(tokens.cache_read)));
    }
    if tokens.cache_write > 0 {
        parts.push(format!("W{}", format_tokens(tokens.cache_write)));
    }
    parts.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_tokens_units() {
        assert_eq!(format_tokens(0), "0");
        assert_eq!(format_tokens(999), "999");
        assert_eq!(format_tokens(1_000), "1.0K");
        assert_eq!(format_tokens(12_345), "12.3K");
        assert_eq!(format_tokens(999_999), "1000.0K");
        assert_eq!(format_tokens(1_000_000), "1.0M");
        assert_eq!(format_tokens(18_500_000), "18.5M");
    }

    #[test]
    fn format_stats_empty() {
        assert_eq!(format_stats(&SessionStats::default()), "");
    }

    #[test]
    fn format_stats_basic() {
        let stats = SessionStats {
            tokens: TokenStats {
                input: 12_000,
                output: 81_000,
                cache_read: 18_000_000,
                cache_write: 353_000,
            },
            cost: 13.434,
            context_window: 200_000,
            context_tokens: Some(36_800),
            ..Default::default()
        };
        let s = format_stats(&stats);
        assert!(s.contains("↑12.0K"), "got: {s}");
        assert!(s.contains("↓81.0K"), "got: {s}");
        assert!(s.contains("R18.0M"), "got: {s}");
        assert!(s.contains("W353.0K"), "got: {s}");
        assert!(s.contains("$13.434"), "got: {s}");
        assert!(s.contains("18.4%/200.0K"), "got: {s}");
    }

    #[test]
    fn format_stats_unknown_context() {
        let stats = SessionStats {
            context_window: 200_000,
            context_tokens: None,
            ..Default::default()
        };
        let s = format_stats(&stats);
        assert!(s.contains("?/200.0K"), "got: {s}");
    }

    #[test]
    fn format_token_stats_mixed() {
        let t = TokenStats {
            input: 1_200,
            output: 900,
            cache_read: 0,
            cache_write: 3_500,
        };
        let s = format_token_stats(&t);
        assert!(s.contains("↑1.2K"), "got: {s}");
        assert!(s.contains("↓900"), "got: {s}");
        assert!(s.contains("W3.5K"), "got: {s}");
        assert!(!s.contains('R'), "got: {s}");
    }
}
