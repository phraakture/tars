use super::{ToolDef, ToolOutput};
use base64::Engine;
use tars_base::{CancelToken, ImageContent, TextContent, Tool, ToolResultContent};

pub const MAX_PATHS: usize = 20;
pub const MAX_TOTAL_BYTES: usize = 256 * 1024;
pub const MAX_IMAGE_BYTES: usize = 5 * 1024 * 1024;
pub const MAX_TOTAL_IMAGE_BYTES: usize = 20 * 1024 * 1024;

fn image_mime_from_path(path: &str) -> Option<&'static str> {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())?;
    match ext.as_str() {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        _ => None,
    }
}

fn mime_label(mime: &str) -> &'static str {
    match mime {
        "image/png" => "PNG",
        "image/jpeg" => "JPEG",
        "image/gif" => "GIF",
        "image/webp" => "WEBP",
        _ => "image",
    }
}

pub fn tool_def() -> ToolDef {
    ToolDef {
        tool: Tool {
            name: "read".into(),
            description:
                "Read the contents of one or more files. Supports offset/limit for large files (applied per file). Each line in the body is prefixed with `<hash>§` — a stable per-line anchor (FNV-1a 8 hex; `.n` suffix for duplicate lines) you can use with the `edit` tool's anchor shape."
                    .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": MAX_PATHS,
                        "description": "Paths to the files to read (1–20)."
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from (1-indexed). Applied to each file independently."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read per file."
                    }
                },
                "required": ["paths"]
            }),
        },
        execute: Box::new(execute),
        prepare_arguments: Some(Box::new(prepare_arguments)),
    }
}

fn prepare_arguments(mut args: serde_json::Value) -> serde_json::Value {
    let Some(obj) = args.as_object_mut() else {
        return args;
    };
    if obj.contains_key("paths") {
        return args;
    }
    let Some(path_val) = obj.get("path") else {
        return args;
    };
    if !path_val.is_string() {
        return args;
    }
    let Some(path_val) = obj.remove("path") else {
        return args;
    };
    obj.insert(
        "paths".to_string(),
        serde_json::Value::Array(vec![path_val]),
    );
    args
}

enum FileBody {
    Text(String),
    Image {
        data_b64: String,
        mime: &'static str,
        raw_bytes: usize,
        args_ignored: bool,
    },
    Error(String),
}

struct FileRead {
    path_str: String,
    body: FileBody,
    total_lines: usize,
    bytes: usize,
    range: Option<(usize, usize)>,
}

fn read_one(
    cwd: &str,
    path_str: &str,
    offset: usize,
    limit: Option<usize>,
    remaining_bytes: usize,
) -> FileRead {
    let path = super::resolve_path(cwd, path_str);

    if let Some(mime) = image_mime_from_path(path_str) {
        let args_ignored = offset != 1 || limit.is_some();
        let meta = match std::fs::metadata(&path) {
            Ok(m) => m,
            Err(e) => {
                return FileRead {
                    path_str: path_str.to_string(),
                    body: FileBody::Error(format!("failed to read {}: {}", path.display(), e)),
                    total_lines: 0,
                    bytes: 0,
                    range: None,
                };
            }
        };
        let raw_bytes = meta.len() as usize;
        if raw_bytes > MAX_IMAGE_BYTES {
            return FileRead {
                path_str: path_str.to_string(),
                body: FileBody::Error(format!(
                    "image {} is {} bytes, exceeds per-image cap of {} bytes",
                    path.display(),
                    raw_bytes,
                    MAX_IMAGE_BYTES,
                )),
                total_lines: 0,
                bytes: 0,
                range: None,
            };
        }
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                return FileRead {
                    path_str: path_str.to_string(),
                    body: FileBody::Error(format!("failed to read {}: {}", path.display(), e)),
                    total_lines: 0,
                    bytes: 0,
                    range: None,
                };
            }
        };
        let data_b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
        let charge = data_b64.len();
        if charge > remaining_bytes {
            return FileRead {
                path_str: path_str.to_string(),
                body: FileBody::Error(format!(
                    "image {} ({} bytes base64) exceeds remaining image budget for this call",
                    path.display(),
                    charge,
                )),
                total_lines: 0,
                bytes: 0,
                range: None,
            };
        }
        return FileRead {
            path_str: path_str.to_string(),
            body: FileBody::Image {
                data_b64,
                mime,
                raw_bytes,
                args_ignored,
            },
            total_lines: 0,
            bytes: charge,
            range: None,
        };
    }

    let content = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(e) => {
            return FileRead {
                path_str: path_str.to_string(),
                body: FileBody::Error(format!("failed to read {}: {}", path.display(), e)),
                total_lines: 0,
                bytes: 0,
                range: None,
            };
        }
    };

    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len();
    let start = (offset.max(1) - 1).min(total);
    let end = match limit {
        Some(l) => (start + l).min(total),
        None => total,
    };

    let all_anchors = super::line_hash::hash_lines_with_disambiguators(&lines);
    let selected = &lines[start..end];
    let selected_anchors = &all_anchors[start..end];
    let mut body = super::line_hash::format_hashed(selected, selected_anchors);

    if end < total {
        body.push_str(&format!(
            "\n\n[{} more lines in file. Use offset={} to continue.]",
            total - end,
            end + 1,
        ));
    }

    let bytes = body.len();
    let (body, bytes) = if bytes > remaining_bytes {
        let mut cut = remaining_bytes.min(body.len());
        while cut > 0 && !body.is_char_boundary(cut) {
            cut -= 1;
        }
        let mut truncated: String = body[..cut].to_string();
        truncated.push_str("\n\n[truncated: per-call byte cap reached]");
        let trunc_bytes = truncated.len();
        (truncated, trunc_bytes)
    } else {
        (body, bytes)
    };

    let range = if start == 0 && end == total {
        None
    } else {
        Some((start + 1, end))
    };

    FileRead {
        path_str: path_str.to_string(),
        body: FileBody::Text(body),
        total_lines: total,
        bytes,
        range,
    }
}

fn execute(args: serde_json::Value, cwd: &str, _cancel: &CancelToken) -> ToolOutput {
    let Some(paths_val) = args.get("paths") else {
        return ToolOutput::error("missing 'paths' argument (expected an array of strings)");
    };
    let Some(paths_arr) = paths_val.as_array() else {
        return ToolOutput::error("'paths' must be an array of strings");
    };
    if paths_arr.is_empty() {
        return ToolOutput::error("'paths' array is empty — provide at least one path");
    }
    if paths_arr.len() > MAX_PATHS {
        return ToolOutput::error(format!(
            "too many paths: {} (max {})",
            paths_arr.len(),
            MAX_PATHS
        ));
    }

    let mut paths: Vec<String> = Vec::with_capacity(paths_arr.len());
    for (i, v) in paths_arr.iter().enumerate() {
        let Some(s) = v.as_str() else {
            return ToolOutput::error(format!("paths[{}] is not a string", i));
        };
        paths.push(s.to_string());
    }

    let offset = args
        .get("offset")
        .and_then(|o| o.as_u64())
        .unwrap_or(1)
        .max(1) as usize;
    let limit = args
        .get("limit")
        .and_then(|l| l.as_u64())
        .map(|l| l as usize);

    let n_paths = paths.len();
    let mut results: Vec<FileRead> = Vec::with_capacity(n_paths);
    let mut used_text_bytes: usize = 0;
    let mut used_image_bytes: usize = 0;
    let mut cap_reached = false;
    let mut skipped_after_cap: usize = 0;

    for path_str in &paths {
        if cap_reached {
            skipped_after_cap += 1;
            continue;
        }
        let is_image = image_mime_from_path(path_str).is_some();
        let remaining = if is_image {
            MAX_TOTAL_IMAGE_BYTES.saturating_sub(used_image_bytes)
        } else {
            MAX_TOTAL_BYTES.saturating_sub(used_text_bytes)
        };
        let fr = read_one(cwd, path_str, offset, limit, remaining);
        match &fr.body {
            FileBody::Image { .. } => {
                used_image_bytes = used_image_bytes.saturating_add(fr.bytes);
                if used_image_bytes >= MAX_TOTAL_IMAGE_BYTES {
                    cap_reached = true;
                }
            }
            FileBody::Text(_) => {
                used_text_bytes = used_text_bytes.saturating_add(fr.bytes);
                if used_text_bytes >= MAX_TOTAL_BYTES {
                    cap_reached = true;
                }
            }
            FileBody::Error(_) => {}
        }
        results.push(fr);
    }

    if n_paths == 1 {
        let fr = results.into_iter().next().expect("one result for one path");
        match fr.body {
            FileBody::Text(body) => {
                let summary = match fr.range {
                    None => format!("read: {} ({} lines)", fr.path_str, fr.total_lines),
                    Some((s, e)) => format!(
                        "read: {} (lines {}-{}, {} total)",
                        fr.path_str, s, e, fr.total_lines
                    ),
                };
                ToolOutput::text(body).with_summary(summary)
            }
            FileBody::Image {
                data_b64,
                mime,
                raw_bytes,
                args_ignored,
            } => {
                let summary = format!(
                    "image: {} ({}, {} KB)",
                    fr.path_str,
                    mime_label(mime),
                    raw_bytes / 1024
                );
                let mut out = ToolOutput::image(data_b64, mime.into()).with_summary(summary);
                if args_ignored {
                    out.content.insert(
                        0,
                        ToolResultContent::Text(TextContent {
                            text: "[note: offset/limit ignored for image inputs]".into(),
                            text_signature: None,
                        }),
                    );
                }
                out
            }
            FileBody::Error(msg) => ToolOutput::error(msg),
        }
    } else {
        let mut content: Vec<ToolResultContent> = Vec::new();
        let mut buf = String::new();
        let flush = |buf: &mut String, content: &mut Vec<ToolResultContent>| {
            if !buf.is_empty() {
                content.push(ToolResultContent::Text(TextContent {
                    text: std::mem::take(buf),
                    text_signature: None,
                }));
            }
        };

        let mut total_lines = 0usize;
        let mut errors = 0usize;
        let mut successes = 0usize;
        let mut images = 0usize;
        for (i, fr) in results.iter().enumerate() {
            if i > 0 {
                buf.push_str("\n\n");
            }
            buf.push_str(&format!("===== {} =====\n", fr.path_str));
            match &fr.body {
                FileBody::Text(body) => {
                    buf.push_str(body);
                    total_lines += fr.total_lines;
                    successes += 1;
                }
                FileBody::Image {
                    data_b64,
                    mime,
                    raw_bytes,
                    args_ignored,
                } => {
                    buf.push_str(&format!(
                        "image: {} ({}, {} KB)\n",
                        fr.path_str,
                        mime_label(mime),
                        raw_bytes / 1024
                    ));
                    if *args_ignored {
                        buf.push_str("[note: offset/limit ignored for image inputs]\n");
                    }
                    flush(&mut buf, &mut content);
                    content.push(ToolResultContent::Image(ImageContent {
                        data: data_b64.clone(),
                        mime_type: (*mime).into(),
                    }));
                    successes += 1;
                    images += 1;
                }
                FileBody::Error(msg) => {
                    buf.push_str("error: ");
                    buf.push_str(msg);
                    errors += 1;
                }
            }
        }
        if cap_reached && skipped_after_cap > 0 {
            buf.push_str(&format!(
                "\n\n[truncated: byte cap reached, {} file(s) not read]",
                skipped_after_cap
            ));
        }
        flush(&mut buf, &mut content);

        let mut summary = format!("read: {} files ({} total lines", n_paths, total_lines);
        if images > 0 {
            summary.push_str(&format!(
                ", {} image{}",
                images,
                if images == 1 { "" } else { "s" }
            ));
        }
        if errors > 0 {
            summary.push_str(&format!(
                ", {} error{}",
                errors,
                if errors == 1 { "" } else { "s" }
            ));
        }
        summary.push(')');

        ToolOutput {
            content,
            is_error: successes == 0,
            summary: Some(summary),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tars_base::CancelToken;

    fn write_file(dir: &std::path::Path, name: &str, content: &str) -> std::path::PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, content).expect("write test file");
        p
    }

    #[test]
    fn single_file_hashed() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = write_file(dir.path(), "f.txt", "line1\nline2\n");
        let out = execute(
            serde_json::json!({"paths": [p.to_str().unwrap()]}),
            "/tmp",
            &CancelToken::new(),
        );
        assert!(!out.is_error);
        let text = out.content[0].text().to_string();
        let h1 = crate::tools::line_hash::fnv1a_8hex("line1");
        let h2 = crate::tools::line_hash::fnv1a_8hex("line2");
        assert_eq!(text, format!("{h1}§line1\n{h2}§line2"));
    }

    #[test]
    fn offset_limit() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = write_file(dir.path(), "f.txt", "a\nb\nc\nd\n");
        let out = execute(
            serde_json::json!({"paths": [p.to_str().unwrap()], "offset": 2, "limit": 2}),
            "/tmp",
            &CancelToken::new(),
        );
        assert!(!out.is_error);
        let text = out.content[0].text().to_string();
        let hb = crate::tools::line_hash::fnv1a_8hex("b");
        let hc = crate::tools::line_hash::fnv1a_8hex("c");
        assert!(text.contains(&format!("{hb}§b")));
        assert!(text.contains(&format!("{hc}§c")));
        assert!(!text.contains("§a"));
    }
}
