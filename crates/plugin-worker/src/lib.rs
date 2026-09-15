//! In-process worker tools for tars: the built-in tool implementations
//! (`bash`, `read`, `write`, `edit`) and their shared registry/dispatch.
//!
//! These same definitions are consumed by the in-process executor and by the
//! `tars worker` plugin binary.

pub mod tools;
pub mod worker;

pub use tools::{ToolDef, ToolOutput, default_tools, execute_tool, tool_schemas};
pub use worker::InProcessWorker;
