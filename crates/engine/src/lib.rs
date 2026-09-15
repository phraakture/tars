//! Core agent loop and model provider abstraction for tars.
//!
//! The engine is a leaf: it never imports the server. It turns a `Context`
//! (system prompt + messages + tools) into a stream of model events, dispatches
//! tool calls through a `ToolExecutor`, and loops until the model stops using
//! tools.

mod provider;
pub use provider::{EventReceiver, EventSender, Provider, ProviderRegistry, STREAM_CAPACITY};
