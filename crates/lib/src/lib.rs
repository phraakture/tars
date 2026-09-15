//! The tars server daemon.
//!
//! Owns SQLite persistence, the Unix-socket request dispatcher, the agent
//! runner that bridges `Request::Chat` to an engine loop, and the subprocess
//! plugin manager. Also bundles the `tars worker` plugin binary.

pub mod db;
pub mod server;
