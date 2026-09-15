use async_trait::async_trait;
use tars_base::{CancelToken, StreamEvent, ToolCall, ToolResultMessage};

#[async_trait]
pub trait ToolExecutor: Send {
    async fn execute(
        &mut self,
        tool_call: &ToolCall,
        output_tx: &tokio::sync::mpsc::Sender<String>,
        cancel: &CancelToken,
    ) -> tars_base::Result<ToolResultMessage>;
}

/// Spawn a forwarder that relays `String` output deltas from `output_rx` into
/// `event_tx` as `StreamEvent::ToolOutputDelta` for `tool_call_id`.
///
/// Returns a `JoinHandle` that resolves when `output_rx` is closed and all
/// pending deltas have been forwarded. Callers should `await` it after the
/// tool execution completes (after dropping `output_tx`).
pub fn spawn_output_forwarder(
    mut output_rx: tokio::sync::mpsc::Receiver<String>,
    tool_call_id: String,
    event_tx: tokio::sync::mpsc::Sender<StreamEvent>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        while let Some(delta) = output_rx.recv().await {
            let _ = event_tx
                .send(StreamEvent::ToolOutputDelta {
                    tool_call_id: tool_call_id.clone(),
                    delta,
                })
                .await;
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StreamingExecutor;

    #[async_trait]
    impl ToolExecutor for StreamingExecutor {
        async fn execute(
            &mut self,
            tool_call: &ToolCall,
            output_tx: &tokio::sync::mpsc::Sender<String>,
            _cancel: &CancelToken,
        ) -> tars_base::Result<ToolResultMessage> {
            let _ = output_tx.send("chunk1\n".to_string()).await;
            let _ = output_tx.send("chunk2\n".to_string()).await;
            Ok(ToolResultMessage::success(
                tool_call.id.clone(),
                tool_call.name.clone(),
                "done",
            ))
        }
    }

    #[tokio::test]
    async fn forwarder_relays_deltas_then_result() {
        let (output_tx, output_rx) = tokio::sync::mpsc::channel(8);
        let (event_tx, mut event_rx) = tokio::sync::mpsc::channel(16);
        let forwarder = spawn_output_forwarder(output_rx, "tc1".into(), event_tx.clone());

        let mut exec = StreamingExecutor;
        let tc = ToolCall {
            id: "tc1".into(),
            name: "bash".into(),
            arguments: serde_json::json!({}),
        };
        let result = exec
            .execute(&tc, &output_tx, &CancelToken::new())
            .await
            .unwrap();
        drop(output_tx);
        forwarder.await.unwrap();

        let mut deltas = Vec::new();
        while let Ok(ev) = event_rx.try_recv() {
            if let StreamEvent::ToolOutputDelta { delta, .. } = ev {
                deltas.push(delta);
            }
        }
        assert_eq!(deltas, vec!["chunk1\n", "chunk2\n"]);
        assert_eq!(result.text(), "done");
        assert!(!result.is_error);
    }
}
