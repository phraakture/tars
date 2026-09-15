use async_trait::async_trait;
use tars_base::{CancelToken, ToolCall, ToolResultMessage};

#[async_trait]
pub trait ToolExecutor: Send {
    async fn execute(
        &mut self,
        tool_call: &ToolCall,
        output_tx: &tokio::sync::mpsc::Sender<String>,
        cancel: &CancelToken,
    ) -> tars_base::Result<ToolResultMessage>;
}
