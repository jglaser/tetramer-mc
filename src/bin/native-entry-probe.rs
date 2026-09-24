//! JSONL parity probe. Requires an explicit caller assertion of hard validity.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    io::{self, BufRead, Write},
    path::PathBuf,
};
use tetramer_mc::{
    math::Pose,
    native_entry::{AnchorMatches, CompleteNativeEntry, NativeDecision},
};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    definition: PathBuf,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    id: Value,
    #[serde(default)]
    anchor: Option<Pose>,
    moving: Pose,
    #[serde(default)]
    hard_valid: bool,
}
fn evaluate(model: &CompleteNativeEntry, request: Request) -> Result<Value> {
    ensure!(
        request.hard_valid,
        "caller must assert hard_valid=true after whole-body hard/wall checks; the native observer cannot certify them"
    );
    let decision = if let Some(anchor) = request.anchor {
        let matches = model.classify_pair(anchor, request.moving)?;
        let native_any = !matches.is_empty();
        NativeDecision {
            native_any,
            matched_anchor_indices: if native_any { vec![0] } else { vec![] },
            per_anchor: vec![AnchorMatches {
                anchor_index: 0,
                matched_motif_ids: matches.iter().map(|m| m.motif_id).collect(),
                matches,
            }],
        }
    } else {
        model.classify(request.moving)?
    };
    let mut result = serde_json::to_value(decision)?;
    result
        .as_object_mut()
        .unwrap()
        .insert("id".to_owned(), request.id);
    Ok(result)
}
fn main() -> Result<()> {
    let args = Args::parse();
    let model = CompleteNativeEntry::load(&args.definition)?;
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    for (line_index, line) in stdin.lock().lines().enumerate() {
        let line = line.context("read native-entry probe JSONL")?;
        let value = serde_json::from_str::<Value>(&line);
        let id = value
            .as_ref()
            .ok()
            .and_then(|v| v.get("id"))
            .cloned()
            .unwrap_or(Value::Null);
        let result = value
            .context("parse probe JSON")
            .and_then(|v| serde_json::from_value::<Request>(v).context("parse probe request"))
            .and_then(|request| evaluate(&model, request));
        let output = match result {
            Ok(value) => value,
            Err(error) => json!({"id":id,"error":format!("line {}: {error:#}",line_index+1)}),
        };
        serde_json::to_writer(&mut stdout, &output)?;
        writeln!(stdout)?;
        stdout.flush()?;
    }
    Ok(())
}
