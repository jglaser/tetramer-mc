//! Embed the exact application sources used for this executable, so runtime
//! provenance does not accidentally archive a newer working-tree revision.
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
};

fn rust_files(directory: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(directory).expect("source directory") {
        let path = entry.expect("source entry").path();
        if path.is_dir() {
            rust_files(&path, files);
        } else if path.extension().is_some_and(|e| e == "rs") {
            files.push(path);
        }
    }
}
fn main() {
    let mut files = vec![
        PathBuf::from("Cargo.toml"),
        PathBuf::from("Cargo.lock"),
        PathBuf::from("build.rs"),
        PathBuf::from("vendor/README.md"),
    ];
    rust_files(Path::new("src"), &mut files);
    files.sort();
    let mut sources = BTreeMap::new();
    for path in files {
        println!("cargo:rerun-if-changed={}", path.display());
        let bytes = fs::read(&path).expect("read source");
        sources.insert(path.to_string_lossy().to_string(),serde_json::json!({"sha256":format!("{:x}",Sha256::digest(&bytes)),"text":String::from_utf8(bytes).expect("UTF-8 source")}));
    }
    println!("cargo:rerun-if-changed=src");
    for git_rel in ["HEAD", "logs/HEAD"] {
        if let Ok(out) = std::process::Command::new("git")
            .args(["rev-parse", "--git-path", git_rel])
            .output()
            && out.status.success()
        {
            let p = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !p.is_empty() {
                println!("cargo:rerun-if-changed={p}");
            }
        }
    }
    let git_hash = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let git_dirty = std::process::Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);
    println!("cargo:rustc-env=GIT_HASH={git_hash}");
    println!("cargo:rustc-env=GIT_DIRTY={git_dirty}");
    let bundle = serde_json::json!({"schema":1,"git_hash":git_hash,"git_dirty":git_dirty,"files":sources,"third_party":"Vendored hoomd-rs revision recorded in vendor/README.md; registry versions in Cargo.lock"});
    fs::write(
        PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("source-bundle.json"),
        serde_json::to_vec_pretty(&bundle).unwrap(),
    )
    .unwrap();
}
