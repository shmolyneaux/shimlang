/**
 * GC mark-phase microbenchmark.
 *
 * Loads a script, executes it once to build a live object graph, then times the
 * GC mark phase over that graph repeatedly (see `Interpreter::bench_gc_mark`).
 * Marking frees nothing, so the live set is identical on every pass.
 *
 * Usage: gc_bench <script.shm> [iterations=200] [warmup=3]
 * Prints the per-iteration mark time (ms) to stderr as `<per_iter_ms> <script>`,
 * keeping stdout free for the script's own output.
 */
use std::process::ExitCode;

use shimlang::*;

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: gc_bench <script.shm> [iterations] [warmup]")?;
    let iterations: u32 = args.next().map(|s| s.parse().unwrap()).unwrap_or(200);
    let warmup: u32 = args.next().map(|s| s.parse().unwrap()).unwrap_or(3);

    let contents = std::fs::read(&path).map_err(|e| format!("{path}: {e}"))?;

    let ast = shimlang::ast_from_text(&contents).map_err(|msg| format!("Parse Error:\n{msg}"))?;
    let program = shimlang::compile_ast(&ast)?;
    let mut interpreter = shimlang::Interpreter::create(&Config::default(), program);

    interpreter.execute()?;

    let elapsed = interpreter.bench_gc_mark(warmup, iterations);
    let per_iter_ms = elapsed.as_secs_f64() * 1000.0 / (iterations as f64);
    eprintln!("{per_iter_ms:.4} {path}");

    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("{msg}");
            ExitCode::from(1)
        }
    }
}
