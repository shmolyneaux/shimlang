use std::env;
use std::fs::File;
use std::io::Read;
use std::process::ExitCode;

use std::io;
use std::io::Write;

use shimlang::*;

#[derive(Debug, PartialEq)]
#[derive(Default)]
enum Command {
    Parse,
    #[default]
    Execute,
    Spans,
    Compile,
}


#[derive(Debug, Default)]
struct Args {
    // Scripts to execute/hot reload in order
    scripts: Vec<String>,
    gc: bool,
    script_on_command_line: bool,
    hot_reload: bool,
    command: Command,
}

/// An ordered list of script snapshots for a hot-reload session. Each snapshot
/// is a full version of the program; the reload driver runs one, calls its
/// `loop` (if defined), then swaps in the next while preserving interpreter
/// state between them.
#[derive(Debug, Default)]
struct HotReloadSession {
    // Populated by the CLI; consumed by the (not-yet-implemented) reload driver.
    #[allow(dead_code)]
    scripts: Vec<String>,
}

fn print_help() {
    println!("Usage: shm [OPTIONS] [FILE]...");
    println!();
    println!("Shimlang interpreter and compiler");
    println!();
    println!("Arguments:");
    println!("  [FILE]...           Script file to execute (or script content with -c).");
    println!("                      With --hot-reload, multiple files are run as a");
    println!("                      hot-reload session, each a successive snapshot.");
    println!();
    println!("Options:");
    println!(
        "  -c                  Treat positional argument as script content instead of file path"
    );
    println!("  --hot-reload        Run the given files as an ordered hot-reload session");
    println!("  --gc                Run garbage collector after execution");
    println!("  --parse             Parse the script and check syntax without execution");
    println!("  --spans             Display lexical spans (tokens) from the script");
    println!("  --compile           Compile and display the bytecode assembly");
    println!("  --help              Display this help message");
    println!();
    println!("If no FILE is provided, starts an interactive REPL.");
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();

    for (idx, arg) in env::args().enumerate() {
        // Skip the name of the executable, which is the first arg
        if idx == 0 {
            continue;
        } else if arg == "--help" || arg == "-h" {
            print_help();
            std::process::exit(0);
        } else if !arg.starts_with('-') {
            args.scripts.push(arg.clone());
        } else if arg == "--gc" {
            args.gc = true;
        } else if arg == "-c" {
            args.script_on_command_line = true;
        } else if arg == "--hot-reload" {
            args.hot_reload = true;
        } else if arg == "--parse" {
            if args.command != Command::default() {
                return Err(format!("Attempted to set command multiple times! {}", arg));
            }
            args.command = Command::Parse;
        } else if arg == "--spans" {
            if args.command != Command::default() {
                return Err(format!("Attempted to set command multiple times! {}", arg));
            }
            args.command = Command::Spans;
        } else if arg == "--compile" {
            if args.command != Command::default() {
                return Err(format!("Attempted to set command multiple times! {}", arg));
            }
            args.command = Command::Compile;
        } else {
            return Err(format!("Unknown args {}", arg));
        }
    }

    Ok(args)
}

fn run() -> Result<(), String> {
    let args = parse_args()?;

    if args.hot_reload {
        // Hot-reload session: the positional scripts are successive snapshots of
        // the program, run in order with interpreter state preserved between
        // them. Collect them for the reload driver. The actual reload/execution
        // loop is not wired up yet.
        let session = HotReloadSession {
            scripts: args.scripts.clone(),
        };
        let _ = session;
        // TODO: drive the hot-reload loop over `session.scripts`.
        return Ok(());
    }

    if args.hot_reload {
        if !matches!(args.command, Command::Execute) {
            return Err("--hot-reload not compatible with other flags".to_string());
        }

        let ast = shimlang::ast_from_text(b"").unwrap();
        let program = shimlang::compile_ast(&ast)?;
        let mut interpreter = shimlang::Interpreter::create(&Config::default(), program);

        for (idx, script) in args.scripts.iter().enumerate() {
            let mut file = File::open(&script).map_err(|e| format!("{:?}", e))?;
            let mut contents = Vec::new();
            file.read_to_end(&mut contents)
                .map_err(|e| format!("{:?}", e))?;
            match std::str::from_utf8(&contents) {
                Ok(_) => (),
                Err(e) => return Err(format!("Script is not utf8 {:?}", e)),
            }

            if idx == 0 {
                let ast = match shimlang::ast_from_text(&contents) {
                    Ok(ast) => ast,
                    Err(msg) => {
                        eprintln!("Parse Error:\n{msg}");
                        return Err("Failed to parse script".to_string());
                    }
                };
                let program = shimlang::compile_ast(&ast)?;
                // Create a new interpreter for the initial load
                interpreter = shimlang::Interpreter::create(&Config::default(), program);
                match interpreter.execute() {
                    Ok(_) => {
                        if args.gc {
                            interpreter.gc();
                        }
                    }
                    Err(msg) => {
                        eprintln!("{msg}");
                        return Err(String::new());
                    }
                };
            } else {
                interpreter.hot_reload_from_script(contents)?;
                // Execute the new script in the existing interpreter
                // Get the AST of the old script
                // - Find the _top-level_ struct definitions and variable assignments
                //   of the old script (maybe also functions so that function
                //   references that are passed around are updated too?)
                // - Create a map of old-struct to new-struct
                // - Transform all the top-level values (recursively) to match
                //   the new struct data shape
                // - Assign the transformed values to the new environment
            }

            match interpreter.get_from_root_env(b"loop") {
                Some(ShimValue::None) | None => (),
                // This can be any callable, we'll just return an Err if it's not
                Some(func) => {
                    let mut args = ArgBundle::new();
                    match func.call(&mut interpreter, &mut args)? {
                        CallResult::ReturnValue(_) => (),
                        CallResult::PC(pc, captured_scope) => {
                            let mut new_env = Environment::with_scope(captured_scope);
                            interpreter.execute_bytecode_extended(
                                &mut (pc as usize),
                                args,
                                &mut new_env,
                            )?;
                        }
                    }
                }
            }
        }
    }

    if let Some(pos) = args.scripts.into_iter().next() {
        let contents = if !args.script_on_command_line {
            let mut file = File::open(&pos).map_err(|e| format!("{:?}", e))?;
            let mut contents = Vec::new();
            file.read_to_end(&mut contents)
                .map_err(|e| format!("{:?}", e))?;
            contents
        } else {
            pos.into_bytes()
        };

        match std::str::from_utf8(&contents) {
            Ok(_) => (),
            Err(e) => return Err(format!("Script is not utf8 {:?}", e)),
        }

        match args.command {
            Command::Execute => {
                let ast = match shimlang::ast_from_text(&contents) {
                    Ok(ast) => ast,
                    Err(msg) => {
                        eprintln!("Parse Error:\n{msg}");
                        return Err("Failed to parse script".to_string());
                    }
                };
                let program = shimlang::compile_ast(&ast)?;
                let mut interpreter = shimlang::Interpreter::create(&Config::default(), program);
                match interpreter.execute() {
                    Ok(_) => {
                        if args.gc {
                            interpreter.gc();
                        }
                    }
                    Err(msg) => {
                        eprintln!("{msg}");
                        return Err(String::new());
                    }
                };
            }
            Command::Parse => {
                match shimlang::ast_from_text(&contents) {
                    Ok(program) => program,
                    Err(msg) => {
                        eprintln!("Parse Error:\n{msg}");
                        return Err("Failed to parse script".to_string());
                    }
                };
            }
            Command::Spans => {
                let tokens = shimlang::lex(&contents)?;
                for span in tokens.spans() {
                    println!(
                        "{}",
                        std::str::from_utf8(&contents[(span.start as usize)..(span.end as usize)])
                            .map_err(|e| format!("{:?}", e))?
                    );
                }
            }
            Command::Compile => {
                let ast = match shimlang::ast_from_text(&contents) {
                    Ok(ast) => ast,
                    Err(msg) => {
                        eprintln!("Parse Error:\n{msg}");
                        return Err("Failed to parse script".to_string());
                    }
                };
                let program = shimlang::compile_ast(&ast)?;
                shimlang::print_asm(&program.bytecode);
            }
        }
    } else {
        let ast = shimlang::ast_from_text(b"").unwrap();
        let program = shimlang::compile_ast(&ast)?;
        let mut interpreter = shimlang::Interpreter::create(&Config::default(), program);

        let mut pc = 0;

        loop {
            let mut input = String::new();

            print!(">>> ");
            io::stdout().flush().unwrap();

            io::stdin()
                .read_line(&mut input)
                .expect("Failed to read line");

            let ast = match shimlang::ast_from_text(&input.into_bytes()) {
                Ok(ast) => ast,
                Err(msg) => {
                    eprintln!("{msg}");
                    return Err(String::new());
                }
            };
            let program = shimlang::compile_ast(&ast)?;

            interpreter.append_program(program)?;
            match interpreter.execute_at(&mut pc) {
                Ok(shimlang::ShimValue::None) => (),
                Ok(val) => {
                    println!("{}", val.to_string(&mut interpreter));
                }
                Err(msg) => {
                    eprintln!("{msg}");
                    return Err(String::new());
                }
            };
        }
    }

    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            if !msg.is_empty() {
                eprintln!("{msg}");
            }
            ExitCode::from(1)
        }
    }
}
