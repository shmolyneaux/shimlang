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
    pos: Option<String>,
    gc: bool,
    script_on_command_line: bool,
    command: Command,
}

fn print_help() {
    println!("Usage: shm [OPTIONS] [FILE]");
    println!();
    println!("Shimlang interpreter and compiler");
    println!();
    println!("Arguments:");
    println!("  [FILE]              Script file to execute (or script content with -c)");
    println!();
    println!("Options:");
    println!(
        "  -c                  Treat positional argument as script content instead of file path"
    );
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
            if let Some(existing_positional_arg) = args.pos {
                return Err(format!(
                    "Found multiple positional arguments {} and {}",
                    existing_positional_arg, arg
                ));
            } else {
                args.pos = Some(arg.clone());
            }
        } else if arg == "--gc" {
            args.gc = true;
        } else if arg == "-c" {
            args.script_on_command_line = true;
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

    if let Some(pos) = args.pos {
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
                let mut env = shimlang::Environment::new_with_builtins(&mut interpreter);
                let mut pc = 0;
                match interpreter.execute_bytecode_extended(
                    &mut pc,
                    shimlang::ArgBundle::new(),
                    &mut env,
                ) {
                    Ok(_) => {
                        if args.gc {
                            interpreter.gc(&env);
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
        let mut env = shimlang::Environment::new_with_builtins(&mut interpreter);

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
            match interpreter.execute_bytecode_extended(
                &mut pc,
                shimlang::ArgBundle::new(),
                &mut env,
            ) {
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
