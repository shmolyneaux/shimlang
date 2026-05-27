/**
 * This is for debugging GC issues. It runds the script in `contents` multiple
 * times and GC's inbetween.
 */
use std::process::ExitCode;

use shimlang::*;

fn run() -> Result<(), String> {
    let contents: &[u8] = br#"
        struct Point {
            x,
            y
        }

        let some_p0 = Point(0, 1);
        let some_p1 = Point(0, 1);
        let some_p2 = Point(0, 1);
        let some_p3 = Point(0, 1);

        fn rounds() {
            let d = dict();
            for i in 0..1000 {
                d[i] = Point(i*2, i*3);
            }
        }

        rounds();

        print("done");
    "#;

    let ast = match shimlang::ast_from_text(contents) {
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

    for _ in 0..3 {
        match interpreter.execute_bytecode_extended(&mut pc, shimlang::ArgBundle::new(), &mut env) {
            Ok(_) => {
                interpreter.gc(&env);
            }
            Err(msg) => {
                eprintln!("{msg}");
                return Err(String::new());
            }
        };
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
