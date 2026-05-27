use crate::lex::debug_u8s;
use crate::parse::*;
use crate::runtime::ShimValue;
use crate::runtime::format_float;

// NOTE: When adding new bytecodes, also update format_asm() below.
#[repr(u8)]
pub(crate) enum ByteCode {
    NoOp,
    Pad0,
    Pad1,
    Pad2,
    Pad3,
    Pad4,
    Pad5,
    Pad6,
    Pad7,
    Pad8,
    Pad9,
    UnpackArgs = 128,
    AssignArg,
    Pop,
    Add,
    Sub,
    Equal,
    NotEqual,
    Multiply,
    Divide,
    Modulus,
    GT,
    Gte,
    LT,
    Lte,
    In,
    Not,
    Negate,
    // And,
    // ToString,
    // ToBool,
    // JumpZ,
    // JumpNZ,
    Copy,
    CopyFrom,
    LiteralShimValue,
    LiteralString,
    LiteralNone,
    CreateFn,
    CreateList,
    CreateStruct,
    CreateTuple,
    UnpackTuple,
    VariableDeclaration,
    Assignment,
    VariableLoad,
    GetAttr,
    SetAttr,
    StartScope,
    StartCapturedScope,
    EndScope,
    LoopStart,
    LoopEnd,
    Stringify,
    Break,
    Continue,
    Call,
    AttrCall,
    Index,
    SetIndex,
    Return,
    Jmp,
    JmpUp,
    JmpNZ,
    JmpZ,
    JmpInitArg,
    Range,
}

pub struct Program {
    pub bytecode: Vec<u8>,
    pub(crate) spans: Vec<Span>,
    pub(crate) script: Vec<u8>,
}

pub fn compile_ast(ast: &Ast) -> Result<Program, String> {
    let mut program = Vec::new();
    let ast_span = Span {
        start: 0,
        end: ast.script.len() as u32,
    };
    compile_block_inner(&ast.block, true, ast_span, &mut program)?;
    let (bytecode, spans): (Vec<u8>, Vec<Span>) = program.into_iter().unzip();
    Ok(Program {
        bytecode,
        spans,
        script: ast.script.clone(),
    })
}

pub fn u16_to_u8s(val: u16) -> [u8; 2] {
    [(val >> 8) as u8, (val & 0xff) as u8]
}

pub fn u8s_to_u16(val: [u8; 2]) -> u16 {
    ((val[0] as u16) << 8) + val[1] as u16
}

pub fn compile_fn_body_inner(
    pos_args_required: &[Vec<u8>],
    pos_args_optional: &[(Vec<u8>, ExprNode)],
    body: &Block,
    fn_span: Span,
) -> Result<Vec<(u8, Span)>, String> {
    let mut asm = if block_captures_env(body) {
        vec![(ByteCode::StartCapturedScope as u8, fn_span)]
    } else {
        vec![(ByteCode::StartScope as u8, fn_span)]
    };

    asm.push((ByteCode::UnpackArgs as u8, fn_span));
    asm.push((pos_args_required.len() as u8, fn_span));
    asm.push((pos_args_optional.len() as u8, fn_span));

    for param in pos_args_required.iter() {
        asm.push((
            param.len().try_into().expect("Param len should into u8"),
            fn_span,
        ));
        for b in param {
            asm.push((*b, fn_span));
        }
    }

    for (param, _) in pos_args_optional.iter() {
        asm.push((
            param.len().try_into().expect("Param len should into u8"),
            fn_span,
        ));
        for b in param {
            asm.push((*b, fn_span));
        }
    }

    for (idx, (_param, expr)) in pos_args_optional.iter().enumerate() {
        let jmp_idx = asm.len();
        asm.push((ByteCode::JmpInitArg as u8, expr.span));
        asm.push((0, expr.span));
        asm.push((0, expr.span));

        asm.extend(compile_expression(expr)?);
        asm.push((ByteCode::AssignArg as u8, expr.span));
        asm.push((idx as u8, expr.span));

        let expr_offset = u16_to_u8s(asm.len() as u16 - jmp_idx as u16);
        asm[jmp_idx + 1].0 = expr_offset[0];
        asm[jmp_idx + 2].0 = expr_offset[1];
    }

    for stmt in body.stmts.iter() {
        asm.extend(compile_statement(stmt)?);
    }

    if let Some(expr) = &body.last_expr {
        let val: Option<&ExprNode> = Some(expr);
        asm.extend(compile_return(&val, expr.span)?);
    } else {
        let needs_implicit_return = if body.stmts.len() > 1 {
            !matches!(&body.stmts[body.stmts.len() - 1].data, Statement::Return(_))
        } else {
            true
        };

        if needs_implicit_return {
            let expr = ExprNode {
                data: Expression::Primary(Primary::None),
                span: fn_span,
            };
            let val: Option<&ExprNode> = Some(&expr);
            asm.extend(compile_return(&val, fn_span)?);
        }
    }

    if asm.len() > u16::MAX as usize {
        return Err(format!("Function has more than {} instructions", u16::MAX));
    }
    Ok(asm)
}

pub fn compile_fn_expression(
    pos_args_required: &[Vec<u8>],
    pos_args_optional: &[(Vec<u8>, ExprNode)],
    body: &Block,
    fn_span: Span,
) -> Result<Vec<(u8, Span)>, String> {
    // This will be replaced with a relative jump to after the function
    // declaration
    let mut asm = vec![(ByteCode::Jmp as u8, fn_span), (0, fn_span), (0, fn_span)];
    asm.extend(compile_fn_body_inner(
        pos_args_required,
        pos_args_optional,
        body,
        fn_span,
    )?);

    // Fix the jump offset at the function declaration now that we know
    // the size of the body
    let pc_offset = asm.len() as u16;
    asm[1].0 = (pc_offset >> 8) as u8;
    asm[2].0 = (pc_offset & 0xff) as u8;

    // Assign the value to the ident
    let pc_offset = asm.len() as u16 - 3;
    asm.push((ByteCode::CreateFn as u8, fn_span));
    asm.push(((pc_offset >> 8) as u8, fn_span));
    asm.push(((pc_offset & 0xff) as u8, fn_span));

    Ok(asm)
}

pub fn compile_fn(func: &Fn, fn_span: Span) -> Result<Vec<(u8, Span)>, String> {
    let ident = if let Some(ident) = &func.ident {
        ident
    } else {
        return Err("No ident for function declaration!".to_string());
    };

    let mut asm = compile_fn_expression(
        &func.pos_args_required,
        &func.pos_args_optional,
        &func.body,
        fn_span,
    )?;

    asm.push((ByteCode::VariableDeclaration as u8, fn_span));
    asm.push((
        ident.len().try_into().expect("Ident len should into u8"),
        fn_span,
    ));
    for b in ident.iter() {
        asm.push((*b, fn_span));
    }

    Ok(asm)
}

pub fn compile_fn_body(func: &Fn, fn_span: Span) -> Result<Vec<(u8, Span)>, String> {
    compile_fn_body_inner(
        &func.pos_args_required,
        &func.pos_args_optional,
        &func.body,
        fn_span,
    )
}

pub fn compile_return(expr: &Option<&ExprNode>, span: Span) -> Result<Vec<(u8, Span)>, String> {
    let mut res = Vec::new();
    if let Some(expr) = expr {
        res.extend(compile_expression(expr)?);
    } else {
        res.push((ByteCode::LiteralNone as u8, span));
    }
    res.push((ByteCode::Return as u8, span));
    Ok(res)
}

fn compound_op_bytecode(op: &CompoundOp) -> u8 {
    match op {
        CompoundOp::Add => ByteCode::Add as u8,
        CompoundOp::Subtract => ByteCode::Sub as u8,
        CompoundOp::Multiply => ByteCode::Multiply as u8,
        CompoundOp::Divide => ByteCode::Divide as u8,
        CompoundOp::Modulus => ByteCode::Modulus as u8,
    }
}

pub fn compile_statement(stmt_node: &StatementNode) -> Result<Vec<(u8, Span)>, String> {
    let stmt_span = stmt_node.span;
    match &stmt_node.data {
        Statement::Let(ident, expr) => {
            // Expression evaluates to a value that's on the top of the stack
            let mut expr_asm = compile_expression(expr)?;
            // When getting VariableDeclaration the next byte is the length of
            // the identifier, followed by the
            expr_asm.push((ByteCode::VariableDeclaration as u8, expr.span));
            expr_asm.push((
                ident.len().try_into().expect("Ident len should into u8"),
                expr.span,
            ));
            for b in ident.iter() {
                expr_asm.push((*b, expr.span));
            }

            Ok(expr_asm)
        }
        Statement::Assignment(ident, expr) => {
            let mut expr_asm = compile_expression(expr)?;
            expr_asm.push((ByteCode::Assignment as u8, expr.span));
            expr_asm.push((
                ident.len().try_into().expect("Ident len should into u8"),
                expr.span,
            ));
            for b in ident.iter() {
                expr_asm.push((*b, expr.span));
            }

            Ok(expr_asm)
        }
        Statement::AttributeAssignment(obj_expr, ident, expr) => {
            let mut expr_asm = compile_expression(obj_expr)?;
            expr_asm.extend(compile_expression(expr)?);
            expr_asm.push((ByteCode::SetAttr as u8, expr.span));
            expr_asm.push((
                ident.len().try_into().expect("Ident len should into u8"),
                expr.span,
            ));
            for b in ident.iter() {
                expr_asm.push((*b, expr.span));
            }

            Ok(expr_asm)
        }
        Statement::IndexAssignment(obj_expr, index_expr, expr) => {
            let mut expr_asm = compile_expression(obj_expr)?;
            expr_asm.extend(compile_expression(index_expr)?);
            expr_asm.extend(compile_expression(expr)?);
            expr_asm.push((ByteCode::SetIndex as u8, expr.span));

            Ok(expr_asm)
        }
        Statement::CompoundAssignment(ident, op, rhs) => {
            // Desugar: x += e  →  x = x + e
            // Emit: load x, compile rhs, binary op, assign x
            let op_bytecode = compound_op_bytecode(op);
            let mut asm = Vec::new();
            asm.push((ByteCode::VariableLoad as u8, rhs.span));
            asm.push((
                ident.len().try_into().expect("Ident len should into u8"),
                rhs.span,
            ));
            for b in ident.iter() {
                asm.push((*b, rhs.span));
            }
            asm.extend(compile_expression(rhs)?);
            asm.push((op_bytecode, rhs.span));
            asm.push((ByteCode::Assignment as u8, rhs.span));
            asm.push((
                ident.len().try_into().expect("Ident len should into u8"),
                rhs.span,
            ));
            for b in ident.iter() {
                asm.push((*b, rhs.span));
            }
            Ok(asm)
        }
        Statement::CompoundAttributeAssignment(obj_expr, ident, op, rhs) => {
            // Desugar: obj.attr += e  →  obj.attr = obj.attr + e
            // Emit: compile obj, Copy, GetAttr, compile rhs, binary op, SetAttr
            let op_bytecode = compound_op_bytecode(op);
            let mut asm = compile_expression(obj_expr)?;
            asm.push((ByteCode::Copy as u8, rhs.span));
            asm.push((ByteCode::GetAttr as u8, rhs.span));
            asm.push((
                ident.len().try_into().expect("Ident len should into u8"),
                rhs.span,
            ));
            for b in ident.iter() {
                asm.push((*b, rhs.span));
            }
            asm.extend(compile_expression(rhs)?);
            asm.push((op_bytecode, rhs.span));
            asm.push((ByteCode::SetAttr as u8, rhs.span));
            asm.push((
                ident.len().try_into().expect("Ident len should into u8"),
                rhs.span,
            ));
            for b in ident.iter() {
                asm.push((*b, rhs.span));
            }
            Ok(asm)
        }
        Statement::CompoundIndexAssignment(obj_expr, index_expr, op, rhs) => {
            // Desugar: obj[idx] += e  →  obj[idx] = obj[idx] + e
            // Emit: compile obj, compile idx, CopyFrom(1), CopyFrom(1), Index, compile rhs, binary op, SetIndex
            let op_bytecode = compound_op_bytecode(op);
            let mut asm = compile_expression(obj_expr)?;
            asm.extend(compile_expression(index_expr)?);
            asm.push((ByteCode::CopyFrom as u8, rhs.span));
            asm.push((1, rhs.span));
            asm.push((ByteCode::CopyFrom as u8, rhs.span));
            asm.push((1, rhs.span));
            asm.push((ByteCode::Index as u8, rhs.span));
            asm.extend(compile_expression(rhs)?);
            asm.push((op_bytecode, rhs.span));
            asm.push((ByteCode::SetIndex as u8, rhs.span));
            Ok(asm)
        }
        Statement::Fn(func) => compile_fn(func, stmt_span),
        Statement::Struct(Struct {
            ident,
            members_required,
            members_optional,
            methods,
        }) => {
            let mut asm = vec![
                (ByteCode::CreateStruct as u8, stmt_span),
                (0, stmt_span),
                (0, stmt_span),
            ];
            asm.push((
                (members_required.len() + members_optional.len()) as u8,
                stmt_span,
            ));

            // The +1 is for the constructor
            asm.push(((methods.len() + 1) as u8, stmt_span));

            // Add struct name
            asm.push((
                ident
                    .len()
                    .try_into()
                    .expect("Struct name len should into u8"),
                stmt_span,
            ));
            for b in ident.iter() {
                asm.push((*b, stmt_span));
            }

            let member_names: Vec<Vec<u8>> = members_required
                .iter()
                .cloned()
                .chain(members_optional.iter().map(|(x, _)| x.clone()))
                .collect();

            for member in member_names.iter() {
                asm.push((
                    member
                        .len()
                        .try_into()
                        .expect("Member ident len should into u8"),
                    stmt_span,
                ));
                for b in member.iter() {
                    asm.push((*b, stmt_span));
                }
            }

            let arg_list = member_names
                .into_iter()
                .map(|name| Node {
                    data: Expression::Primary(Primary::Identifier(name)),
                    span: stmt_span,
                })
                .collect();

            #[allow(clippy::type_complexity)]
            let mut method_defs: Vec<(&[u8], Vec<(u8, Span)>)> = Vec::new();
            method_defs.push((
                b"__init__",
                compile_fn_body_inner(
                    members_required,
                    members_optional,
                    &Block {
                        stmts: Vec::new(),
                        last_expr: Some(Box::new(Node {
                            data: Expression::Call(
                                Box::new(Node {
                                    data: Expression::Primary(Primary::Identifier(ident.clone())),
                                    span: stmt_span,
                                }),
                                arg_list,
                                Vec::new(),
                            ),
                            span: stmt_span,
                        })),
                    },
                    stmt_span,
                )?,
            ));
            for method in methods {
                let ident = if let Some(ident) = &method.ident {
                    ident
                } else {
                    return Err("Method does not have ident!".to_string());
                };
                method_defs.push((ident, compile_fn_body(method, stmt_span)?));
            }

            let mut jump_asm_idx = Vec::new();
            for (ident, _method_def) in method_defs.iter() {
                jump_asm_idx.push(asm.len());
                asm.push((0, stmt_span));
                asm.push((0, stmt_span));
                asm.push((
                    ident
                        .len()
                        .try_into()
                        .expect("Method ident len should into u8"),
                    stmt_span,
                ));
                for b in ident.iter() {
                    asm.push((*b, stmt_span));
                }
            }

            for (method_idx, (_, method_def)) in method_defs.into_iter().enumerate() {
                let jump_idx = jump_asm_idx[method_idx];
                let pc_offset = asm.len() as u16;
                asm[jump_idx].0 = (pc_offset >> 8) as u8;
                asm[jump_idx + 1].0 = (pc_offset & 0xff) as u8;
                asm.extend(method_def);
            }

            let pc_offset = asm.len() as u16;
            asm[1].0 = (pc_offset >> 8) as u8;
            asm[2].0 = (pc_offset & 0xff) as u8;

            asm.push((ByteCode::VariableDeclaration as u8, stmt_span));
            asm.push((
                ident.len().try_into().expect("Ident len should into u8"),
                stmt_span,
            ));

            for b in ident.iter() {
                asm.push((*b, stmt_span));
            }

            Ok(asm)
        }
        Statement::For(idents, expr, body) => {
            let mut asm = compile_expression(expr)?;

            // Call .iter() and put its .next on the top of the stack
            {
                asm.push((ByteCode::GetAttr as u8, expr.span));
                asm.extend([
                    (4, expr.span),
                    (b'i', expr.span),
                    (b't', expr.span),
                    (b'e', expr.span),
                    (b'r', expr.span),
                ]);

                asm.push((ByteCode::Call as u8, expr.span));
                asm.push((0, expr.span));
                asm.push((0, expr.span));

                asm.push((ByteCode::GetAttr as u8, expr.span));
                asm.extend([
                    (4, expr.span),
                    (b'n', expr.span),
                    (b'e', expr.span),
                    (b'x', expr.span),
                    (b't', expr.span),
                ]);
            }

            let loop_start_idx = asm.len();
            asm.extend(vec![
                (ByteCode::LoopStart as u8, stmt_span),
                (0, stmt_span),
                (0, stmt_span),
            ]);

            // Copy the .next bound method and call it
            asm.push((ByteCode::Copy as u8, expr.span));
            asm.push((ByteCode::Call as u8, expr.span));
            asm.push((0, expr.span));
            asm.push((0, expr.span));

            // Copy the result of .next() so we can check if it's None
            asm.push((ByteCode::Copy as u8, expr.span));
            asm.push((ByteCode::LiteralNone as u8, expr.span));
            asm.push((ByteCode::Equal as u8, expr.span));

            // Jump to `LoopEnd` if calling .next() returns None
            let none_check_idx = asm.len();
            asm.push((ByteCode::JmpNZ as u8, stmt_span));
            asm.push((0, stmt_span));
            asm.push((0, stmt_span));

            if block_captures_env(body) {
                asm.push((ByteCode::StartCapturedScope as u8, stmt_span));
            } else {
                asm.push((ByteCode::StartScope as u8, stmt_span));
            }

            if idents.len() != 1 {
                let tuple_size_u8s = u16_to_u8s(idents.len() as u16);
                asm.push((ByteCode::UnpackTuple as u8, stmt_span));
                asm.push((tuple_size_u8s[0], stmt_span));
                asm.push((tuple_size_u8s[1], stmt_span));
            }

            for ident in idents {
                asm.push((ByteCode::VariableDeclaration as u8, stmt_span));
                asm.push((
                    ident
                        .len()
                        .try_into()
                        .expect("For loop ident len should into u8"),
                    stmt_span,
                ));
                for b in ident {
                    asm.push((*b, stmt_span));
                }
            }

            asm.extend(compile_block(body, false, stmt_span)?);

            asm.push((ByteCode::EndScope as u8, stmt_span));

            // Jump to start of loop to check the condition again
            let loop_start_offset = u16_to_u8s(asm.len() as u16 - loop_start_idx as u16 - 3);
            asm.push((ByteCode::JmpUp as u8, stmt_span));
            asm.push((loop_start_offset[0], stmt_span));
            asm.push((loop_start_offset[1], stmt_span));

            // This is the offset from none_check_idx that will get us
            // out of the loop. JmpNZ lands on a Pop that discards the
            // leftover None left on the stack by the iterator check
            // (Copy/None/Equal consumed only the duplicate). `break`
            // jumps to LoopEnd below instead, where that None is not
            // on the stack.
            let pc_offset = u16_to_u8s(asm.len() as u16 - none_check_idx as u16);
            asm[none_check_idx + 1].0 = pc_offset[0];
            asm[none_check_idx + 2].0 = pc_offset[1];

            asm.push((ByteCode::Pop as u8, stmt_span));

            let loop_end = u16_to_u8s(asm.len() as u16 - loop_start_idx as u16);
            asm[loop_start_idx + 1].0 = loop_end[0];
            asm[loop_start_idx + 2].0 = loop_end[1];

            asm.push((ByteCode::LoopEnd as u8, stmt_span));

            // Remove the `iter` method from the stack
            asm.push((ByteCode::Pop as u8, stmt_span));

            Ok(asm)
        }
        Statement::While(conditional, body) => {
            let mut asm = vec![
                (ByteCode::LoopStart as u8, stmt_span),
                (0, stmt_span),
                (0, stmt_span),
            ];
            asm.extend(compile_expression(conditional)?);

            // Jump to `LoopEnd` if the condition is falsy
            let conditional_check_idx = asm.len();
            asm.push((ByteCode::JmpZ as u8, stmt_span));
            asm.push((0, stmt_span));
            asm.push((0, stmt_span));

            asm.extend(compile_block(body, false, stmt_span)?);

            // Jump to start of loop to check the condition again
            let loop_start_offset = u16_to_u8s(asm.len() as u16 - 3);
            asm.push((ByteCode::JmpUp as u8, stmt_span));
            asm.push((loop_start_offset[0], stmt_span));
            asm.push((loop_start_offset[1], stmt_span));

            // This is the offset from conditional_check_idx that will get us
            // out of the loop
            let pc_offset = u16_to_u8s(asm.len() as u16 - conditional_check_idx as u16);
            asm[conditional_check_idx + 1].0 = pc_offset[0];
            asm[conditional_check_idx + 2].0 = pc_offset[1];

            let loop_end = u16_to_u8s(asm.len() as u16);
            asm[1].0 = loop_end[0];
            asm[2].0 = loop_end[1];

            asm.push((ByteCode::LoopEnd as u8, stmt_span));

            Ok(asm)
        }
        Statement::Break => Ok(vec![(ByteCode::Break as u8, stmt_span)]),
        Statement::Continue => Ok(vec![(ByteCode::Continue as u8, stmt_span)]),
        Statement::Return(expr) => compile_return(&expr.as_ref(), stmt_span),
        Statement::If(conditional, if_body, else_body) => {
            compile_if(conditional, if_body, else_body, false, stmt_span)
        }
        Statement::Expression(expr) => {
            // Expression evaluates to a value that's on the top of the stack
            let mut expr_asm = compile_expression(expr)?;
            // Pop the value since it's not used
            expr_asm.push((ByteCode::Pop as u8, expr.span));

            Ok(expr_asm)
        }
    }
}

pub fn compile_block_inner(
    block: &Block,
    is_expr: bool,
    block_span: Span,
    asm: &mut Vec<(u8, Span)>,
) -> Result<(), String> {
    for stmt in block.stmts.iter() {
        asm.extend(compile_statement(stmt)?);
    }
    if let Some(last_expr) = &block.last_expr {
        asm.extend(compile_expression(last_expr)?);
    } else {
        asm.push((ByteCode::LiteralNone as u8, block_span));
    }
    if !is_expr {
        asm.push((ByteCode::Pop as u8, block_span));
    }

    Ok(())
}

pub fn statement_captures_env(stmt: &Statement) -> bool {
    match stmt {
        Statement::Fn(..) => true,
        Statement::Struct(..) => true,
        Statement::Let(_ident, expr) => expression_captures_env(&expr.data),
        Statement::Assignment(_ident, expr) => expression_captures_env(&expr.data),
        Statement::AttributeAssignment(obj, _ident, expr) => {
            expression_captures_env(&obj.data) || expression_captures_env(&expr.data)
        }
        Statement::IndexAssignment(obj, index_expr, expr) => {
            expression_captures_env(&obj.data)
                || expression_captures_env(&index_expr.data)
                || expression_captures_env(&expr.data)
        }
        Statement::CompoundAssignment(_ident, _op, expr) => expression_captures_env(&expr.data),
        Statement::CompoundAttributeAssignment(obj, _ident, _op, expr) => {
            expression_captures_env(&obj.data) || expression_captures_env(&expr.data)
        }
        Statement::CompoundIndexAssignment(obj, index_expr, _op, expr) => {
            expression_captures_env(&obj.data)
                || expression_captures_env(&index_expr.data)
                || expression_captures_env(&expr.data)
        }
        Statement::If(cond, if_block, else_block) => {
            expression_captures_env(&cond.data)
                || block_captures_env(if_block)
                || block_captures_env(else_block)
        }
        Statement::For(_ident, expr, block) => {
            expression_captures_env(&expr.data) || block_captures_env(block)
        }
        Statement::While(cond, block) => {
            expression_captures_env(&cond.data) || block_captures_env(block)
        }
        Statement::Break => false,
        Statement::Continue => false,
        Statement::Expression(expr) => expression_captures_env(&expr.data),
        Statement::Return(expr) => match expr {
            Some(expr) => expression_captures_env(&expr.data),
            None => false,
        },
    }
}

pub fn expression_captures_env(input_expr: &Expression) -> bool {
    match input_expr {
        Expression::Primary(Primary::None) => false,
        Expression::Primary(Primary::Integer(_)) => false,
        Expression::Primary(Primary::Float(_)) => false,
        Expression::Primary(Primary::Identifier(_)) => false,
        Expression::Primary(Primary::Bool(_)) => false,
        Expression::Primary(Primary::String(_)) => false,
        Expression::Primary(Primary::List(lst)) => {
            for expr in lst.iter() {
                if expression_captures_env(&expr.data) {
                    return true;
                }
            }
            false
        }
        Expression::Primary(Primary::Tuple(tpl)) => {
            for expr in tpl.iter() {
                if expression_captures_env(&expr.data) {
                    return true;
                }
            }
            false
        }
        Expression::Primary(Primary::Expression(expr)) => expression_captures_env(&expr.data),
        Expression::BooleanOp(op) => {
            let exprs = op.exprs();
            expression_captures_env(&exprs.0.data) || expression_captures_env(&exprs.1.data)
        }
        Expression::BinaryOp(op) => {
            let exprs = op.exprs();
            expression_captures_env(&exprs.0.data) || expression_captures_env(&exprs.1.data)
        }
        Expression::UnaryOp(op) => match op {
            UnaryOp::Not(expr) => expression_captures_env(&expr.data),
            UnaryOp::Negate(expr) => expression_captures_env(&expr.data),
        },
        Expression::Stringify(expr) => expression_captures_env(&expr.data),
        Expression::Call(func, args, kwargs) => {
            if expression_captures_env(&func.data) {
                return true;
            }
            for arg_expr in args.iter() {
                if expression_captures_env(&arg_expr.data) {
                    return true;
                }
            }
            for (_ident, kwarg_expr) in kwargs.iter() {
                if expression_captures_env(&kwarg_expr.data) {
                    return true;
                }
            }
            false
        }
        Expression::Index(obj, index) => {
            expression_captures_env(&obj.data) || expression_captures_env(&index.data)
        }
        Expression::Attribute(obj, _attr) => expression_captures_env(&obj.data),
        Expression::Block(block) => block_captures_env(block),
        Expression::If(cond, if_block, else_block) => {
            expression_captures_env(&cond.data)
                || block_captures_env(if_block)
                || block_captures_env(else_block)
        }
        Expression::Fn(..) => true,
    }
}

pub fn block_captures_env(block: &Block) -> bool {
    for stmt in block.stmts.iter() {
        if statement_captures_env(&stmt.data) {
            return true;
        }
    }
    if let Some(last_expr) = &block.last_expr {
        if expression_captures_env(&last_expr.data) {
            return true;
        }
    }

    false
}

pub fn compile_block(
    block: &Block,
    is_expr: bool,
    block_span: Span,
) -> Result<Vec<(u8, Span)>, String> {
    let mut asm = Vec::new();

    // If we don't define any new variables we don't need the overhead of creating a new scope
    // `let` directly defines a new variable
    // `for` implicitly defines an iteration variable. That will be rough for nested for loops, so we'll
    // likely want to be able to hoist that into a new parent scope
    /*
    for y in rows {
      for x in cols { // <--- This `x` is a new variable that would
                      //      make the outer loop require StartScope/EndScope,
                      //      but x could be defined before the outer loop in
                      //      the bytcode to avoid this inefficiency.
        foo(x, y);
      }
    }
    */

    // We can go further than this by creating a separate HeapScope and StackScope.
    // We only need the HeapScope if the scope is captured by a function definition,
    // closure, or struct definition. The vast majority of scopes should be StackScopes.
    // Even further, the HeapScope might only include captured variables

    let mut needs_new_scope = false;
    for stmt in &block.stmts {
        let stmt = &stmt.data;
        match stmt {
            Statement::Let(..) | Statement::For(..) | Statement::Fn(..) | Statement::Struct(..) => {
                needs_new_scope = true;
                break;
            }
            _ => (),
        }
    }

    if needs_new_scope {
        // When a block starts a new Environment is created. However,
        // when the block ends the Environment only needs to stick
        // around in memory if it's captured by a Struct/fn. We can free
        // it immediately (or re-use it) if there are no function
        // declarations or struct declarations inside the block.
        //
        // Even that's a bit pessimistic since it's not the whole
        // environment (with parents) that needs to be kept and some
        // function declarations don't need to capture their environment.
        if block_captures_env(block) {
            asm.push((ByteCode::StartCapturedScope as u8, block_span));
        } else {
            asm.push((ByteCode::StartScope as u8, block_span));
        }
    }

    compile_block_inner(block, is_expr, block_span, &mut asm)?;

    if needs_new_scope {
        asm.push((ByteCode::EndScope as u8, block_span));
    }

    Ok(asm)
}

pub fn compile_if(
    conditional: &ExprNode,
    if_body: &Block,
    else_body: &Block,
    is_expr: bool,
    span: Span,
) -> Result<Vec<(u8, Span)>, String> {
    let mut asm = compile_expression(conditional)?;
    let conditional_check_idx = asm.len();
    asm.push((ByteCode::JmpZ as u8, span));
    asm.push((0, span));
    asm.push((0, span));
    asm.extend(compile_block(if_body, is_expr, span)?);
    asm.push((ByteCode::Jmp as u8, span));
    asm.push((0, span));
    asm.push((0, span));
    // We jump to here when the condition is false
    let else_case_start_idx = asm.len();

    asm.extend(compile_block(else_body, is_expr, span)?);

    // Offset from conditional to the else branch
    let else_jump_offset = u16_to_u8s(else_case_start_idx as u16 - conditional_check_idx as u16);
    asm[conditional_check_idx + 1].0 = else_jump_offset[0];
    asm[conditional_check_idx + 2].0 = else_jump_offset[1];

    // Offset from the end of the if branch to after the else branch
    let if_jump_offset = u16_to_u8s(asm.len() as u16 - else_case_start_idx as u16 + 3);

    asm[else_case_start_idx - 2].0 = if_jump_offset[0];
    asm[else_case_start_idx - 1].0 = if_jump_offset[1];

    Ok(asm)
}

pub fn compile_expression(expr: &ExprNode) -> Result<Vec<(u8, Span)>, String> {
    let span = expr.span;
    match &expr.data {
        Expression::Primary(Primary::None) => {
            let val = ShimValue::None;
            let mut res = vec![(ByteCode::LiteralShimValue as u8, expr.span)];
            for b in val.to_bytes().into_iter() {
                res.push((b, expr.span));
            }
            Ok(res)
        }
        Expression::Primary(Primary::Bool(b)) => {
            let val = ShimValue::Bool(*b);
            let mut res = vec![(ByteCode::LiteralShimValue as u8, expr.span)];
            for b in val.to_bytes().into_iter() {
                res.push((b, expr.span));
            }
            Ok(res)
        }
        Expression::Primary(Primary::Integer(i)) => {
            let val = ShimValue::Integer(*i);
            let mut res = vec![(ByteCode::LiteralShimValue as u8, expr.span)];
            for b in val.to_bytes().into_iter() {
                res.push((b, expr.span));
            }
            Ok(res)
        }
        Expression::Primary(Primary::Float(f)) => {
            let val = ShimValue::Float(*f);
            let mut res = vec![(ByteCode::LiteralShimValue as u8, expr.span)];
            for b in val.to_bytes().into_iter() {
                res.push((b, expr.span));
            }
            Ok(res)
        }
        Expression::Primary(Primary::Identifier(ident)) => {
            let mut res = Vec::new();
            res.push((ByteCode::VariableLoad as u8, expr.span));
            res.push((
                ident.len().try_into().expect("Ident len should into u8"),
                expr.span,
            ));
            for b in ident.iter() {
                res.push((*b, expr.span));
            }
            Ok(res)
        }
        Expression::Primary(Primary::String(s)) => {
            let mut res = Vec::new();
            res.push((ByteCode::LiteralString as u8, expr.span));
            res.push((s.len().try_into().expect("Ident should into u8"), expr.span));
            for b in s.iter() {
                res.push((*b, expr.span));
            }
            Ok(res)
        }
        Expression::Primary(Primary::Tuple(items)) => {
            let mut res = Vec::new();
            for expr in items {
                res.extend(compile_expression(expr)?);
            }
            res.push((ByteCode::CreateTuple as u8, expr.span));
            let len: u16 = items.len().try_into().expect("Tuple should fit into u16");
            res.push(((len >> 8) as u8, expr.span));
            res.push(((len & 0xff) as u8, expr.span));
            Ok(res)
        }
        Expression::Primary(Primary::List(items)) => {
            let mut res = Vec::new();
            for expr in items {
                res.extend(compile_expression(expr)?);
            }
            res.push((ByteCode::CreateList as u8, expr.span));
            let len: u16 = items.len().try_into().expect("List should fit into u16");
            res.push(((len >> 8) as u8, expr.span));
            res.push(((len & 0xff) as u8, expr.span));
            Ok(res)
        }
        Expression::Primary(Primary::Expression(expr)) => compile_expression(expr),
        Expression::BooleanOp(BooleanOp::And(a, b)) => {
            let mut asm = compile_expression(a)?;
            asm.push((ByteCode::Copy as u8, span));

            let short_circuit_idx = asm.len();
            asm.push((ByteCode::JmpZ as u8, span));
            asm.push((0, span));
            asm.push((0, span));

            // Since the result of a is truthy we get rid of it and return the result of b
            asm.push((ByteCode::Pop as u8, span));

            asm.extend(compile_expression(b)?);

            let short_circuit_offset = u16_to_u8s(asm.len() as u16 - short_circuit_idx as u16);
            asm[short_circuit_idx + 1].0 = short_circuit_offset[0];
            asm[short_circuit_idx + 2].0 = short_circuit_offset[1];

            Ok(asm)
        }
        Expression::BooleanOp(BooleanOp::Or(a, b)) => {
            let mut asm = compile_expression(a)?;
            asm.push((ByteCode::Copy as u8, span));

            let short_circuit_idx = asm.len();
            asm.push((ByteCode::JmpNZ as u8, span));
            asm.push((0, span));
            asm.push((0, span));

            // Since the result of a is falsy we get rid of it and return the result of b
            asm.push((ByteCode::Pop as u8, span));

            asm.extend(compile_expression(b)?);

            let short_circuit_offset = u16_to_u8s(asm.len() as u16 - short_circuit_idx as u16);
            asm[short_circuit_idx + 1].0 = short_circuit_offset[0];
            asm[short_circuit_idx + 2].0 = short_circuit_offset[1];

            Ok(asm)
        }
        Expression::BinaryOp(op) => {
            let (opcode, a, b) = match op {
                BinaryOp::Add(a, b) => (ByteCode::Add, a, b),
                BinaryOp::Subtract(a, b) => (ByteCode::Sub, a, b),
                BinaryOp::Equal(a, b) => (ByteCode::Equal, a, b),
                BinaryOp::NotEqual(a, b) => (ByteCode::NotEqual, a, b),
                BinaryOp::Multiply(a, b) => (ByteCode::Multiply, a, b),
                BinaryOp::Modulus(a, b) => (ByteCode::Modulus, a, b),
                BinaryOp::Divide(a, b) => (ByteCode::Divide, a, b),
                BinaryOp::GT(a, b) => (ByteCode::GT, a, b),
                BinaryOp::Gte(a, b) => (ByteCode::Gte, a, b),
                BinaryOp::LT(a, b) => (ByteCode::LT, a, b),
                BinaryOp::Lte(a, b) => (ByteCode::Lte, a, b),
                BinaryOp::In(a, b) => (ByteCode::In, a, b),
                BinaryOp::Range(a, b) => (ByteCode::Range, a, b),
            };
            let mut res = compile_expression(a)?;
            res.extend(compile_expression(b)?);
            res.push((opcode as u8, expr.span));
            Ok(res)
        }
        Expression::UnaryOp(op) => {
            let (opcode, a) = match op {
                UnaryOp::Not(a) => (ByteCode::Not, a),
                UnaryOp::Negate(a) => (ByteCode::Negate, a),
            };
            let mut res = compile_expression(a)?;
            res.push((opcode as u8, expr.span));
            Ok(res)
        }
        Expression::Stringify(expr) => {
            let mut asm = compile_expression(expr)?;
            asm.push((ByteCode::Stringify as u8, expr.span));
            Ok(asm)
        }
        Expression::Index(obj_expr, index_expr) => {
            let mut asm = compile_expression(obj_expr)?;
            asm.extend(compile_expression(index_expr)?);
            asm.push((ByteCode::Index as u8, expr.span));

            Ok(asm)
        }
        Expression::Call(expr, args, kwargs) => {
            if let Expression::Attribute(obj_expr, attr_ident) = &expr.data {
                // Method calling optimization.
                // Normally, accessing a method on an instance of an object would allocate a 2-word object
                // in memory that's immediately discarded after the call. That's pretty rough from a
                // fragmentation and memory-use perspective to need to allocate for every single method
                // call.

                // This optimzation uses a fused op-code whenever a `Call` is done on an accessed attribute.
                // Instead of allocating for a BoundMethod, it executes ShimValue::call_attr

                // Get the object
                let mut res = compile_expression(obj_expr)?;

                // Get the args
                for arg_expr in args.iter() {
                    res.extend(compile_expression(arg_expr)?);
                }

                // Get the kwargs
                for (ident, kwarg_expr) in kwargs.iter() {
                    res.push((ByteCode::LiteralString as u8, kwarg_expr.span));
                    res.push((
                        ident.len().try_into().expect("Ident should into u8"),
                        kwarg_expr.span,
                    ));
                    for b in ident.iter() {
                        res.push((*b, kwarg_expr.span));
                    }

                    res.extend(compile_expression(kwarg_expr)?);
                }

                // Then call a particular attribute with the arguments.
                // The ident is last since it's a variable length and that feels right.
                // This attribute access it technically out-of-order for this optimization,
                // (arguments are evaluated before the attribute is accessed). People will
                // just need to deal with that since this is too important of an optimzation
                // to ignore.
                res.push((ByteCode::AttrCall as u8, span));
                res.push((args.len() as u8, span));
                res.push((kwargs.len() as u8, span));
                res.push((
                    attr_ident
                        .len()
                        .try_into()
                        .expect("Ident len should into u8"),
                    span,
                ));
                for b in attr_ident.iter() {
                    res.push((*b, span));
                }
                Ok(res)
            } else {
                // First we evaluate the thing that needs to be called
                let mut res = compile_expression(expr)?;

                // Then we evaluate each argument
                for arg_expr in args.iter() {
                    res.extend(compile_expression(arg_expr)?);
                }

                for (ident, kwarg_expr) in kwargs.iter() {
                    res.push((ByteCode::LiteralString as u8, kwarg_expr.span));
                    res.push((
                        ident.len().try_into().expect("Ident should into u8"),
                        kwarg_expr.span,
                    ));
                    for b in ident.iter() {
                        res.push((*b, kwarg_expr.span));
                    }

                    res.extend(compile_expression(kwarg_expr)?);
                }

                res.push((ByteCode::Call as u8, span));
                res.push((args.len() as u8, span));
                res.push((kwargs.len() as u8, span));
                Ok(res)
            }
        }
        Expression::Attribute(expr, ident) => {
            let mut res = compile_expression(expr)?;
            res.push((ByteCode::GetAttr as u8, span));
            res.push((
                ident.len().try_into().expect("Ident len should into u8"),
                span,
            ));
            for b in ident.iter() {
                res.push((*b, span));
            }
            Ok(res)
        }
        Expression::Block(block) => compile_block(block, true, span),
        Expression::If(conditional, if_body, else_body) => {
            compile_if(conditional, if_body, else_body, true, span)
        }
        Expression::Fn(func) => compile_fn_expression(
            &func.pos_args_required,
            &func.pos_args_optional,
            &func.body,
            span,
        ),
    }
}

pub fn eprint_asm(bytes: &[u8]) {
    eprintln!("{}", format_asm(bytes));
}

pub fn print_asm(bytes: &[u8]) {
    println!("{}", format_asm(bytes));
}

pub fn format_asm(bytes: &[u8]) -> String {
    let mut out = String::new();

    let mut idx = 0;
    while idx < bytes.len() {
        let b = &bytes[idx];
        let start_idx = idx;

        out.push_str(&format!("{start_idx:4}:  "));

        if *b == ByteCode::Jmp as u8 {
            let offset = ((bytes[idx + 1] as usize) << 8) + bytes[idx + 2] as usize;
            let target = idx + offset;
            out.push_str(&format!("JMP -> {}", target));
            idx += 2;
        } else if *b == ByteCode::VariableDeclaration as u8 {
            let len = bytes[idx + 1] as usize;
            let slice = &bytes[idx + 2..idx + 2 + len];
            out.push_str(&format!(r#"let "{}""#, debug_u8s(slice)));
            idx += len + 1;
        } else if *b == ByteCode::NoOp as u8 {
            out.push_str("no-op");
        } else if *b == ByteCode::Pop as u8 {
            out.push_str("pop");
        } else if *b == ByteCode::Assignment as u8 {
            let len = bytes[idx + 1] as usize;
            let slice = &bytes[idx + 2..idx + 2 + len];
            out.push_str(&format!(r#"assign "{}""#, debug_u8s(slice)));
            idx += len + 1;
        } else if *b == ByteCode::Call as u8 {
            let arg_count = bytes[idx + 1] as usize;
            let kwarg_count = bytes[idx + 2] as usize;
            out.push_str(&format!("call args={}  kwargs={}", arg_count, kwarg_count));
            idx += 2;
        } else if *b == ByteCode::AttrCall as u8 {
            let arg_count = bytes[idx + 1] as usize;
            let kwarg_count = bytes[idx + 2] as usize;
            let len = bytes[idx + 3] as usize;
            let slice = &bytes[idx + 4..idx + 4 + len];
            out.push_str(&format!(
                "attr_call .{} args={}  kwargs={}",
                debug_u8s(slice),
                arg_count,
                kwarg_count
            ));
            idx += 3 + len;
        } else if *b == ByteCode::Not as u8 {
            out.push_str("Not");
        } else if *b == ByteCode::GT as u8 {
            out.push_str("GT");
        } else if *b == ByteCode::Gte as u8 {
            out.push_str("GTE");
        } else if *b == ByteCode::LT as u8 {
            out.push_str("LT");
        } else if *b == ByteCode::Lte as u8 {
            out.push_str("LTE");
        } else if *b == ByteCode::In as u8 {
            out.push_str("In");
        } else if *b == ByteCode::Negate as u8 {
            out.push_str("negate");
        } else if *b == ByteCode::Index as u8 {
            out.push_str("index");
        } else if *b == ByteCode::SetIndex as u8 {
            out.push_str("set_index");
        } else if *b == ByteCode::Add as u8 {
            out.push_str("add");
        } else if *b == ByteCode::Sub as u8 {
            out.push_str("sub");
        } else if *b == ByteCode::Multiply as u8 {
            out.push_str("multiply");
        } else if *b == ByteCode::Divide as u8 {
            out.push_str("divide");
        } else if *b == ByteCode::Modulus as u8 {
            out.push_str("modulus");
        } else if *b == ByteCode::Equal as u8 {
            out.push_str("equal");
        } else if *b == ByteCode::NotEqual as u8 {
            out.push_str("not_equal");
        } else if *b == ByteCode::JmpZ as u8 {
            let offset = ((bytes[idx + 1] as usize) << 8) + bytes[idx + 2] as usize;
            let target = idx + offset;
            out.push_str(&format!("JMPZ -> {}", target));
            idx += 2;
        } else if *b == ByteCode::JmpNZ as u8 {
            let offset = ((bytes[idx + 1] as usize) << 8) + bytes[idx + 2] as usize;
            let target = idx + offset;
            out.push_str(&format!("JMPNZ -> {}", target));
            idx += 2;
        } else if *b == ByteCode::JmpUp as u8 {
            let offset = ((bytes[idx + 1] as usize) << 8) + bytes[idx + 2] as usize;
            let target = idx.saturating_sub(offset);
            out.push_str(&format!("JMPUP -> {}", target));
            idx += 2;
        } else if *b == ByteCode::JmpInitArg as u8 {
            let offset = ((bytes[idx + 1] as usize) << 8) + bytes[idx + 2] as usize;
            let target = idx + offset;
            out.push_str(&format!("JmpInitArg -> {}", target));
            idx += 2;
        } else if *b == ByteCode::UnpackArgs as u8 {
            let required_arg_count = bytes[idx + 1] as usize;
            let optional_arg_count = bytes[idx + 2] as usize;

            let mut param_names = Vec::new();
            let mut param_idx = idx + 3;
            for _ in 0..(required_arg_count + optional_arg_count) {
                let len = bytes[param_idx] as usize;
                let param_name = &bytes[param_idx + 1..param_idx + 1 + len];
                param_names.push(debug_u8s(param_name).to_string());
                param_idx += 1 + len;
            }

            out.push_str(&format!(
                "unpack_args required={} optional={} [{}]",
                required_arg_count,
                optional_arg_count,
                param_names.join(", ")
            ));
            idx = param_idx - 1;
        } else if *b == ByteCode::AssignArg as u8 {
            out.push_str("assign arg");
        } else if *b == ByteCode::CreateFn as u8 {
            let instruction_offset = ((bytes[idx + 1] as u16) << 8) + bytes[idx + 2] as u16;
            // The function points backwards by this offset
            let target_pc = idx.saturating_sub(instruction_offset as usize);
            out.push_str(&format!("CreateFn -> PC {}", target_pc));
            idx += 2;
        } else if *b == ByteCode::CreateStruct as u8 {
            let member_count = bytes[idx + 3];
            let method_count = bytes[idx + 4];

            let mut parse_idx = idx + 5;

            // Read struct name
            let name_len = bytes[parse_idx];
            let name = &bytes[parse_idx + 1..parse_idx + 1 + name_len as usize];
            parse_idx = parse_idx + 1 + name_len as usize;

            // Read member names
            let mut member_names = Vec::new();
            for _ in 0..member_count {
                let ident_len = bytes[parse_idx];
                let ident = &bytes[parse_idx + 1..parse_idx + 1 + ident_len as usize];
                member_names.push(debug_u8s(ident).to_string());
                parse_idx = parse_idx + 1 + ident_len as usize;
            }

            // Read method names and PCs
            let mut methods = Vec::new();
            for _ in 0..method_count {
                let method_pc =
                    idx + ((bytes[parse_idx] as usize) << 8) + bytes[parse_idx + 1] as usize;
                parse_idx += 2;

                let ident_len = bytes[parse_idx];
                let ident = &bytes[parse_idx + 1..parse_idx + 1 + ident_len as usize];
                methods.push(format!("{}@{}", debug_u8s(ident), method_pc));
                parse_idx = parse_idx + 1 + ident_len as usize;
            }

            out.push_str(&format!(
                "CreateStruct \"{}\" members=[{}] methods=[{}]",
                debug_u8s(name),
                member_names.join(", "),
                methods.join(", ")
            ));

            // Skip to the end of the struct header (not the entire definition)
            // This allows the method bodies to be disassembled normally
            // Note: The outer loop will add 1, so we subtract 1 here
            idx = parse_idx - 1;
        } else if *b == ByteCode::GetAttr as u8 {
            let len = bytes[idx + 1] as usize;
            let slice = &bytes[idx + 2..idx + 2 + len];
            out.push_str(&format!("get .{}", debug_u8s(slice)));
            idx += len + 1;
        } else if *b == ByteCode::SetAttr as u8 {
            let len = bytes[idx + 1] as usize;
            let slice = &bytes[idx + 2..idx + 2 + len];
            out.push_str(&format!("set .{}", debug_u8s(slice)));
            idx += len + 1;
        } else if *b == ByteCode::VariableLoad as u8 {
            let len = bytes[idx + 1] as usize;
            let slice = &bytes[idx + 2..idx + 2 + len];
            out.push_str(&format!(r#"load "{}""#, debug_u8s(slice)));
            idx += len + 1;
        } else if *b == ByteCode::Break as u8 {
            out.push_str("break");
        } else if *b == ByteCode::Continue as u8 {
            out.push_str("continue");
        } else if *b == ByteCode::LiteralShimValue as u8 {
            let shim_bytes: [u8; 8] = bytes[idx + 1..idx + 9].try_into().unwrap();
            let val = ShimValue::from_bytes(shim_bytes);
            let val_str = match val {
                ShimValue::Integer(i) => format!("{}", i),
                ShimValue::Float(f) => format_float(f),
                ShimValue::Bool(true) => "true".to_string(),
                ShimValue::Bool(false) => "false".to_string(),
                ShimValue::None => "None".to_string(),
                ShimValue::Unit => "Unit".to_string(),
                _ => format!("{:?}", val),
            };
            out.push_str(&format!("ShimValue {}", val_str));
            idx += 8;
        } else if *b == ByteCode::LiteralString as u8 {
            let len = bytes[idx + 1] as usize;
            let slice = &bytes[idx + 2..idx + 2 + len];
            out.push_str(&format!(r#"String "{}""#, debug_u8s(slice)));
            idx += len + 1;
        } else if *b == ByteCode::LiteralNone as u8 {
            out.push_str("None");
        } else if *b == ByteCode::CreateList as u8 {
            let list_size = ((bytes[idx + 1] as usize) << 8) + bytes[idx + 2] as usize;
            out.push_str(&format!("CreateList size={}", list_size));
            idx += 2;
        } else if *b == ByteCode::CreateTuple as u8 {
            let list_size = ((bytes[idx + 1] as usize) << 8) + bytes[idx + 2] as usize;
            out.push_str(&format!("CreateTuple size={}", list_size));
            idx += 2;
        } else if *b == ByteCode::UnpackTuple as u8 {
            let list_size = ((bytes[idx + 1] as usize) << 8) + bytes[idx + 2] as usize;
            out.push_str(&format!("UnpackTuple size={}", list_size));
            idx += 2;
        } else if *b == ByteCode::Copy as u8 {
            out.push_str("Copy");
        } else if *b == ByteCode::CopyFrom as u8 {
            let offset = bytes[idx + 1];
            out.push_str(&format!("CopyFrom offset={}", offset));
            idx += 1;
        } else if *b == ByteCode::LoopStart as u8 {
            let offset = ((bytes[idx + 1] as usize) << 8) + bytes[idx + 2] as usize;
            let target = idx + offset;
            out.push_str(&format!("Loop Start -> {}", target));
            idx += 2;
        } else if *b == ByteCode::LoopEnd as u8 {
            out.push_str("Loop End");
        } else if *b == ByteCode::Stringify as u8 {
            out.push_str("stringify");
        } else if *b == ByteCode::StartScope as u8 {
            out.push_str("start_scope");
        } else if *b == ByteCode::StartCapturedScope as u8 {
            out.push_str("start_captured_scope");
        } else if *b == ByteCode::EndScope as u8 {
            out.push_str("end_scope");
        } else if *b == ByteCode::Return as u8 {
            out.push_str("return");
        } else if *b == ByteCode::Range as u8 {
            out.push_str("Range");
        } else {
            // Unformatted byte (including Pad0-Pad9) - show the decimal value
            out.push_str(&format!("{b:3}  "));
        }

        out.push('\n');
        idx += 1;
    }
    out
}
