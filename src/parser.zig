const std = @import("std");

pub fn bufPrintC(buf: []u8, comptime fmt: []const u8, args: anytype) std.fmt.BufPrintError!usize {
    const text = try std.fmt.bufPrint(buf, fmt, args);
    return text.len;
}

pub const Ast = struct {
    allocator: std.mem.Allocator,

    declarations: []Declaration,

    pub fn deinit(_: Ast) void {}

    pub fn print(self: Ast, allocator: std.mem.Allocator) void {
        const empty_buf: []u8 = "";

        var buf = allocator.alloc(u8, 1024) catch empty_buf;
        defer allocator.free(buf);

        const count = self.fmt(buf) catch {
            std.debug.print("Ran out of buffer space\n", .{});
            return;
        };
        std.debug.print("{s}\n", .{buf[0..count]});
    }

    pub fn fmt(self: Ast, bufIn: []u8) std.fmt.BufPrintError!usize {
        var buf = bufIn;
        const startingSize = buf.len;
        for (self.declarations) |decl| {
            switch (decl) {
                DeclarationTag.variable_declaration => {
                    try bufPushFmt(&buf, "let {s} = ", .{decl.variable_declaration.name});

                    if (decl.variable_declaration.expr) |expr| {
                        try expr.pushFmt(&buf);
                    }

                    try bufPushText(&buf, ";\n");
                },
                DeclarationTag.statement_declaration => {
                    try decl.statement_declaration.pushFmt(&buf);
                },
                DeclarationTag.function_declaration => {
                    try bufPushFmt(&buf, "TODO__FUNC {any}", .{decl});
                },
                DeclarationTag.struct_declaration => {
                    try bufPushFmt(&buf, "TODO__STRUCT {any}", .{decl});
                },
            }
        }
        return startingSize - buf.len;
    }
};

// Ast: Statement*
// Statement: Expression ";" | Declaration
// Expression: "(" Expression ")" | Expression "-" Expression | Expression "+" Expression | Identifier | Constant
// Declaration: "let" Identifier ("=" Expression) ";"
// Identifier: [a-zA-Z_][a-zA-Z_0-9]*
// Constant: ConstantInt | ConstantFloat | ConstantStr
// ConstantInt: "-"? [0-9]+
// ConstantFloat: "-"? [0-9]* "." [0-9]*
// ConstantStr: """ [^"]* """

// function       → IDENTIFIER "(" parameters? ")" block ;
// parameters     → IDENTIFIER ( "," IDENTIFIER )* ;
// arguments      → expression ( "," expression )* ;

// declaration    → classDecl
//                | funDecl
//                | varDecl
//                | statement ;
//
// classDecl      → "class" IDENTIFIER ( "<" IDENTIFIER )?
//                  "{" function* "}" ;
// funDecl        → "fn" function ;
// varDecl        → "let" IDENTIFIER ( "=" expression )? ";" ;

pub const DeclarationTag = enum {
    struct_declaration,
    function_declaration,
    variable_declaration,
    statement_declaration,
};

pub const Declaration = union(DeclarationTag) {
    struct_declaration: StructDeclaration,
    function_declaration: FunctionDeclaration,
    variable_declaration: VariableDeclaration,
    statement_declaration: Statement,
};

pub const StructDeclaration = struct {
    name: []const u8,
    members: [][]const u8,
    methods: []FunctionDeclaration,
};

pub const FunctionDeclaration = struct {
    name: []const u8,
    parameters: [][]const u8,
    body: BlockStatement,
};

pub const VariableDeclaration = struct {
    name: []const u8,
    expr: ?Expression,
};

// expression     → assignment ;
//
// assignment     → ( call "." )? IDENTIFIER "=" assignment
//                | logic_or ;
//
//                logic_or       → logic_and ( "or" logic_and )* ;
//                logic_and      → equality ( "and" equality )* ;
//                equality       → comparison ( ( "!=" | "==" ) comparison )* ;
//                comparison     → term ( ( ">" | ">=" | "<" | "<=" ) term )* ;
//                term           → factor ( ( "-" | "+" ) factor )* ;
//                factor         → unary ( ( "/" | "*" ) unary )* ;
//
//                unary          → ( "!" | "-" ) unary | call ;
//                call           → primary ( "(" arguments? ")" | "." IDENTIFIER )* ;
//                primary        → "true" | "false" | "null"
//                               | NUMBER | STRING | IDENTIFIER | "(" expression ")"

pub const BinaryOp = struct {
    a: *Expression,
    b: *Expression,
};

pub const UnaryOp = struct {
    a: *Expression,
};

pub const PrimaryTag = enum {
    intLiteral,
    floatLiteral,
    boolLiteral,
    nullLiteral,
    stringLiteral,
    listLiteral,
    identifier,
};

pub const Primary = union(PrimaryTag) {
    intLiteral: i64,
    floatLiteral: f64,
    boolLiteral: bool,
    nullLiteral: void,
    stringLiteral: []const u8,
    listLiteral: []const Expression,
    identifier: []const u8,

    fn pushFmt(self: Primary, buf: *[]u8) std.fmt.BufPrintError!void {
        const len = try self.fmt(buf.*);
        buf.* = buf.*[len..];
    }

    fn fmt(self: Primary, bufIn: []u8) std.fmt.BufPrintError!usize {
        var buf = bufIn;
        const startingSize = buf.len;
        switch (self) {
            PrimaryTag.listLiteral => {
                try bufPushText(&buf, "[");
                try bufPushArgList(&buf, self.listLiteral);
                try bufPushText(&buf, "]");
            },
            PrimaryTag.intLiteral => {
                try bufPushFmt(&buf, "{any}", .{self.intLiteral});
            },
            PrimaryTag.identifier => {
                try bufPushText(&buf, self.identifier);
            },
            else => {
                try bufPushFmt(&buf, "TODO_PRIM {any}", .{self});
            },
        }
        return startingSize - buf.len;
    }
};

pub const CallInfo = struct {
    callee: *Expression,
    args: []const Expression,
};

pub const IndexInfo = struct {
    callee: *Expression,
    index: *Expression,
};

fn bufPushText(buf: *[]u8, text: []const u8) !void {
    if (text.len > buf.*.len) {
        return error.NoSpaceLeft;
    }
    for (0..text.len) |idx| {
        buf.*[idx] = text[idx];
    }

    buf.* = buf.*[text.len..];
}

pub fn bufPushFmt(buf: *[]u8, comptime fmt: []const u8, args: anytype) std.fmt.BufPrintError!void {
    const text = try std.fmt.bufPrint(buf.*, fmt, args);
    buf.* = buf.*[text.len..];
}

pub const ExpressionTag = enum {
    logicalOr,
    logicalAnd,
    equal,
    notEqual,
    gt,
    gte,
    lt,
    lte,
    minus,
    plus,
    divide,
    multiply,
    not,
    negation,
    call,
    index,
    primary,
};

pub const Expression = union(ExpressionTag) {
    logicalOr: BinaryOp,
    logicalAnd: BinaryOp,
    equal: BinaryOp,
    notEqual: BinaryOp,
    gt: BinaryOp,
    gte: BinaryOp,
    lt: BinaryOp,
    lte: BinaryOp,
    minus: BinaryOp,
    plus: BinaryOp,
    divide: BinaryOp,
    multiply: BinaryOp,
    not: UnaryOp,
    negation: UnaryOp,
    call: CallInfo,
    index: IndexInfo,
    primary: Primary,

    fn pushFmt(self: Expression, buf: *[]u8) std.fmt.BufPrintError!void {
        const len = try self.fmt(buf.*);
        buf.* = buf.*[len..];
    }

    fn fmt(self: Expression, bufIn: []u8) !usize {
        var buf = bufIn;
        const startingSize = buf.len;
        switch (self) {
            ExpressionTag.logicalOr => {
                try bufPushText(&buf, "(");
                try self.logicalOr.a.pushFmt(&buf);
                try bufPushText(&buf, " or ");
                try self.logicalOr.b.pushFmt(&buf);
                try bufPushText(&buf, ")");
            },
            ExpressionTag.primary => {
                try self.primary.pushFmt(&buf);
            },
            ExpressionTag.call => {
                // Only wrap the function being called if it's not a trivial value
                switch (self.call.callee.*) {
                    ExpressionTag.primary => {
                        try self.call.callee.pushFmt(&buf);
                    },
                    else => {
                        try bufPushText(&buf, "(");
                        try self.call.callee.pushFmt(&buf);
                        try bufPushText(&buf, ")");
                    },
                }

                try bufPushText(&buf, "(");
                try bufPushArgList(&buf, self.call.args);
                try bufPushText(&buf, ")");
            },
            ExpressionTag.index => {
                // Only wrap the value being index if it's not a trivial value
                switch (self.index.callee.*) {
                    ExpressionTag.primary => {
                        try self.index.callee.pushFmt(&buf);
                    },
                    else => {
                        try bufPushText(&buf, "(");
                        try self.index.callee.pushFmt(&buf);
                        try bufPushText(&buf, ")");
                    },
                }

                try bufPushText(&buf, "[");
                try self.index.index.pushFmt(&buf);
                try bufPushText(&buf, "]");
            },
            else => {
                try bufPushFmt(&buf, "TODO_EXPRESSION {any}", .{self});
            },
        }
        return startingSize - buf.len;
    }
};

fn bufPushArgList(buf: *[]u8, exprs: []const Expression) std.fmt.BufPrintError!void {
    for (exprs, 0..) |arg, idx| {
        try arg.pushFmt(buf);

        if (idx != exprs.len - 1) {
            try bufPushText(buf, ", ");
        }
    }
}

pub const TokenTag = enum {
    forKeyword,
    ifKeyword,
    letKeyword,
    fnKeyword,
    whileKeyword,
    returnKeyword,
    structKeyword,
    identifier,
    plus,
    minus,
    bang,
    star,
    div,
    bangEqual,
    doubleEqual,
    equal,
    lt,
    gt,
    lte,
    gte,
    dot,
    nullLiteral,
    boolLiteral,
    intLiteral,
    floatLiteral,
    lCurly,
    rCurly,
    lParen,
    rParen,
    lSquare,
    rSquare,
    comma,
    semicolon,
    stringLiteral,
    doublePipe,
    EOF,
};

pub const Token = union(TokenTag) {
    forKeyword: void,
    ifKeyword: void,
    letKeyword: void,
    fnKeyword: void,
    whileKeyword: void,
    returnKeyword: void,
    structKeyword: void,
    identifier: []const u8,
    plus: void,
    minus: void,
    bang: void,
    star: void,
    div: void,
    bangEqual: void,
    doubleEqual: void,
    equal: void,
    lt: void,
    gt: void,
    lte: void,
    gte: void,
    dot: void,
    nullLiteral: void,
    boolLiteral: bool,
    intLiteral: i64,
    floatLiteral: f64,
    lCurly: void,
    rCurly: void,
    lParen: void,
    rParen: void,
    lSquare: void,
    rSquare: void,
    comma: void,
    semicolon: void,
    stringLiteral: []const u8,
    doublePipe: void,
    EOF: void,

    pub fn eql(self: Token, other: Token) bool {
        if (@as(TokenTag, self) != @as(TokenTag, other)) {
            return false;
        }

        switch (self) {
            TokenTag.identifier => {
                return std.mem.eql(u8, self.identifier, other.identifier);
            },
            TokenTag.stringLiteral => {
                return std.mem.eql(u8, self.stringLiteral, other.stringLiteral);
            },
            else => {},
        }

        return true;
    }
};

pub const TokenizationError = error{
    UnknownCharacter,
};

pub fn tokenize(allocator: std.mem.Allocator, programText: []const u8) ![]const Token {
    var tokens = std.ArrayList(Token).init(allocator);

    var idx: usize = 0;

    outer: while (idx < programText.len) {
        const token: Token = switch (programText[idx]) {
            '+' => Token{ .plus = {} },
            '-' => Token{ .minus = {} },
            '*' => Token{ .star = {} },
            '.' => Token{ .dot = {} },
            '{' => Token{ .lCurly = {} },
            '}' => Token{ .rCurly = {} },
            '(' => Token{ .lParen = {} },
            ')' => Token{ .rParen = {} },
            '[' => Token{ .lSquare = {} },
            ']' => Token{ .rSquare = {} },
            ',' => Token{ .comma = {} },
            ';' => Token{ .semicolon = {} },
            '|' => blk: {
                if (programText.len - idx >= 2 and programText[idx + 1] == '|') {
                    idx = idx + 1;
                    break :blk Token{ .doublePipe = {} };
                } else {
                    return error.ExpectedDoublePipe;
                }
            },
            '/' => blk: {
                if (programText.len - idx >= 2 and programText[idx + 1] == '/') {
                    // This is a comment. Skip text until the end of the line.
                    while (idx < programText.len and programText[idx] != '\n') {
                        idx = idx + 1;
                    }
                    continue :outer;
                } else {
                    break :blk Token{ .div = {} };
                }
            },
            '!' => blk: {
                if (programText.len - idx >= 2 and programText[idx + 1] == '=') {
                    idx = idx + 1;
                    break :blk Token{ .bangEqual = {} };
                } else {
                    break :blk Token{ .bang = {} };
                }
            },
            '=' => blk: {
                if (programText.len - idx >= 2 and programText[idx + 1] == '=') {
                    idx = idx + 1;
                    break :blk Token{ .doubleEqual = {} };
                } else {
                    break :blk Token{ .equal = {} };
                }
            },
            '<' => blk: {
                if (programText.len - idx >= 2 and programText[idx + 1] == '=') {
                    idx = idx + 1;
                    break :blk Token{ .lte = {} };
                } else {
                    break :blk Token{ .lt = {} };
                }
            },
            '>' => blk: {
                if (programText.len - idx >= 2 and programText[idx + 1] == '=') {
                    idx = idx + 1;
                    break :blk Token{ .gte = {} };
                } else {
                    break :blk Token{ .gt = {} };
                }
            },
            ' ', '\t', '\r', '\n' => {
                idx = idx + 1;
                continue :outer;
            },
            '"' => blk: {
                // Start at the first character, not the quote
                const start_idx = idx + 1;
                idx = idx + 2;
                while (idx < programText.len) {
                    if (programText[idx] == '"') {
                        break :blk Token{ .stringLiteral = programText[start_idx..idx] };
                    } else {
                        idx = idx + 1;
                    }
                }

                return error.UnclosedQuote;
            },
            'A'...'Z', 'a'...'z', '_' => blk: {
                const start_idx = idx;
                idx = idx + 1;
                while (idx < programText.len) {
                    switch (programText[idx]) {
                        'A'...'Z', 'a'...'z', '_', '0'...'9' => {
                            idx += 1;
                        },
                        else => {
                            break;
                        },
                    }
                }

                const tk = inner_blk: {
                    // Check for keywords
                    if (idx - start_idx == 2) {
                        if (std.mem.eql(u8, programText[start_idx..idx], "if")) {
                            break :inner_blk Token{ .ifKeyword = {} };
                        } else if (std.mem.eql(u8, programText[start_idx..idx], "fn")) {
                            break :inner_blk Token{ .fnKeyword = {} };
                        }
                    } else if (idx - start_idx == 3) {
                        if (std.mem.eql(u8, programText[start_idx..idx], "let")) {
                            break :inner_blk Token{ .letKeyword = {} };
                        } else if (std.mem.eql(u8, programText[start_idx..idx], "for")) {
                            break :inner_blk Token{ .forKeyword = {} };
                        }
                    } else if (idx - start_idx == 5) {
                        if (std.mem.eql(u8, programText[start_idx..idx], "while")) {
                            break :inner_blk Token{ .whileKeyword = {} };
                        }
                    } else if (idx - start_idx == 6) {
                        if (std.mem.eql(u8, programText[start_idx..idx], "struct")) {
                            break :inner_blk Token{ .structKeyword = {} };
                        } else if (std.mem.eql(u8, programText[start_idx..idx], "return")) {
                            break :inner_blk Token{ .returnKeyword = {} };
                        }
                    }
                    break :inner_blk Token{ .identifier = programText[start_idx..idx] };
                };

                // Make up for the `idx += 1` below
                idx -= 1;
                break :blk tk;
            },
            '0'...'9' => blk: {
                // TODO: parse float
                var num: i64 = programText[idx] - '0';
                idx = idx + 1;
                while (idx < programText.len and '0' <= programText[idx] and programText[idx] <= '9') {
                    num = num * 10 + programText[idx] - '0';
                    idx += 1;
                }
                // Make up for the `idx += 1` below
                idx -= 1;
                break :blk Token{ .intLiteral = num };
            },
            else => {
                std.debug.print("Unknown character {any}", .{programText[idx]});
                @breakpoint();
                @trap();
                //return TokenizationError.UnknownCharacter;
            },
        };
        try tokens.append(token);
        idx += 1;
    }

    try tokens.append(Token{ .EOF = {} });

    return tokens.items;
}

const tokenEq = Token{ .equal = {} };
const tokenLet = Token{ .letKeyword = {} };
const tokenFn = Token{ .fnKeyword = {} };
const tokenComma = Token{ .comma = {} };
const tokenPlus = Token{ .plus = {} };
const tokenSemicolon = Token{ .semicolon = {} };
const tokenRSquare = Token{ .rSquare = {} };
const tokenLSquare = Token{ .lSquare = {} };
const tokenRParen = Token{ .rParen = {} };
const tokenLParen = Token{ .lParen = {} };
const tokenRCurly = Token{ .rCurly = {} };
const tokenLCurly = Token{ .lCurly = {} };

pub fn parseProgramText(allocator: std.mem.Allocator, programText: []const u8) !Ast {
    const tokens = try tokenize(allocator, programText);
    return parseTokens(allocator, tokens);
}

pub fn parseTokens(allocator: std.mem.Allocator, inputTokens: []const Token) ParseError!Ast {
    var declarations = std.ArrayList(Declaration).init(allocator);
    var tokens = TokenStream.new(inputTokens);

    while (!tokens.match(Token.EOF)) {
        const result = parseDeclaration(allocator, &tokens);
        if (result) |declaration| {
            try declarations.append(declaration);
        } else |err| {
            std.debug.print("Tokens remaining  {any}\n{any}\n", .{ tokens.tokens, err });
            return err;
        }
    }

    return Ast{ .allocator = allocator, .declarations = declarations.items };
}

pub fn parseDeclaration(allocator: std.mem.Allocator, tokens: *TokenStream) !Declaration {
    if (tokens.match(TokenTag.EOF)) {
        return error.TokensExhausted;
    }

    if (tokens.match(TokenTag.letKeyword)) {
        return parseVariableDeclaration(allocator, tokens);
    }

    if (tokens.match(TokenTag.structKeyword)) {
        return parseStructDeclaration(allocator, tokens);
    }

    if (tokens.match(TokenTag.fnKeyword)) {
        return parseFnDeclaration(allocator, tokens);
    }

    return parseStatementDeclaration(allocator, tokens);
}

pub fn parseStructDeclaration(_: std.mem.Allocator, _: *TokenStream) !Declaration {
    // NOTE: `struct` already consumed

    return error.NotImplemented;
}

pub fn parseFnDeclaration(_: std.mem.Allocator, _: *TokenStream) !Declaration {
    // NOTE: `fn` already consumed

    return error.NotImplemented;
}

// statement      → exprStmt
//                | forStmt
//                | ifStmt
//                | returnStmt
//                | whileStmt
//                | block ;
//
// exprStmt       → expression ";" ;
// forStmt        → "for" "(" ( varDecl | exprStmt | ";" )
//                            expression? ";"
//                            expression? ")" statement ;
// ifStmt         → "if" "(" expression ")" statement
//                  ( "else" statement )? ;
// returnStmt     → "return" expression? ";" ;
// whileStmt      → "while" "(" expression ")" statement ;
// block          → "{" declaration* "}" ;

pub const StatementTag = enum {
    expression_statement,
    for_statement,
    if_statement,
    return_statement,
    while_statement,
    block_statement,
};

pub const Statement = union(StatementTag) {
    expression_statement: *Expression,
    for_statement: ForStatement,
    if_statement: IfStatement,
    return_statement: ReturnStatement,
    while_statement: WhileStatement,
    block_statement: BlockStatement,

    fn pushFmt(self: Statement, buf: *[]u8) std.fmt.BufPrintError!void {
        const len = try self.fmt(buf.*);
        buf.* = buf.*[len..];
    }

    fn fmt(self: Statement, bufIn: []u8) std.fmt.BufPrintError!usize {
        var buf = bufIn;
        const startingSize = buf.len;
        switch (self) {
            StatementTag.expression_statement => {
                try self.expression_statement.*.pushFmt(&buf);
                try bufPushText(&buf, ";");
            },
            else => {
                try bufPushFmt(&buf, "TODO_STATEMENT {any}", .{self});
            },
        }
        return startingSize - buf.len;
    }
};

const ForStatement = struct {};

const IfStatement = struct {
    expr: Expression,
    block: BlockStatement,
};

const ReturnStatement = struct {
    expr: ?*Expression,
};

const WhileStatement = struct {
    expr: Expression,
    block: BlockStatement,
};

const BlockStatement = struct {
    decl: []Declaration,
};

const ParseError = error{
    NotImplemented,
    OutOfMemory,
    TokensExhausted,
    NoTokensRemaining,
    MismatchedTokenTag,
    UnexpectedToken,
    TODO,
};

pub fn parseStatementDeclaration(allocator: std.mem.Allocator, tokens: *TokenStream) !Declaration {
    if (tokens.match(TokenTag.forKeyword)) {
        return error.NotImplemented;
    }

    if (tokens.match(TokenTag.ifKeyword)) {
        return error.NotImplemented;
    }

    if (tokens.match(TokenTag.whileKeyword)) {
        return error.NotImplemented;
    }

    if (tokens.match(TokenTag.returnKeyword)) {
        if (tokens.match(tokenSemicolon)) {
            return Declaration{ .statement_declaration = Statement{ .return_statement = .{ .expr = null } } };
        }

        const expr = try parseExpression(allocator, tokens);
        _ = try tokens.consume(tokenSemicolon);

        return Declaration{ .statement_declaration = Statement{ .return_statement = .{ .expr = expr } } };
    }

    const expr = try parseExpression(allocator, tokens);
    _ = try tokens.consume(tokenSemicolon);
    return Declaration{ .statement_declaration = Statement{ .expression_statement = expr } };
}

pub fn parseVariableDeclaration(allocator: std.mem.Allocator, tokens: *TokenStream) !Declaration {
    // NOTE: `let` already consumed

    const identToken = try tokens.consume(TokenTag.identifier);

    var expression: ?Expression = null;
    if (tokens.match(TokenTag.equal)) {
        expression = (try parseExpression(allocator, tokens)).*;
    }

    _ = try tokens.consume(tokenSemicolon);

    return .{ .variable_declaration = .{ .name = identToken.identifier, .expr = expression } };
}

pub fn parseExpressionVal(allocator: std.mem.Allocator, tokens: *TokenStream) ParseError!Expression {
    return (try parseLogicOr(allocator, tokens)).*;
}

pub fn parseExpression(allocator: std.mem.Allocator, tokens: *TokenStream) ParseError!*Expression {
    return parseLogicOr(allocator, tokens);
}
//                logic_or       → logic_and ( "or" logic_and )* ;
//                logic_and      → equality ( "and" equality )* ;
//                equality       → comparison ( ( "!=" | "==" ) comparison )* ;
//                comparison     → term ( ( ">" | ">=" | "<" | "<=" ) term )* ;
//                term           → factor ( ( "-" | "+" ) factor )* ;
//                factor         → unary ( ( "/" | "*" ) unary )* ;
//
//                unary          → ( "!" | "-" ) unary | call ;
//                call           → primary ( "(" arguments? ")" | "." IDENTIFIER )* ;
//                primary        → "true" | "false" | "null"
//                               | NUMBER | STRING | IDENTIFIER | "(" expression ")"

pub fn parseLogicOr(allocator: std.mem.Allocator, tokens: *TokenStream) !*Expression {
    const a = try parseLogicAnd(allocator, tokens);
    if (tokens.match(Token.doublePipe)) {
        const b = try parseLogicAnd(allocator, tokens);
        const expr = try allocator.create(Expression);
        expr.* = Expression{ .logicalOr = .{ .a = a, .b = b } };
        return expr;
    }
    return a;
}

pub fn parseLogicAnd(allocator: std.mem.Allocator, tokens: *TokenStream) !*Expression {
    const a = try parseEquality(allocator, tokens);
    if (tokens.match(Token.doubleEqual)) {
        const b = try parseEquality(allocator, tokens);
        const expr = try allocator.create(Expression);
        expr.* = Expression{ .logicalAnd = .{ .a = a, .b = b } };
        return expr;
    }
    return a;
}

pub fn parseEquality(allocator: std.mem.Allocator, tokens: *TokenStream) !*Expression {
    const a = try parseComparison(allocator, tokens);
    if (tokens.match(Token.bangEqual)) {
        const b = try parseComparison(allocator, tokens);
        const expr = try allocator.create(Expression);
        expr.* = Expression{ .notEqual = .{ .a = a, .b = b } };
        return expr;
    } else if (tokens.match(Token.doubleEqual)) {
        const b = try parseComparison(allocator, tokens);
        const expr = try allocator.create(Expression);
        expr.* = Expression{ .equal = .{ .a = a, .b = b } };
        return expr;
    }
    return a;
}

pub fn parseComparison(allocator: std.mem.Allocator, tokens: *TokenStream) !*Expression {
    const a = try parseTerm(allocator, tokens);
    if (tokens.match(Token.gt)) {
        const b = try parseTerm(allocator, tokens);
        const expr = try allocator.create(Expression);
        expr.* = Expression{ .gt = .{ .a = a, .b = b } };
        return expr;
    } else if (tokens.match(Token.gte)) {
        const b = try parseTerm(allocator, tokens);
        const expr = try allocator.create(Expression);
        expr.* = Expression{ .gte = .{ .a = a, .b = b } };
        return expr;
    } else if (tokens.match(Token.lt)) {
        const b = try parseTerm(allocator, tokens);
        const expr = try allocator.create(Expression);
        expr.* = Expression{ .lt = .{ .a = a, .b = b } };
        return expr;
    } else if (tokens.match(Token.lte)) {
        const b = try parseTerm(allocator, tokens);
        const expr = try allocator.create(Expression);
        expr.* = Expression{ .lte = .{ .a = a, .b = b } };
        return expr;
    }
    return a;
}

pub fn parseTerm(allocator: std.mem.Allocator, tokens: *TokenStream) !*Expression {
    const a = try parseFactor(allocator, tokens);
    if (tokens.match(Token.plus)) {
        const b = try parseFactor(allocator, tokens);
        const expr = try allocator.create(Expression);
        expr.* = Expression{ .plus = .{ .a = a, .b = b } };
        return expr;
    } else if (tokens.match(Token.minus)) {
        const b = try parseFactor(allocator, tokens);
        const expr = try allocator.create(Expression);
        expr.* = Expression{ .minus = .{ .a = a, .b = b } };
        return expr;
    }
    return a;
}

pub fn parseFactor(allocator: std.mem.Allocator, tokens: *TokenStream) !*Expression {
    const a = try parseUnary(allocator, tokens);
    if (tokens.match(Token.div)) {
        const b = try parseUnary(allocator, tokens);
        const expr = try allocator.create(Expression);
        expr.* = Expression{ .divide = .{ .a = a, .b = b } };
        return expr;
    } else if (tokens.match(Token.star)) {
        const b = try parseUnary(allocator, tokens);
        const expr = try allocator.create(Expression);
        expr.* = Expression{ .multiply = .{ .a = a, .b = b } };
        return expr;
    }
    return a;
}

pub fn parseUnary(allocator: std.mem.Allocator, tokens: *TokenStream) !*Expression {
    if (tokens.match(Token.minus)) {
        const a = try parseUnary(allocator, tokens);
        const expr = try allocator.create(Expression);
        expr.* = Expression{ .negation = .{ .a = a } };
        return expr;
    } else if (tokens.match(Token.bang)) {
        const a = try parseUnary(allocator, tokens);
        const expr = try allocator.create(Expression);
        expr.* = Expression{ .not = .{ .a = a } };
        return expr;
    }
    return try parseCall(allocator, tokens);
}

pub fn parseCall(allocator: std.mem.Allocator, tokens: *TokenStream) !*Expression {
    const callee = try parsePrimary(allocator, tokens);
    if (tokens.match(Token.lParen)) {
        const args = try parseArguments(allocator, tokens);
        _ = try tokens.consume(Token.rParen);

        const expr = try allocator.create(Expression);
        expr.* = Expression{ .call = CallInfo{ .callee = callee, .args = args } };
        return expr;
    }

    if (tokens.match(Token.lSquare)) {
        const index = try parseExpression(allocator, tokens);
        _ = try tokens.consume(Token.rSquare);

        const expr = try allocator.create(Expression);
        expr.* = Expression{ .index = IndexInfo{ .callee = callee, .index = index } };
        return expr;
    }

    return callee;
}

pub fn parsePrimary(allocator: std.mem.Allocator, tokens: *TokenStream) !*Expression {
    if (tokens.captureMatch(Token.intLiteral)) |val| {
        const expr = try allocator.create(Expression);
        expr.* = Expression{ .primary = .{ .intLiteral = val.intLiteral } };
        return expr;
    }

    if (tokens.captureMatch(Token.floatLiteral)) |val| {
        const expr = try allocator.create(Expression);
        expr.* = Expression{ .primary = .{ .floatLiteral = val.floatLiteral } };
        return expr;
    }

    if (tokens.captureMatch(Token.boolLiteral)) |val| {
        const expr = try allocator.create(Expression);
        expr.* = Expression{ .primary = .{ .boolLiteral = val.boolLiteral } };
        return expr;
    }

    if (tokens.match(Token.nullLiteral)) {
        const expr = try allocator.create(Expression);
        expr.* = Expression{ .primary = .{ .nullLiteral = {} } };
        return expr;
    }

    if (tokens.captureMatch(Token.stringLiteral)) |val| {
        const expr = try allocator.create(Expression);
        expr.* = Expression{ .primary = .{ .stringLiteral = val.stringLiteral } };
        return expr;
    }

    if (tokens.captureMatch(Token.identifier)) |val| {
        const expr = try allocator.create(Expression);
        expr.* = Expression{ .primary = .{ .identifier = val.identifier } };
        return expr;
    }

    if (tokens.match(Token.lSquare)) {
        const args = try parseArguments(allocator, tokens);
        _ = try tokens.consume(Token.rSquare);

        const expr = try allocator.create(Expression);
        expr.* = Expression{ .primary = .{ .listLiteral = args } };
        return expr;
    }

    if (tokens.match(Token.lParen)) {
        const expr = parseExpression(allocator, tokens);
        _ = try tokens.consume(Token.rParen);

        return expr;
    }

    return error.UnexpectedToken;
}

pub fn parseArguments(allocator: std.mem.Allocator, tokens: *TokenStream) ![]const Expression {
    var args = std.ArrayList(Expression).init(allocator);
    while (true) {
        try args.append(try parseExpressionVal(allocator, tokens));
        if (tokens.match(tokenComma)) {
            continue;
        } else {
            // TODO: support trailing comma
            break;
        }
    }

    return args.items;
}

const TokenStream = struct {
    tokens: []const Token,

    pub fn new(tokens: []const Token) TokenStream {
        return TokenStream{ .tokens = tokens };
    }

    pub fn pop(self: *TokenStream) !Token {
        if (self.tokens.len == 0) {
            return error.NoTokensRemaining;
        }

        const token = self.tokens[0];
        self.*.tokens = self.tokens[1..];
        return token;
    }

    /// Consumes token if present returning the token, null otherwise.
    pub fn captureMatch(self: *TokenStream, tag: TokenTag) ?Token {
        if (self.*.tokens.len == 0) {
            return null;
        }

        if (@as(TokenTag, self.*.tokens[0]) == tag) {
            const token = self.tokens[0];
            self.*.tokens = self.*.tokens[1..];
            return token;
        }

        return null;
    }

    /// Consumes token if present returning true. Return false otherwise.
    pub fn match(self: *TokenStream, tag: TokenTag) bool {
        if (self.*.tokens.len == 0) {
            return false;
        }

        if (@as(TokenTag, self.*.tokens[0]) == tag) {
            self.*.tokens = self.*.tokens[1..];
            return true;
        }

        return false;
    }

    /// Unconditionally consumes a token if it matches, returning an error
    pub fn consume(self: *TokenStream, tag: TokenTag) !Token {
        if (self.tokens.len == 0) {
            return error.NoTokensRemaining;
        }

        if (@as(TokenTag, self.tokens[0]) != tag) {
            std.debug.print("Expected token {any} found {any}\n", .{ tag, self.tokens[0] });
            return error.MismatchedTokenTag;
        }

        const token = self.tokens[0];
        self.*.tokens = self.tokens[1..];
        return token;
    }
};

pub fn testTokenization(script: []const u8, expectedTokens: []const Token) !void {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const tokens = try tokenize(allocator, script);

    try expectEqualTokens(tokens, expectedTokens);
}

/// Copied from std.testing.SliceDiffer
fn SliceDiffer(comptime T: type) type {
    return struct {
        start_index: usize,
        expected: []const T,
        actual: []const T,
        ttyconf: std.io.tty.Config,

        const Self = @This();

        pub fn write(self: Self, writer: anytype) !void {
            for (self.expected, 0..) |value, i| {
                const full_index = self.start_index + i;
                const diff = if (i < self.actual.len) !(if (T == Token) self.actual[i].eql(value) else std.meta.eql(self.actual[i], value)) else true;
                if (diff) try self.ttyconf.setColor(writer, .red);
                if (@typeInfo(T) == .Pointer) {
                    try writer.print("[{}]{*}: {any}\n", .{ full_index, value, value });
                } else {
                    try writer.print("[{}]: {any}\n", .{ full_index, value });
                }
                if (diff) try self.ttyconf.setColor(writer, .reset);
            }
        }
    };
}

/// Copied from std.testing.expectEqualSlices
pub fn expectEqualTokens(expected: []const Token, actual: []const Token) !void {
    if (expected.ptr == actual.ptr and expected.len == actual.len) {
        return;
    }
    const diff_index: usize = diff_index: {
        const shortest = @min(expected.len, actual.len);
        var index: usize = 0;
        while (index < shortest) : (index += 1) {
            if (!actual[index].eql(expected[index])) break :diff_index index;
        }
        break :diff_index if (expected.len == actual.len) return else shortest;
    };

    if (!std.testing.backend_can_print) {
        return error.TestExpectedEqual;
    }

    std.debug.print("slices differ. first difference occurs at index {d} (0x{X})\n", .{ diff_index, diff_index });

    // TODO: Should this be configurable by the caller?
    const max_lines: usize = 16;
    const max_window_size: usize = max_lines;

    // Print a maximum of max_window_size items of each input, starting just before the
    // first difference to give a bit of context.
    var window_start: usize = 0;
    if (@max(actual.len, expected.len) > max_window_size) {
        const alignment = 2;
        window_start = std.mem.alignBackward(usize, diff_index - @min(diff_index, alignment), alignment);
    }
    const expected_window = expected[window_start..@min(expected.len, window_start + max_window_size)];
    const expected_truncated = window_start + expected_window.len < expected.len;
    const actual_window = actual[window_start..@min(actual.len, window_start + max_window_size)];
    const actual_truncated = window_start + actual_window.len < actual.len;

    const stderr = std.io.getStdErr();
    const ttyconf = std.io.tty.detectConfig(stderr);
    var differ = SliceDiffer(Token){
        .start_index = window_start,
        .expected = expected_window,
        .actual = actual_window,
        .ttyconf = ttyconf,
    };

    // Print indexes as hex for slices of u8 since it's more likely to be binary data where
    // that is usually useful.
    const index_fmt = "{}";

    std.debug.print("\n============ expected this output: =============  len: {} (0x{X})\n\n", .{ expected.len, expected.len });
    if (window_start > 0) {
        std.debug.print("... truncated ...\n", .{});
    }
    differ.write(stderr.writer()) catch {};
    if (expected_truncated) {
        const num_missing_items = expected.len - (window_start + expected_window.len);
        std.debug.print("... truncated, remaining items: " ++ index_fmt ++ " ...\n", .{num_missing_items});
    }

    // now reverse expected/actual and print again
    differ.expected = actual_window;
    differ.actual = expected_window;
    std.debug.print("\n============= instead found this: ==============  len: {} (0x{X})\n\n", .{ actual.len, actual.len });
    if (window_start > 0) {
        std.debug.print("... truncated ...\n", .{});
    }
    differ.write(stderr.writer()) catch {};
    if (actual_truncated) {
        const num_missing_items = actual.len - (window_start + actual_window.len);
        std.debug.print("... truncated, remaining items: " ++ index_fmt ++ " ...\n", .{num_missing_items});
    }
    std.debug.print("\n================================================\n\n", .{});

    return error.TestExpectedEqual;
}

test "basic tokenization" {
    try testTokenization(
        \\let lst = [1, 2, 3];
        \\print(lst[0]);
    ,
        &[_]Token{
            // let lst = [1, 2, 3];
            tokenLet,
            Token{ .identifier = "lst" },
            tokenEq,
            tokenLSquare,
            Token{ .intLiteral = 1 },
            tokenComma,
            Token{ .intLiteral = 2 },
            tokenComma,
            Token{ .intLiteral = 3 },
            tokenRSquare,
            tokenSemicolon,

            // print(lst[0]);
            Token{ .identifier = "print" },
            tokenLParen,
            Token{ .identifier = "lst" },
            tokenLSquare,
            Token{ .intLiteral = 0 },
            tokenRSquare,
            tokenRParen,
            tokenSemicolon,
        },
    );

    try testTokenization("let+let;", &[_]Token{
        tokenLet,
        tokenPlus,
        tokenLet,
        tokenSemicolon,
    });
}
