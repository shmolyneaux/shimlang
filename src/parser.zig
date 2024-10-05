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

        var buf = allocator.alloc(u8, 1 << 16) catch empty_buf;
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

pub const Declaration = struct {
    decl: union(DeclarationTag) {
        struct_declaration: StructDeclaration,
        function_declaration: FunctionDeclaration,
        variable_declaration: VariableDeclaration,
        statement_declaration: Statement,
    },
    startIdx: usize,
    endIdx: usize,
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

pub const PropInfo = struct {
    obj: *Expression,
    name: []const u8,
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
    getProp,
    primary,
};

pub const ExpressionInner = union(ExpressionTag) {
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
    getProp: PropInfo,
    primary: Primary,
};

pub const Expression = struct {
    expr: ExpressionInner,
    startIdx: usize,
    endIdx: usize,

    fn pushFmt(self: Expression, buf: *[]u8) std.fmt.BufPrintError!void {
        const len = try self.fmt(buf.*);
        buf.* = buf.*[len..];
    }

    fn fmt(self: Expression, bufIn: []u8) !usize {
        var buf = bufIn;
        const startingSize = buf.len;
        switch (self.expr) {
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
    orKeyword,
    andKeyword,
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
    orKeyword: void,
    andKeyword: void,
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

const TokenLoc = struct {
    token: Token,
    start: usize,
    end: usize,
};

pub fn tokenize(allocator: std.mem.Allocator, programText: []const u8) ![]const TokenLoc {
    var tokens = std.ArrayList(TokenLoc).init(allocator);

    var idx: usize = 0;

    outer: while (idx < programText.len) {
        const tokenStartIdx = idx;
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
                idx = start_idx;
                while (idx < programText.len) {
                    if (programText[idx] == '"') {
                        break :blk Token{ .stringLiteral = programText[start_idx..idx] };
                    } else {
                        idx = idx + 1;
                    }
                }

                std.debug.print("Got text: {s}", .{programText[start_idx..]});
                std.debug.print("Text remaining: {s}", .{programText[idx..]});
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
                        } else if (std.mem.eql(u8, programText[start_idx..idx], "or")) {
                            break :inner_blk Token{ .orKeyword = {} };
                        }
                    } else if (idx - start_idx == 3) {
                        if (std.mem.eql(u8, programText[start_idx..idx], "let")) {
                            break :inner_blk Token{ .letKeyword = {} };
                        } else if (std.mem.eql(u8, programText[start_idx..idx], "for")) {
                            break :inner_blk Token{ .forKeyword = {} };
                        } else if (std.mem.eql(u8, programText[start_idx..idx], "and")) {
                            break :inner_blk Token{ .andKeyword = {} };
                        }
                    } else if (idx - start_idx == 4) {
                        if (std.mem.eql(u8, programText[start_idx..idx], "true")) {
                            break :inner_blk Token{ .boolLiteral = true };
                        }
                    } else if (idx - start_idx == 5) {
                        if (std.mem.eql(u8, programText[start_idx..idx], "while")) {
                            break :inner_blk Token{ .whileKeyword = {} };
                        } else if (std.mem.eql(u8, programText[start_idx..idx], "false")) {
                            break :inner_blk Token{ .boolLiteral = false };
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
                const startIdx = idx;
                var foundDecimal = false;
                while (idx < programText.len) {
                    if (programText[idx] == '.') {
                        if (foundDecimal) {
                            // This is a second dot in a number, so we'll ignore it
                            // and probably barf at the parsing stage
                            break;
                        }
                        foundDecimal = true;
                    } else if (programText[idx] < '0' or programText[idx] > '9') {
                        break;
                    }
                    idx += 1;
                }

                var token: Token = undefined;
                if (foundDecimal) {
                    const num = try std.fmt.parseFloat(f64, programText[startIdx..idx]);
                    token = Token{ .floatLiteral = num };
                } else {
                    const num = try std.fmt.parseInt(i64, programText[startIdx..idx], 10);
                    token = Token{ .intLiteral = num };
                }

                // Make up for the `idx += 1` below
                idx -= 1;
                break :blk token;
            },
            else => {
                std.debug.print("Unknown character {any}", .{programText[idx]});
                @breakpoint();
                @trap();
                //return TokenizationError.UnknownCharacter;
            },
        };
        try tokens.append(.{ .token = token, .start = tokenStartIdx, .end = idx });
        idx += 1;
    }

    try tokens.append(.{ .token = Token{ .EOF = {} }, .start = idx, .end = idx });

    return tokens.items;
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
    assignment_statement,
    for_statement,
    if_statement,
    return_statement,
    while_statement,
    block_statement,
};

pub const AssignmentStatement = struct {
    obj: ?Expression,
    ident: []const u8,
    expr: Expression,
};

pub const Statement = union(StatementTag) {
    expression_statement: *Expression,
    assignment_statement: AssignmentStatement,
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

pub const ParseErrorInfo = struct { idx: usize };

pub const Parser = struct {
    allocator: std.mem.Allocator,

    // The start index of the most-recently-consumed token
    prevIdxStart: usize,
    // The end index of the most-recently-consumed token
    prevIdx: usize,

    tokens: TokenStream,
    programText: []const u8,
    errInfo: ?ParseErrorInfo,

    pub fn new(allocator: std.mem.Allocator, programText: []const u8) !Parser {
        const inputTokens = try tokenize(allocator, programText);
        const tokens = TokenStream.new(inputTokens);
        return .{ .allocator = allocator, .prevIdxStart = 0, .prevIdx = 0, .tokens = tokens, .programText = programText, .errInfo = null };
    }

    fn captureMatch(self: *Parser, tag: TokenTag) ?Token {
        const res = self.tokens.captureMatch(tag);
        if (res) |token| {
            self.prevIdxStart = token.start;
            self.prevIdx = token.end;
            return token.token;
        } else {
            return null;
        }
    }

    fn match(self: *Parser, tag: TokenTag) bool {
        const res = self.tokens.captureMatch(tag);
        if (res) |token| {
            self.prevIdxStart = token.start;
            self.prevIdx = token.end;
            return true;
        } else {
            return false;
        }
    }

    fn consume(self: *Parser, tag: TokenTag) !Token {
        const token = try self.tokens.consume(tag);
        self.prevIdxStart = token.start;
        self.prevIdx = token.end;
        return token.token;
    }

    pub fn parseProgramText(self: *Parser) !Ast {
        return self.parseTokens() catch |err| {
            if (self.errInfo) |info| {
                std.debug.print("Error at idx {}: {}\n", .{ info, err });

                var printColumn: ?usize = null;
                var column: usize = 0;
                for (self.programText, 0..) |c, idx| {
                    std.debug.print("{c}", .{c});
                    if (idx == info.idx) {
                        printColumn = column;
                    }
                    if (c == '\n') {
                        if (printColumn) |spaceCount| {
                            for (0..spaceCount) |_| {
                                std.debug.print(" ", .{});
                            }
                            std.debug.print("^\n", .{});
                        }
                        column = 0;
                    } else {
                        column += 1;
                    }
                }
                std.debug.print("\n", .{});
            } else {
                std.debug.print("Unexpected parser error {}\n", .{err});
            }
            return err;
        };
    }

    pub fn parseTokens(self: *Parser) ParseError!Ast {
        var declarations = std.ArrayList(Declaration).init(self.allocator);

        while (!self.match(Token.EOF)) {
            const result = self.parseDeclaration();
            if (result) |declaration| {
                try declarations.append(declaration);
            } else |err| {
                if (self.tokens.tokens.len != 0) {
                    self.errInfo = ParseErrorInfo{
                        .idx = self.tokens.tokens[0].start,
                    };
                }
                return err;
            }
        }

        return Ast{ .allocator = self.allocator, .declarations = declarations.items };
    }

    pub fn parseDeclaration(self: *Parser) !Declaration {
        if (self.match(TokenTag.EOF)) {
            return error.TokensExhausted;
        }

        if (self.match(TokenTag.letKeyword)) {
            return self.parseVariableDeclaration();
        }

        if (self.match(TokenTag.structKeyword)) {
            return self.parseStructDeclaration();
        }

        if (self.match(TokenTag.fnKeyword)) {
            return self.parseFnDeclaration();
        }

        return self.parseStatementDeclaration();
    }

    pub fn parseStructDeclaration(_: *Parser) !Declaration {
        // NOTE: `struct` already consumed

        std.debug.print("line: {}  struct parsing not implemented\n", .{@src().line});
        return error.NotImplemented;
    }

    pub fn parseFnDeclaration(_: *Parser) !Declaration {
        // NOTE: `fn` already consumed

        std.debug.print("line: {}  function parsing not implemented\n", .{@src().line});
        return error.NotImplemented;
    }

    fn newDecl(self: *Parser, startIdx: usize, info: anytype) Declaration {
        if (@TypeOf(info) == Statement) {
            return Declaration{ .startIdx = startIdx, .endIdx = self.prevIdx, .decl = .{
                .statement_declaration = info,
            } };
        } else {
            @compileError("Can only use Statement for now");
        }
    }

    fn newExpr(self: *Parser, startIdx: usize, expr: anytype) Expression {
        return Expression{ .startIdx = startIdx, .endIdx = self.prevIdx, .expr = expr };
    }

    // The start idx of the next token
    fn nextIdx(self: *Parser) usize {
        if (self.tokens.tokens.len != 0) {
            return self.tokens.tokens[0].start;
        } else {
            return self.programText.len;
        }
    }

    pub fn parseStatementDeclaration(self: *Parser) ParseError!Declaration {
        const idx = self.nextIdx();
        if (self.match(TokenTag.forKeyword)) {
            std.debug.print("line: {}  for loop parsing not implemented\n", .{@src().line});
            return error.NotImplemented;
        }

        if (self.match(TokenTag.ifKeyword)) {
            const expr = try self.parseExpressionVal();

            _ = try self.consume(TokenTag.lCurly);

            var declList = std.ArrayList(Declaration).init(self.allocator);
            while (!self.match(TokenTag.rCurly)) {
                const decl = try self.parseDeclaration();
                try declList.append(decl);
            }

            // TODO: implement else

            return self.newDecl(idx, Statement{
                .if_statement = .{
                    .expr = expr,
                    .block = .{ .decl = declList.items },
                },
            });
        }

        if (self.match(TokenTag.whileKeyword)) {
            const expr = try self.parseExpressionVal();
            var declList = std.ArrayList(Declaration).init(self.allocator);

            _ = try self.consume(TokenTag.lCurly);

            while (!self.match(TokenTag.rCurly)) {
                const decl = try self.parseDeclaration();
                try declList.append(decl);
            }

            return Declaration{ .startIdx = idx, .endIdx = self.prevIdx, .decl = .{
                .statement_declaration = Statement{
                    .while_statement = .{
                        .expr = expr,
                        .block = .{ .decl = declList.items },
                    },
                },
            } };
        }

        if (self.match(TokenTag.returnKeyword)) {
            if (self.match(TokenTag.semicolon)) {
                return self.newDecl(idx, Statement{ .return_statement = .{ .expr = null } });
            }

            const expr = try self.parseExpression();
            _ = try self.consume(TokenTag.semicolon);

            return self.newDecl(idx, Statement{ .return_statement = .{ .expr = expr } });
        }

        const expr = try self.parseExpression();

        switch (expr.*.expr) {
            ExpressionTag.primary => |prim| {
                switch (prim) {
                    PrimaryTag.identifier => |ident| {
                        if (self.match(TokenTag.equal)) {
                            const valExpr = try self.parseExpression();
                            _ = try self.consume(TokenTag.semicolon);
                            return self.newDecl(idx, Statement{ .assignment_statement = .{
                                .obj = null,
                                .ident = ident,
                                .expr = valExpr.*,
                            } });
                        }
                    },
                    else => {},
                }
            },
            ExpressionTag.getProp => |propInfo| {
                if (self.match(TokenTag.equal)) {
                    const valExpr = try self.parseExpression();
                    _ = try self.consume(TokenTag.semicolon);
                    return self.newDecl(idx, Statement{ .assignment_statement = .{
                        .obj = propInfo.obj.*,
                        .ident = propInfo.name,
                        .expr = valExpr.*,
                    } });
                }
            },
            // TODO: assign to indexed list
            else => {},
        }

        _ = try self.consume(TokenTag.semicolon);
        return self.newDecl(idx, Statement{ .expression_statement = expr });
    }

    pub fn parseVariableDeclaration(self: *Parser) !Declaration {
        // NOTE: `let` already consumed

        const identToken = try self.consume(TokenTag.identifier);

        var expression: ?Expression = null;
        if (self.match(TokenTag.equal)) {
            expression = (try self.parseExpression()).*;
        }

        _ = try self.consume(TokenTag.semicolon);

        return .{ .startIdx = self.prevIdxStart, .endIdx = self.prevIdx, .decl = .{ .variable_declaration = .{ .name = identToken.identifier, .expr = expression } } };
    }

    pub fn parseExpressionVal(self: *Parser) ParseError!Expression {
        return (try self.parseLogicOr()).*;
    }

    pub fn parseExpression(self: *Parser) ParseError!*Expression {
        return self.parseLogicOr();
    }

    const parseLogicOr = parseBinaryOp(parseLogicAnd, .{
        .logicalOr = Token.orKeyword,
    });
    const parseLogicAnd = parseBinaryOp(parseEquality, .{
        .logicalAnd = Token.andKeyword,
    });
    const parseEquality = parseBinaryOp(parseComparison, .{
        .notEqual = Token.bangEqual,
        .equal = Token.doubleEqual,
    });
    const parseComparison = parseBinaryOp(parseTerm, .{
        .gt = Token.gt,
        .gte = Token.gte,
        .lt = Token.lte,
        .lte = Token.lte,
    });
    const parseTerm = parseBinaryOp(parseFactor, .{
        .plus = Token.plus,
        .minus = Token.minus,
    });
    const parseFactor = parseBinaryOp(parseUnary, .{
        .divide = Token.div,
        .multiply = Token.star,
    });

    pub fn parseUnary(self: *Parser) !*Expression {
        const idx = self.nextIdx();
        if (self.match(Token.minus)) {
            const a = try self.parseUnary();
            const expr = try self.allocator.create(Expression);
            expr.* = self.newExpr(idx, .{ .negation = .{ .a = a } });
            return expr;
        } else if (self.match(Token.bang)) {
            const a = try self.parseUnary();
            const expr = try self.allocator.create(Expression);
            expr.* = self.newExpr(idx, .{ .not = .{ .a = a } });
            return expr;
        }
        return try self.parseCall();
    }

    pub fn parseCall(self: *Parser) !*Expression {
        const idx = self.nextIdx();
        var expr = try self.parsePrimary();
        while (true) {
            if (self.match(Token.lParen)) {
                if (!self.match(Token.rParen)) {
                    const args = try self.parseArguments();
                    _ = try self.consume(Token.rParen);

                    const callee = expr;
                    expr = try self.allocator.create(Expression);
                    expr.* = self.newExpr(idx, .{ .call = CallInfo{ .callee = callee, .args = args } });
                } else {
                    const args = try self.allocator.alloc(Expression, 0);
                    const callee = expr;
                    expr = try self.allocator.create(Expression);
                    expr.* = self.newExpr(idx, .{ .call = CallInfo{ .callee = callee, .args = args } });
                }
            } else if (self.match(Token.lSquare)) {
                const index = try self.parseExpression();
                _ = try self.consume(Token.rSquare);

                const callee = expr;
                expr = try self.allocator.create(Expression);
                expr.* = self.newExpr(idx, .{ .index = IndexInfo{ .callee = callee, .index = index } });
            } else if (self.match(Token.dot)) {
                const ident = try self.consume(Token.identifier);
                const callee = expr;
                expr = try self.allocator.create(Expression);
                expr.* = self.newExpr(idx, .{ .getProp = .{ .obj = callee, .name = ident.identifier } });
            } else {
                break;
            }
        }

        return expr;
    }

    pub fn parsePrimary(self: *Parser) !*Expression {
        const idx = self.nextIdx();
        if (self.captureMatch(Token.intLiteral)) |val| {
            const expr = try self.allocator.create(Expression);
            expr.* = self.newExpr(idx, .{ .primary = .{ .intLiteral = val.intLiteral } });
            return expr;
        }

        if (self.captureMatch(Token.floatLiteral)) |val| {
            const expr = try self.allocator.create(Expression);
            expr.* = self.newExpr(idx, .{ .primary = .{ .floatLiteral = val.floatLiteral } });
            return expr;
        }

        if (self.captureMatch(Token.boolLiteral)) |val| {
            const expr = try self.allocator.create(Expression);
            expr.* = self.newExpr(idx, .{ .primary = .{ .boolLiteral = val.boolLiteral } });
            return expr;
        }

        if (self.match(Token.nullLiteral)) {
            const expr = try self.allocator.create(Expression);
            expr.* = self.newExpr(idx, .{ .primary = .{ .nullLiteral = {} } });
            return expr;
        }

        if (self.captureMatch(Token.stringLiteral)) |val| {
            const expr = try self.allocator.create(Expression);
            expr.* = self.newExpr(idx, .{ .primary = .{ .stringLiteral = val.stringLiteral } });
            return expr;
        }

        if (self.captureMatch(Token.identifier)) |val| {
            const expr = try self.allocator.create(Expression);
            expr.* = self.newExpr(idx, .{ .primary = .{ .identifier = val.identifier } });
            return expr;
        }

        if (self.match(Token.lSquare)) {
            var args: []const Expression = undefined;
            if (!self.match(Token.rSquare)) {
                args = try self.parseArguments();
                _ = try self.consume(Token.rSquare);
            } else {
                args = try self.allocator.alloc(Expression, 0);
            }

            const expr = try self.allocator.create(Expression);
            expr.* = self.newExpr(idx, .{ .primary = .{ .listLiteral = args } });
            return expr;
        }

        if (self.match(Token.lParen)) {
            // NOTE: this doesn't (yet?) support the unit type ()
            const expr = self.parseExpression();
            _ = try self.consume(Token.rParen);

            return expr;
        }

        return error.UnexpectedToken;
    }

    pub fn parseArguments(self: *Parser) ![]const Expression {
        var args = std.ArrayList(Expression).init(self.allocator);
        while (true) {
            try args.append(try self.parseExpressionVal());
            if (self.match(TokenTag.comma)) {
                continue;
            } else {
                // TODO: support trailing comma
                break;
            }
        }

        return args.items;
    }
};
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

pub fn parseBinaryOp(
    comptime higherPrecedence: fn (*Parser) ParseError!*Expression,
    comptime opTable: anytype,
) fn (*Parser) ParseError!*Expression {
    return struct {
        fn parser(self: *Parser) ParseError!*Expression {
            const idx = self.nextIdx();
            var expr = try higherPrecedence(self);

            while (true) {
                var shouldBreak = true;
                inline for (std.meta.fields(@TypeOf(opTable))) |field| {
                    if (self.match(@field(opTable, field.name))) {
                        const a = expr;
                        const b = try higherPrecedence(self);

                        expr = try self.allocator.create(Expression);
                        expr.* = self.newExpr(idx, @unionInit(ExpressionInner, field.name, .{ .a = a, .b = b }));

                        shouldBreak = false;
                        break;
                    }
                }
                if (shouldBreak) {
                    break;
                }
            }
            return expr;
        }
    }.parser;
}

const TokenStream = struct {
    tokens: []const TokenLoc,

    pub fn new(tokens: []const TokenLoc) TokenStream {
        return TokenStream{ .tokens = tokens };
    }

    /// Consumes token if present returning the token, null otherwise.
    pub fn captureMatch(self: *TokenStream, tag: TokenTag) ?TokenLoc {
        if (self.*.tokens.len == 0) {
            return null;
        }

        if (@as(TokenTag, self.*.tokens[0].token) == tag) {
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

        if (@as(TokenTag, self.*.tokens[0].token) == tag) {
            self.*.tokens = self.*.tokens[1..];
            return true;
        }

        return false;
    }

    /// Unconditionally consumes a token if it matches, returning an error
    pub fn consume(self: *TokenStream, tag: TokenTag) !TokenLoc {
        if (self.tokens.len == 0) {
            return error.NoTokensRemaining;
        }

        if (@as(TokenTag, self.tokens[0].token) != tag) {
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
    const tokenEq = Token{ .equal = {} };
    const tokenLet = Token{ .letKeyword = {} };
    const tokenComma = Token{ .comma = {} };
    const tokenPlus = Token{ .plus = {} };
    const tokenSemicolon = Token{ .semicolon = {} };
    const tokenRSquare = Token{ .rSquare = {} };
    const tokenLSquare = Token{ .lSquare = {} };
    const tokenRParen = Token{ .rParen = {} };
    const tokenLParen = Token{ .lParen = {} };

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
