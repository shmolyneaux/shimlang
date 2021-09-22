const build_options = @import("build_options");
const std = @import("std");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;

// Limited precision version numbers have never been a problem for anyone /s
const VersionInfo = struct {
    major: u8,
    minor: u8,
    patch: u8,
    release_info: []const u8,

    pub fn version_string(comptime self: VersionInfo) []const u8 {
        return self.release_info;
    }
};

pub const version: VersionInfo = .{
    .major = 0,
    .minor = 0,
    .patch = 0,
    .release_info = "unreleased",
};

// Static parser types

const File = struct {
    shebang: ArrayList(u8),
    ast: Ast,
};

const Ast = struct {
    allocator: *Allocator,
    stmts: ArrayList(Statement),

    pub fn deinit(self: Ast) void {
        for (self.stmts.items) |stmt| {
            stmt.deinit(self.allocator);
        }
        self.stmts.deinit();
    }

    pub fn format(
        self: Ast,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        try writer.writeAll("AST([");

        var first = true;
        for (self.stmts.items) |stmt| {
            if (!first) {
                try writer.writeAll(",");
            }
            try writer.print("{}", .{stmt});
            first = false;
        }

        try writer.writeAll("])");
    }
};

// TODO: Multiple sorts of statements (if, for, use, while, etc.)
const Statement = struct {
    expr: Expression,

    pub fn deinit(self: Statement, allocator: *Allocator) void {
        self.expr.deinit(allocator);
    }

    pub fn format(
        self: Statement,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        try writer.print("STATEMENT({})", .{self.expr});
    }
};

const UnaryOperator = enum {
    UnaryNegation,
    UnaryNot,
};
const UnaryExpr = struct {
    expr: u8,
    op: UnaryOperator,
};

const BinaryOperator = enum {
    BinaryAdd,
    BinaryMul,
};

const BinaryExpr = struct {
    left: *Expression,
    op: BinaryOperator,
    right: *Expression,

    // TODO: We assume that some parent structure provides the allocator that
    // was used to originally create these expressions.
    pub fn deinit(self: BinaryExpr, allocator: *Allocator) void {
        allocator.destroy(self.left);
        allocator.destroy(self.right);
    }
};

const ExpressionTag = enum {
    int_literal,
    string_literal,
    unary,
    binary,
    call,
};
const Expression = union(ExpressionTag) {
    int_literal: u8,
    string_literal: []const u8,
    unary: UnaryExpr,
    binary: BinaryExpr,
    call: bool,

    pub fn deinit(self: Expression, allocator: *Allocator) void {
        switch (self) {
            ExpressionTag.int_literal => |_| {},
            ExpressionTag.string_literal => |_| {},
            ExpressionTag.unary => |_| {},
            ExpressionTag.binary => |bexpr| bexpr.deinit(allocator),
            ExpressionTag.call => |_| {},
        }
    }

    pub fn format(
        self: Expression,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = self;
        _ = fmt;
        _ = options;

        try writer.print("EXPRESSION({s})", .{"foo"});
    }
};

const IntLiteral = struct {
    // TODO: bigint
    val: u8,
};

const OperandTag = enum {
    add,
    sub,
};

const Operand = union(OperandTag) {
    add: [2]u8,
    sub: [2]u8,
};

// Runtime types

const ShimUnit = struct {};

const ValueTag = enum {
    // Fixed length ints
    shim_u8,
    // shim_u16,
    // shim_u32,
    // shim_u64,
    // shim_i8,
    // shim_i16,
    // shim_i32,
    // shim_i64,

    // shim_bool

    // Arbitrary sized ints
    // shim_biguint,
    // shim_bigint,

    // String
    shim_str,

    // This is the value returned by statements
    shim_unit,

    shim_error,

    // ... everything else
    shim_obj,
};
const ShimValue = union(ValueTag) {
    shim_u8: u8,
    shim_str: bool,
    shim_unit: ShimUnit,
    shim_error: ShimUnit,
    shim_obj: ShimUnit,
};

// Functions!

pub fn run_text(allocator: *Allocator, text: []const u8) !void {
    var ast = try parse_text(allocator, text);
    defer ast.deinit();

    std.debug.print("{any}\n", .{ast});
    var result = interpret_ast(ast);

    std.debug.print("{}\n", .{result});
}

pub fn parse_text(allocator: *Allocator, text: []const u8) !Ast {
    var remaining_text = text[0..];

    var stmts = ArrayList(Statement).init(allocator);
    _ = stmts;

    if (try parse_expression(allocator, &remaining_text)) |expr| {
        try stmts.append(Statement{ .expr = expr });
    }

    var left = try std.fmt.parseInt(u8, text[6..7], 10);
    var right = try std.fmt.parseInt(u8, text[10..11], 10);
    var op = BinaryOperator.BinaryAdd;

    var list = ArrayList(Statement).init(allocator);

    var left_expr: *Expression = try allocator.create(Expression);
    left_expr.* = Expression{ .int_literal = left };
    var right_expr: *Expression = try allocator.create(Expression);
    right_expr.* = Expression{ .int_literal = right };

    try list.append(Statement{ .expr = Expression{
        .binary = .{ .left = left_expr, .op = op, .right = right_expr },
    } });

    return Ast{ .allocator = allocator, .stmts = list };
}

pub fn parse_expression(allocator: *Allocator, text: *[]const u8) !?Expression {
    _ = allocator;
    _ = text;

    // TODO: this is how we consume text functionally
    text.* = text.*[1..];
    std.log.info("blah: {s}", .{text.*});

    return null;
}

pub fn interpret_ast(ast: Ast) void {
    for (ast.stmts.items) |stmt| {
        _ = interpret_stmt(stmt);
    }
}

pub fn interpret_stmt(stmt: Statement) ShimValue {
    _ = interpret_expr(&stmt.expr);
    return ShimValue{ .shim_unit = ShimUnit{} };
}

pub fn interpret_expr(expr: *const Expression) ShimValue {
    switch (expr.*) {
        ExpressionTag.int_literal => |v| return ShimValue{ .shim_u8 = v },
        ExpressionTag.string_literal => |_| return ShimValue{ .shim_str = false },
        ExpressionTag.unary => |_| return ShimValue{ .shim_str = false },
        ExpressionTag.binary => |bexpr| {
            var left = switch (interpret_expr(bexpr.left)) {
                ValueTag.shim_u8 => |v| v,
                else => return ShimValue{ .shim_unit = ShimUnit{} },
            };
            var right = switch (interpret_expr(bexpr.right)) {
                ValueTag.shim_u8 => |v| v,
                else => return ShimValue{ .shim_unit = ShimUnit{} },
            };
            switch (bexpr.op) {
                BinaryOperator.BinaryAdd => return ShimValue{ .shim_u8 = left + right },
                BinaryOperator.BinaryMul => return ShimValue{ .shim_u8 = left * right },
            }
        },
        ExpressionTag.call => |_| return ShimValue{ .shim_str = false },
    }
}

// TODO: this is how we could conditionally add comments to the AST
// const build_options = @import("build_options");
//
// const Ast = blk: {
//     if (build_options.include_comments_in_ast) {
//         break :blk struct {
//             foo: u8,
//             comment: u8,
//         };
//     } else {
//         break :blk struct {
//             foo: u8,
//         };
//     }
// };
