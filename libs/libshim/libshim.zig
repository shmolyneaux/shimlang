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

const File = struct {
    shebang: ArrayList(u8),
    ast: Ast,
};

const Ast = struct {
    stmts: ArrayList(Statement),

    pub fn deinit(self: Ast) void {
        self.stmts.deinit();
    }
};

// TODO: Multiple sorts of statements (if, for, use, while, etc.)
const Statement = struct { expr: Expression };

const Expression = struct {
    left: IntLiteral,
    op: Operand,
    right: IntLiteral,
};

const IntLiteral = struct {
    // TODO: bigint
    val: u8,
};

const Operand = struct {};

pub fn run_text(allocator: *Allocator, text: *const [12:0]u8) !void {
    var ast = try parse_text(allocator, text);
    defer ast.deinit();
    var result = interpret_ast(ast);

    std.debug.print("{}\n", .{result});
}

pub fn parse_text(allocator: *Allocator, text: *const [12:0]u8) !Ast {
    var left = try std.fmt.parseInt(u8, text[6..7], 10);
    var op = Operand{};
    var right = try std.fmt.parseInt(u8, text[10..11], 10);

    var list = ArrayList(Statement).init(allocator);
    try list.append(Statement{ .expr = Expression{
        .left = IntLiteral{ .val = left },
        .op = op,
        .right = IntLiteral{ .val = right },
    } });
    return Ast{ .stmts = list };
}

pub fn interpret_ast(ast: Ast) u8 {
    var stmt = ast.stmts.items[0];
    var expr = stmt.expr;
    var left = expr.left.val;
    //var op = expr.op;
    var right = expr.right.val;

    return left + right;
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
