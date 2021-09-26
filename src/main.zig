const std = @import("std");
const libshim = @import("libshim");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const leaked = gpa.deinit();
        if (leaked) {
            std.log.err("Leaked: {}", .{leaked});
        }
    }

    try libshim.run_text(&gpa.allocator, "print(3 + 3 + 2);");
}
