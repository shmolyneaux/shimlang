# GC mark-phase benchmarks

Each `.shm` here builds a large, fully-live object graph and then leaves it
reachable from the root environment, so the garbage collector's mark phase has
to walk the whole thing. The workloads deliberately stress different parts of
the object-graph walk:

| workload          | exercises                                             |
| ----------------- | ----------------------------------------------------- |
| `struct_list`     | structs with nested lists/strings + a linked chain    |
| `dicts`           | many dicts holding lists and strings                  |
| `tuples`          | many small tuples                                     |
| `sets`            | many sets (and their backing dicts)                   |
| `closures`        | closures capturing scopes (functions + environments)  |
| `methods`         | structs, struct defs, and bound methods               |
| `deep_chain`      | a deep linked list (worklist depth, not width)        |
| `aliased`         | many references to one shared object (mark dedup)      |

Run them with the harness one directory up:

```sh
python3 benchmark_scripts/gc_bench.py                 # current build
python3 benchmark_scripts/gc_bench.py --baseline PATH # vs another gc_bench binary
```

The harness times only the mark phase (via `Interpreter::bench_gc_mark`, driven
by the `gc_bench` binary), reporting the minimum per-iteration time over several
trials.
