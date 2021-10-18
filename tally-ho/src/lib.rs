/// A type that can have and break cycles.
pub trait Manage {
    // TODO: it seems like we could do some lifetime stuff to make sure that
    // we're only tracing things that are relevant to _this_ Collector
    /// Updates `traces` with all the GC-tracked references directly accessible
    /// to this object.
    fn trace(&mut self, traces: &mut Vec<&mut Gc<Self>>);

    /// Removes outbound GC-tracked references from this object to remove it
    /// from a cycle.
    fn cycle_break(&mut self);
}

pub struct Gc<T: ?Sized> {
    // TODO: GcBox
    value: Box<T>,
    strong: usize,
    // TODO: it seems like we'll only ever have 1, which we can always keep track
    // of. Is this needed?
    // weak: usize,
    tally: usize,
}

impl<T> Gc<T> {
    fn clone(&self) -> Gc<T> {
        // Increment strong
        todo!()
    }

    fn downgrade(&self) -> GcWeak<T> {
        todo!()
    }

    fn tally(&mut self) {
        // if self.tally == self.strong {
        //     self.value.cycle_break();
        // }
        todo!()
    }

    fn clear_tally(&mut self) {
        // if self.tally == self.strong {
        //     self.value.cycle_break();
        // }
        todo!()
    }
}

struct GcWeak<T> {
    inner: Gc<T>
}

pub struct Collector<T: Manage> {
    items: Vec<GcWeak<T>>
}

impl<T: Manage> Collector<T> {
    pub fn new() -> Self {
        let items = Vec::new();
        Self { items }
    }

    pub fn create(&mut self, value: T) -> Gc<T> {
        let item = Gc {
            value: Box::new(value),
            strong: 1,
            //weak: 0,
            tally: 0,
        };
        // TODO: use the free list
        self.items.push(item.clone().downgrade());

        item
    }

    pub fn collect_cycles(&mut self) {
        let mut items_to_tally = Vec::new();

        // Trace each value that we're managing. If any item has a strong count
        // that is accounted for by _just_ finding direct references from other
        // GC-tracked items, that means it's part of an unreachable cycle. If
        // A strong count is greater than this tally, it means that there's
        // a reachable reference to that item outside of `self.items`.
        //
        // We remove cycles immediately when found.
        for item in self.items.iter_mut() {
            item.inner.value.trace(&mut items_to_tally);
            for thing_to_tally in items_to_tally.iter_mut() {
                thing_to_tally.tally();
            }
        }

        // TODO: Make a free list
        for item in self.items.iter_mut() {
            item.inner.clear_tally();
            // Increment strong
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
