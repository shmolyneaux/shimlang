use std::cell::{Cell, RefCell, Ref, RefMut};
use std::rc::{Rc, Weak};

/// A type that can have and break cycles.
pub trait Manage {
    // TODO: it seems like we could do some lifetime stuff to make sure that
    // we're only tracing things that are relevant to _this_ Collector
    /// Updates `traces` with all the GC-tracked references directly accessible
    /// to this object.
    fn trace<'a>(&'a self, traces: &mut Vec<&'a Gc<Self>>);

    /// Removes outbound GC-tracked references from this object to remove it
    /// from a cycle.
    fn cycle_break(&mut self);
}

pub struct Gc<T: ?Sized> {
    value: Rc<(Cell<usize>, RefCell<T>)>,
}

impl<T> Gc<T> {
    fn new(value: T) -> Self {
        Self {
            value: Rc::new((Cell::new(0), RefCell::new(value))),
        }
    }

    fn from_weak(weak: &Weak<(Cell<usize>, RefCell<T>)>) -> Option<Self> {
        if let Some(value) = weak.upgrade() {
            Some(Self { value })
        } else {
            None
        }
    }

    fn inner_mut(&mut self) -> RefMut<'_, T> {
        self.value.1.borrow_mut()
    }

    fn inner(&self) -> Ref<'_, T> {
        self.value.1.borrow()
    }

    fn tally(&self) {
        let new_tally = self.value.as_ref().0.get() + 1;
        self.value.as_ref().0.set(new_tally);
    }

    fn tally_dec(&self) {
        let new_tally = self.value.as_ref().0.get() - 1;
        self.value.as_ref().0.set(new_tally);
    }

    fn should_cleanup(&self) -> bool {
        // NOTE: This could include the reference that the collector uses while
        // tracing the Rc
        self.value.as_ref().0.get() == Rc::strong_count(&self.value)
    }

    fn clear_tally(&self) {
        self.value.as_ref().0.set(0);
    }
}

impl<T: ?Sized> Clone for Gc<T> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone()
        }
    }
}

pub struct Collector<T: Manage> {
    // TODO: these should be weak RC's so that objects without cycles get freed
    // mor eagerly
    items: Vec<Weak<(Cell<usize>, RefCell<T>)>>
}

impl<T: Manage> Collector<T> {
    pub fn new() -> Self {
        let items = Vec::new();
        Self { items }
    }

    pub fn manage(&mut self, value: T) -> Gc<T> {
        let item = Gc::new(value);
        self.items.push(Rc::downgrade(&item.value));

        item
    }

    pub fn live_count(&self) -> usize {
        self.items.iter().filter(|w| w.strong_count() > 0).count()
    }

    pub fn collect_cycles(&mut self) {
        // Trace each value that we're managing. If any item has a strong count
        // that is accounted for by _just_ finding direct references from other
        // GC-tracked items, that means it's part of an unreachable cycle. If
        // A strong count is greater than this tally, it means that there's
        // a reachable reference to that item outside of `self.items`.
        //
        // We remove cycles immediately when found.
        for mut item in self.items.iter_mut().filter_map(|v| Gc::from_weak(v)) {
            // Increment the tally to account for the Gc we just created
            item.tally();

            // TODO: how to we move this out of the loop so that we don't have
            // to constantly create/resize it?
            let mut items_to_tally = Vec::new();

            let item_ref = item.inner();
            item_ref.trace(&mut items_to_tally);

            let mut drop_item = false;
            for thing_to_tally in items_to_tally.iter_mut() {
                thing_to_tally.tally();
                if thing_to_tally.should_cleanup() {
                    // Avoid re-borrowing item
                    // TODO: put in free list
                    if Rc::ptr_eq(&item.value, &thing_to_tally.value) {
                        drop_item = true;
                    } else {
                        thing_to_tally.value.1.borrow_mut().cycle_break();
                    }
                }
            }

            std::mem::drop(item_ref);

            if drop_item {
                item.inner_mut().cycle_break();
            }

            // Decrement the tally as the Gc we created in the predicate gets dropped
            item.tally_dec();
        }

        for item in self.items.iter_mut().filter_map(|v| Gc::from_weak(v)) {
            item.clear_tally();
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Gc, Collector, Manage};

    enum Value {
        Int(u8),
        List(Vec<Gc<Value>>),
    }

    impl Value {
        fn push(&mut self, value: Gc<Value>) {
            match self {
                Value::Int(_) => panic!("Cannot push to int"),
                Value::List(v) => {
                    v.push(value)
                }
            }
        }
    }

    impl Manage for Value {
        fn trace<'a>(&'a self, traces: &mut Vec<&'a Gc<Self>>) {
            match self {
                Value::Int(_) => {}
                Value::List(v) => {
                    *traces = v.iter().collect()
                }
            }
        }

        fn cycle_break(&mut self) {
            match self {
                Value::Int(_) => {}
                Value::List(v) => {
                    v.clear()
                }
            }
        }
    }

    #[test]
    fn it_works() {
        let mut collector = Collector::new();

        assert_eq!(collector.live_count(), 0);
        {
            let mut value = collector.manage(Value::List(Vec::new()));
            // Create a cycle. Have the first element of the list reference the list
            let self_reference = value.clone();
            value.inner_mut().push(self_reference);

            assert_eq!(collector.live_count(), 1);
        }

        // This should still be 1 even though value was dropped (since there's a cycle)
        assert_eq!(collector.live_count(), 1);

        collector.collect_cycles();

        // The cycle should now be cleaned up
        assert_eq!(collector.live_count(), 0);
    }
}
