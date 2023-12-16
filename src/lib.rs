/*! 
A trait for universal reference-counted lock
with interior mutability
allowing multiple readers or a single writer,
which may represent either `Rc<RefCell<T>>` or `Arc<RwLock<T>>`.

# Basics
## Motivation
The `Rc<RefCell<T>>` and `Arc<RwLock<T>>` are idiomatic ways of expressing
interior mutability in Rust in the single-threaded and multi-threaded
scenarious respectively. Both types provide essentially the same interface
with multiple readers or a single writer for their managed data.

However, despite their similarity, these types do not share any common trait
and thus can't be used in a generic way: one can't define a function argument
or a struct field that could accept either `Rc<RefCell<T>>` or `Arc<RwLock<T>>`.
This leads to code duplication if it is not known in advance which kind of
smart pointer (Rc or Arc) will be passed.

`UniRcLock` solves this problem by providing a common trait
for `Rc<RefCell<T>>` and `Arc<RwLock<T>>` so that they could be used in
a generic way in single-threaded and multi-threaded scenarious alike.

## Performance
`UniRcLock` is a zero-cost abstraction.

## Limitations
An ability to recover from lock poisoning in `RwLock<T>` is lost
when using `UniRcLock`. The methods `read()` and `write()` will panic if
the lock is poisoned.

# Examples

A generic function which accepts both `Rc<RefCell<T>>` and `Arc<RwLock<T>>`:
```
# use std::{rc::Rc, cell::RefCell, sync::{Arc, RwLock}};
# use uni_rc_lock::UniRcLock;
#
#[derive(Debug)]
struct Foo(i32);

fn incr_foo(v: impl UniRcLock<Foo>){
    v.write().0 += 1;
}

// Using Rc
let ptr1 = Rc::new(RefCell::new(Foo(0)));
incr_foo(ptr1.clone());
println!("After increment: {:?}", ptr1);

// Using Arc
let ptr2 = Arc::new(RwLock::new(Foo(0)));
incr_foo(ptr2.clone());
println!("After increment: {:?}", ptr2);
```
Example of generic struct, which can hold either `Rc<RefCell<T>>` or `Arc<RwLock<T>>`:
```
# use std::{rc::Rc, cell::RefCell, sync::{Arc, RwLock}};
# use uni_rc_lock::UniRcLock;
# use std::thread;
#
// A user struct
struct State {val: i32}

// Generic wrapper
#[derive(Debug,Clone)]
struct StateHandler<T: UniRcLock<State>> {
    state: T,
}

impl<T: UniRcLock<State>> StateHandler<T> {
    // Constructor taking either Rc<RefCell<T>>` or `Arc<RwLock<T>>
    fn new(val: T) -> Self {
        Self{state: val}
    }
}

// Using with Rc
{
    let st = Rc::new(RefCell::new(State { val: 42 }));
    let st_handler = StateHandler::new(st);
    st_handler.state.write().val += 1;
    println!("{}", st_handler.state.read().val);
}

// Using with Arc in exactly the same way
{
    let st = Arc::new(RwLock::new(State { val: 42 }));
    let st_handler = StateHandler::new(st);
    st_handler.state.write().val += 1;
    println!("{}", st_handler.state.read().val);
}

// Using in multiple threads with Arc
{
    let st = Arc::new(RwLock::new(State { val: 42 }));

    let st_handler = StateHandler::new(st);
    let threads: Vec<_> = (0..10)
        .map(|i| {
            let h = st_handler.clone();
            thread::spawn(move || {
                h.state.write().val += 1;
                println!("Thread #{i} incremented");
            })
        })
        .collect();

    for t in threads {
        t.join().unwrap();
    }
 
    println!("Result: {}", st_handler.state.read().val);
}
```
Expectibly, this example won't compile with `Rc` since it doesn't implement `Send`.
*/

//===============================================================

use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::{
    cell::{Ref, RefCell, RefMut},
    ops::Deref,
    ops::DerefMut,
    rc::Rc,
};

/// A common trait for `Rc<RefCell<T>>` and `Arc<RwLock<T>>` 
pub trait UniRcLock<T>: Clone {
    type OutRead<'a>: Deref<Target = T> where Self: 'a;
    type OutWrite<'a>: DerefMut<Target = T> where Self: 'a;
    /// Obtain a scoped guard for reading
    fn read<'a>(&'a self) -> Self::OutRead<'a>;
    /// Obtain a scoped guard for writing
    fn write<'a>(&'a self) -> Self::OutWrite<'a>;
}

// Implementation for Rc<RefCell<T>>
impl<T> UniRcLock<T> for Rc<RefCell<T>> {
    type OutRead<'a> = Ref<'a, T> where T: 'a;
    type OutWrite<'a> = RefMut<'a, T> where T: 'a;

    fn read<'a>(&'a self) -> Self::OutRead<'a> {
        Rc::deref(self).borrow()
    }

    fn write<'a>(&'a self) -> Self::OutWrite<'a> {
        Rc::deref(self).borrow_mut()
    }
}

// Implementation for Arc<RwLock<T>>
impl<T> UniRcLock<T> for Arc<RwLock<T>> {
    type OutRead<'a> = RwLockReadGuard<'a, T> where T: 'a;
    type OutWrite<'a> = RwLockWriteGuard<'a, T> where T: 'a;

    fn read<'a>(&'a self) -> Self::OutRead<'a> {
        Arc::deref(self)
            .read()
            .expect("Read lock should not be poisoned")
    }

    fn write<'a>(&'a self) -> Self::OutWrite<'a> {
        Arc::deref(self)
            .write()
            .expect("Write lock should not be poisoned")
    }
}

#[cfg(test)]
mod tests {
    use std::{
        cell::RefCell,
        rc::Rc,
        sync::{Arc, RwLock},
    };

    use super::UniRcLock;

    #[derive(Debug)]
    struct State {
        val: i32,
    }
    //-----------------
    // Generic handler
    //-----------------
    #[derive(Debug, Clone)]
    struct StateHandler<T: UniRcLock<State>> {
        state: T,
    }

    impl<T: UniRcLock<State>> StateHandler<T> {
        fn new(val: T) -> Self {
            Self { state: val }
        }
    }

    #[test]
    fn rc() {
        let st1 = Rc::new(RefCell::new(State { val: 42 }));
        st1.write().val += 1;
        println!("{:?}", st1.read());
    }
    #[test]
    fn arc() {
        let st2 = Arc::new(RwLock::new(State { val: 42 }));
        st2.write().val += 1;
        println!("{:?}", st2.read());
    }

    #[test]
    fn rc_handler() {
        let st1 = Rc::new(RefCell::new(State { val: 42 }));
        let sth1 = StateHandler::new(st1);
        println!("{:?}", sth1.state);
    }

    #[test]
    fn arc_handler() {
        let st2 = Arc::new(RwLock::new(State { val: 42 }));
        let sth2 = StateHandler::new(st2);
        sth2.state.write().val += 1;
        println!("{:?}", sth2.state);
    }

    #[test]
    fn test_func() {
        fn incr(p: impl UniRcLock<State>) {
            p.write().val += 1;
        }

        let ptr = Rc::new(RefCell::new(State { val: 0 }));
        incr(ptr.clone());
        println!("{:?}", ptr.read().val);
    }

    #[test]
    fn threads_test_arc() {
        use std::thread;
        let st2 = Arc::new(RwLock::new(State { val: 42 }));
        let sth2 = StateHandler::new(st2);

        let threads: Vec<_> = (0..10)
            .map(|i| {
                let h = sth2.clone();
                thread::spawn(move || {
                    h.state.write().val += 1;
                    println!("Thread #{i} incremented");
                })
            })
            .collect();

        for t in threads {
            t.join().unwrap();
        }

        println!("Result: {}", sth2.state.read().val);
    }
}
