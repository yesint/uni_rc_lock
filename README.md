This crate provides a trait for universal reference-counted lock
with internal mutability
allowing multiple readers or a single writer,
which may represent either `Rc<RefCell<T>>` or `Arc<RwLock<T>>`.
