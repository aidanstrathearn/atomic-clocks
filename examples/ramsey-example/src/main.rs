#[cfg(not(target_arch = "wasm32"))]
fn main() -> myplotlib::NativeResult {
    ramsey_example::run_native()
}

#[cfg(target_arch = "wasm32")]
fn main() {}
