mod mts;
mod mts_velocity;
mod units;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> myplotlib::Result {
    myplotlib::run_native(mts_velocity::definition())
}

#[cfg(target_arch = "wasm32")]
pub use wasm_bindgen_rayon::init_thread_pool;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(js_name = startApp)]
pub fn start_app() -> Result<(), wasm_bindgen::JsValue> {
    myplotlib::run_web(mts_velocity::definition())
}

#[cfg(target_arch = "wasm32")]
fn main() {}
