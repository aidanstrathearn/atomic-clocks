mod app;
mod controls;
mod gradients;
mod parallel;
mod probe;
mod response;
#[cfg(test)]
mod tests;
mod units;

#[cfg(not(target_arch = "wasm32"))]
pub fn run_native() -> myplotlib::NativeResult {
    myplotlib::run_native(app::definition())
}

#[cfg(target_arch = "wasm32")]
pub use wasm_bindgen_rayon::init_thread_pool;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(js_name = mountApp)]
pub async fn mount_app(
    canvas: web_sys::HtmlCanvasElement,
) -> Result<myplotlib::WebHandle, wasm_bindgen::JsValue> {
    myplotlib::mount_web(canvas, app::definition()).await
}
