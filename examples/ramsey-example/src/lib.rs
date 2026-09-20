mod adev_plot;
mod app;
mod controls;
mod feedback;
mod feedback_plot;
mod frequency_response_plot;
mod nyquist_plot;
mod params;
mod ramsey;
mod response_plot;
mod signal_plot;

#[cfg(not(target_arch = "wasm32"))]
pub fn run_native() -> myplotlib::NativeResult {
    myplotlib::run_native(app::definition())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(js_name = mountApp)]
pub async fn mount_app(
    canvas: web_sys::HtmlCanvasElement,
) -> Result<myplotlib::WebHandle, wasm_bindgen::JsValue> {
    myplotlib::mount_web(canvas, app::definition()).await
}
