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
fn main() -> myplotlib::NativeResult {
    app::run_native()
}

#[cfg(target_arch = "wasm32")]
fn main() {
    app::run_web().expect("failed to start the web app");
}
