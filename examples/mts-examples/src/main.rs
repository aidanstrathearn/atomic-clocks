mod mts;
mod mts_velocity;
mod units;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> myplotlib::Result {
    myplotlib::run_native(mts_velocity::definition())
}

#[cfg(target_arch = "wasm32")]
fn main() {
    myplotlib::run_web(mts_velocity::definition()).expect("failed to start the web app");
}
