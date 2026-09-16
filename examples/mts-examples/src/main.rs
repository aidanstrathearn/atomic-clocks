mod mts;
#[path = "mts-velocity.rs"]
mod mts_velocity;

fn main() -> myplotlib::Result {
    myplotlib::AppMenu::new("MTS examples")
        .app("MTS", mts::definition())
        .app("MTS velocity", mts_velocity::definition())
        .run_native()
}
