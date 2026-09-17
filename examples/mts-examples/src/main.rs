mod mts;
mod mts_velocity;
mod units;

fn main() -> myplotlib::Result {
    myplotlib::run_native(mts_velocity::definition())
}
