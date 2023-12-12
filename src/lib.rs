pub mod day1;
pub mod day2;
pub mod day3;
pub mod day4;
pub mod day5_common;
#[cfg_attr(feature = "cuda", path = "day5_cuda.rs")]
#[cfg_attr(not(feature = "cuda"), path = "day5_cpu.rs")]
pub mod day5;
pub mod file_utils;
