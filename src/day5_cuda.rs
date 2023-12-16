use crate::day5_common::{Garden, LocMap};
use crate::file_utils;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{DevicePtr, DeviceRepr, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::cmp::min;
use std::time::Instant;

//const INPUT_FILE_PATH: &'static str = "input/day5_sample.txt";
const INPUT_FILE_PATH: &'static str = "input/day5.txt";

#[repr(C)]
#[derive(Clone)]
struct LocMapComponentDevice {
    from: u32,
    to: u32,
    offset: i32,
}

unsafe impl DeviceRepr for LocMapComponentDevice {}

#[repr(C)]
struct GardenDevice {
    seed_input: CUdeviceptr, // pointer of Vec<u32>
    seed_input_size: u32,
    seed_to_soil: CUdeviceptr, // pointer of Vec<LocMapComponentDevice>
    seed_to_soil_size: u32,
    soil_to_fertilizer: CUdeviceptr, // pointer of Vec<LocMapComponentDevice>
    soil_to_fertilizer_size: u32,
    fertilizer_to_water: CUdeviceptr, // pointer of Vec<LocMapComponentDevice>
    fertilizer_to_water_size: u32,
    water_to_light: CUdeviceptr, // pointer of Vec<LocMapComponentDevice>
    water_to_light_size: u32,
    light_to_temperature: CUdeviceptr, // pointer of Vec<LocMapComponentDevice>
    light_to_temperature_size: u32,
    temperature_to_humidity: CUdeviceptr, // pointer of Vec<LocMapComponentDevice>
    temperature_to_humidity_size: u32,
    humidity_to_location: CUdeviceptr, // pointer of Vec<LocMapComponentDevice>
    humidity_to_location_size: u32,
}

unsafe impl DeviceRepr for GardenDevice {}

fn locmap_to_c_repr(locmap: LocMap) -> Vec<LocMapComponentDevice> {
    locmap
        .components
        .iter()
        .map(|x| LocMapComponentDevice {
            from: x.from as u32,
            to: x.to as u32,
            offset: x.offset as i32,
        })
        .collect()
}

pub fn solve() {
    println!("Day 5 part 2 solution (with CUDA):");

    let reader = file_utils::input_reader(INPUT_FILE_PATH);
    let garden = Garden::from_reader(reader);

    let dev = cudarc::driver::CudaDevice::new(0).unwrap();

    let seed_input: Vec<u32> = garden.seed_input.into_iter().map(|i| i as u32).collect();
    let mut n: u32 = 0;
    for i in (0..seed_input.len()).step_by(2) {
        n += seed_input[i + 1]
    }
    let batches: u32 = min(96_000_000, n).try_into().unwrap();
    let batch_size: u32 = (n / batches as u32).try_into().unwrap();

    let cfg = LaunchConfig::for_num_elems(batches.try_into().unwrap());
    let cfg = LaunchConfig {
        grid_dim: cfg.grid_dim,
        block_dim: cfg.block_dim,
        shared_mem_bytes: cfg.block_dim.0 * 4, // one u32 for each thread in the block
    };

    println!(
        "n: {:?}, batches: {:?}, batch_size: {:?}, cuda config: {:?}",
        n, batches, batch_size, &cfg
    );

    let seed_to_soil = locmap_to_c_repr(garden.seed_to_soil);
    let soil_to_fertilizer = locmap_to_c_repr(garden.soil_to_fertilizer);
    let fertilizer_to_water = locmap_to_c_repr(garden.fertilizer_to_water);
    let water_to_light = locmap_to_c_repr(garden.water_to_light);
    let light_to_temperature = locmap_to_c_repr(garden.light_to_temperature);
    let temperature_to_humidity = locmap_to_c_repr(garden.temperature_to_humidity);
    let humidity_to_location = locmap_to_c_repr(garden.humidity_to_location);

    let out_local_mins = vec![u32::MAX; cfg.grid_dim.0.try_into().unwrap()];

    let d_seed_input = dev.htod_copy(seed_input.clone()).unwrap();
    let d_seed_to_soil_input = dev.htod_copy(seed_to_soil.clone()).unwrap();
    let d_soil_to_fertilizer_input = dev.htod_copy(soil_to_fertilizer.clone()).unwrap();
    let d_fertilizer_to_water_input = dev.htod_copy(fertilizer_to_water.clone()).unwrap();
    let d_water_to_light_input = dev.htod_copy(water_to_light.clone()).unwrap();
    let d_light_to_temperature_input = dev.htod_copy(light_to_temperature.clone()).unwrap();
    let d_temperature_to_humidity_input = dev.htod_copy(temperature_to_humidity.clone()).unwrap();
    let d_humidity_to_location_input = dev.htod_copy(humidity_to_location.clone()).unwrap();

    let d_garden = GardenDevice {
        seed_input: *d_seed_input.device_ptr(),
        seed_input_size: seed_input.len() as u32,
        seed_to_soil: *d_seed_to_soil_input.device_ptr(),
        seed_to_soil_size: seed_to_soil.len() as u32,
        soil_to_fertilizer: *d_soil_to_fertilizer_input.device_ptr(),
        soil_to_fertilizer_size: soil_to_fertilizer.len() as u32,
        fertilizer_to_water: *d_fertilizer_to_water_input.device_ptr(),
        fertilizer_to_water_size: fertilizer_to_water.len() as u32,
        water_to_light: *d_water_to_light_input.device_ptr(),
        water_to_light_size: water_to_light.len() as u32,
        light_to_temperature: *d_light_to_temperature_input.device_ptr(),
        light_to_temperature_size: light_to_temperature.len() as u32,
        temperature_to_humidity: *d_temperature_to_humidity_input.device_ptr(),
        temperature_to_humidity_size: temperature_to_humidity.len() as u32,
        humidity_to_location: *d_humidity_to_location_input.device_ptr(),
        humidity_to_location_size: humidity_to_location.len() as u32,
    };

    let mut d_out_local_mins = dev.htod_copy(out_local_mins).unwrap();

    dev.load_ptx(
        Ptx::from_file("./src/day5_kernel.ptx"),
        "aoc",
        &["calc_dest_min"],
    )
    .unwrap();
    let calc_dest_min_kernel = dev.get_func("aoc", "calc_dest_min").unwrap();

    let start = Instant::now();
    unsafe { calc_dest_min_kernel.launch(cfg, (&mut d_out_local_mins, d_garden, batch_size, n)) }
        .unwrap();

    let min = dev
        .dtoh_sync_copy(&d_out_local_mins)
        .unwrap()
        .into_iter()
        .min()
        .unwrap();

    println!(
        "CUDA calculation and copy back to host took: {:?}",
        start.elapsed()
    );
    println!("Part 2: {:?}", min);
}
