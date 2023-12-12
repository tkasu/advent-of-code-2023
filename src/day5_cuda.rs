use crate::day5_common::{Garden, LocMap};
use crate::file_utils;

use cudarc::driver::{DeviceRepr, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::cmp::min;
use std::time::Instant;

//const INPUT_FILE_PATH: &'static str = "input/day5_sample.txt";
const INPUT_FILE_PATH: &'static str = "input/day5.txt";

#[repr(C)]
struct LocMapComponentC {
    from: u64,
    to: u64,
    offset: i64,
}

unsafe impl DeviceRepr for LocMapComponentC {}

#[repr(C)]
struct LocMapSizes {
    seed_to_soil: u32,
    soil_to_fertilizer: u32,
    fertilizer_to_water: u32,
    water_to_light: u32,
    light_to_temperature: u32,
    temperature_to_humidity: u32,
    humidity_to_location: u32,
}

unsafe impl DeviceRepr for LocMapSizes {}

fn locmap_to_c_repr(locmap: LocMap) -> Vec<LocMapComponentC> {
    locmap
        .components
        .iter()
        .map(|x| LocMapComponentC {
            from: x.from as u64,
            to: x.to as u64,
            offset: x.offset as i64,
        })
        .collect()
}

pub fn solve() {
    println!("Day 5 part 2 solution (with CUDA):");

    let reader = file_utils::input_reader(INPUT_FILE_PATH);
    let garden = Garden::from_reader(reader);

    let dev = cudarc::driver::CudaDevice::new(0).unwrap();

    let seed_input: Vec<u64> = garden.seed_input;
    let mut n: u64 = 0;
    for i in (0..seed_input.len()).step_by(2) {
        n += seed_input[i + 1]
    }
    let batches: u32 = min(960000, n / 10).try_into().unwrap();
    let batch_size: u32 = (n / batches as u64).try_into().unwrap();

    println!(
        "n: {:?}, batches: {:?}, batch_size: {:?}",
        n, batches, batch_size
    );

    let seed_to_soil = locmap_to_c_repr(garden.seed_to_soil);
    let soil_to_fertilizer = locmap_to_c_repr(garden.soil_to_fertilizer);
    let fertilizer_to_water = locmap_to_c_repr(garden.fertilizer_to_water);
    let water_to_light = locmap_to_c_repr(garden.water_to_light);
    let light_to_temperature = locmap_to_c_repr(garden.light_to_temperature);
    let temperature_to_humidity = locmap_to_c_repr(garden.temperature_to_humidity);
    let humidity_to_location = locmap_to_c_repr(garden.humidity_to_location);

    let sizes = LocMapSizes {
        seed_to_soil: seed_to_soil.len() as u32,
        soil_to_fertilizer: soil_to_fertilizer.len() as u32,
        fertilizer_to_water: fertilizer_to_water.len() as u32,
        water_to_light: water_to_light.len() as u32,
        light_to_temperature: light_to_temperature.len() as u32,
        temperature_to_humidity: temperature_to_humidity.len() as u32,
        humidity_to_location: humidity_to_location.len() as u32,
    };

    let out_local_mins = vec![u64::MAX; batches.try_into().unwrap()];

    let d_seed_input = dev.htod_copy(seed_input).unwrap();
    let d_seed_to_soil_input = dev.htod_copy(seed_to_soil).unwrap();
    let d_soil_to_fertilizer_input = dev.htod_copy(soil_to_fertilizer).unwrap();
    let d_fertilizer_to_water_input = dev.htod_copy(fertilizer_to_water).unwrap();
    let d_water_to_light_input = dev.htod_copy(water_to_light).unwrap();
    let d_light_to_temperature_input = dev.htod_copy(light_to_temperature).unwrap();
    let d_temperature_to_humidity_input = dev.htod_copy(temperature_to_humidity).unwrap();
    let d_humidity_to_location_input = dev.htod_copy(humidity_to_location).unwrap();

    let mut d_out_local_mins = dev.htod_copy(out_local_mins).unwrap();

    dev.load_ptx(
        Ptx::from_file("./src/day5_kernel.ptx"),
        "aoc",
        &["calc_dest_min"],
    )
    .unwrap();
    let calc_dest_min_kernel = dev.get_func("aoc", "calc_dest_min").unwrap();
    let cfg = LaunchConfig::for_num_elems(batches.try_into().unwrap());

    let start = Instant::now();
    unsafe {
        calc_dest_min_kernel.launch(
            cfg,
            (
                &mut d_out_local_mins,
                &d_seed_input,
                &d_seed_to_soil_input,
                &d_soil_to_fertilizer_input,
                &d_fertilizer_to_water_input,
                &d_water_to_light_input,
                &d_light_to_temperature_input,
                &d_temperature_to_humidity_input,
                &d_humidity_to_location_input,
                sizes,
                batch_size,
                n,
            ),
        )
    }
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
