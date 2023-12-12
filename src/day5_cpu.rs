use crate::day5_common::Garden;
use crate::file_utils;

use std::fs::File;
use std::io::BufReader;

//const INPUT_FILE_PATH: &'static str = "input/day5_sample.txt";
const INPUT_FILE_PATH: &'static str = "input/day5.txt";

fn part1(reader: BufReader<File>) {
    let garden = Garden::from_reader(reader);
    let seed_dest_locs = garden.seed_dest_locs();
    let lowest_loc = seed_dest_locs.into_iter().min().unwrap();
    println!("Part 1: {:?}", lowest_loc);
}

fn part2(reader: BufReader<File>) {
    let garden = Garden::from_reader(reader);
    let loweset_loc = garden.seed_range_min_loc();
    println!("Part 2: {:?}", loweset_loc);
}

pub fn solve() {
    println!("Day 5 solutions:");
    let reader = file_utils::input_reader(INPUT_FILE_PATH);
    part1(reader);

    let reader = file_utils::input_reader(INPUT_FILE_PATH);
    part2(reader);
}
