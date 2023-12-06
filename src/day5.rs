use crate::file_utils;

use std::fs::File;
use std::io::{prelude::*, BufReader};

use rayon::prelude::*;

//const INPUT_FILE_PATH: &'static str = "input/day5_sample.txt";
const INPUT_FILE_PATH: &'static str = "input/day5.txt";

#[derive(Debug)]
struct Garden {
    seed_input: Vec<u64>,
    seed_to_soil: LocMap,
    soil_to_fertilizer: LocMap,
    fertilizer_to_water: LocMap,
    water_to_light: LocMap,
    light_to_temperature: LocMap,
    temperature_to_humidity: LocMap,
    humidity_to_location: LocMap,
}

impl Garden {
    fn seed_dest_locs(&self) -> Vec<u64> {
        self.seed_input
            .clone()
            .into_iter()
            .map(|seed| self.dest_loc(seed))
            .collect()
    }

    fn seed_range_min_loc(&self) -> u64 {
        let pairs: Vec<&[u64]> = self.seed_input.chunks(2).collect();
        let min_loc: u64 = pairs
            .into_iter()
            .map(|pair| {
                println!("Starting pair: {:?}", pair);
                let range_start = *pair.first().unwrap();
                let range_end = range_start + pair.last().unwrap();
                let range: Vec<u64> = (range_start..range_end).collect();
                let range_min = range.par_iter().map(| i| self.dest_loc(*i)).min().unwrap();
                println!("Pair {:?} done, local_min: {}", pair, range_min);
                range_min
            })
            .min()
            .unwrap();
        min_loc
    }

    fn dest_loc(&self, seed: u64) -> u64 {
        let map_chain = [
            &self.seed_to_soil,
            &self.soil_to_fertilizer,
            &self.fertilizer_to_water,
            &self.water_to_light,
            &self.light_to_temperature,
            &self.temperature_to_humidity,
            &self.humidity_to_location,
        ];
        map_chain
            .into_iter()
            .fold(seed, |acc, nxt| nxt.next_loc(acc))
    }

    fn filter_map_lines(lines: &Vec<String>, map_name: &str) -> Vec<String> {
        let mut mby_line_start: Option<usize> = None;
        let mut mby_line_end: Option<usize> = None;
        for (i, line) in lines.iter().enumerate() {
            if mby_line_start.is_none() && line.starts_with(map_name) {
                mby_line_start = Some(i + 1);
            } else if mby_line_start.is_some() && line == "" {
                mby_line_end = Some(i);
                break;
            } else if mby_line_start.is_some() && line.ends_with("map:") {
                // Found some other map, logic error
                panic!("Did not find line end for {}", map_name);
            }
        }
        let line_start =
            mby_line_start.expect(format!("Did not find line start for {}", map_name).as_str());
        let line_end = mby_line_end.unwrap_or(lines.len());
        let map_lines = &lines[line_start..line_end];
        map_lines.to_vec()
    }

    fn from_reader(reader: BufReader<File>) -> Self {
        let lines: Vec<String> = reader.lines().map(|l| l.unwrap()).collect();

        let seeds: &Vec<u64> = &lines
            .first()
            .unwrap()
            .strip_prefix("seeds: ")
            .unwrap()
            .to_string()
            .split(" ")
            .map(|s| s.parse::<u64>().unwrap())
            .collect();

        let seed_to_soil_lines = Self::filter_map_lines(&lines, "seed-to-soil");
        let soil_to_fertilizer_lines = Self::filter_map_lines(&lines, "soil-to-fertilizer");
        let fertilizer_to_water_lines = Self::filter_map_lines(&lines, "fertilizer-to-water");
        let water_to_light_lines = Self::filter_map_lines(&lines, "water-to-light");
        let light_to_temperature_lines = Self::filter_map_lines(&lines, "light-to-temperature");
        let temperature_to_humidity_lines =
            Self::filter_map_lines(&lines, "temperature-to-humidity");
        let humidity_to_location_lines = Self::filter_map_lines(&lines, "humidity-to-location");

        let seed_to_soil = LocMap::from_lines(seed_to_soil_lines);
        let soil_to_fertilizer = LocMap::from_lines(soil_to_fertilizer_lines);
        let fertilizer_to_water = LocMap::from_lines(fertilizer_to_water_lines);
        let water_to_light = LocMap::from_lines(water_to_light_lines);
        let light_to_temperature = LocMap::from_lines(light_to_temperature_lines);
        let temperature_to_humidity = LocMap::from_lines(temperature_to_humidity_lines);
        let humidity_to_location = LocMap::from_lines(humidity_to_location_lines);

        Self {
            seed_input: seeds.clone(),
            seed_to_soil,
            soil_to_fertilizer,
            fertilizer_to_water,
            water_to_light,
            light_to_temperature,
            temperature_to_humidity,
            humidity_to_location,
        }
    }
}

#[derive(Debug)]
struct LocMap {
    components: Vec<LocMapComponent>,
}

impl LocMap {
    fn next_loc(&self, cur: u64) -> u64 {
        self.components
            .clone()
            .into_iter()
            .find_map(|map_comp| map_comp.next_loc(cur))
            .unwrap_or(cur)
    }

    fn from_lines(lines: Vec<String>) -> Self {
        let components: Vec<LocMapComponent> = lines
            .into_iter()
            .map(|s| LocMapComponent::from_string(s))
            .collect();
        Self { components }
    }
}

#[derive(Copy, Clone, Debug)]
struct LocMapComponent {
    from: u64,
    to: u64,
    offset: i64,
}

impl LocMapComponent {
    fn next_loc(&self, cur: u64) -> Option<u64> {
        if cur >= self.from && cur <= self.to {
            Some(((cur as i64) + (self.offset as i64)).try_into().unwrap())
        } else {
            None
        }
    }

    fn from_string(s: String) -> Self {
        let parts: Vec<&str> = s.split(" ").collect();
        assert!(parts.len() == 3, "Expected three numbers, got {:?}", parts);

        let dest_range_start: u64 = parts[0]
            .parse()
            .expect(format!("Expected number, got: {}", parts[0]).as_str());
        let source_range_start: u64 = parts[1]
            .parse()
            .expect(format!("Expected number, got: {}", parts[1]).as_str());
        let range_length: u64 = parts[2]
            .parse()
            .expect(format!("Expected number, got: {}", parts[2]).as_str());

        let from = source_range_start;
        let to = source_range_start + range_length - 1;
        let offset: i64 = (dest_range_start as i64) - (source_range_start as i64);
        Self { from, to, offset }
    }
}

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
