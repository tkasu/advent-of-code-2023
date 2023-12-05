use crate::file_utils;

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{prelude::*, BufReader};

//const INPUT_FILE_PATH: &'static str = "input/day3_sample.txt";
const INPUT_FILE_PATH: &'static str = "input/day3.txt";

#[derive(Debug)]
struct Engine {
    schemantic: Vec<Vec<Visual>>,
    schemantic_width: usize,
    schemantic_height: usize,
}

impl Engine {
    fn new(schemantic: Vec<Vec<Visual>>) -> Self {
        let width: usize = schemantic.first().unwrap().len().try_into().unwrap();
        let height: usize = schemantic.len().try_into().unwrap();
        Self {
            schemantic,
            schemantic_width: width,
            schemantic_height: height,
        }
    }

    fn get(&self, x: isize, y: isize) -> Option<Visual> {
        if x < 0 {
            None
        } else if y < 0 {
            None
        } else if x >= self.schemantic_width.try_into().unwrap() {
            None
        } else if y >= self.schemantic_height.try_into().unwrap() {
            None
        } else {
            let xu: usize = x.try_into().unwrap();
            let yu: usize = y.try_into().unwrap();
            Some(self.schemantic[yu][xu])
        }
    }

    fn has_adjacent_symbol(&self, x: usize, y: usize) -> bool {
        fn is_symbol(ov: Option<Visual>) -> bool {
            ov.map(|s| s.is_symbol()).unwrap_or(false)
        }

        let xi: isize = x.try_into().unwrap();
        let yi: isize = y.try_into().unwrap();

        let adjacent_coords = [
            (xi - 1, yi - 1),
            (xi - 1, yi),
            (xi - 1, yi + 1),
            (xi, yi + 1),
            (xi + 1, yi + 1),
            (xi + 1, yi),
            (xi + 1, yi - 1),
            (xi, yi - 1),
        ];

        adjacent_coords
            .into_iter()
            .any(|(_x, _y)| is_symbol(self.get(_x, _y)))
    }

    fn adjacent_star_coords(&self, x: usize, y: usize) -> HashSet<(usize, usize)> {
        fn is_gear(ov: Option<Visual>) -> bool {
            ov == Some(Visual::Symbol('*'))
        }

        let xi: isize = x.try_into().unwrap();
        let yi: isize = y.try_into().unwrap();

        let adjacent_coords = [
            (xi - 1, yi - 1),
            (xi - 1, yi),
            (xi - 1, yi + 1),
            (xi, yi + 1),
            (xi + 1, yi + 1),
            (xi + 1, yi),
            (xi + 1, yi - 1),
            (xi, yi - 1),
        ];

        adjacent_coords
            .into_iter()
            .filter(|(_x, _y)| is_gear(self.get(*_x, *_y)))
            .map(|(_x, _y)| (_x as usize, _y as usize))
            .collect::<HashSet<(usize, usize)>>()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Visual {
    NumPart(usize),
    Symbol(char),
    Empty,
}

impl Visual {
    fn is_symbol(&self) -> bool {
        match self {
            Self::Symbol(_) => true,
            _ => false,
        }
    }
}

fn parse_line(line: String) -> Vec<Visual> {
    line.chars()
        .map(|c| match c {
            '.' => Visual::Empty,
            d if d.is_digit(10) => Visual::NumPart(d.to_digit(10).unwrap().try_into().unwrap()),
            c => Visual::Symbol(c),
        })
        .collect::<Vec<Visual>>()
}

fn part1(reader: BufReader<File>) {
    let schemantic: Vec<Vec<Visual>> = reader
        .lines()
        .map(|line| parse_line(line.unwrap()))
        .collect();
    let engine = Engine::new(schemantic);

    fn flush_state(numbers: &mut Vec<usize>, num_string: &mut String, adjacent_flag: &mut bool) {
        if !num_string.is_empty() {
            if *adjacent_flag {
                let part_number: usize = num_string.parse().unwrap();
                numbers.push(part_number);
            }
            *adjacent_flag = false;
            *num_string = String::from("");
        }
    }

    let mut part_numbers: Vec<usize> = vec![];
    let mut cur_num_string = String::from("");
    let mut has_adjacent = false;
    for (y, vs) in engine.schemantic.iter().enumerate() {
        flush_state(&mut part_numbers, &mut cur_num_string, &mut has_adjacent);
        for (x, v) in vs.iter().enumerate() {
            match v {
                Visual::NumPart(num) => {
                    let num_c = char::from_digit(num.clone().try_into().unwrap(), 10).unwrap();
                    cur_num_string.push(num_c);
                    if has_adjacent || engine.has_adjacent_symbol(x, y) {
                        has_adjacent = true;
                    }
                }
                _ => {
                    flush_state(&mut part_numbers, &mut cur_num_string, &mut has_adjacent);
                }
            }
        }
    }
    flush_state(&mut part_numbers, &mut cur_num_string, &mut has_adjacent);

    let part_numbers_sum: usize = part_numbers.into_iter().sum();
    println!("Part1: {:?}", part_numbers_sum)
}

fn part2(reader: BufReader<File>) {
    let schemantic: Vec<Vec<Visual>> = reader
        .lines()
        .map(|line| parse_line(line.unwrap()))
        .collect();
    let engine = Engine::new(schemantic);

    fn flush_state(
        gear_nums: &mut HashMap<(usize, usize), Vec<usize>>,
        num_string: &mut String,
        cur_adjacent_gears: &mut HashSet<(usize, usize)>,
    ) {
        if !num_string.is_empty() {
            if !cur_adjacent_gears.is_empty() {
                let part_number: usize = num_string.parse().unwrap();
                for coord in cur_adjacent_gears.iter() {
                    gear_nums.entry(*coord).or_insert(vec![]).push(part_number);
                }
            }
            HashSet::clear(cur_adjacent_gears);
            *num_string = String::from("");
        }
    }

    let mut gear_adjacent_nums: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    let mut cur_adjacent_gears: HashSet<(usize, usize)> = HashSet::new();
    let mut cur_num_string = String::from("");

    for (y, vs) in engine.schemantic.iter().enumerate() {
        flush_state(
            &mut gear_adjacent_nums,
            &mut cur_num_string,
            &mut cur_adjacent_gears,
        );
        for (x, v) in vs.iter().enumerate() {
            match v {
                Visual::NumPart(num) => {
                    let num_c = char::from_digit(num.clone().try_into().unwrap(), 10).unwrap();
                    cur_num_string.push(num_c);

                    let adjacent_gears = engine.adjacent_star_coords(x, y);
                    cur_adjacent_gears.extend(adjacent_gears)
                }
                _ => {
                    flush_state(
                        &mut gear_adjacent_nums,
                        &mut cur_num_string,
                        &mut cur_adjacent_gears,
                    );
                }
            }
        }
    }
    flush_state(
        &mut gear_adjacent_nums,
        &mut cur_num_string,
        &mut cur_adjacent_gears,
    );

    let gear_ratios_sum: usize = gear_adjacent_nums
        .values()
        .filter(|vals| vals.len() == 2)
        .map(|vals| vals.first().unwrap() * vals.last().unwrap())
        .sum();

    println!("Part2: {:?}", gear_ratios_sum);
}

pub fn solve() {
    println!("Day 3 solutions:");
    let reader = file_utils::input_reader(INPUT_FILE_PATH);
    part1(reader);

    let reader = file_utils::input_reader(INPUT_FILE_PATH);
    part2(reader);
}
