use crate::file_utils;

use std::fs::File;
use std::io::{prelude::*, BufReader};

//const INPUT_FILE_PATH: &'static str = "input/day3_sample.txt";
const INPUT_FILE_PATH: &'static str = "input/day3.txt";

#[derive(Debug)]
struct Engine {
    schemantic: Vec<Vec<Visual>>,
}

impl Engine {
    fn get(&self, x: isize, y: isize) -> Option<Visual> {
        if x < 0 {
            None
        } else if y < 0 {
            None
        } else if x >= self.schemantic.first().unwrap().len().try_into().unwrap() {
            None
        } else if y >= self.schemantic.len().try_into().unwrap() {
            None
        } else {
            let xu: usize = x.try_into().unwrap();
            let yu: usize = y.try_into().unwrap();
            Some(self.schemantic[yu][xu])
        }
    }

    fn has_adjacent_symbol(&self, x: usize, y: usize) -> bool {
        // top left
        let xi: isize = x.try_into().unwrap();
        let yi: isize = y.try_into().unwrap();

        self.get(xi - 1, yi - 1) == Some(Visual::Symbol)
            || self.get(xi - 1, yi) == Some(Visual::Symbol)
            || self.get(xi - 1, yi + 1) == Some(Visual::Symbol)
            || self.get(xi, yi + 1) == Some(Visual::Symbol)
            || self.get(xi + 1, yi + 1) == Some(Visual::Symbol)
            || self.get(xi + 1, yi) == Some(Visual::Symbol)
            || self.get(xi + 1, yi - 1) == Some(Visual::Symbol)
            || self.get(xi, yi - 1) == Some(Visual::Symbol)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Visual {
    NumPart(usize),
    Symbol,
    Empty,
}

fn parse_line(line: String) -> Vec<Visual> {
    line.chars()
        .map(|c| match c {
            '.' => Visual::Empty,
            d if d.is_digit(10) => Visual::NumPart(d.to_digit(10).unwrap().try_into().unwrap()),
            _ => Visual::Symbol,
        })
        .collect::<Vec<Visual>>()
}

fn part1(reader: BufReader<File>) {
    let schemantic: Vec<Vec<Visual>> = reader
        .lines()
        .map(|line| parse_line(line.unwrap()))
        .collect();
    let engine = Engine { schemantic };

    let mut part_numbers: Vec<usize> = vec![];
    let mut cur_num_parts = String::from("");
    let mut has_adjacent = false;
    for (y, vs) in engine.schemantic.iter().enumerate() {
        if !cur_num_parts.is_empty() {
            if has_adjacent {
                let part_number: usize = cur_num_parts.parse().unwrap();
                part_numbers.push(part_number);
            }
            has_adjacent = false;
            cur_num_parts = String::from("");
        }
        for (x, v) in vs.iter().enumerate() {
            match v {
                Visual::NumPart(num) => {
                    let num_c = char::from_digit(num.clone().try_into().unwrap(), 10).unwrap();
                    cur_num_parts.push(num_c);
                    if has_adjacent || engine.has_adjacent_symbol(x, y) {
                        has_adjacent = true;
                    }
                }
                _ => {
                    if !cur_num_parts.is_empty() {
                        if has_adjacent {
                            let part_number: usize = cur_num_parts.parse().unwrap();
                            part_numbers.push(part_number);
                        }
                        has_adjacent = false;
                        cur_num_parts = String::from("");
                    }
                }
            }
        }
    }
    let part_numbers_sum: usize = part_numbers.into_iter().sum();
    println!("Part1: {:?}", part_numbers_sum)
}

pub fn solve() {
    println!("Day 3 solutions:");
    let reader = file_utils::input_reader(INPUT_FILE_PATH);
    part1(reader)
}
