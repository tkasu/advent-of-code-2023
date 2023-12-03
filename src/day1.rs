use crate::file_utils;

use std::collections::HashMap;
use std::fs::File;
use std::io::{prelude::*, BufReader};

//const INPUT_FILE_PATH: &'static str = "input/day1_sample.txt";
//const INPUT_FILE_PATH: &'static str = "input/day1_sample_2.txt";
const INPUT_FILE_PATH: &'static str = "input/day1.txt";

fn part1(reader: BufReader<File>) {
    let digits = reader
        .lines()
        .into_iter()
        .map(|line| line.unwrap().chars().filter(|c| c.is_digit(10)).collect())
        .collect::<Vec<String>>();

    let calibration_values = digits
        .into_iter()
        .map(|s| format!("{}{}", s.chars().next().unwrap(), s.chars().last().unwrap()))
        .map(|s| s.parse::<i32>().unwrap())
        .collect::<Vec<i32>>();

    let sum: i32 = calibration_values.into_iter().sum();
    println!("Part 1: {}", sum);
}

fn next_digit(rem_line: &String) -> Option<u32> {
    let digit_map: HashMap<&str, u32> = HashMap::from([
        ("one", 1),
        ("two", 2),
        ("three", 3),
        ("four", 4),
        ("five", 5),
        ("six", 6),
        ("seven", 7),
        ("eight", 8),
        ("nine", 9),
    ]);
    if let Some(digit) = rem_line.chars().next().unwrap().to_digit(10) {
        Some(digit)
    } else {
        let digit_key = digit_map
            .keys()
            .into_iter()
            .find(|k| rem_line.starts_with(*k));
        match digit_key {
            Some(key) => Some(digit_map.get(key).unwrap().clone()),
            None => None,
        }
    }
}

fn parse_line(line: String) -> u32 {
    let mut digits: Vec<u32> = vec![];
    let mut rem_path = line;

    while rem_path.len() > 0 {
        let maybe_digit = next_digit(&rem_path);
        if let Some(digit) = maybe_digit {
            digits.push(digit)
        };
        rem_path = rem_path[1..].to_string();
    }
    digits.first().unwrap() * 10 + digits.last().unwrap()
}

fn part2(reader: BufReader<File>) {
    let mut sum: u32 = 0;
    for line in reader.lines() {
        sum += parse_line(line.unwrap());
    }
    println!("Part 2: {:?}", sum);
}

pub fn solve() {
    println!("Day 1 solutions:");
    let reader = file_utils::input_reader(INPUT_FILE_PATH);
    //file_utils::print_reader(reader);
    part1(reader);

    let reader = file_utils::input_reader(INPUT_FILE_PATH);
    part2(reader);
}
