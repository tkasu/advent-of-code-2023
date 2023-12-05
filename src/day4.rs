use crate::file_utils;

use std::fs::File;
use std::io::{prelude::*, BufReader};

//const INPUT_FILE_PATH: &'static str = "input/day4_sample.txt";
const INPUT_FILE_PATH: &'static str = "input/day4.txt";

#[derive(Debug)]
struct Cards {
    id: usize,
    numbers: Vec<usize>,
    winning_numbers: Vec<usize>,
}

impl Cards {
    fn winning_numbers_included(&self) -> Vec<usize> {
        self.numbers
            .clone()
            .into_iter()
            .filter(|num| self.winning_numbers.contains(num))
            .collect()
    }

    fn from_line(line: String) -> Self {
        fn parse_nums(s: &str) -> Vec<usize> {
            s.trim()
                .split(" ")
                .filter(|s| !s.is_empty())
                .map(|s| usize::from_str_radix(s, 10).unwrap())
                .collect::<Vec<usize>>()
        }

        let id: usize = line
            .split(":")
            .next()
            .unwrap()
            .split(" ")
            .last()
            .unwrap()
            .parse()
            .unwrap();

        let all_numbers_string = line.split(":").last().unwrap();
        let numbers: Vec<usize> = parse_nums(all_numbers_string.split("|").next().unwrap());
        let winning_numbers: Vec<usize> = parse_nums(all_numbers_string.split("|").last().unwrap());

        Self {
            id,
            numbers,
            winning_numbers,
        }
    }
}

fn part1(reader: BufReader<File>) {
    let card_games: Vec<Cards> = reader
        .lines()
        .map(|line| Cards::from_line(line.unwrap()))
        .collect();
    let winning_numbers: Vec<usize> = card_games
        .into_iter()
        .map(|cards| cards.winning_numbers_included())
        .map(|cards| cards.len())
        .filter(|len| len > &0)
        .map(|len| (2 as usize).pow((len - 1).try_into().unwrap()))
        .collect();
    let sum: usize = winning_numbers.into_iter().sum();
    println!("Part2: {:?}", sum);
}

pub fn solve() {
    println!("Day 4 solutions:");
    let reader = file_utils::input_reader(INPUT_FILE_PATH);
    //file_utils::print_reader(reader)
    part1(reader)
}
