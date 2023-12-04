use crate::file_utils;

use std::fs::File;
use std::io::{prelude::*, BufReader};

//const INPUT_FILE_PATH: &'static str = "input/day2_sample.txt";
const INPUT_FILE_PATH: &'static str = "input/day2.txt";

#[derive(Debug)]
struct Game {
    id: u32,
    sets: Vec<Set>,
}

impl Game {
    fn can_be_played_with(&self, red: u32, blue: u32, green: u32) -> bool {
        self.sets
            .clone()
            .into_iter()
            .all(|set| set.red <= red && set.blue <= blue && set.green <= green)
    }

    fn power(&self) -> u32 {
        let red_max = self.sets.clone().into_iter().map(|s| s.red).max().unwrap();
        let blue_max = self.sets.clone().into_iter().map(|s| s.blue).max().unwrap();
        let green_max = self
            .sets
            .clone()
            .into_iter()
            .map(|s| s.green)
            .max()
            .unwrap();
        red_max * blue_max * green_max
    }

    fn from_line(line: String) -> Game {
        let id = line
            .split(":")
            .next()
            .unwrap()
            .split(" ")
            .last()
            .unwrap()
            .parse::<u32>()
            .unwrap();

        let sets_row_string = line.split(":").nth(1).unwrap().trim();
        let set_strings: Vec<Vec<&str>> = sets_row_string
            .split(";")
            .map(|s| s.split(",").map(|cs| cs.trim()).collect())
            .collect();

        let sets: Vec<Set> = set_strings
            .into_iter()
            .map(|s| Set::from_strings(s))
            .collect::<Vec<Set>>();
        Game { id, sets }
    }
}

#[derive(Debug, Clone)]
struct Set {
    red: u32,
    blue: u32,
    green: u32,
}

impl Set {
    fn parse_color(xs: &Vec<&str>, color: &str) -> u32 {
        xs.into_iter()
            .filter(|s| s.contains(color))
            .map(|s| s.split(" ").next().unwrap().parse::<u32>().unwrap())
            .next()
            .or(Some(0))
            .unwrap()
    }

    pub fn from_strings(xs: Vec<&str>) -> Self {
        let red = Self::parse_color(&xs, "red");
        let blue = Self::parse_color(&xs, "blue");
        let green = Self::parse_color(&xs, "green");

        Self {
            red: red,
            blue: blue,
            green: green,
        }
    }
}

fn part1(reader: BufReader<File>) {
    let games = reader
        .lines()
        .into_iter()
        .map(|l| Game::from_line(l.unwrap()));
    let possible_games = games.filter(|game| game.can_be_played_with(12, 14, 13));
    let sum_of_ids: u32 = possible_games.map(|game| game.id).sum();
    println!("Part 1: {:?}", sum_of_ids);
}

fn part2(reader: BufReader<File>) {
    let games = reader
        .lines()
        .into_iter()
        .map(|l| Game::from_line(l.unwrap()));
    let power: u32 = games.map(|g| g.power()).sum();
    println!("Part 2: {:?}", power);
}

pub fn solve() {
    println!("Day 2 solutions:");
    let reader = file_utils::input_reader(INPUT_FILE_PATH);
    part1(reader);

    let reader = file_utils::input_reader(INPUT_FILE_PATH);
    part2(reader);
}
