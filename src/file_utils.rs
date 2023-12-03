use std::fs::File;
use std::io::{prelude::*, BufReader};

pub fn input_reader(path: &str) -> BufReader<File> {
    let file = File::open(path).unwrap();
    BufReader::new(file)
}

pub fn print_reader(reader: BufReader<File>) {
    for line in reader.lines() {
        println!("{}", line.unwrap());
    }
}
