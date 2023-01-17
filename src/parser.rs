use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Debug)]
pub struct PafRecord {
    pub qname: String,
    pub qlen: u64,
    pub qstart: u64,
    pub qend: u64,
    pub tname: String,
    pub tlen: u64,
    pub tstart: u64,
    pub tend: u64,
    pub nb_matches: u64,
}

impl PafRecord {
    pub fn from_paf_line(line: String) -> Self {
        let split_line = line.split("\t").collect::<Vec<&str>>();

        Self {
            qname: String::from(split_line[0]),
            qlen: split_line[1].parse::<u64>().unwrap(),
            qstart: split_line[2].parse::<u64>().unwrap(),
            qend: split_line[3].parse::<u64>().unwrap(),
            tname: String::from(split_line[5]),
            tlen: split_line[6].parse::<u64>().unwrap(),
            tstart: split_line[7].parse::<u64>().unwrap(),
            tend: split_line[8].parse::<u64>().unwrap(),
            nb_matches: split_line[9].parse::<u64>().unwrap(),
        }
    }
}

pub fn parse_paf(input_paf: &String) -> Vec<(String, Vec<PafRecord>)> {
    let paf = File::open(input_paf);
    let reader = BufReader::new(paf.unwrap());

    let mut records_hash: HashMap<String, Vec<PafRecord>> = HashMap::new();
    for line in reader.lines() {
        let record = PafRecord::from_paf_line(line.unwrap());
        match records_hash.get_mut(&record.tname) {
            Some(records) => records.push(record),
            None => {
                records_hash.insert(record.tname.clone(), vec![record]);
            }
        }
    }

    sort_records_hash(&mut records_hash);
    Vec::from_iter(records_hash)
}

pub fn sort_records_hash(records_hash: &mut HashMap<String, Vec<PafRecord>>) {
    for (_, records) in records_hash.iter_mut() {
        sort_records_by_tstart(records);
    }
}

pub fn sort_records_by_tstart(records: &mut Vec<PafRecord>) {
    records.sort_by(|a, b| a.tstart.partial_cmp(&b.tstart).unwrap());
}
