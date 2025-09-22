use std::collections::HashMap;
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::num::ParseIntError;

#[derive(Debug)]
pub struct PafRecord {
    pub qname: String,
    pub qlen: u64,
    pub qstart: u64,
    pub qend: u64,
    pub strand: char,
    pub tname: String,
    pub tlen: u64,
    pub tstart: u64,
    pub tend: u64,
    pub nb_matches: u64,
    pub is_best_matching_chr: bool,
}

impl PafRecord {
    pub fn from_paf_line(line_number: usize, line: &str) -> Result<Self, PafError> {
        let split_line = line.split('\t').collect::<Vec<&str>>();
        const REQUIRED_FIELDS: usize = 10;

        if split_line.len() < REQUIRED_FIELDS {
            return Err(PafError::MissingColumns {
                line: line_number,
                expected: REQUIRED_FIELDS,
                found: split_line.len(),
            });
        }

        let strand_str = split_line[4];
        let strand = strand_str
            .chars()
            .next()
            .ok_or_else(|| PafError::InvalidStrand {
                line: line_number,
                value: strand_str.to_string(),
            })?;

        if !matches!(strand, '+' | '-') {
            return Err(PafError::InvalidStrand {
                line: line_number,
                value: strand_str.to_string(),
            });
        }

        Ok(Self {
            qname: String::from(split_line[0]),
            qlen: parse_u64(split_line[1], "query length", line_number)?,
            qstart: parse_u64(split_line[2], "query start", line_number)?,
            qend: parse_u64(split_line[3], "query end", line_number)?,
            strand,
            tname: String::from(split_line[5]),
            tlen: parse_u64(split_line[6], "target length", line_number)?,
            tstart: parse_u64(split_line[7], "target start", line_number)?,
            tend: parse_u64(split_line[8], "target end", line_number)?,
            nb_matches: parse_u64(split_line[9], "number of matches", line_number)?,
            is_best_matching_chr: false,
        })
    }
}

pub fn parse_paf(
    input_paf: &str,
    min_aln_size: u64,
) -> Result<Vec<(String, Vec<PafRecord>)>, PafError> {
    let reader = BufReader::new(File::open(input_paf).map_err(PafError::Io)?);

    let mut records_hash: HashMap<String, Vec<PafRecord>> = HashMap::new();
    for (line_number, line) in reader.lines().enumerate() {
        let raw_line = line.map_err(PafError::Io)?;
        if raw_line.trim().is_empty() {
            continue;
        }

        let record = PafRecord::from_paf_line(line_number + 1, &raw_line)?;

        let aln_size = record.tend as i64 - record.tstart as i64;
        if i64::abs(aln_size) >= min_aln_size as i64 {
            match records_hash.get_mut(&record.tname) {
                Some(records) => records.push(record),
                None => {
                    records_hash.insert(record.tname.clone(), vec![record]);
                }
            }
        }
    }

    sort_records_hash(&mut records_hash);
    Ok(Vec::from_iter(records_hash))
}

fn parse_u64(value: &str, field: &'static str, line_number: usize) -> Result<u64, PafError> {
    value.parse::<u64>().map_err(|source| PafError::InvalidNumber {
        line: line_number,
        field,
        value: value.to_string(),
        source,
    })
}

#[derive(Debug)]
pub enum PafError {
    Io(std::io::Error),
    MissingColumns {
        line: usize,
        expected: usize,
        found: usize,
    },
    InvalidNumber {
        line: usize,
        field: &'static str,
        value: String,
        source: ParseIntError,
    },
    InvalidStrand {
        line: usize,
        value: String,
    },
}

impl Display for PafError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::MissingColumns { line, expected, found } => write!(
                f,
                "line {line}: expected at least {expected} tab-separated fields, found {found}"
            ),
            Self::InvalidNumber { line, field, value, .. } => write!(
                f,
                "line {line}: could not parse {field} value '{value}' as an integer"
            ),
            Self::InvalidStrand { line, value } => write!(
                f,
                "line {line}: strand must be '+' or '-', found '{value}'"
            ),
        }
    }
}

impl Error for PafError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::InvalidNumber { source, .. } => Some(source),
            _ => None,
        }
    }
}

pub fn sort_records_hash(records_hash: &mut HashMap<String, Vec<PafRecord>>) {
    for (_, records) in records_hash.iter_mut() {
        sort_records_by_tstart(records);
    }
}

pub fn sort_records_by_tstart(records: &mut [PafRecord]) {
    records.sort_by(|a, b| a.tstart.partial_cmp(&b.tstart).unwrap());
}
