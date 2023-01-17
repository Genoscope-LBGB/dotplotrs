#[warn(clippy::all, clippy::pedantic)]
mod cli;
mod config;
mod dotplot;
mod parser;

use cli::{parse_args, setup_logging};
use dotplot::Dotplot;
use log::{debug, info, LevelFilter};
use parser::{parse_paf, sort_records_hash};

fn main() {
    let config = parse_args();
    match config.debug {
        true => setup_logging(LevelFilter::Debug),
        false => setup_logging(LevelFilter::Info),
    }

    info!("Reading PAF file: {}", config.paf);
    let mut records = parse_paf(&config.paf);
    debug!("Sorting records");
    sort_records_hash(&mut records);

    info!("Building the dotplot");
    let mut dotplot = Dotplot::new(&config);
    dotplot.draw(&records);

    dotplot.save();
}
