#[warn(clippy::all, clippy::pedantic)]
mod cli;
mod config;
mod dotplot;
mod parser;

use cli::{parse_args, setup_logging};
use dotplot::Dotplot;
use log::{debug, info, LevelFilter};
use parser::parse_paf;

fn main() {
    let config = parse_args();
    match config.debug {
        true => setup_logging(LevelFilter::Debug),
        false => setup_logging(LevelFilter::Info),
    }

    info!("Reading PAF file: {}", config.paf);
    let records = parse_paf(&config.paf, config.min_aln_size);
    debug!("Sorting records");

    info!("Building the dotplot");
    let mut dotplot = Dotplot::new(&config);
    dotplot.draw(records);

    dotplot.save();
}
