#[warn(clippy::all, clippy::pedantic)]
mod cli;
mod config;
mod dotplot;
mod parser;

use cli::parse_args;
use dotplot::Dotplot;
use parser::parse_paf;

fn main() {
    let config = parse_args();

    println!("Reading PAF file: {}", config.paf);
    let records = parse_paf(&config.paf);

    println!("Building the dotplot");
    let dotplot = Dotplot::new(&config);
    dotplot.save();
}
