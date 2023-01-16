#[warn(clippy::all, clippy::pedantic)]
mod cli;
mod config;
mod dotplot;

use cli::parse_args;
use dotplot::Dotplot;

fn main() {
    let config = parse_args();
    let dotplot = Dotplot::new(config);
    dotplot.save();
}
