#[warn(clippy::all, clippy::pedantic)]
mod cli;
mod config;
mod dotplot;

use cli::parse_cli;
use dotplot::Dotplot;

fn main() {
    let config = parse_cli();
    let dotplot = Dotplot::new(config);
    dotplot.save();
}
