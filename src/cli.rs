use crate::config::Config;
use clap::{command, value_parser, Arg};

pub fn parse_args() -> Config {
    let args = command!()
        .arg(Arg::new("paf").short('p').long("paf").required(true))
        .arg(
            Arg::new("height")
                .long("height")
                .required(true)
                .value_parser(value_parser!(u32))
                .help("Height of the image"),
        )
        .arg(
            Arg::new("width")
                .long("width")
                .required(true)
                .value_parser(value_parser!(u32))
                .help("Width of the image"),
        )
        .arg(
            Arg::new("fontsize")
                .long("font-size")
                .required(false)
                .default_value("24")
                .value_parser(value_parser!(u32))
                .help("Font size to use"),
        )
        .arg(
            Arg::new("marginx")
                .long("margin-x")
                .required(false)
                .default_value("0.05")
                .value_parser(value_parser!(f32))
                .help("Percentage of the image that will stay blank on the x-axis"),
        )
        .arg(
            Arg::new("marginy")
                .long("margin-y")
                .required(false)
                .default_value("0.05")
                .value_parser(value_parser!(f32))
                .help("Percentage of the image that will stay blank on the y-axis"),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .default_value("dotplot.png")
                .help("Name of the output file"),
        )
        .get_matches();

    Config {
        paf: args.get_one::<String>("paf").unwrap().clone(),
        height: args.get_one::<u32>("height").unwrap().to_owned(),
        width: args.get_one::<u32>("width").unwrap().to_owned(),
        margin_x: args.get_one::<f32>("marginx").unwrap().to_owned(),
        margin_y: args.get_one::<f32>("marginy").unwrap().to_owned(),
        font_size: args.get_one::<u32>("fontsize").unwrap().to_owned(),
        output: args.get_one::<String>("output").unwrap().clone(),
    }
}
