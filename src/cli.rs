use crate::config::Config;
use clap::{command, value_parser, Arg};
use fern::colors::{Color, ColoredLevelConfig};

pub fn parse_args() -> Config {
    let args = command!()
        .arg(Arg::new("paf").short('p').long("paf").required(true))
        .arg(
            Arg::new("minalnsize")
                .long("min-aln-size")
                .required(false)
                .value_parser(value_parser!(f32))
                .default_value("0.0")
                .help("Only displays alignment longer than this value"),
        )
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
        .arg(
            Arg::new("debug")
                .long("debug")
                .action(clap::ArgAction::SetTrue),
        )
        .get_matches();

    Config {
        paf: args.get_one::<String>("paf").unwrap().clone(),
        min_aln_size: args.get_one::<f32>("minalnsize").unwrap().to_owned(),
        height: args.get_one::<u32>("height").unwrap().to_owned(),
        width: args.get_one::<u32>("width").unwrap().to_owned(),
        margin_x: args.get_one::<f32>("marginx").unwrap().to_owned(),
        margin_y: args.get_one::<f32>("marginy").unwrap().to_owned(),
        font_size: args.get_one::<u32>("fontsize").unwrap().to_owned(),
        output: args.get_one::<String>("output").unwrap().clone(),
        debug: args.get_flag("debug"),
    }
}

pub fn setup_logging(log_level: log::LevelFilter) {
    // configure colors for the whole line
    let colors_line = ColoredLevelConfig::new()
        .error(Color::Red)
        .warn(Color::Yellow)
        // we actually don't need to specify the color for debug and info, they are white by default
        .info(Color::White)
        .debug(Color::White)
        // depending on the terminals color scheme, this is the same as the background color
        .trace(Color::BrightBlack);

    // configure colors for the name of the level.
    // since almost all of them are the same as the color for the whole line, we
    // just clone `colors_line` and overwrite our changes
    let colors_level = colors_line.info(Color::Green);
    // here we set up our fern Dispatch
    fern::Dispatch::new()
        .format(move |out, message, record| {
            out.finish(format_args!(
                "{color_line}[{date}] [{target}] [{level}{color_line}] {message}\x1B[0m",
                color_line = format_args!(
                    "\x1B[{}m",
                    colors_line.get_color(&record.level()).to_fg_str()
                ),
                date = chrono::Local::now().format("%d-%m-%Y %H:%M:%S"),
                target = record.target(),
                level = colors_level.color(record.level()),
                message = message,
            ));
        })
        // set the default log level. to filter out verbose log messages from dependencies, set
        // this to Warn and overwrite the log level for your crate.
        .level(log_level)
        // change log levels for individual modules. Note: This looks for the record's target
        // field which defaults to the module path but can be overwritten with the `target`
        // parameter:
        // `info!(target="special_target", "This log message is about special_target");`
        .level_for("pretty_colored", log::LevelFilter::Trace)
        // output to stdout
        .chain(std::io::stderr())
        .apply()
        .unwrap();
}
