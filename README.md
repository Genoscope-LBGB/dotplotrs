# Dotplotrs - Easily create dotplots from PAF file

To install, clone the repository and run `cargo build --release`:
```
git clone https://github.com/Genoscope-LBGB/dotplotrs
cd dotplotrs
cargo build --release
```
If everything went well, you should have a `dotplotrs` binary in the `target/release` directory.

## Usage
```
dotplotrs -h
Usage: dotplotrs [OPTIONS] --paf <paf>

Options:
  -p, --paf <paf>
  -m, --min-aln-size <minalnsize>       Only displays alignment longer than this value [default: 0]
      --height <height>                 Height of the image [default: 2000]
      --width <width>                   Width of the image [default: 2000]
      --font-size <fontsize>            Font size to use [default: 24]
      --margin-x <marginx>              Percentage of the image that will stay blank on the x-axis [default: 0.05]
      --margin-y <marginy>              Percentage of the image that will stay blank on the y-axis [default: 0.05]
  -o, --output <output>                 Name of the output file [default: dotplot.png]
      --debug
      --line-thickness <linethickness>  Thickness of lines (doubled for best matching chromosomes) [default: 1]
  -h, --help                            Print help
  -V, --version                         Print version
```