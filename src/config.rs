pub struct Config {
    pub paf: String,
    pub min_aln_size: u64,
    pub height: u32,
    pub width: u32,
    pub margin_x: f32,
    pub margin_y: f32,
    pub output: String,
    pub debug: bool,
    pub no_color: bool,
    pub line_thickness: u32,
    pub theme: Theme,
    pub bubble_min_sequence_size: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Theme {
    Light,
    Dark,
}

impl Theme {
    pub fn background_color(self) -> [u8; 4] {
        match self {
            Theme::Light => [255, 255, 255, 255],
            Theme::Dark => [2, 8, 23, 255],
        }
    }

    pub fn foreground_color(self) -> [u8; 4] {
        match self {
            Theme::Light => [0, 0, 0, 255],
            Theme::Dark => [250, 250, 250, 255],
        }
    }
}
