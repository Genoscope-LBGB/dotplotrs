use crate::config::Config;
use image::{Rgb, RgbImage};

pub struct Dotplot {
    config: Config,
    plot: RgbImage,
}

impl Dotplot {
    pub fn new(config: Config) -> Self {
        let plot = RgbImage::new(config.width, config.height);

        let mut dotplot = Self { config, plot };
        dotplot.init_plot();
        dotplot
    }

    fn init_plot(&mut self) {
        self.init_background();
    }

    fn init_background(&mut self) {
        for x in 0..self.config.width {
            for y in 0..self.config.height {
                self.plot.put_pixel(x, y, Rgb([255, 255, 255]));
            }
        }
    }

    pub fn save(&self) {
        self.plot.save(&self.config.output).unwrap();
    }
}
