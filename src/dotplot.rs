use crate::config::Config;
use image::{Rgb, RgbImage};

pub struct Dotplot<'a> {
    config: &'a Config,
    plot: RgbImage,
}

impl<'a> Dotplot<'a> {
    pub fn new(config: &'a Config) -> Self {
        let plot = RgbImage::new(config.width, config.height);

        let mut dotplot = Self { config, plot };
        dotplot.init_plot();
        dotplot
    }

    // Initializes the dotplot with a blank background and empty axes
    fn init_plot(&mut self) {
        self.init_background();
        self.init_axes_lines();
    }

    // Initialize the background to white
    fn init_background(&mut self) {
        for x in 0..self.config.width {
            for y in 0..self.config.height {
                self.plot.put_pixel(x, y, Rgb([255, 255, 255]));
            }
        }
    }

    // Draws blank axes
    fn init_axes_lines(&mut self) {
        let offset_x = (self.config.width as f32 * self.config.margin_x) as u32;
        let offset_y = (self.config.height as f32 * self.config.margin_y) as u32;

        let y_min = offset_y;
        let y_max = self.config.height - offset_y;
        for x in offset_x..(self.config.width - offset_x) {
            self.plot.put_pixel(x, y_min, Rgb([0, 0, 0]));
            self.plot.put_pixel(x, y_max, Rgb([0, 0, 0]));
        }

        let x_min = offset_x;
        let x_max = self.config.width - offset_x;
        for y in offset_y..(self.config.height - offset_y) {
            self.plot.put_pixel(x_min, y, Rgb([0, 0, 0]));
            self.plot.put_pixel(x_max, y, Rgb([0, 0, 0]));
        }
    }

    // Saves the plot to a file
    pub fn save(&self) {
        self.plot.save(&self.config.output).unwrap();
    }
}
