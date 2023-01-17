use std::collections::HashMap;

use crate::{config::Config, parser::PafRecord};
use image::{Rgb, RgbImage};

pub struct Dotplot<'a> {
    config: &'a Config,
    start_x: u32,
    end_x: u32,
    start_y: u32,
    end_y: u32,
    plot: RgbImage,
}

impl<'a> Dotplot<'a> {
    pub fn new(config: &'a Config) -> Self {
        let plot = RgbImage::new(config.width, config.height);

        let offset_x = (config.width as f32 * config.margin_x) as u32;
        let offset_y = (config.height as f32 * config.margin_y) as u32;
        let start_x = offset_x;
        let end_x = config.width - offset_x;
        let start_y = offset_y;
        let end_y = config.height - offset_y;

        let mut dotplot = Self {
            config,
            start_x,
            end_x,
            start_y,
            end_y,
            plot,
        };

        dotplot.init_plot();
        dotplot
    }

    pub fn draw(&mut self, records: &HashMap<String, Vec<PafRecord>>) {}

    // Gets the porsition of each target on the x-axis
    fn record_target_to_coords(records: &HashMap<String, Vec<PafRecord>>) {}

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
        for x in self.start_x..self.end_x {
            self.plot.put_pixel(x, self.start_y, Rgb([0, 0, 0]));
            self.plot.put_pixel(x, self.end_y, Rgb([0, 0, 0]));
        }

        for y in self.start_y..self.end_y {
            self.plot.put_pixel(self.start_x, y, Rgb([0, 0, 0]));
            self.plot.put_pixel(self.end_x, y, Rgb([0, 0, 0]));
        }
    }

    // Saves the plot to a file
    pub fn save(&self) {
        self.plot.save(&self.config.output).unwrap();
    }
}
