use std::collections::HashMap;

use crate::{config::Config, parser::PafRecord};
use image::{Rgb, RgbImage};
use imageproc::drawing::draw_line_segment_mut;

pub struct TargetCoord {
    pub start: u32,
    pub end: u32,
}

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

    pub fn draw(&mut self, records_hash: &HashMap<String, Vec<PafRecord>>) {
        let target_coords = self.targets_to_coords(records_hash);
        self.draw_target_ticks(target_coords);
    }

    // Gets the porsition of each target on the x-axis
    fn targets_to_coords(
        &self,
        records_hash: &HashMap<String, Vec<PafRecord>>,
    ) -> HashMap<String, TargetCoord> {
        let targets_sizes = self.get_target_sizes_in_bp(records_hash);
        let total_size = targets_sizes.iter().fold(0 as u64, |acc, x| acc + x.1);

        let mut targets_sizes_vec = Vec::from_iter(&targets_sizes);
        targets_sizes_vec.sort_by(|&(_, tlen_1), &(_, tlen_2)| tlen_2.cmp(tlen_1));

        let axis_size = self.end_x - self.start_x;
        let px_per_bp = axis_size as f64 / total_size as f64;

        let mut coords: HashMap<String, TargetCoord> = HashMap::new();
        let mut last_pos: u32 = self.start_x;
        for (target, size) in targets_sizes_vec.iter() {
            let segment_size = (**size as f64 * px_per_bp) as u32;
            let end_coord = last_pos + segment_size;
            let target_coords = TargetCoord {
                start: last_pos,
                end: end_coord,
            };
            coords.insert((*target).clone(), target_coords);
            last_pos = end_coord;
        }

        coords
    }

    // Gets all targets sizes
    fn get_target_sizes_in_bp(
        &self,
        records_hash: &HashMap<String, Vec<PafRecord>>,
    ) -> HashMap<String, u64> {
        let mut targets_sizes: HashMap<String, u64> = HashMap::new();

        for (tname, records) in records_hash.iter() {
            match targets_sizes.get_mut(tname) {
                Some(_) => panic!("Multiple targets have the same name! {}", tname),
                None => {
                    targets_sizes.insert(tname.clone(), records[0].tlen);
                }
            }
        }

        targets_sizes
    }

    fn draw_target_ticks(&mut self, coords: HashMap<String, TargetCoord>) {
        for (target, TargetCoord { start, end }) in coords.iter() {
            if (*start > self.start_x) {
                draw_line_segment_mut(
                    &mut self.plot,
                    (*start as f32, self.end_y as f32),
                    (*start as f32, self.end_y as f32 + 10.0),
                    Rgb([0, 0, 0]),
                );
            }
        }
    }

    // Saves the plot to a file
    pub fn save(&self) {
        self.plot.save(&self.config.output).unwrap();
    }
}
