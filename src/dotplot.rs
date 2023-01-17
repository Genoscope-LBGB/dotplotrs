use std::collections::HashMap;

use crate::{config::Config, parser::PafRecord};
use image::{Rgb, RgbImage};
use imageproc::drawing::draw_line_segment_mut;

pub struct TargetCoord {
    pub start: u32,
    pub end: u32,
}

pub struct QueryCoord {
    pub start: f32,
    pub end: f32,
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

    pub fn draw(&mut self, records: &Vec<(String, Vec<PafRecord>)>) {
        let target_coords = self.targets_to_coords(records);
        self.draw_target_ticks(&target_coords);

        let query_coords = self.queries_to_coords(records);
        self.draw_query_ticks(&query_coords);
    }

    // Gets the porsition of each target on the x-axis
    fn targets_to_coords(
        &self,
        records: &Vec<(String, Vec<PafRecord>)>,
    ) -> HashMap<String, TargetCoord> {
        let targets_sizes = self.get_target_sizes_in_bp(records);
        let total_size = targets_sizes.iter().fold(0 as u64, |acc, x| acc + x.1);

        let axis_size = self.end_x - self.start_x;
        let px_per_bp = axis_size as f64 / total_size as f64;

        let mut coords: HashMap<String, TargetCoord> = HashMap::new();
        let mut last_pos: u32 = self.start_x;
        for (target, size) in targets_sizes.iter() {
            let segment_size = (*size as f64 * px_per_bp) as u32;
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

    // Gets the porsition of each target on the x-axis
    fn queries_to_coords(
        &self,
        records: &Vec<(String, Vec<PafRecord>)>,
    ) -> HashMap<String, QueryCoord> {
        let query_sizes = self.get_query_sizes_in_bp(records);
        let total_size = query_sizes.iter().fold(0 as u64, |acc, x| acc + x.1);

        let axis_size = self.end_y - self.start_y;
        let px_per_bp = axis_size as f64 / total_size as f64;

        let mut coords = HashMap::new();
        let mut last_pos = self.start_y as f32;
        for (query, size) in query_sizes.iter() {
            let segment_size = (*size as f64 * px_per_bp) as f32;
            let end_coord = last_pos + segment_size;
            let query_coords = QueryCoord {
                start: last_pos,
                end: end_coord,
            };
            coords.insert(query.clone(), query_coords);
            println!("{} {}", size, segment_size);
            last_pos = end_coord;
        }

        coords
    }

    // Gets all targets sizes
    fn get_target_sizes_in_bp(
        &self,
        records_vec: &Vec<(String, Vec<PafRecord>)>,
    ) -> Vec<(String, u64)> {
        let mut targets_sizes: HashMap<String, u64> = HashMap::new();

        for (tname, records) in records_vec.iter() {
            targets_sizes
                .entry(tname.clone())
                .or_insert(records[0].tlen);
        }

        let mut targets_sizes_vec = Vec::from_iter(targets_sizes);
        targets_sizes_vec.sort_by(|&(_, tlen_1), &(_, tlen_2)| tlen_2.cmp(&tlen_1));
        targets_sizes_vec
    }

    // Gets all query sizes
    fn get_query_sizes_in_bp(
        &self,
        records_vec: &Vec<(String, Vec<PafRecord>)>,
    ) -> Vec<(String, u64)> {
        let mut query_sizes: HashMap<String, u64> = HashMap::new();

        for (_, records) in records_vec.iter() {
            for record in records.iter() {
                query_sizes
                    .entry(record.qname.clone())
                    .or_insert(record.qlen);
            }
        }

        let mut query_sizes_vec = Vec::from_iter(query_sizes);
        query_sizes_vec.sort_by(|&(_, qlen_1), &(_, qlen_2)| qlen_1.cmp(&qlen_2));
        query_sizes_vec
    }

    fn draw_target_ticks(&mut self, coords: &HashMap<String, TargetCoord>) {
        for (_, TargetCoord { start, end }) in coords.iter() {
            if *start > self.start_x {
                draw_line_segment_mut(
                    &mut self.plot,
                    (*start as f32, self.end_y as f32),
                    (*start as f32, self.end_y as f32 + 10.0),
                    Rgb([0, 0, 0]),
                );
            }
        }
    }

    fn draw_query_ticks(&mut self, coords: &HashMap<String, QueryCoord>) {
        for (_, QueryCoord { start, end }) in coords.iter() {
            if *start < self.end_y as f32 {
                draw_line_segment_mut(
                    &mut self.plot,
                    (self.start_x as f32, *start),
                    (self.start_x as f32 - 10.0, *start),
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
