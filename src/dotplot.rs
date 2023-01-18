use std::collections::HashMap;

use crate::{config::Config, parser::PafRecord};
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_line_segment_mut, draw_text_mut};
use imageproc::geometric_transformations::{rotate, Interpolation};
use num_traits::NumCast;
use rusttype::{Font, Scale};

pub struct TargetCoord {
    pub start: f32,
    pub end: f32,
}

pub struct QueryCoord {
    pub start: f32,
    pub end: f32,
}

pub struct Dotplot<'a> {
    config: &'a Config,
    start_x: f32,
    end_x: f32,
    start_y: f32,
    end_y: f32,
    plot: RgbImage,
}

impl<'a> Dotplot<'a> {
    pub fn new(config: &'a Config) -> Self {
        let plot = RgbImage::new(config.width, config.height);

        let offset_x = config.width as f32 * config.margin_x;
        let offset_y = config.height as f32 * config.margin_y;
        let start_x = offset_x;
        let end_x = config.width as f32 - offset_x;
        let start_y = offset_y;
        let end_y = config.height as f32 - offset_y;

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
        draw_line_segment_mut(
            &mut self.plot,
            (self.start_x, self.start_y),
            (self.end_x, self.start_y),
            Rgb([0, 0, 0]),
        );

        draw_line_segment_mut(
            &mut self.plot,
            (self.start_x, self.end_y),
            (self.end_x, self.end_y),
            Rgb([0, 0, 0]),
        );

        draw_line_segment_mut(
            &mut self.plot,
            (self.start_x, self.start_y),
            (self.start_x, self.end_y),
            Rgb([0, 0, 0]),
        );

        draw_line_segment_mut(
            &mut self.plot,
            (self.end_x, self.start_y),
            (self.end_x, self.end_y),
            Rgb([0, 0, 0]),
        );
    }

    pub fn draw(&mut self, records: &[(String, Vec<PafRecord>)]) {
        let target_coords = self.targets_to_coords(records);
        self.draw_target_ticks(&target_coords);

        let query_coords = self.queries_to_coords(records);
        self.draw_query_ticks(&query_coords);

        self.draw_alignments(records, &target_coords, &query_coords);

        self.draw_target_names(&target_coords);
        self.draw_query_names(&query_coords);
    }

    fn draw_alignments(
        &mut self,
        records_vec: &[(String, Vec<PafRecord>)],
        target_coords: &HashMap<String, TargetCoord>,
        query_coords: &HashMap<String, QueryCoord>,
    ) {
        for (_, records) in records_vec.iter() {
            for record in records.iter() {
                self.draw_alignment(record, target_coords, query_coords);
            }
        }
    }

    fn draw_alignment(
        &mut self,
        record: &PafRecord,
        target_coords: &HashMap<String, TargetCoord>,
        query_coords: &HashMap<String, QueryCoord>,
    ) {
        let tcoords = target_coords.get(&record.tname).unwrap();
        let tstart_px = map_range(
            record.tstart as f32,
            1.0,
            record.tlen as f32,
            tcoords.start as f32,
            tcoords.end as f32,
        );
        let tend_px = map_range(
            record.tend as f32,
            1.0,
            record.tlen as f32,
            tcoords.start as f32,
            tcoords.end as f32,
        );

        let qcoords = query_coords.get(&record.qname).unwrap();
        let qstart_px = map_range(
            record.qstart as f32,
            1.0,
            record.qlen as f32,
            qcoords.start as f32,
            qcoords.end as f32,
        );
        let qend_px = map_range(
            record.qend as f32,
            1.0,
            record.qlen as f32,
            qcoords.start as f32,
            qcoords.end as f32,
        );

        draw_line_segment_mut(
            &mut self.plot,
            (tstart_px, qstart_px),
            (tend_px, qend_px),
            Rgb([100, 0, 0]),
        );
    }

    // Gets the position of each target on the x-axis
    fn targets_to_coords(
        &self,
        records: &[(String, Vec<PafRecord>)],
    ) -> HashMap<String, TargetCoord> {
        let targets_sizes = self.get_target_sizes_in_bp(records);
        let total_size = targets_sizes.iter().fold(0_u64, |acc, x| acc + x.1);

        let axis_size = self.end_x - self.start_x;
        let px_per_bp = axis_size as f64 / total_size as f64;

        let mut coords: HashMap<String, TargetCoord> = HashMap::new();
        let mut last_pos = self.start_x;
        for (target, size) in targets_sizes.iter() {
            let segment_size = (*size as f64 * px_per_bp) as f32;
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
        records_vec: &[(String, Vec<PafRecord>)],
    ) -> HashMap<String, QueryCoord> {
        let targets_sizes = self.get_target_sizes_in_bp(records_vec);
        let query_sizes = self.get_query_sizes_in_bp(records_vec);
        let total_size = query_sizes.iter().fold(0_u64, |acc, x| acc + x.1);

        let axis_size = self.end_y - self.start_y;
        let px_per_bp = axis_size as f64 / total_size as f64;

        let mut coords = HashMap::new();
        let mut last_pos = self.start_y;
        for (tname, _) in targets_sizes {
            for (tname_records, records) in records_vec.iter() {
                if *tname_records != tname {
                    continue;
                }

                for record in records.iter() {
                    let segment_size = (record.qlen as f64 * px_per_bp) as f32;
                    let end_coord = last_pos + segment_size;
                    let query_coords = QueryCoord {
                        start: last_pos,
                        end: end_coord,
                    };
                    match coords.get(&record.qname) {
                        Some(_) => {}
                        None => {
                            coords.insert(record.qname.clone(), query_coords);
                            last_pos = end_coord;
                        }
                    }
                }
            }
        }

        coords
    }

    // Gets all targets sizes
    fn get_target_sizes_in_bp(
        &self,
        records_vec: &[(String, Vec<PafRecord>)],
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
        records_vec: &[(String, Vec<PafRecord>)],
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
        for (_, TargetCoord { start, end: _ }) in coords.iter() {
            if *start > self.start_x {
                draw_line_segment_mut(
                    &mut self.plot,
                    (*start, self.end_y),
                    (*start, self.end_y + 10.0),
                    Rgb([0, 0, 0]),
                );
            }
        }
    }

    fn draw_query_ticks(&mut self, coords: &HashMap<String, QueryCoord>) {
        for (_, QueryCoord { start, end: _ }) in coords.iter() {
            if *start < self.end_y {
                draw_line_segment_mut(
                    &mut self.plot,
                    (self.start_x, *start),
                    (self.start_x - 10.0, *start),
                    Rgb([0, 0, 0]),
                );
            }
        }
    }

    fn draw_target_names(&mut self, target_coords: &HashMap<String, TargetCoord>) {
        let font = Vec::from(include_bytes!("../FiraCode-Regular.ttf") as &[u8]);
        let font = Font::try_from_vec(font).unwrap();

        let height = 12.4;
        let scale = Scale {
            x: height * 2.0,
            y: height,
        };

        for (target, TargetCoord { start, end }) in target_coords.iter() {
            let middle_x = (end - start) / 2.0;

            let text = Self::get_text(target, *start, *end, height);
            // let mut text = String::from(&target[..]);
            // if text_size_x >= (end - start) {
            //     let nb_chars = usize::min(((end - start) / height) as usize, target.len());
            //     if nb_chars < 4 {
            //         text = String::new();
            //     } else {
            //         text = String::from(&target[0..(nb_chars - 3)]) + "...";
            //     }
            let text_size_x = (text.len() as f32) * height;
            // }

            draw_text_mut(
                &mut self.plot,
                Rgb([0, 0, 0]),
                (*start + (middle_x - (text_size_x / 2.0))) as i32,
                (self.end_y + 10.0) as i32,
                scale,
                &font,
                &text,
            );
        }
    }

    fn draw_query_names(&mut self, query_coords: &HashMap<String, QueryCoord>) {
        self.rotate_image(-std::f32::consts::FRAC_PI_2);

        let font = Vec::from(include_bytes!("../FiraCode-Regular.ttf") as &[u8]);
        let font = Font::try_from_vec(font).unwrap();
        let height = 12.4;
        let scale = Scale {
            x: height * 2.0,
            y: height,
        };

        for (query, QueryCoord { start, end }) in query_coords.iter() {
            let middle_x = (end - start) / 2.0;
            let text = Self::get_text(query, *start, *end, height);
            let text_size_x = (text.len() as f32) * height;

            draw_text_mut(
                &mut self.plot,
                Rgb([0, 0, 0]),
                (*start + (middle_x - (text_size_x / 2.0))) as i32,
                (self.end_y + 10.0) as i32,
                scale,
                &font,
                &text,
            );
        }

        self.rotate_image(std::f32::consts::FRAC_PI_2);
    }

    fn get_text(query: &String, start: f32, end: f32, height: f32) -> String {
        let text_size_x = (query.len() as f32) * height;
        let mut text = String::from(&query[..]);

        if text_size_x >= (end - start) {
            let nb_chars = usize::min(((end - start) / height) as usize, query.len());

            if nb_chars < 4 {
                text = String::new();
            } else {
                text = String::from(&query[0..(nb_chars - 3)]) + "...";
            }
        }

        text
    }

    fn rotate_image(&mut self, angle: f32) {
        let center_x = self.config.width as f32 / 2.0;
        let center_y = self.config.height as f32 / 2.0;

        self.plot = rotate(
            &self.plot,
            (center_x, center_y),
            angle,
            Interpolation::Bicubic,
            Rgb([0, 0, 0]),
        );
    }

    // Saves the plot to a file
    pub fn save(&self) {
        self.plot.save(&self.config.output).unwrap();
    }
}

// Taken from nannou::math
pub fn map_range<X, Y>(val: X, in_min: X, in_max: X, out_min: Y, out_max: Y) -> Y
where
    X: NumCast,
    Y: NumCast,
{
    macro_rules! unwrap_or_panic {
        ($result:expr, $arg:expr) => {
            $result.unwrap_or_else(|| panic!("[map_range] failed to cast {} arg to `f64`", $arg))
        };
    }

    let val_f: f64 = unwrap_or_panic!(NumCast::from(val), "first");
    let in_min_f: f64 = unwrap_or_panic!(NumCast::from(in_min), "second");
    let in_max_f: f64 = unwrap_or_panic!(NumCast::from(in_max), "third");
    let out_min_f: f64 = unwrap_or_panic!(NumCast::from(out_min), "fourth");
    let out_max_f: f64 = unwrap_or_panic!(NumCast::from(out_max), "fifth");

    NumCast::from((val_f - in_min_f) / (in_max_f - in_min_f) * (out_max_f - out_min_f) + out_min_f)
        .unwrap_or_else(|| panic!("[map_range] failed to cast result to target type"))
}
