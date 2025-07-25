use crate::{config::Config, parser::PafRecord};
use ab_glyph::PxScale;
use image::{Pixel, Rgba, RgbaImage};
use imageproc::drawing::{
    draw_antialiased_line_segment_mut, draw_filled_rect_mut, draw_hollow_rect_mut, draw_line_segment_mut, draw_text_mut, text_size
};
use imageproc::geometric_transformations::{rotate, Interpolation};
use imageproc::rect::Rect;
use num_traits::NumCast;
use std::collections::HashMap;

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
    plot: RgbaImage,
}

impl<'a> Dotplot<'a> {
    pub fn new(config: &'a Config) -> Self {
        let plot = RgbaImage::new(config.width, config.height);

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

    // Initializes the background to white
    fn init_background(&mut self) {
        draw_filled_rect_mut(
            &mut self.plot,
            Rect::at(0, 0).of_size(self.config.width, self.config.height),
            Rgba([255, 255, 255, 255]),
        );
    }

    // Draws blank axes
    fn init_axes_lines(&mut self) {
        draw_hollow_rect_mut(
            &mut self.plot,
            Rect::at(self.start_x as i32, self.start_y as i32).of_size(
                (self.end_x - self.start_x) as u32,
                (self.end_y - self.start_y) as u32,
            ),
            Rgba([0, 0, 0, 255]),
        );
    }

    pub fn draw(&mut self, mut records: Vec<(String, Vec<PafRecord>)>) {
        let target_coords = self.targets_to_coords(&records);
        self.draw_target_ticks(&target_coords);

        let query_coords = self.queries_to_coords(&mut records);
        self.draw_query_ticks(&query_coords);

        self.draw_alignments(&records, &target_coords, &query_coords);

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
            tcoords.start,
            tcoords.end,
        );
        let tend_px = map_range(
            record.tend as f32,
            1.0,
            record.tlen as f32,
            tcoords.start,
            tcoords.end,
        );

        let qcoords = query_coords.get(&record.qname).unwrap();
        let (qstart_px, qend_px) = if record.strand == '-' {
            // For reverse strand, flip the query coordinates
            let qend_px = map_range(
                record.qstart as f32,
                1.0,
                record.qlen as f32,
                qcoords.start,
                qcoords.end,
            );
            let qstart_px = map_range(
                record.qend as f32,
                1.0,
                record.qlen as f32,
                qcoords.start,
                qcoords.end,
            );
            (qstart_px, qend_px)
        } else {
            // For forward strand, use coordinates as-is
            let qstart_px = map_range(
                record.qstart as f32,
                1.0,
                record.qlen as f32,
                qcoords.start,
                qcoords.end,
            );
            let qend_px = map_range(
                record.qend as f32,
                1.0,
                record.qlen as f32,
                qcoords.start,
                qcoords.end,
            );
            (qstart_px, qend_px)
        };

        let thickness = if record.is_best_matching_chr { 
            self.config.line_thickness * 4
        } else {
            self.config.line_thickness
        };
        
        // Calculate the line direction vector
        let dx = tend_px - tstart_px;
        let dy = qend_px - qstart_px;
        let line_length = (dx * dx + dy * dy).sqrt();
        
        // Normalize the direction vector
        let nx = dx / line_length;
        let ny = dy / line_length;
        
        // Calculate the perpendicular vector (rotate 90 degrees)
        let px = -ny;
        let py = nx;
        
        // Draw parallel lines to create a thicker line
        for offset in 1..=thickness {
            // Calculate offset distance from the center line
            let offset_dist = (offset as f32 - thickness as f32 / 2.0) * 0.5;
            
            // Calculate the offset points
            let offset_x = px * offset_dist;
            let offset_y = py * offset_dist;
            
            draw_antialiased_line_segment_mut(
                &mut self.plot,
                ((tstart_px + offset_x) as i32, (qstart_px + offset_y) as i32),
                ((tend_px + offset_x) as i32, (qend_px + offset_y) as i32),
                Rgba([0, 0, 0, 255]),
                Self::interpolate,
            );
            
        }
    }

    fn interpolate<P: Pixel>(left: P, right: P, _left_weight: f32) -> P 
        where
        P::Subpixel: Into<f32> + imageproc::definitions::Clamp<f32>,
        {
        imageproc::pixelops::interpolate(left, right, 2.0)
    }

    // Gets the position of each target on the x-axis, sorted by size
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

    // Gets the porsition of each query on the y-axis, sorted by gravity
    fn queries_to_coords(
        &self,
        records_vec: &mut [(String, Vec<PafRecord>)],
    ) -> HashMap<String, QueryCoord> {
        let targets_sizes = self.get_target_sizes_in_bp(records_vec);
        let query_sizes = self.get_query_sizes_in_bp(records_vec);
        let total_size = query_sizes.iter().fold(0_u64, |acc, x| acc + x.1);

        let best_matching_chrs = Self::get_best_matching_chrs(records_vec);

        let axis_size = self.end_y - self.start_y;
        let px_per_bp = axis_size as f64 / total_size as f64;

        let mut coords = HashMap::new();
        let mut last_pos = self.end_y;
        for (tname, _) in targets_sizes {
            for (tname_records, records) in records_vec.iter_mut() {
                if *tname_records != tname {
                    continue;
                }

                for record in records.iter_mut() {
                    let best_matching_chr = best_matching_chrs.get(&record.qname).unwrap();
                    if *best_matching_chr != tname {
                        continue;
                    }
                    record.is_best_matching_chr = true;

                    let segment_size = (record.qlen as f64 * px_per_bp) as f32;
                    let end_coord = last_pos - segment_size;
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
    ) -> HashMap<String, u64> {
        let mut query_sizes: HashMap<String, u64> = HashMap::new();

        for (_, records) in records_vec.iter() {
            for record in records.iter() {
                query_sizes
                    .entry(record.qname.clone())
                    .or_insert(record.qlen);
            }
        }

        query_sizes
    }

    // Get the best matching chromosome for each query
    fn get_best_matching_chrs(records_vec: &[(String, Vec<PafRecord>)]) -> HashMap<String, String> {
        log::debug!("Finding best matching chromosome");

        let gravities = Self::compute_gravity(records_vec);
        let mut best_gravity = HashMap::new();
        let mut best_matching_chr = HashMap::new();

        for ((target, query), gravity) in gravities.iter() {
            let g = best_gravity.entry(query).or_insert(gravity);

            if gravity >= *g {
                best_gravity
                    .entry(query)
                    .and_modify(|grav| *grav = gravity);

                best_matching_chr
                    .entry(query.clone())
                    .and_modify(|t| *t = target.clone())
                    .or_insert(target.clone());
            }
        }

        best_matching_chr
    }

    fn compute_gravity(records_vec: &[(String, Vec<PafRecord>)]) -> HashMap<(String, String), u64> {
        let mut gravity: HashMap<(String, String), u64> = HashMap::new(); 

        for (target, records) in records_vec.iter() {
            for record in records.iter() {
                let gravity_compound = record.nb_matches.pow(2);

                gravity
                    .entry((target.clone(), record.qname.clone()))
                    .and_modify(|g| *g += gravity_compound)
                    .or_insert(gravity_compound);
            }
        }

        gravity
    }

    fn draw_target_ticks(&mut self, coords: &HashMap<String, TargetCoord>) {
        for (_, TargetCoord { start, end: _ }) in coords.iter() {
            if *start > self.start_x {
                draw_line_segment_mut(
                    &mut self.plot,
                    (*start, self.end_y),
                    (*start, self.end_y + 10.0),
                    Rgba([0, 0, 0, 255]),
                );

                let grid_line_size = self.config.height as f32 * 0.0025;
                let mut y = self.start_y;
                while y + grid_line_size < self.end_y {
                    draw_line_segment_mut(
                        &mut self.plot,
                        (*start, y),
                        (*start, y + grid_line_size),
                        Rgba([0, 0, 0, 255]),
                    );
                    y += 2.0 * grid_line_size;
                } 
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
                    Rgba([0, 0, 0, 255]),
                );

                let grid_line_size = self.config.width as f32 * 0.0025;
                let mut x = self.start_x;
                while x + grid_line_size < self.end_x {
                    draw_line_segment_mut(
                        &mut self.plot,
                        (x, *start),
                        (x + grid_line_size, *start),
                        Rgba([0, 0, 0, 255]),
                    );
                    x += 2.0 * grid_line_size;
                } 
            }
        }
    }

    fn draw_target_names(&mut self, target_coords: &HashMap<String, TargetCoord>) {
        let font = Vec::from(include_bytes!("../FiraCode-Regular.ttf") as &[u8]);
        let font = ab_glyph::FontVec::try_from_vec(font).unwrap();
        let height = 12.4;
        let scale = PxScale {
            x: height,
            y: height,
        };

        let mut offset = 5.0;
        let mut target_coords_sorted = Vec::from_iter(target_coords);
        target_coords_sorted
            .sort_by(|a, b| (a.1.start).partial_cmp(&(b.1.start)).unwrap());
        
        for (target, TargetCoord { start, end }) in target_coords_sorted.iter() {
            let middle_x = (end + start) / 2.0;
            let text = Self::get_text(target, *start, *end, height);
            let text_size = text_size(scale, &font, &text);

            offset = -offset;

            let word_start = (middle_x - (text_size.0 as f32 / 2.0)) as i32;
            draw_text_mut(
                &mut self.plot,
                Rgba([0, 0, 0, 255]),
                word_start,
                (self.end_y + 10.0 + offset) as i32,
                scale,
                &font,
                &text,
            );
        }
    }

    fn draw_query_names(&mut self, query_coords: &HashMap<String, QueryCoord>) {
        self.rotate_image(-std::f32::consts::FRAC_PI_2);

        let font = Vec::from(include_bytes!("../FiraCode-Regular.ttf") as &[u8]);
        let font = ab_glyph::FontVec::try_from_vec(font).unwrap();
        let height = 12.4;
        let scale = PxScale {
            x: height,
            y: height,
        };

        let mut offset = 5.0;

        let mut query_coords_sorted = Vec::from_iter(query_coords);
        query_coords_sorted
            .sort_by(|a, b| (a.1.start).partial_cmp(&(b.1.start)).unwrap());
        for (query, QueryCoord { start, end }) in query_coords_sorted.iter() {
            let middle_x = (end + start) / 2.0;
            let text = Self::get_text(query, *end, *start, height); 
            let text_size = text_size(scale, &font, &text);
            if !text.is_empty() {
                offset = -offset;
            }

            let word_start = (middle_x - (text_size.0 as f32 / 2.0)) as i32;
            draw_text_mut(
                &mut self.plot,
                Rgba([0, 0, 0, 255]),
                word_start,
                (self.end_y + 10.0 + offset) as i32,
                scale,
                &font,
                &text,
            );
        }

        self.rotate_image(std::f32::consts::FRAC_PI_2);
    }

    fn get_text(query: &str, start: f32, end: f32, height: f32) -> String {
        let text_size_x = (query.len() as f32) * height;
        let mut text = String::from(query);

        if text_size_x >= (end - start) {
            let nb_chars = usize::min(((end - start) / height) as usize, query.len());
            // let nb_chars = query.len() + 3;

            if nb_chars < 2 {
                text = String::new();
            } 
            // else {
            //     text = String::from(&query[0..(nb_chars - 3)]) + "...";
            // }
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
            Rgba([0, 0, 0, 255]),
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
