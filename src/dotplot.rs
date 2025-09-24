use crate::{config::Config, parser::PafRecord};
use ab_glyph::{FontVec, PxScale};
use image::{imageops::overlay, Pixel, Rgba, RgbaImage};
use imageproc::drawing::{
    draw_antialiased_line_segment_mut, draw_filled_circle_mut, draw_filled_rect_mut,
    draw_hollow_rect_mut, draw_line_segment_mut, draw_text_mut, text_size,
};
use imageproc::geometric_transformations::{rotate, Interpolation};
use imageproc::rect::Rect;
use num_traits::NumCast;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const SIGNIFICANCE_THRESHOLD: f64 = 0.01;

#[derive(Clone, Copy)]
struct SignificanceBin {
    upper_bound: f64,
    color: [u8; 4],
    label: &'static str,
}

const SIGNIFICANCE_BINS: [SignificanceBin; 5] = [
    SignificanceBin {
        upper_bound: 1e-25,
        color: [45, 0, 75, 255],
        label: "≤ 1e-25",
    },
    SignificanceBin {
        upper_bound: 1e-20,
        color: [84, 26, 139, 255],
        label: "≤ 1e-20",
    },
    SignificanceBin {
        upper_bound: 1e-15,
        color: [115, 52, 180, 255],
        label: "≤ 1e-15",
    },
    SignificanceBin {
        upper_bound: 1e-10,
        color: [170, 85, 203, 255],
        label: "≤ 1e-10",
    },
    SignificanceBin {
        upper_bound: SIGNIFICANCE_THRESHOLD,
        color: [224, 140, 222, 255],
        label: "< 0.01",
    },
];

fn non_significant_color() -> Rgba<u8> {
    Rgba([180, 180, 180, 255])
}

#[derive(Debug, Clone)]
struct PairSummary {
    anchors: u64,
    raw_p_value: f64,
    corrected_p_value: f64,
}

impl PairSummary {
    fn is_significant(&self) -> bool {
        self.corrected_p_value < SIGNIFICANCE_THRESHOLD
    }

    fn color(&self) -> Rgba<u8> {
        if !self.is_significant() {
            return non_significant_color();
        }

        let bin = SIGNIFICANCE_BINS
            .iter()
            .find(|bin| self.corrected_p_value <= bin.upper_bound)
            .unwrap_or_else(|| SIGNIFICANCE_BINS.last().unwrap());
        Rgba(bin.color)
    }
}

#[derive(Debug, Clone)]
struct SyntenyAnalysis {
    pair_summaries: HashMap<String, HashMap<String, PairSummary>>,
    target_totals: HashMap<String, u64>,
    query_totals: HashMap<String, u64>,
    total_anchors: u64,
    num_tests: usize,
}

impl SyntenyAnalysis {
    fn new(records: &[(String, Vec<PafRecord>)]) -> Self {
        let mut pair_counts: HashMap<(String, String), u64> = HashMap::new();
        let mut target_totals: HashMap<String, u64> = HashMap::new();
        let mut query_totals: HashMap<String, u64> = HashMap::new();

        let mut total_anchors = 0_u64;

        for (target, records) in records {
            for record in records {
                *pair_counts
                    .entry((target.clone(), record.qname.clone()))
                    .or_insert(0) += 1;
                *target_totals.entry(target.clone()).or_insert(0) += 1;
                *query_totals.entry(record.qname.clone()).or_insert(0) += 1;
                total_anchors += 1;
            }
        }

        let target_names: Vec<String> = target_totals.keys().cloned().collect();
        let query_names: Vec<String> = query_totals.keys().cloned().collect();

        let num_tests = target_names.len().saturating_mul(query_names.len());

        if total_anchors == 0 || num_tests == 0 {
            return Self {
                pair_summaries: HashMap::new(),
                target_totals,
                query_totals,
                total_anchors,
                num_tests,
            };
        }

        let total_usize: usize = total_anchors
            .try_into()
            .expect("Total number of anchors exceeds usize range");
        let ln_factorials = cumulative_ln_factorials(total_usize);

        let mut pair_summaries: HashMap<String, HashMap<String, PairSummary>> = HashMap::new();
        for target in &target_names {
            for query in &query_names {
                let anchors = *pair_counts
                    .get(&(target.clone(), query.clone()))
                    .unwrap_or(&0);
                let row_total = *target_totals.get(target).unwrap_or(&0);
                let col_total = *query_totals.get(query).unwrap_or(&0);

                let raw_p_value = fisher_exact_greater(
                    anchors,
                    row_total,
                    col_total,
                    total_anchors,
                    &ln_factorials,
                );

                let corrected = (raw_p_value * num_tests as f64).min(1.0);
                pair_summaries.entry(target.clone()).or_default().insert(
                    query.clone(),
                    PairSummary {
                        anchors,
                        raw_p_value,
                        corrected_p_value: corrected,
                    },
                );
            }
        }

        Self {
            pair_summaries,
            target_totals,
            query_totals,
            total_anchors,
            num_tests,
        }
    }

    fn summary_for(&self, target: &str, query: &str) -> Option<&PairSummary> {
        self.pair_summaries
            .get(target)
            .and_then(|inner| inner.get(query))
    }

    fn color_for(&self, target: &str, query: &str) -> Rgba<u8> {
        self.summary_for(target, query)
            .map(PairSummary::color)
            .unwrap_or_else(non_significant_color)
    }
}

fn cumulative_ln_factorials(n: usize) -> Vec<f64> {
    let mut ln_factorials = Vec::with_capacity(n + 1);
    ln_factorials.push(0.0);
    for k in 1..=n {
        let previous = ln_factorials[k - 1];
        ln_factorials.push(previous + (k as f64).ln());
    }
    ln_factorials
}

fn ln_combination(n: u64, k: u64, ln_factorials: &[f64]) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    let n_usize = n as usize;
    let k_usize = k as usize;
    ln_factorials[n_usize] - ln_factorials[k_usize] - ln_factorials[n_usize - k_usize]
}

fn fisher_exact_greater(
    a: u64,
    row_total: u64,
    col_total: u64,
    grand_total: u64,
    ln_factorials: &[f64],
) -> f64 {
    if row_total == 0 || col_total == 0 {
        return 1.0;
    }

    let max_k = row_total.min(col_total);
    let min_k = row_total
        .saturating_add(col_total)
        .saturating_sub(grand_total);
    if a > max_k {
        return 0.0;
    }
    let start_k = a.max(min_k);
    let mut cumulative = 0.0;
    for k in start_k..=max_k {
        let log_p = ln_combination(col_total, k, ln_factorials)
            + ln_combination(grand_total - col_total, row_total - k, ln_factorials)
            - ln_combination(grand_total, row_total, ln_factorials);
        cumulative += log_p.exp();
    }
    cumulative.clamp(0.0, 1.0)
}

pub struct TargetCoord {
    pub start: f32,
    pub end: f32,
}

impl TargetCoord {
    fn map_position(&self, position: u64, total_length: u64) -> f32 {
        map_range(
            position as f32,
            0.0,
            total_length as f32,
            self.start,
            self.end,
        )
    }
}

pub struct QueryCoord {
    pub start: f32,
    pub end: f32,
}

impl QueryCoord {
    fn map_position(&self, position: u64, total_length: u64) -> f32 {
        map_range(
            position as f32,
            0.0,
            total_length as f32,
            self.start,
            self.end,
        )
    }
}

const LABEL_TEXT_HEIGHT: f32 = 12.4;

pub struct Dotplot<'a> {
    config: &'a Config,
    start_x: f32,
    end_x: f32,
    start_y: f32,
    end_y: f32,
    plot: RgbaImage,
    bubble_plot: Option<RgbaImage>,
    foreground_color: Rgba<u8>,
    background_color: Rgba<u8>,
    font: FontVec,
    text_scale: PxScale,
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

        let foreground_color = Rgba(config.theme.foreground_color());
        let background_color = Rgba(config.theme.background_color());
        let font = Self::load_font();
        let text_scale = PxScale {
            x: LABEL_TEXT_HEIGHT,
            y: LABEL_TEXT_HEIGHT,
        };

        let mut dotplot = Self {
            config,
            start_x,
            end_x,
            start_y,
            end_y,
            plot,
            bubble_plot: None,
            foreground_color,
            background_color,
            font,
            text_scale,
        };

        dotplot.init_plot();
        dotplot
    }

    fn load_font() -> FontVec {
        let bytes = include_bytes!("../FiraCode-Regular.ttf");
        FontVec::try_from_vec(bytes.to_vec()).expect("failed to load built-in font")
    }

    // Initializes the dotplot with a blank background and empty axes
    fn init_plot(&mut self) {
        self.init_background();
        self.init_axes_lines();
    }

    // Initializes the background with the selected theme color
    fn init_background(&mut self) {
        draw_filled_rect_mut(
            &mut self.plot,
            Rect::at(0, 0).of_size(self.config.width, self.config.height),
            self.background_color,
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
            self.foreground_color,
        );
    }

    pub fn draw(&mut self, mut records: Vec<(String, Vec<PafRecord>)>) {
        let analysis = SyntenyAnalysis::new(&records);
        let target_coords = self.targets_to_coords(&records);
        if target_coords.is_empty() {
            log::warn!("No alignments available after filtering; nothing to draw");
            return;
        }
        self.draw_target_ticks(&target_coords);

        let query_coords = self.queries_to_coords(&mut records);
        if query_coords.is_empty() {
            log::warn!("No query coordinates available; nothing to draw");
            return;
        }
        self.draw_query_ticks(&query_coords);

        self.draw_alignments(&records, &target_coords, &query_coords, &analysis);

        self.draw_target_names(&target_coords);
        self.draw_query_names(&query_coords);

        self.bubble_plot = self.build_bubble_grid(&analysis, &target_coords, &query_coords);
    }

    fn draw_alignments(
        &mut self,
        records_vec: &[(String, Vec<PafRecord>)],
        target_coords: &HashMap<String, TargetCoord>,
        query_coords: &HashMap<String, QueryCoord>,
        analysis: &SyntenyAnalysis,
    ) {
        for (_, records) in records_vec.iter() {
            for record in records.iter() {
                self.draw_alignment(record, target_coords, query_coords, analysis);
            }
        }
    }

    fn draw_alignment(
        &mut self,
        record: &PafRecord,
        target_coords: &HashMap<String, TargetCoord>,
        query_coords: &HashMap<String, QueryCoord>,
        analysis: &SyntenyAnalysis,
    ) {
        let Some(tcoords) = target_coords.get(&record.tname) else {
            log::warn!(
                "Missing target coordinates for '{}'; skipping alignment",
                record.tname
            );
            return;
        };
        let tstart_px = tcoords.map_position(record.tstart, record.tlen);
        let tend_px = tcoords.map_position(record.tend, record.tlen);

        let Some(qcoords) = query_coords.get(&record.qname) else {
            log::warn!(
                "Missing query coordinates for '{}'; skipping alignment",
                record.qname
            );
            return;
        };
        let (qstart_px, qend_px) = Self::query_pixel_range(record, qcoords);

        let color = analysis.color_for(&record.tname, &record.qname);

        // Calculate the line direction vector
        let dx = tend_px - tstart_px;
        let dy = qend_px - qstart_px;
        let line_length_sq = dx * dx + dy * dy;

        let thickness = self.config.line_thickness;
        if line_length_sq <= f32::EPSILON {
            self.draw_alignment_point(tstart_px, qstart_px, thickness.max(1), color);
            return;
        }

        let line_length = line_length_sq.sqrt();

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
                color,
                Self::interpolate,
            );
        }
    }

    fn query_pixel_range(record: &PafRecord, coords: &QueryCoord) -> (f32, f32) {
        match record.strand {
            '-' => (
                coords.map_position(record.qend, record.qlen),
                coords.map_position(record.qstart, record.qlen),
            ),
            _ => (
                coords.map_position(record.qstart, record.qlen),
                coords.map_position(record.qend, record.qlen),
            ),
        }
    }

    fn interpolate<P: Pixel>(left: P, right: P, left_weight: f32) -> P
    where
        P::Subpixel: Into<f32> + imageproc::definitions::Clamp<f32>,
    {
        imageproc::pixelops::interpolate(left, right, left_weight)
    }

    fn draw_alignment_point(&mut self, x: f32, y: f32, thickness: u32, color: Rgba<u8>) {
        let radius = ((thickness as f32 / 2.0).ceil() as i32).max(0);
        let center_x = x.round() as i32;
        let center_y = y.round() as i32;

        let width = self.plot.width() as i32;
        let height = self.plot.height() as i32;

        for dx in -radius..=radius {
            for dy in -radius..=radius {
                let px = center_x + dx;
                let py = center_y + dy;
                if px >= 0 && px < width && py >= 0 && py < height {
                    let (px_u32, py_u32) = (px as u32, py as u32);
                    self.plot.put_pixel(px_u32, py_u32, color);
                }
            }
        }
    }

    // Gets the position of each target on the x-axis, sorted by size
    fn targets_to_coords(
        &self,
        records: &[(String, Vec<PafRecord>)],
    ) -> HashMap<String, TargetCoord> {
        let targets_sizes = self.get_target_sizes_in_bp(records);
        let total_size = targets_sizes.iter().fold(0_u64, |acc, x| acc + x.1);

        let axis_size = self.end_x - self.start_x;
        if total_size == 0 {
            return HashMap::new();
        }

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
        if total_size == 0 {
            return HashMap::new();
        }

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
            if let Some(record) = records.first() {
                targets_sizes.entry(tname.clone()).or_insert(record.tlen);
            }
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
                best_gravity.entry(query).and_modify(|grav| *grav = gravity);

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
        for (_, TargetCoord { start, end: _ }) in
            Self::sorted_coordinates(coords, |coord| coord.start)
        {
            if *start > self.start_x {
                draw_line_segment_mut(
                    &mut self.plot,
                    (*start, self.end_y),
                    (*start, self.end_y + 10.0),
                    self.foreground_color,
                );

                let grid_line_size = self.config.height as f32 * 0.0025;
                let mut y = self.start_y;
                while y + grid_line_size < self.end_y {
                    draw_line_segment_mut(
                        &mut self.plot,
                        (*start, y),
                        (*start, y + grid_line_size),
                        self.foreground_color,
                    );
                    y += 2.0 * grid_line_size;
                }
            }
        }
    }

    fn draw_query_ticks(&mut self, coords: &HashMap<String, QueryCoord>) {
        for (_, QueryCoord { start, end: _ }) in
            Self::sorted_coordinates(coords, |coord| coord.start)
        {
            if *start < self.end_y {
                draw_line_segment_mut(
                    &mut self.plot,
                    (self.start_x, *start),
                    (self.start_x - 10.0, *start),
                    self.foreground_color,
                );

                let grid_line_size = self.config.width as f32 * 0.0025;
                let mut x = self.start_x;
                while x + grid_line_size < self.end_x {
                    draw_line_segment_mut(
                        &mut self.plot,
                        (x, *start),
                        (x + grid_line_size, *start),
                        self.foreground_color,
                    );
                    x += 2.0 * grid_line_size;
                }
            }
        }
    }

    fn draw_target_names(&mut self, target_coords: &HashMap<String, TargetCoord>) {
        let mut offset = 5.0;
        let scale = self.text_scale;
        let height = scale.y;

        for (target, TargetCoord { start, end }) in
            Self::sorted_coordinates(target_coords, |coord| coord.start)
        {
            let middle_x = (end + start) / 2.0;
            let text = Self::get_text(target, *start, *end, height);
            let text_size = text_size(scale, &self.font, &text);

            offset = -offset;

            let word_start = (middle_x - (text_size.0 as f32 / 2.0)) as i32;
            draw_text_mut(
                &mut self.plot,
                self.foreground_color,
                word_start,
                (self.end_y + 10.0 + offset) as i32,
                scale,
                &self.font,
                &text,
            );
        }
    }

    fn draw_query_names(&mut self, query_coords: &HashMap<String, QueryCoord>) {
        let width = self.plot.width();
        let height = self.plot.height();
        let transparent = Rgba([0, 0, 0, 0]);
        let center = (width as f32 / 2.0, height as f32 / 2.0);

        let base_overlay = RgbaImage::from_pixel(width, height, transparent);
        let mut rotated_overlay = rotate(
            &base_overlay,
            center,
            -std::f32::consts::FRAC_PI_2,
            Interpolation::Bicubic,
            transparent,
        );

        let scale = self.text_scale;
        let height = scale.y;

        let mut offset = 5.0;

        for (query, QueryCoord { start, end }) in
            Self::sorted_coordinates(query_coords, |coord| coord.start)
        {
            let middle_y = (end + start) / 2.0;
            let text = Self::get_text(query, *end, *start, height);
            if text.is_empty() {
                continue;
            }

            offset = -offset;

            let text_size = text_size(scale, &self.font, &text);
            let target_center_x = self.start_x - 10.0;
            let target_center_y = middle_y + offset;

            // Translate the desired label centre (after rotation) back into the
            // unrotated coordinate space so the label lands next to the axis
            // once we rotate the overlay by -90°.
            let source_center_x = center.0 + center.1 - target_center_y;
            let source_center_y = target_center_x - center.0 + center.1;

            let draw_x = (source_center_x - text_size.0 as f32 / 2.0).round() as i32;
            let draw_y = (source_center_y - text_size.1 as f32 / 2.0).round() as i32;

            draw_text_mut(
                &mut rotated_overlay,
                self.foreground_color,
                draw_x,
                draw_y,
                scale,
                &self.font,
                &text,
            );
        }

        let query_label_overlay = rotate(
            &rotated_overlay,
            center,
            -std::f32::consts::FRAC_PI_2,
            Interpolation::Bicubic,
            transparent,
        );

        overlay(&mut self.plot, &query_label_overlay, 0, 0);
    }

    fn build_bubble_grid(
        &self,
        analysis: &SyntenyAnalysis,
        target_coords: &HashMap<String, TargetCoord>,
        query_coords: &HashMap<String, QueryCoord>,
    ) -> Option<RgbaImage> {
        if analysis.total_anchors == 0
            || analysis.num_tests == 0
            || target_coords.is_empty()
            || query_coords.is_empty()
        {
            return None;
        }

        let target_order: Vec<String> =
            Self::sorted_coordinates(target_coords, |coord| coord.start)
                .into_iter()
                .map(|(name, _)| name.clone())
                .collect();
        let query_order: Vec<String> = Self::sorted_coordinates(query_coords, |coord| coord.start)
            .into_iter()
            .map(|(name, _)| name.clone())
            .collect();

        if target_order.is_empty() || query_order.is_empty() {
            return None;
        }

        let max_cells = usize::max(target_order.len(), query_order.len()).max(1) as f32;
        let mut cell_size = (self.config.width as f32 / (max_cells + 5.0)).clamp(40.0, 140.0);
        if cell_size.is_nan() || cell_size <= 0.0 {
            cell_size = 60.0;
        }
        let left_margin = cell_size * 2.4;
        let top_margin = cell_size * 2.0;
        let legend_height = (SIGNIFICANCE_BINS.len() as f32 + 1.5) * (cell_size * 0.6);

        let grid_width = (target_order.len() as f32) * cell_size;
        let grid_height = (query_order.len() as f32) * cell_size;

        let image_width = (left_margin + grid_width + cell_size * 2.0).ceil() as u32;
        let image_height = (top_margin + grid_height + legend_height + cell_size).ceil() as u32;

        let mut bubble_plot =
            RgbaImage::from_pixel(image_width, image_height, self.background_color);

        let grid_left = left_margin;
        let grid_top = top_margin;
        let grid_right = grid_left + grid_width;
        let grid_bottom = grid_top + grid_height;

        draw_hollow_rect_mut(
            &mut bubble_plot,
            Rect::at(grid_left.round() as i32, grid_top.round() as i32)
                .of_size(grid_width.round() as u32, grid_height.round() as u32),
            self.foreground_color,
        );

        for (col_idx, _) in target_order.iter().enumerate() {
            let x = grid_left + (col_idx as f32) * cell_size;
            draw_line_segment_mut(
                &mut bubble_plot,
                (x, grid_top),
                (x, grid_bottom),
                self.foreground_color,
            );
        }

        for (row_idx, _) in query_order.iter().enumerate() {
            let y = grid_top + (row_idx as f32) * cell_size;
            draw_line_segment_mut(
                &mut bubble_plot,
                (grid_left, y),
                (grid_right, y),
                self.foreground_color,
            );
        }

        let max_anchor = analysis
            .pair_summaries
            .values()
            .flat_map(|inner| inner.values())
            .filter(|summary| summary.is_significant())
            .map(|summary| summary.anchors)
            .max()
            .unwrap_or(0)
            .max(1);
        let max_radius = (cell_size * 0.45).max(5.0);
        let bubble_text_scale = PxScale {
            x: (cell_size * 0.35).clamp(12.0, 28.0),
            y: (cell_size * 0.35).clamp(12.0, 28.0),
        };

        for (col_idx, target) in target_order.iter().enumerate() {
            for (row_idx, query) in query_order.iter().enumerate() {
                let Some(summary) = analysis.summary_for(target, query) else {
                    continue;
                };

                let center_x = grid_left + (col_idx as f32 + 0.5) * cell_size;
                let center_y = grid_top + (row_idx as f32 + 0.5) * cell_size;

                if summary.is_significant() && summary.anchors > 0 {
                    let radius_ratio = (summary.anchors as f32 / max_anchor as f32).sqrt();
                    let radius = (radius_ratio * max_radius).max(3.0);
                    let color = summary.color();
                    draw_filled_circle_mut(
                        &mut bubble_plot,
                        (center_x.round() as i32, center_y.round() as i32),
                        radius.round() as i32,
                        color,
                    );
                } else {
                    let text = "n.s.";
                    let (text_w, text_h) = text_size(bubble_text_scale, &self.font, text);
                    let draw_x = (center_x - text_w as f32 / 2.0).round() as i32;
                    let draw_y = (center_y - text_h as f32 / 2.0).round() as i32;
                    draw_text_mut(
                        &mut bubble_plot,
                        self.foreground_color,
                        draw_x,
                        draw_y,
                        bubble_text_scale,
                        &self.font,
                        text,
                    );
                }
            }
        }

        let label_scale = PxScale {
            x: (cell_size * 0.35).clamp(12.0, 26.0),
            y: (cell_size * 0.35).clamp(12.0, 26.0),
        };

        for (col_idx, target) in target_order.iter().enumerate() {
            let center_x = grid_left + (col_idx as f32 + 0.5) * cell_size;
            let text = target.as_str();
            let (text_w, text_h) = text_size(label_scale, &self.font, text);
            let draw_x = (center_x - text_w as f32 / 2.0).round() as i32;
            let draw_y = (grid_bottom + cell_size * 0.2) as i32;
            draw_text_mut(
                &mut bubble_plot,
                self.foreground_color,
                draw_x,
                draw_y,
                label_scale,
                &self.font,
                text,
            );
        }

        for (row_idx, query) in query_order.iter().enumerate() {
            let center_y = grid_top + (row_idx as f32 + 0.5) * cell_size;
            let text = query.as_str();
            let (text_w, text_h) = text_size(label_scale, &self.font, text);
            let draw_x = (grid_left - cell_size * 0.3 - text_w as f32).round() as i32;
            let draw_y = (center_y - text_h as f32 / 2.0).round() as i32;
            draw_text_mut(
                &mut bubble_plot,
                self.foreground_color,
                draw_x,
                draw_y,
                label_scale,
                &self.font,
                text,
            );
        }

        let legend_x = grid_left;
        let mut legend_y = grid_bottom + cell_size * 0.7;
        let swatch_size = (cell_size * 0.4).clamp(12.0, 24.0);
        let legend_scale = PxScale {
            x: (cell_size * 0.32).clamp(11.0, 24.0),
            y: (cell_size * 0.32).clamp(11.0, 24.0),
        };

        draw_text_mut(
            &mut bubble_plot,
            self.foreground_color,
            legend_x.round() as i32,
            (legend_y - cell_size * 0.4) as i32,
            legend_scale,
            &self.font,
            "Fisher P (corrected)",
        );

        for bin in SIGNIFICANCE_BINS.iter() {
            let rect = Rect::at(legend_x.round() as i32, legend_y.round() as i32)
                .of_size(swatch_size.round() as u32, swatch_size.round() as u32);
            draw_filled_rect_mut(&mut bubble_plot, rect, Rgba(bin.color));

            let text_x = legend_x + swatch_size + cell_size * 0.2;
            draw_text_mut(
                &mut bubble_plot,
                self.foreground_color,
                text_x.round() as i32,
                legend_y.round() as i32,
                legend_scale,
                &self.font,
                bin.label,
            );

            legend_y += swatch_size + cell_size * 0.2;
        }

        let rect = Rect::at(legend_x.round() as i32, legend_y.round() as i32)
            .of_size(swatch_size.round() as u32, swatch_size.round() as u32);
        draw_filled_rect_mut(&mut bubble_plot, rect, non_significant_color());
        let text_x = legend_x + swatch_size + cell_size * 0.2;
        draw_text_mut(
            &mut bubble_plot,
            self.foreground_color,
            text_x.round() as i32,
            legend_y.round() as i32,
            legend_scale,
            &self.font,
            "n.s.",
        );

        Some(bubble_plot)
    }

    fn sorted_coordinates<'a, T, F>(
        coords: &'a HashMap<String, T>,
        key_fn: F,
    ) -> Vec<(&'a String, &'a T)>
    where
        F: Fn(&T) -> f32,
    {
        let mut entries: Vec<_> = coords.iter().collect();
        entries.sort_by(|a, b| {
            let left = key_fn(a.1);
            let right = key_fn(b.1);
            match left.partial_cmp(&right) {
                Some(Ordering::Equal) => a.0.cmp(b.0),
                Some(order) => order,
                None => a.0.cmp(b.0),
            }
        });
        entries
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

    // Saves the plot to a file
    pub fn save(&self) {
        self.plot.save(&self.config.output).unwrap();

        if let Some(bubble_plot) = &self.bubble_plot {
            let bubble_path = bubble_output_path(&self.config.output);
            bubble_plot.save(&bubble_path).unwrap();
            log::info!("Saved bubble grid to {}", bubble_path.to_string_lossy());
        }
    }
}

fn bubble_output_path(output: &str) -> PathBuf {
    let path = Path::new(output);
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy())
        .unwrap_or_else(|| "dotplot".into());

    let mut filename = format!("{}_bubble", stem);
    if let Some(extension) = path.extension() {
        filename.push('.');
        filename.push_str(&extension.to_string_lossy());
    }

    match path.parent() {
        Some(parent) if parent != Path::new("") => parent.join(filename),
        _ => PathBuf::from(filename),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fisher_exact_handles_empty_rows_and_columns() {
        let ln_factorials = cumulative_ln_factorials(0);
        let p = fisher_exact_greater(0, 0, 5, 5, &ln_factorials);
        assert_eq!(p, 1.0);

        let ln_factorials = cumulative_ln_factorials(5);
        let p = fisher_exact_greater(0, 5, 0, 5, &ln_factorials);
        assert_eq!(p, 1.0);
    }

    #[test]
    fn fisher_exact_matches_hypergeom_tail_for_extreme_case() {
        let ln_factorials = cumulative_ln_factorials(20);
        let p = fisher_exact_greater(10, 10, 10, 20, &ln_factorials);
        let expected = 1.0 / 184_756.0; // Choose(20,10)
        let diff = (p - expected).abs();
        assert!(diff < 1e-12, "diff={diff}, p={p}, expected={expected}");
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
