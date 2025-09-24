use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ab_glyph::{FontVec, PxScale};
use image::{Rgba, RgbaImage};
use imageproc::drawing::{
    draw_filled_circle_mut, draw_hollow_rect_mut, draw_line_segment_mut, draw_text_mut, text_size,
};
use imageproc::rect::Rect;

use crate::dotplot::SyntenyAnalysis;

pub struct BubblePlotBuilder<'a> {
    plot_width: u32,
    foreground_color: Rgba<u8>,
    background_color: Rgba<u8>,
    font: &'a FontVec,
}

impl<'a> BubblePlotBuilder<'a> {
    pub fn new(
        plot_width: u32,
        foreground_color: Rgba<u8>,
        background_color: Rgba<u8>,
        font: &'a FontVec,
    ) -> Self {
        Self {
            plot_width,
            foreground_color,
            background_color,
            font,
        }
    }

    pub fn build(
        &self,
        analysis: &SyntenyAnalysis,
        target_order: &[String],
        query_order: &[String],
        target_colors: &HashMap<String, Rgba<u8>>,
        non_significant: Rgba<u8>,
    ) -> Option<RgbaImage> {
        if !analysis.has_data() || target_order.is_empty() || query_order.is_empty() {
            return None;
        }

        let max_cells = usize::max(target_order.len(), query_order.len()).max(1) as f32;
        let mut cell_size = (self.plot_width as f32 / (max_cells + 5.0)).clamp(40.0, 140.0);
        if cell_size.is_nan() || cell_size <= 0.0 {
            cell_size = 60.0;
        }
        let left_margin = cell_size * 1.8;
        let top_margin = cell_size * 1.25;
        let grid_width = (target_order.len() as f32) * cell_size;
        let grid_height = (query_order.len() as f32) * cell_size;

        let image_width = (left_margin + grid_width + cell_size * 0.8).ceil() as u32;
        let image_height = (top_margin + grid_height + cell_size * 0.85).ceil() as u32;

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

        let max_anchor = analysis.max_significant_anchor_count();
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

                let anchors = summary.anchor_count();

                if summary.is_significant() && anchors > 0 {
                    let radius_ratio = (anchors as f32 / max_anchor as f32).sqrt();
                    let radius = (radius_ratio * max_radius).max(3.0);
                    let chromosome_color = target_colors
                        .get(target)
                        .copied()
                        .unwrap_or(self.foreground_color);
                    let color =
                        analysis.color_for(target, query, chromosome_color, non_significant);
                    draw_filled_circle_mut(
                        &mut bubble_plot,
                        (center_x.round() as i32, center_y.round() as i32),
                        radius.round() as i32,
                        color,
                    );
                } else {
                    let text = "n.s.";
                    let (text_w, text_h) = text_size(bubble_text_scale, self.font, text);
                    let draw_x = (center_x - text_w as f32 / 2.0).round() as i32;
                    let draw_y = (center_y - text_h as f32 / 2.0).round() as i32;
                    draw_text_mut(
                        &mut bubble_plot,
                        self.foreground_color,
                        draw_x,
                        draw_y,
                        bubble_text_scale,
                        self.font,
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
            let (text_w, _text_h) = text_size(label_scale, self.font, text);
            let draw_x = (center_x - text_w as f32 / 2.0).round() as i32;
            let draw_y = (grid_bottom + cell_size * 0.2) as i32;
            draw_text_mut(
                &mut bubble_plot,
                self.foreground_color,
                draw_x,
                draw_y,
                label_scale,
                self.font,
                text,
            );
        }

        for (row_idx, query) in query_order.iter().enumerate() {
            let center_y = grid_top + (row_idx as f32 + 0.5) * cell_size;
            let text = query.as_str();
            let (text_w, text_h) = text_size(label_scale, self.font, text);
            let draw_x = (grid_left - cell_size * 0.3 - text_w as f32).round() as i32;
            let draw_y = (center_y - text_h as f32 / 2.0).round() as i32;
            draw_text_mut(
                &mut bubble_plot,
                self.foreground_color,
                draw_x,
                draw_y,
                label_scale,
                self.font,
                text,
            );
        }

        Some(bubble_plot)
    }
}

pub fn bubble_output_path(output: &str) -> PathBuf {
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
