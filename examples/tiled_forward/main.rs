// Copyright 2015 The Gfx-rs Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


extern crate cgmath;
extern crate env_logger;
#[macro_use]
extern crate gfx;
extern crate gfx_window_glutin;
extern crate glutin;
extern crate time;
extern crate rand;
extern crate genmesh;
extern crate noise;

use rand::Rng;
use cgmath::{SquareMatrix, Array, Matrix, Matrix4, Point3, Vector, zero, vec2, Vector2, Vector3, vec4, Vector4, EuclideanVector};
use cgmath::{Transform, AffineMatrix3};

pub use gfx::format::{DepthStencil, Depth, I8Scaled, Srgb8, Rgba8, ChannelType};
use gfx::traits::{Device, Factory, FactoryExt};
use gfx::handle::{ShaderResourceView, Buffer, Texture};
use gfx::tex::Kind;

use genmesh::{Vertices, Triangulate};
use genmesh::generators::{SharedVertex, IndexedPolygon};
use time::precise_time_s;
use noise::{Seed, perlin2};

pub mod grid;

pub use grid::{LightInfo, GridTileData};

// Remember to also change the constants in the shaders
const NUM_LIGHTS: usize = 100;

gfx_vertex_struct!(TerrainVertex {
    pos: [f32; 3] = "a_Pos",
    normal: [f32; 3] = "a_Normal",
    color: [f32; 3] = "a_Color",
});

gfx_pipeline!(terrain {
    vbuf: gfx::VertexBuffer<TerrainVertex> = (),
    model: gfx::Global<[[f32; 4]; 4]> = "u_Model",
    view: gfx::Global<[[f32; 4]; 4]> = "u_View",
    proj: gfx::Global<[[f32; 4]; 4]> = "u_Proj",
    light_buf: gfx::ConstantBuffer<grid::LightInfo> = "u_LightData",
    light_grid: gfx::ConstantBuffer<grid::GridTileData> = "u_LightGrid",
    light_index_tex: gfx::ShaderResource<i32> = "u_LightIndexTex",

    out_color: gfx::RenderTarget<Rgba8> = "o_Color",
    out_depth: gfx::DepthTarget<Depth> = gfx::preset::depth::LESS_EQUAL_WRITE,
});

fn calculate_normal(seed: &Seed, x: f32, y: f32)-> [f32; 3] {
    // determine sample points
    let s_x0 = x - 0.001;
    let s_x1 = x + 0.001;
    let s_y0 = y - 0.001;
    let s_y1 = y + 0.001;

    // calculate gradient in point
    let dzdx = (perlin2(seed, &[s_x1, y]) - perlin2(seed, &[s_x0, y]))/(s_x1 - s_x0);
    let dzdy = (perlin2(seed, &[x, s_y1]) - perlin2(seed, &[x, s_y0]))/(s_y1 - s_y0);

    // cross gradient vectors to get normal
    let normal = Vector3::new(1.0, 0.0, dzdx).cross(Vector3::new(0.0, 1.0, dzdy)).normalize();

    return normal.into();
}

fn calculate_color(height: f32) -> [f32; 3] {
    if height > 8.0 {
        [0.9, 0.9, 0.9] // white
    } else if height > 0.0 {
        [0.7, 0.7, 0.7] // greay
    } else if height > -5.0 {
        [0.2, 0.7, 0.2] // green
    } else {
        [0.2, 0.2, 0.7] // blue
    }
}

pub fn main() {
    env_logger::init().unwrap();
    let (window, mut device, mut factory, main_color, main_depth) =
        gfx_window_glutin::init::<Rgba8, Depth>(glutin::WindowBuilder::new()
            .with_title("Tiled forward rendering".to_string())
            .with_dimensions(800, 600)
            .with_gl(glutin::GL_CORE)
    );

    let (w, h) = {
        let (w, h) = window.get_inner_size().unwrap();
        (w as gfx::tex::Size, h as gfx::tex::Size)
    };

    let seed = {
        let rand_seed = rand::thread_rng().gen();
        Seed::new(rand_seed)
    };

    let aspect = w as f32 / h as f32;
    let proj = cgmath::perspective(cgmath::deg(60.0f32), aspect, 0.1, 1000.0);

    let mut light_grid = grid::LightGrid::new(&mut factory, w as i32, h as i32, 32, NUM_LIGHTS as i32);
    let mut lights: Vec<grid::LightInfo> = (0..NUM_LIGHTS).map(|_| {
        grid::LightInfo { pos: [0.0, 0.0, 0.0], radius: 1.23, color: [1.0, 0.7, 0.8, 1.0] }
    }).collect();

    let terrain_scale = Vector3::new(25.0, 25.0, 25.0);

    for (i, d) in lights.iter_mut().enumerate() {
        let (x, y) = {
            let fi = i as f32;
            // Distribute lights nicely
            let r = 1.0 - (fi*fi) / ((NUM_LIGHTS*NUM_LIGHTS) as f32);
            (r * (0.2 + i as f32).cos(), r * (0.2 + i as f32).sin())
        };
        let h = perlin2(&seed, &[x, y]);

        d.pos[0] = terrain_scale.x * x;
        d.pos[1] = terrain_scale.y * y;
        d.pos[2] = terrain_scale.z * h + 0.5;
    };

    let (terrain_pso, mut terrain_data, terrain_slice) = {
        let plane = genmesh::generators::Plane::subdivide(256, 256);
        let vertex_data: Vec<TerrainVertex> = plane.shared_vertex_iter()
            .map(|(x, y)| {
                let h = terrain_scale.z * perlin2(&seed, &[x, y]);
                TerrainVertex {
                    pos: [terrain_scale.x * x, terrain_scale.y * y, h],
                    normal: calculate_normal(&seed, x, y),
                    color: calculate_color(h),
                }
            })
            .collect();

        let index_data: Vec<u32> = plane.indexed_polygon_iter()
            .triangulate()
            .vertices()
            .map(|i| i as u32)
            .collect();

        let (vbuf, slice) = factory.create_vertex_buffer_indexed(&vertex_data, &index_data[..]);

        let pso = factory.create_pipeline_simple(
            include_bytes!("shader/terrain_150.glslv"),
            include_bytes!("shader/terrain_150.glslf"),
            gfx::state::CullFace::Back,
            terrain::new()
        ).unwrap();

        let data = terrain::Data {
            light_grid: light_grid.buffer.clone(),
            light_buf: light_grid.data.clone(),
            light_index_tex: factory.view_buffer_as_shader_resource(&light_grid.indices).unwrap(),
            vbuf: vbuf,
            model: Matrix4::identity().into(),
            view: Matrix4::identity().into(),
            proj: proj.into(),
            out_color: main_color,
            out_depth: main_depth,
        };

        (pso, data, slice)
    };

    let mut encoder = factory.create_encoder();

    'main: loop {
        // quit when Esc is pressed.
        for event in window.poll_events() {
            use glutin::{Event, VirtualKeyCode};

            match event {
                Event::KeyboardInput(_, _, Some(VirtualKeyCode::Escape)) |
                Event::Closed => break 'main,
                _ => {},
            }
        }

        let time = precise_time_s() as f32;

        // Update camera position
        let view = {
            let cam_pos = {
                // Slowly circle the center
                let x = 0.15;
                let y = 0.15;
                Point3::new(x * 30.0, y * 30.0, 45.0)
            };

            let view: AffineMatrix3<f32> = Transform::look_at(
                cam_pos,
                Point3::new(0.0, 0.0, 0.0),
                Vector3::unit_z(),
            );

            terrain_data.view = view.mat.into();
            view
        };


        light_grid.build(&mut factory, &lights, view, proj, 0.01f32);

        encoder.reset();
        encoder.clear(&terrain_data.out_color, [0.3, 0.3, 0.3, 1.0]);
        encoder.clear_depth(&terrain_data.out_depth, 1.0);

        encoder.draw(&terrain_slice, &terrain_pso, &terrain_data);

        device.submit(encoder.as_buffer());
        window.swap_buffers().unwrap();
        device.cleanup();
    }
}

