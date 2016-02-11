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

use rand::Rng;
use cgmath::{SquareMatrix, Array, Matrix, Matrix4, Point3, Vector, one, zero, vec2, Vector2, Vector3, vec4, Vector4, EuclideanVector};
use cgmath::{Transform, AffineMatrix3};

pub use gfx::format::{DepthStencil, I8Scaled, Rgba8, ChannelType};
use gfx::traits::{Device, Factory, FactoryExt};
use gfx::handle::{ShaderResourceView, Buffer, Texture};
use gfx::tex::Kind;
use gfx;

use genmesh::{Vertices, Triangulate};
use genmesh::generators::{SharedVertex, IndexedPolygon};
use time::precise_time_s;
use noise::{Seed, perlin2};

use std::{mem, cmp};
use std::cmp::Ordering;
use std::marker::PhantomData;

#[derive(PartialEq,PartialOrd)]
struct NonNan(f32);

impl Eq for NonNan {}

impl Ord for NonNan {
    fn cmp(&self, other: &NonNan) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

gfx_constant_struct!(LightInfo {
    pos: [f32; 3],
    radius: f32,
    color: [f32; 4],
});

gfx_constant_struct!(GridTileData {
    offset: i32,
    count: i32,
});

fn update_clip_region_root(normal_xy: f32, light_xy: f32, light_z: f32, radius: f32, scale: f32, min: &mut f32, max: &mut f32) {
    let normal_z = (radius - normal_xy * light_xy) / light_z;
    let pz = (light_xy * light_xy + light_z * light_z - radius * radius) / (light_z - (normal_z / normal_xy) * light_xy);

    if pz < 0f32 {
        let c = -normal_z * scale / normal_xy;

        /*println!("ROOT");
        println!("c: {}", c);
        println!("nc: {}", normal_xy);
        println!("lc: {}", light_xy);
        println!("lz: {}", light_z);
        println!("lightRadius: {}", radius);
        println!("cameraScale: {}", scale);
        println!("nz: {}", normal_z);
        println!("pz: {}", pz);*/
        if normal_xy < 0f32 {
            *min = cmp::max(NonNan(*min), NonNan(c)).0;
        } else {
            *max = cmp::min(NonNan(*max), NonNan(c)).0;
        }
    }
}

fn update_clip_region(light_xy: f32, light_z: f32, radius: f32, scale: f32, min: &mut f32, max: &mut f32) {
    let r = radius * radius;
    let s = light_xy * light_xy + light_z * light_z;
    let dist = r * light_xy * light_xy - s * (r - light_z * light_z);

    if dist >= 0f32 {
        let a = radius * light_xy;
        let b = dist.sqrt();
        let nx0 = (a + b) / s;
        let nx1 = (a - b) / s;

        /*println!("BASE");
        println!("r: {}", r);
        println!("s: {}", s);
        println!("dist: {}", dist);
        println!("a: {}", a);
        println!("b: {}", b);
        println!("nx0: {}", nx0);
        println!("nx1: {}", nx1);*/

        update_clip_region_root(nx0, light_xy, light_z, radius, scale, min, max);
        update_clip_region_root(nx1, light_xy, light_z, radius, scale, min, max);
    }
}

fn compute_clip_region(proj: Matrix4<f32>, point: Point3<f32>, radius: f32, near: f32) -> Vector4<f32> {
    let mut clip_region = vec4(1f32, 1f32, -1f32, -1f32);

    if point.z - radius <= -near {
        let mut clip_min = vec2(-1f32, -1f32);
        let mut clip_max = vec2( 1f32,  1f32);

        update_clip_region(point.x, point.z, radius, proj[0][0], &mut clip_min.x, &mut clip_max.x);
        update_clip_region(point.y, point.z, radius, proj[1][1], &mut clip_min.y, &mut clip_max.y);

        clip_region = vec4(clip_min.x, clip_min.y, clip_max.x, clip_max.y);
    }

    clip_region
}

#[derive(Debug)]
pub struct GridTile {
    min: Vector2<i32>,
    max: Vector2<i32>
}

pub struct LightGrid<R: gfx::Resources, F: Factory<R>> {
    grid_data: Vec<GridTileData>,
    lights: Vec<LightInfo>,
    light_indices: Vec<i32>,
    tiles: Vec<GridTile>,
    width: i32,
    height: i32,
    pub max_width: i32,
    pub max_height: i32,
    tile_size: Vector2<i32>,
    max_lights: i32,
    pub buffer: Buffer<R, GridTileData>,
    pub data: Buffer<R, LightInfo>,
    pub indices: Buffer<R, i32>,
    phantom_factory: PhantomData<F>
}

impl<R: gfx::Resources, F: Factory<R>> LightGrid<R, F> {
    pub fn new(factory: &mut F, width: i32, height: i32, tile_size: i32, max_lights: i32) -> Self {
        let max_width = (width + tile_size - 1) / tile_size;
        let max_height = (height + tile_size - 1) / tile_size;

        println!("oops: {}", (max_width * max_height * max_lights));

        LightGrid {
            grid_data: Vec::with_capacity((max_width * max_height) as usize),
            lights: Vec::new(),
            light_indices: Vec::new(),
            tiles: Vec::new(),
            width: width,
            height: height,
            max_width: max_width,
            max_height: max_height,
            tile_size: Vector2::from_value(tile_size),
            max_lights: max_lights,
            buffer: factory.create_constant_buffer((max_width * max_height) as usize),
            data: factory.create_constant_buffer(max_lights as usize),
            indices: factory.create_constant_buffer((max_width * max_height * max_lights) as usize),
            phantom_factory: PhantomData
        }
    }

    pub fn resize(&mut self, width: i32, height: i32) {

    }

    fn find_bounds(&self, proj: Matrix4<f32>, point: Point3<f32>, radius: f32, near: f32) -> GridTile {
        fn clamp(vec: Vector4<f32>, low: Vector4<f32>, high: Vector4<f32>) -> Vector4<f32> {
            let x = cmp::max(cmp::min(NonNan(vec.x), NonNan(high.x)), NonNan(low.x)).0;
            let y = cmp::max(cmp::min(NonNan(vec.y), NonNan(high.y)), NonNan(low.y)).0;
            let z = cmp::max(cmp::min(NonNan(vec.z), NonNan(high.z)), NonNan(low.z)).0;
            let w = cmp::max(cmp::min(NonNan(vec.w), NonNan(high.w)), NonNan(low.w)).0;

            vec4(x, y, z, w)
        }

        let mut reg = -compute_clip_region(proj, point, radius, near);

        reg.swap_elements(0, 2);
        reg.swap_elements(1, 3);

        reg = reg * 0.5f32;
        reg = reg + 0.5f32;

        reg = clamp(reg, vec4(0.0, 0.0, 0.0, 0.0), vec4(1.0, 1.0, 1.0, 1.0));

        GridTile {
            min: vec2((reg.x * (self.width as f32)) as i32, (reg.y * (self.height as f32)) as i32),
            max: vec2((reg.z * (self.width as f32)) as i32, (reg.w * (self.height as f32)) as i32)
        }
    }

    pub fn build(&mut self, factory: &mut F, lights: &Vec<LightInfo>, view: AffineMatrix3<f32>, proj: Matrix4<f32>, near: f32) {
        fn clamp(vec: Vector2<i32>, low: Vector2<i32>, high: Vector2<i32>) -> Vector2<i32> {
            let x = cmp::min(cmp::max(vec.x, low.x), high.x);
            let y = cmp::min(cmp::max(vec.y, low.y), high.y);

            vec2(x, y)
        }

        if lights.is_empty() {
            return;
        }

        self.lights.clear();
        self.tiles.clear();


        for light in lights {
            let light_pos = Point3::from(light.pos).clone();
            let vp = view.transform_point(light_pos);

            let rect = self.find_bounds(proj, vp, light.radius, near);

            if rect.min.x < rect.max.x && rect.min.y < rect.max.y {
                self.tiles.push(rect);
                self.lights.push(LightInfo {
                    pos: vp.into(),
                    radius: light.radius,
                    color: light.color
                });
            }
        }

    

        let grid_dim = (vec2(self.width, self.height) + self.tile_size - 1) / self.tile_size;

        unsafe { self.grid_data.set_len((self.max_width * self.max_height) as usize); }
        for mut data in &mut self.grid_data {
            data.count = 0;
            data.offset = 0;
        }

        let mut coverage = 0i32;
        for ref rect in self.tiles.iter() {
            let lower = clamp(rect.min / self.tile_size, vec2(0, 0), grid_dim + 1);
            let higher = clamp((rect.max + self.tile_size - 1) / self.tile_size, vec2(0, 0), grid_dim + 1);

            for y in lower.y..higher.y {
                for x in lower.x..higher.x {
                    self.grid_data[(x + y * self.max_width) as usize].count += 1;
                    coverage += 1;
                }
            }
        }

 
        self.light_indices.resize(coverage as usize, 0);

        unsafe { self.light_indices.set_len(coverage as usize); }

        let mut offset = 0i32;
        for y in 0..grid_dim.y {
            for x in 0..grid_dim.x {
                let count = self.grid_data[(x + y * self.max_width) as usize].count;
                self.grid_data[(x + y * self.max_width) as usize].offset = offset + count;
                offset += count;
            }
        }

        println!("{}, {}, {}", self.lights.len(), self.grid_data.len(), self.light_indices.len());
        if !self.tiles.is_empty() {
            let mut light_id = 0i32;
            for ref rect in self.tiles.iter() {
                let lower = clamp(rect.min / self.tile_size, vec2(0, 0), grid_dim + 1);
                let higher = clamp((rect.max + self.tile_size - 1) / self.tile_size, vec2(0, 0), grid_dim + 1);

                for y in lower.y..higher.y {
                    for x in lower.x..higher.x {
                        let offset = self.grid_data[(x + y * self.max_width) as usize].offset - 1;
                        self.light_indices[offset as usize] = light_id;
                        self.grid_data[(x + y * self.max_width) as usize].offset = offset;
                    }
                }

                light_id += 1;
            }

            for mut data in &mut self.grid_data {
                data.count = 1;
                data.offset = 0;
            }

            factory.update_buffer(&self.buffer, &self.grid_data, 0usize).unwrap();
            factory.update_buffer(&self.data, &self.lights, 0usize).unwrap();
            factory.update_buffer(&self.indices, &self.light_indices, 0usize).unwrap();

            println!("{:?}", self.buffer.get_info());
        }
    }
}

