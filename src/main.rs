extern crate rand;
extern crate rayon;

use std::{f64, fmt, io};
use std::ops::{Add, Sub, Mul};
use std::path::Path;
use std::fs::File;
use std::io::prelude::*;
use std::time::Instant;
use rand::prelude::*;
use rayon::prelude::*;

type Rfloat = f64;

const RESOLUTION : usize	= 128;
const RAY_BIAS : Rfloat		= 0.0005;
const SPP : usize			= 16*16; // samples per pixel
const MAX_BOUNCES : usize	= 8;
const MIN_BOUNCES : usize	= 4;
const NUM_AA : usize		= 4;
const INV_AA : Rfloat		= 1.0 / (NUM_AA as Rfloat);

#[derive(Copy, Clone)]
struct Vector 
{
	x : Rfloat,
	y : Rfloat,
	z : Rfloat,
}

enum MaterialType
{
	DIFFUSE,
	GLOSSY,
	MIRROR
}

struct Material 
{
	material_type	: MaterialType,
	diffuse			: Vector,
	emissive		: Vector,
	specular		: Vector,
	exp				: Rfloat,
}

impl Material
{
	fn default() -> Material
	{
		Material { material_type : MaterialType::DIFFUSE, diffuse : Vector::zero(), emissive : Vector::zero(), specular : Vector::zero(), exp : 0.0 }
	}
}

struct Sphere<'a>
{
	radius		: Rfloat,
	center		: Vector,
	material	: &'a Material,

	radius_sqr	: Rfloat,
}

impl<'a> Sphere<'a>
{
	fn new(radius : Rfloat, center : Vector, material : &'a Material) -> Sphere
	{
		Sphere { radius : radius, center : center, material : material, radius_sqr : radius*radius }
	}

	fn intersects(&self, ray : &Ray) -> Rfloat
	{
		let op = self.center - ray.origin;
		let b = dot(&op, &ray.dir);
		let mut d = b * b - dot(&op, &op) + self.radius_sqr;

		if d < 0.0
		{
			return 0.0
		}

		d = d.sqrt();
		let mut t = b - d;

		if t > RAY_BIAS 
		{
			return t
		}

		t = b + d;
		if t > RAY_BIAS 
		{
			return t
		}

		return 0.0
	}
	fn is_light(&self) -> bool
	{
		(dot(&self.material.emissive, &self.material.emissive) > 0.0)
	}
}

struct Ray
{
	origin : Vector,
	dir    : Vector,
}

impl Ray
{
	fn calc_intersection_point(&self, t : Rfloat) -> Vector
	{
		return self.origin + self.dir * t;
	}
}

fn dot(a : &Vector, b : &Vector) -> Rfloat
{
	a.x * b.x + a.y * b.y + a.z * b.z
}

fn cross(a : &Vector, b : &Vector) -> Vector
{
	Vector { x : a.y*b.z - a.z*b.y, y : a.z*b.x - a.x*b.z, z : a.x*b.y - a.y*b.x }
}

fn clamp(x : Rfloat, min : Rfloat, max : Rfloat) -> Rfloat
{
	if x < min
	{
		return min;
	}
	if x > max
	{
		return max;
	}
	return x;
}

fn max(x : Rfloat, y : Rfloat) -> Rfloat
{
	if x > y
	{
		return x;
	}
	else
	{
		return y;
	}
}

impl Vector 
{
	fn new(vx : Rfloat, vy : Rfloat, vz : Rfloat) -> Vector
	{
		Vector { x : vx, y : vy, z : vz }
	}
	fn new_normal(mut vx : Rfloat, mut vy : Rfloat, mut vz : Rfloat) -> Vector
	{
		let len_sqr = vx*vx + vy*vy + vz * vz;
		if len_sqr > f64::EPSILON
		{
			let len = len_sqr.sqrt();
			vx /= len;
			vy /= len;
			vz /= len;
		}
		Vector { x : vx, y : vy, z : vz }
	}
	fn zero() -> Vector
	{
		Vector { x : 0.0, y : 0.0, z : 0.0 }
	}

	fn normalize(&mut self)
	{
		let len_sqr = dot(self, self);
		if len_sqr > f64::EPSILON
		{
			let oo_len = 1.0 / len_sqr.sqrt();
			self.x *= oo_len;
			self.y *= oo_len;
			self.z *= oo_len;
		}
	}

	fn clamp01(&mut self)
	{
		self.x = clamp(self.x, 0.0, 1.0);
		self.y = clamp(self.y, 0.0, 1.0);
		self.z = clamp(self.z, 0.0, 1.0);
	}

	fn get_color(&self) -> (u32, u32, u32)
	{
		let mut color = *self;
		color.clamp01();

		let r = (color.x.powf(0.45) * 255.0 + 0.5) as u32;
		let g = (color.y.powf(0.45) * 255.0 + 0.5) as u32;
		let b = (color.z.powf(0.45) * 255.0 + 0.5) as u32;

		(r, g, b)
	}

	fn vecmul(&self, other : &Vector) -> Vector
	{
		Vector { x : self.x * other.x, y : self.y * other.y, z : self.z * other.z }
	}

	fn set(&mut self, x : Rfloat, y : Rfloat, z : Rfloat)
	{
		self.x = x;
		self.y = y;
		self.z = z;
	}

	fn max_component(&self) -> Rfloat
	{
		max(max(self.x, self.y), self.z)
	}

	fn negate(&mut self)
	{
		self.x = -self.x;
		self.y = -self.y;
		self.z = -self.z;
	}
}

impl Sub for Vector
{
	type Output = Vector;

	fn sub(self, other: Vector) -> Vector
	{
		Vector { x : self.x - other.x, y : self.y - other.y, z : self.z - other.z }
	}
}

impl Add for Vector
{
	type Output = Vector;

	fn add(self, other : Vector) -> Vector
	{
		Vector { x : self.x + other.x, y : self.y + other.y, z : self.z + other.z }
	}
}

impl Mul<Rfloat> for Vector
{
	type Output = Vector;

	fn mul(self, s : Rfloat) -> Vector
	{
		Vector { x : self.x * s, y : self.y * s, z : self.z * s }
	}
}

impl fmt::Display for Vector {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "({}, {}, {})", self.x, self.y, self.z)
	}
}

struct Camera 
{
	forward		: Vector,
	fov_scale	: Rfloat,
}

struct Scene<'a>
{
	objects : Vec<Sphere<'a>>,
	lights	: Vec<usize>,
	camera	: Box<Camera>,
}

impl<'a> Scene<'a>
{
	fn intersect(&self, ray : &Ray) -> Option<(Rfloat, &Sphere)>
	{
		let mut result = None;
		let mut min_t = f64::MAX;

		for sphere in self.objects.iter()
		{
			let t = sphere.intersects(ray);
			if t > 0.0 && t < min_t
			{
				min_t = t;
				result = Some((min_t, sphere));
			}
		}
		(result)
	}
	fn collect_lights(&mut self)
	{
		for (index, object) in self.objects.iter().enumerate()
		{
			if object.is_light()
			{
				self.lights.push(index);
			}

		}
	}
}

struct Context<'a>
{
	scene	: &'a Scene<'a>,
	samples : [Rfloat; SPP * 2]
}

fn rand01<T:Rng>(rng : &mut T) -> Rfloat
{
	(rng.gen::<Rfloat>())
}

// Given 1 axis, returns other 2
fn build_basis(v1 : &Vector) -> (Vector, Vector)
{
	let mut v2 = Vector::zero();

	if v1.x.abs() > v1.y.abs()
	{
		let oo_len = 1.0 / (v1.x * v1.x + v1.z * v1.z).sqrt();
		v2.set(-v1.z*oo_len, 0.0, v1.x*oo_len);
	}
	else
	{
		let oo_len = 1.0 / (v1.y * v1.y + v1.z * v1.z).sqrt();
		v2.set(0.0, v1.z*oo_len, -v1.y*oo_len);
	}
	(v2, cross(v1, &v2))
}

fn transform_to_basis(vin : &Vector, vx : &Vector, vy : &Vector, vz : &Vector) -> Vector
{
	Vector
	{
		x : vx.x*vin.x + vy.x*vin.y + vz.x*vin.z,
		y : vx.y*vin.x + vy.y*vin.y + vz.y*vin.z,
		z : vx.z*vin.x + vy.z*vin.y + vz.z*vin.z 
	}
}

fn reflect(dir : &Vector, n : &Vector) -> Vector
{
	let h = *n * dot(dir, n) * 2.0;
	return h - *dir;
}

fn initialize_samples<T:Rng>(samples : &mut [Rfloat; SPP * 2], rng : &mut T)
{
	let xstrata = (SPP as Rfloat).sqrt();
	let ystrata = (SPP as Rfloat) / xstrata;

	let mut is = 0;

	for ystep in 0..ystrata as i32
	{
		for xstep in 0..xstrata as i32
		{
			let fx = ((xstep as Rfloat) + rand01(rng)) / xstrata;
			let fy = ((ystep as Rfloat) + rand01(rng)) / ystrata;
			samples[is] = fx;
			samples[is + 1] = fy;
			is += 2;
		}
	}
}

impl<'a> Context<'a>
{
	fn initialize_samples(&mut self)
	{
		let mut rng = rand::thread_rng();
		initialize_samples(&mut self.samples, &mut rng);
	}
}

fn sample_hemisphere_cosine(u1 : Rfloat, u2 : Rfloat) -> Vector
{
	let phi = 2.0 * f64::consts::PI * u1;
	let r = u2.sqrt();
	let (s, c) = phi.sin_cos();

	Vector{x : c * r, y : s * r, z : (1.0 - r*r).sqrt()}
}

fn sample_hemisphere_specular(u1 : Rfloat, u2 : Rfloat, exp : Rfloat) -> Vector
{
	let phi = 2.0 * f64::consts::PI * u1;

	let cos_theta = (1.0 - u2).powf(1.0 / (exp + 1.0));
	let sin_theta = (1.0 - cos_theta*cos_theta).sqrt();

	Vector { x : phi.cos() * sin_theta, y: phi.sin() * sin_theta, z : cos_theta }
}


fn interreflect_diffuse(normal : &Vector, intersection_point : &Vector, u1 : Rfloat, u2 : Rfloat) -> Ray
{
	let (v2, v3) = build_basis(normal);

	let sampled_dir = sample_hemisphere_cosine(u1, u2);

	let new_ray = Ray { origin : *intersection_point, 
						dir : transform_to_basis(&sampled_dir, &v2, &v3, normal) };

	return new_ray;
}

fn interreflect_specular(normal : &Vector, intersection_point : &Vector, u1 : Rfloat, u2 : Rfloat, exp : Rfloat,
		new_ray : &mut Ray)
{
	let view = new_ray.dir * -1.0;
	let mut reflected = reflect(&view, normal);
	reflected.normalize();

	let (v2, v3) = build_basis(&reflected);

	let sampled_dir = sample_hemisphere_specular(u1, u2, exp);

	new_ray.origin = *intersection_point;
	new_ray.dir  = transform_to_basis(&sampled_dir, &v2, &v3, &reflected);
}

fn sample_lights(scene : &Scene, intersection : &Vector, normal : &Vector, ray_dir : &Vector, material : &Material) -> Vector
{
	let mut color = Vector::zero();

	for light_index in scene.lights.iter()
	{
		let ref light = scene.objects[*light_index];
		let mut l = light.center - *intersection;
		let light_dist_sqr = dot(&l, &l);
		l.normalize();

		let mut d = dot(normal, &l);

		let shadow_ray = Ray { origin : *intersection, dir : l };
		match scene.intersect(&shadow_ray)
		{
			None => {}
			Some((_, object)) =>
			{
				if object as *const Sphere == light as *const Sphere
				{
					if d > 0.0
					{
						let sin_alpha_max_sqr = light.radius_sqr / light_dist_sqr;
						let cos_alpha_max = (1.0 - sin_alpha_max_sqr).sqrt();

						let omega = 2.0 * (1.0 - cos_alpha_max);
						d *= omega;

						let c = material.diffuse.vecmul(&light.material.emissive);
						color = color + c * d;
					}

					// Specular part
					match material.material_type
					{
						MaterialType::DIFFUSE => {}
						MaterialType::GLOSSY | MaterialType::MIRROR =>
						{
							let reflected = reflect(&l, normal);
							d = -dot(&reflected, ray_dir);
							if d > 0.0
							{
								let smul = d.powf(material.exp);
								let spec_color = material.specular * smul;
								color = color + spec_color;
							}
						}
					}
				}
			}
		}
	}
	(color)
}

fn trace<T:Rng>(ray : &mut Ray, scene : &Scene, samples : &[Rfloat; SPP * 2], mut u1 : Rfloat, mut u2 : Rfloat, rng : &mut T) -> Vector
{
	let mut result = Vector::zero();
	let mut rr_scale = Vector { x : 1.0, y : 1.0, z : 1.0 };
	let mut direct = true;

	for bounce in 0..MAX_BOUNCES
	{
		match scene.intersect(ray)
		{
			None => break,

			Some((t, object)) =>
			{
				let ref material = object.material;
				if direct
				{
					result = result + material.emissive.vecmul(&rr_scale);
				}

				let mut diffuse = material.diffuse;
				let max_diffuse = diffuse.max_component();
				if bounce > MIN_BOUNCES || max_diffuse < f64::EPSILON
				{
					if rand01(rng) > max_diffuse
					{
						break;
					}
					diffuse = diffuse * (1.0 / max_diffuse);
				}

				let intersection_point = ray.calc_intersection_point(t);
				let mut normal = (intersection_point - object.center) * (1.0 / object.radius);
				if dot(&normal, &ray.dir) >= 0.0
				{
					normal.negate();
				}

				match material.material_type
				{
					MaterialType::DIFFUSE =>
					{
						direct = false;
						let direct_light = rr_scale.vecmul(&sample_lights(scene, &intersection_point, &normal, &ray.dir, material));
						result = result + direct_light;

						*ray = interreflect_diffuse(&normal, &intersection_point, u1, u2);
						rr_scale = rr_scale.vecmul(&diffuse);
					}

					MaterialType::GLOSSY =>
					{
						direct = false;
						let direct_light = rr_scale.vecmul(&sample_lights(scene, &intersection_point, &normal, &ray.dir, material));
						result = result + direct_light;

						// Specular/diffuse Russian roulette
						let max_spec = material.specular.max_component();
						let p = max_spec / (max_spec + max_diffuse);
						let smult = 1.0 / p;

						if rand01(rng) > p	// diffuse
						{
							*ray = interreflect_diffuse(&normal, &intersection_point, u1, u2);
							let color = diffuse * (1.0	/ (1.0 - 1.0/smult));
							rr_scale =	rr_scale.vecmul(&color);
						}
						else
						{
							interreflect_specular(&normal, &intersection_point, u1, u2, material.exp, ray);
							let color = material.specular * smult;
							rr_scale = rr_scale.vecmul(&color);
						}
					}

					MaterialType::MIRROR =>
					{
						let view = ray.dir * -1.0;
						let mut reflected = reflect(&view, &normal);
						reflected.normalize();

						ray.origin = intersection_point;
						ray.dir = reflected;

						rr_scale = rr_scale.vecmul(&diffuse);
					}
				}

				let sample_index = rng.gen_range(0, SPP);
				u1 = samples[sample_index*2];
				u2 = samples[sample_index*2+1];
			}
		}
	}
	(result)
}

fn apply_tent_filter(samples : &mut [Rfloat; SPP * 2])
{
	for i in 0..SPP
	{
		let x = samples[i*2+0];
		let y = samples[i*2+1];

		samples[i * 2] = match x {
			x if x < 0.5 => (2.0 * x).sqrt() - 1.0,
			_ => 1.0 - (2.0 - 2.0 * x).sqrt()
		};
		samples[i * 2 + 1] = match y {
			y if y < 0.5 => (2.0 * y).sqrt() - 1.0,
			_ => 1.0 - (2.0 - 2.0 * y).sqrt()
		};
	}
}

fn process_chunk(context : &Context, buffer : &mut [u8], offset : usize)
{
	let res = RESOLUTION as Rfloat;
	let camera = &context.scene.camera;

	let cx = Vector { x : camera.fov_scale, y : 0.0, z : 0.0 };
	let mut cy = cross(&cx, &camera.forward);
	cy.normalize();
	cy = cy * camera.fov_scale;

	let ray_origin = Vector { x : 50.0, y : 52.0, z : 295.6 };

	let mut chunk_samples = [0.0; SPP*2];
	let mut sphere_samples = [0.0; SPP*2];

	let mut rng = rand::thread_rng();

	initialize_samples(&mut chunk_samples, &mut rng);
	apply_tent_filter(&mut chunk_samples);

	let inv_spp = 1.0 / SPP as Rfloat;

	let start_x = offset % RESOLUTION;
	let start_y = offset / RESOLUTION;

	let mut y = start_y;
	let mut x = start_x;
	let pixel_count = buffer.len() / 4;

	for pixel_index in 0..pixel_count
	{
		let pixel_offset = pixel_index * 4;
		initialize_samples(&mut sphere_samples, &mut rng);

		let mut cr = Vector::zero();
		for aa in 0..NUM_AA
		{
			let mut pr = Vector::zero();

			let aax = (aa & 0x1) as Rfloat;
			let aay = (aa >> 1) as Rfloat;

			for s in 0..SPP
			{
				let dx = chunk_samples[s * 2];
				let dy = chunk_samples[s * 2 + 1];

				let px = (((aax + 0.5 + dx) / 2.0) + (x as Rfloat)) / res - 0.5;
				let py = -((((aay + 0.5 + dy) / 2.0) + y as Rfloat) / res - 0.5);

				let ccx = cx * px;
				let ccy = cy * py;

				let mut ray_dir = ccx + ccy + camera.forward;
				ray_dir.normalize();

				let mut ray = Ray{ origin : ray_origin + ray_dir * 136.0, dir : ray_dir};
				let u1 = sphere_samples[s*2];
				let u2 = sphere_samples[s*2+1];

				let r = trace(&mut ray, &context.scene, &context.samples, u1, u2, &mut rng);

				pr = pr + (r * inv_spp);
			}
			cr = cr + (pr * INV_AA);
		}			
		let (r, g, b) = cr.get_color();

		buffer[pixel_offset + 3] = 0xFF;
		buffer[pixel_offset + 0] = b as u8;
		buffer[pixel_offset + 1] = g as u8;
		buffer[pixel_offset + 2] = r as u8;

		x = x + 1;
		if x == RESOLUTION
		{
			x = 0;
			y = y + 1;
		}
	}
}

fn put16(buffer : &mut [u8], v : u16)
{
	buffer[0] = (v & 0xFF) as u8;
	buffer[1] = (v >> 8) as u8;
}

fn write_tga_header(f : &mut File, width : usize, height : usize) -> io::Result<()>
{
	let mut header : [u8; 18] = [0; 18];

	header[2] = 2; // 32-bit
	put16(&mut header[12..], width as u16);
	put16(&mut header[14..], height as u16);
	header[16] = 32;   // BPP
	header[17] = 0x20; // top down, non interlaced

	f.write_all(&header)
}

fn write_tga(fname : &Path, pixels : &[u8], width : usize, height : usize) -> io::Result<()>
{
	let mut file = match File::create(fname) {
		Ok(f) => f,
		Err(e) => panic!("file error: {}", e),
	};

	write_tga_header(&mut file, width, height)?;

	file.write_all(pixels)
}

fn main() 
{
	let fov_scale = (55.0 * f64::consts::PI / 180.0 * 0.5).tan();
	let camera = Camera{ forward : Vector::new_normal(0.0, -0.042612, -1.0), fov_scale: fov_scale };

	use MaterialType::{DIFFUSE, GLOSSY, MIRROR};

	let diffuse_grey = &Material{ material_type : DIFFUSE, diffuse : Vector::new(0.75, 0.75, 0.75), ..Material::default() };
	let diffuse_red = &Material{ material_type: DIFFUSE, diffuse: Vector::new(0.95, 0.15, 0.15), ..Material::default() };
	let diffuse_blue = &Material{ material_type: DIFFUSE, diffuse: Vector::new(0.25, 0.25, 0.7), ..Material::default() };
	let diffuse_black = &Material{ material_type: DIFFUSE, ..Material::default() };
	let diffuse_green = &Material{ material_type: DIFFUSE, diffuse: Vector::new(0.0, 0.55, 14.0/255.0), ..Material::default() };
	let diffuse_white = &Material{ material_type: DIFFUSE, diffuse: Vector::new(0.99, 0.99, 0.99), ..Material::default() };
	let glossy_white = &Material{ material_type: GLOSSY, diffuse: Vector::new(0.3, 0.05, 0.05),
			specular: Vector::new(0.69, 0.69, 0.69), exp: 45.0, emissive: Vector::zero() };
	let white_light = &Material{ material_type: DIFFUSE, emissive: Vector::new(400.0, 400.0, 400.0), ..Material::default() };
	let mirror = &Material{material_type: MIRROR, diffuse: Vector::new(0.999, 0.999, 0.999), ..Material::default() };

	let mut scene = Scene
	{ 
		objects: vec!{ 
			Sphere::new(1e5, Vector::new(1e5 + 1.0, 40.8, 81.6), diffuse_red),
			Sphere::new(1e5, Vector::new(-1e5 + 99.0, 40.8, 81.6), diffuse_blue),
			Sphere::new(1e5, Vector::new(50.0, 40.8, 1e5), diffuse_grey),
			Sphere::new(1e5, Vector::new(50.0, 40.8, -1e5 + 170.0), diffuse_black),
			Sphere::new(1e5, Vector::new(50.0, 1e5, 81.6), diffuse_grey),
			Sphere::new(1e5, Vector::new(50.0, -1e5 + 81.6, 81.6), diffuse_grey),
			Sphere::new(16.5, Vector::new(27.0, 16.5, 57.0), mirror),
			Sphere::new(10.5, Vector::new(17.0, 10.5, 97.0), diffuse_green),
			Sphere::new(16.5, Vector::new(76.0, 16.5, 78.0), glossy_white),
			Sphere::new(8.5, Vector::new(82.0, 8.5, 108.0), diffuse_white),
			Sphere::new(1.5, Vector::new(50.0, 81.6 - 16.5, 81.6), white_light)
		},
		lights: vec!{},
		camera : Box::new(camera)
	};
	scene.collect_lights();

	let start_time = Instant::now();

	let mut context = Context { scene : &scene, samples : [0.0; SPP*2] };
	context.initialize_samples();

	// in pixels
	let num_pixels = RESOLUTION * RESOLUTION;
	let chunk_size = 256usize;

	let mut framebuffer:Vec<u8> = vec![0; num_pixels * 4];

	// Single-threaded
	//process_chunk(&context, framebuffer.as_mut_slice(), 0);

	// Multi-threaded
	framebuffer.par_chunks_mut(chunk_size*4)
		.enumerate()
		.map(|mut x| process_chunk(&context, &mut x.1, x.0*chunk_size))
		.collect::<Vec<_>>();

	let time_taken = Instant::now().duration_since(start_time);
	let time_taken_dbl = time_taken.as_secs() as f64 + time_taken.subsec_nanos() as f64 * 1e-9;

	println!("Tracing took {} seconds", time_taken_dbl);

	if let Err(e) = write_tga(&Path::new("trace.tga"), framebuffer.as_slice(), RESOLUTION, RESOLUTION) {
		println!("Error writing file: {}", e);
	}
}
