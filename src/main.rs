use macroquad::prelude::*;

// Window resolution settings
const WIDTH: i32 = 2880;
const HEIGHT: i32 = 1800;

// Gravitational constant — NOT the real one (6.674e-11).
// In a toy sim we pick a value that makes pixel-scale bodies behave visibly.
// You'll want to tweak this once things are moving.
const G: f32 = 1000.0;

// A single gravitating body.
//
// Notes for a Rust beginner:
// - `pub` isn't needed here because everything lives in one file / one crate.
// - `Vec2` comes from macroquad (re-exported from the `glam` crate). It's a
//   plain `{ x: f32, y: f32 }` with operator overloads, so `a + b`, `a * 2.0`,
//   `a.length()`, `a.normalize()` all work.
// - `#[derive(Clone, Copy)]` lets us pass `Body` by value cheaply. We won't
//   derive Copy once bodies get more complex — for now it's convenient.
#[derive(Clone, Copy, Debug)]
    struct Body {
    pos: Vec2,
    vel: Vec2,
    mass: f32,
    radius: f32,
    color: Color,
}

impl Body {
    fn new(pos: Vec2, vel: Vec2, mass: f32, radius: f32, color: Color) -> Self {
        Self { pos, vel, mass, radius, color }
    }

    fn draw(&self) {
        draw_circle(self.pos.x, self.pos.y, self.radius, self.color);
    }
}

// ---------------------------------------------------------------------------
// PHYSICS — you'll be filling these in.
// ---------------------------------------------------------------------------

/// Compute the gravitational acceleration on each body from every other body.
///
/// Returns a Vec<Vec2> where `accels[i]` is the acceleration to apply to
/// `bodies[i]` this step.
///
/// Why return accelerations instead of mutating velocities directly?
/// Because while we're reading body `i`'s position we also need to read body
/// `j`'s position — if we mutated as we went, body `j` would see body `i`'s
/// *new* state midway through the loop, which is subtly wrong. Computing all
/// accelerations first, then applying them, avoids that.
///
/// TODO(you):
///   for each pair (i, j) where i != j:
///       let r      = bodies[j].pos - bodies[i].pos;
///       let dist_sq = r.length_squared();      // avoids a sqrt
///       let dist    = dist_sq.sqrt();
///       let force_mag = G * bodies[i].mass * bodies[j].mass / dist_sq;
///       let dir = r / dist;                    // unit vector i -> j
///       // F = m * a  =>  a = F / m
///       accels[i] += dir * force_mag / bodies[i].mass;
///
/// Hint: the inner loop can start at `i + 1` and apply equal-and-opposite
/// forces to both bodies — half the work. Do the naive version first.
fn compute_accelerations(bodies: &[Body]) -> Vec<Vec2> {
    let mut accels = vec![Vec2::ZERO; bodies.len()];

    for i in 0..bodies.len() {
        for j in (i + 1)..bodies.len() {
            let r = bodies[j].pos - bodies[i].pos;
            let dist_sq = r.length_squared();
            let dist = dist_sq.sqrt();

            // Softening factor: prevents force from blowing up when two bodies
            // get very close (dist near 0 => division by ~zero).
            let softened_dist_sq = dist_sq + 1.0;

            let force_mag = G * bodies[i].mass * bodies[j].mass / softened_dist_sq;
            let dir = r / dist; // unit vector pointing from i toward j

            // F = m * a  =>  a = F / m
            accels[i] += dir * force_mag / bodies[i].mass;
            accels[j] -= dir * force_mag / bodies[j].mass; // equal and opposite
        }
    }

    accels
}

/// Advance positions and velocities by `dt` seconds using the accelerations.
///
/// This is "semi-implicit Euler" (aka symplectic Euler): update velocity
/// first, then use the *new* velocity to update position. It's one line
/// different from plain Euler but behaves much better for orbits.
///
/// TODO(you):
///   for i in 0..bodies.len():
///       bodies[i].vel += accels[i] * dt;
///       bodies[i].pos += bodies[i].vel * dt;
fn integrate(bodies: &mut [Body], accels: &[Vec2], dt: f32) {
    for (body, &accel) in bodies.iter_mut().zip(accels.iter()) {
        body.vel += accel * dt;  // update velocity first (semi-implicit Euler)
        body.pos += body.vel * dt;
    }
}

/// Detect overlapping bodies and resolve them with an elastic collision.
///
/// Elastic collision between two circles (equal or unequal mass) along the
/// line connecting their centers. Tangential velocity is unchanged; only the
/// component along the collision normal is exchanged.
///
/// TODO(you) — do this in two phases per colliding pair:
///
///   1. POSITIONAL CORRECTION (push them apart so they don't stick):
///        let delta   = b.pos - a.pos;
///        let dist    = delta.length();
///        let overlap = (a.radius + b.radius) - dist;
///        if overlap > 0.0 {
///            let n = delta / dist;   // unit normal, a -> b
///            // push each body away by half the overlap (or weight by mass)
///            a.pos -= n * (overlap * 0.5);
///            b.pos += n * (overlap * 0.5);
///        }
///
///   2. VELOCITY RESPONSE (1D elastic along the normal `n`):
///        let rel_vel    = b.vel - a.vel;
///        let vel_along  = rel_vel.dot(n);
///        if vel_along < 0.0 { /* already separating, skip */ }
///        let m1 = a.mass; let m2 = b.mass;
///        // impulse scalar for a perfectly elastic collision:
///        let j = -(2.0 * vel_along) / (1.0/m1 + 1.0/m2);
///        let impulse = n * j;
///        a.vel -= impulse / m1;
///        b.vel += impulse / m2;
///
/// Rust gotcha you'll hit: you can't hold two `&mut` into the same Vec at
/// once with normal indexing. Use `slice::split_at_mut` to get disjoint
/// mutable halves. Ask me when you get there and I'll show you the trick.
fn resolve_collisions(bodies: &mut [Body]) {
    for i in 0..bodies.len() {
        for j in (i + 1)..bodies.len() {
            // `split_at_mut` splits the slice into two non-overlapping mutable
            // halves at index `j`. This proves to the borrow checker that `a`
            // and `b` are distinct memory locations — something it can't verify
            // if you just do `&mut bodies[i]` and `&mut bodies[j]` separately.
            let (left, right) = bodies.split_at_mut(j);
            let a = &mut left[i];
            let b = &mut right[0]; // right[0] is bodies[j]

            let delta = b.pos - a.pos;
            let dist = delta.length();

            // No collision if they aren't overlapping.
            let overlap = (a.radius + b.radius) - dist;
            if overlap <= 0.0 {
                continue;
            }

            let n = delta / dist; // unit normal pointing from a -> b

            // Phase 1: push apart so they don't tunnel through each other.
            // Weight by mass so the lighter body moves more.
            let total_mass = a.mass + b.mass;
            a.pos -= n * (overlap * b.mass / total_mass);
            b.pos += n * (overlap * a.mass / total_mass);

            // Phase 2: elastic velocity response along n.
            let vel_along = (b.vel - a.vel).dot(n);
            if vel_along >= 0.0 {
                // Already separating — don't apply impulse.
                continue;
            }

            let j = -(2.0 * vel_along) / (1.0 / a.mass + 1.0 / b.mass);
            let impulse = n * j;
            a.vel -= impulse / a.mass;
            b.vel += impulse / b.mass;
        }
    }
}

// ---------------------------------------------------------------------------
// MAIN LOOP
// ---------------------------------------------------------------------------



fn window_conf() -> Conf {
    Conf {
        window_title: "Gravity Simulation".to_owned(),
        window_width: WIDTH,
        window_height: HEIGHT,
        high_dpi: true,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {

    // screen_width/height return logical pixel dimensions at runtime,
    // correctly accounting for DPI scaling on Retina displays.
    let cx = screen_width() / 2.0;
    let cy = screen_height() / 2.0;
    let spread = 100.0;

    // Place 3 bodies at the vertices of an equilateral triangle inscribed
    // in a circle of radius `spread` centered on (cx, cy).
    // Angles are in radians; std::f32::consts::PI gives us π.
    let angle_step = 2.0 * std::f32::consts::PI / 3.0; // 120° in radians
    let start_angle = std::f32::consts::PI / 2.0;       // start at the top (90°)

    let speed = 300.0; // tangential speed — tune to taste

    let mut bodies: Vec<Body> = (0..3).map(|i| {
        let angle = start_angle + i as f32 * angle_step;
        let pos = vec2(cx + spread * angle.cos(), cy - spread * angle.sin());
        // Tangential direction for CCW motion in screen coords (y-down).
        // Derived by differentiating pos w.r.t. angle: d(pos)/dθ = (-sin θ, -cos θ).
        let vel = vec2(-angle.sin(), -angle.cos()) * speed;
        Body::new(pos, vel, 10000.0, 30.0, [YELLOW, BLUE, RED][i])
    }).collect();

    // Number of physics steps per rendered frame. Higher = more accurate
    // collisions but more CPU. 8 is a good balance for this sim.
    const SUBSTEPS: u32 = 8;

    loop {
        // `get_frame_time()` returns the duration of the last frame in seconds.
        // Divide by SUBSTEPS so the total simulated time per frame is unchanged.
        let dt = get_frame_time() / SUBSTEPS as f32;

        // Physics step — run SUBSTEPS times per frame.
        for _ in 0..SUBSTEPS {
            let accels = compute_accelerations(&bodies);
            integrate(&mut bodies, &accels, dt);
            resolve_collisions(&mut bodies);
        }

        // Render.
        clear_background(BLACK);
        for body in &bodies {
            body.draw();
        }

        // Simple HUD.
        draw_text(
            &format!("bodies: {}   fps: {}", bodies.len(), get_fps()),
            10.0,
            20.0,
            20.0,
            WHITE,
        );

        // Hand control back to macroquad to present the frame.
        next_frame().await;
    }
}
