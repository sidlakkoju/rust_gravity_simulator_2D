use macroquad::prelude::*;

// Window resolution settings
const WIDTH: i32 = 2880;
const HEIGHT: i32 = 1800;

// Gravitational constant — not the real one (6.674e-11).
// Tuned to make pixel-scale bodies behave visibly.
const G: f32 = 1000.0;

// A single gravitating body with position, velocity, mass, radius, and color.
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
// PHYSICS
// ---------------------------------------------------------------------------

/// Computes the gravitational acceleration on each body due to all others.
/// Uses Newton's law: F = G * m1 * m2 / d². A softening factor (+1.0) prevents
/// the force from blowing up when two bodies get very close.
/// Each pair is processed once and Newton's third law gives the opposite force for free.
fn compute_accelerations(bodies: &[Body]) -> Vec<Vec2> {
    let mut accels = vec![Vec2::ZERO; bodies.len()];

    for i in 0..bodies.len() {
        for j in (i + 1)..bodies.len() {
            let r = bodies[j].pos - bodies[i].pos;
            let dist_sq = r.length_squared();
            let dist = dist_sq.sqrt();

            let softened_dist_sq = dist_sq + 1.0;

            let force_mag = G * bodies[i].mass * bodies[j].mass / softened_dist_sq;
            let dir = r / dist;

            accels[i] += dir * force_mag / bodies[i].mass;
            accels[j] -= dir * force_mag / bodies[j].mass;
        }
    }

    accels
}

/// Advances each body's velocity then position by `dt` seconds (semi-implicit Euler).
/// Updating velocity before position conserves energy better than plain Euler,
/// keeping orbits stable over time.
fn integrate(bodies: &mut [Body], accels: &[Vec2], dt: f32) {
    for (body, &accel) in bodies.iter_mut().zip(accels.iter()) {
        body.vel += accel * dt;
        body.pos += body.vel * dt;
    }
}

/// Detects overlapping body pairs and resolves them as elastic collisions.
/// Phase 1 pushes the bodies apart (weighted by mass) to eliminate overlap.
/// Phase 2 applies an impulse along the collision normal to reverse their
/// relative approach velocity, conserving both momentum and kinetic energy.
fn resolve_collisions(bodies: &mut [Body]) {
    for i in 0..bodies.len() {
        for j in (i + 1)..bodies.len() {
            let (left, right) = bodies.split_at_mut(j);
            let a = &mut left[i];
            let b = &mut right[0];

            let delta = b.pos - a.pos;
            let dist = delta.length();

            let overlap = (a.radius + b.radius) - dist;
            if overlap <= 0.0 {
                continue;
            }

            let n = delta / dist;

            // Phase 1: positional correction.
            let total_mass = a.mass + b.mass;
            a.pos -= n * (overlap * b.mass / total_mass);
            b.pos += n * (overlap * a.mass / total_mass);

            // Phase 2: velocity impulse.
            let vel_along = (b.vel - a.vel).dot(n);
            if vel_along >= 0.0 {
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
    let cx = screen_width() / 2.0;
    let cy = screen_height() / 2.0;
    let spread = 100.0;

    // Place 3 bodies at the vertices of an equilateral triangle inscribed
    // in a circle of radius `spread`, each with a tangential velocity for
    // counter-clockwise orbit.
    let angle_step = 2.0 * std::f32::consts::PI / 3.0;
    let start_angle = std::f32::consts::PI / 2.0;
    let speed = 300.0;

    let mut bodies: Vec<Body> = (0..3).map(|i| {
        let angle = start_angle + i as f32 * angle_step;
        let pos = vec2(cx + spread * angle.cos(), cy - spread * angle.sin());
        let vel = vec2(-angle.sin(), -angle.cos()) * speed;
        Body::new(pos, vel, 10000.0, 30.0, [YELLOW, BLUE, RED][i])
    }).collect();

    // Run multiple physics steps per frame to reduce energy drift during
    // close encounters and collisions.
    const SUBSTEPS: u32 = 8;

    loop {
        let dt = get_frame_time() / SUBSTEPS as f32;

        for _ in 0..SUBSTEPS {
            let accels = compute_accelerations(&bodies);
            integrate(&mut bodies, &accels, dt);
            resolve_collisions(&mut bodies);
        }

        clear_background(BLACK);
        for body in &bodies {
            body.draw();
        }

        draw_text(
            &format!("bodies: {}   fps: {}", bodies.len(), get_fps()),
            10.0,
            20.0,
            20.0,
            WHITE,
        );

        next_frame().await;
    }
}
