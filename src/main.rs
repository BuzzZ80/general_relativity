use std::f32::consts::PI;

use nalgebra::{Matrix4, Vector4};

const DT: f32 = 0.0001;
const STEPSIZE: f32 = 0.0001;
const MAXSTEPS: usize = 1500;
const NUM_PATHS: i32 = 10;

const RS: f32 = 1.;

const DATAPOINTS: usize = 50;


fn main() {
    println!("r = \\sqrt{{{:.3} - z^2}}", RS.powi(2)); // for desmos3d visualization
    // loop through multiple trajectories
    for n in 0..NUM_PATHS {
        // Initial conditions go into the integration state variable
        let mut state = [
            Vector4::new(1., 3., PI/2., PI / 2.),
            Vector4::new(f32::sqrt(2.) / 2., -f32::sqrt(2.) / 2., (n as f32 - 5.) / 10., 0.)
        ];

        // Save the states until the end so they can be printed in a convenient format
        let mut saved_states: Vec<[Vector4<f32>;2]> = Vec::with_capacity(DATAPOINTS);
        'outer: for _ in 0..DATAPOINTS {
            saved_states.push(state); // save this step

            // compute many small time steps for one data point
            for _ in 0..MAXSTEPS {
                if let Some(newstate) = geodesic_step(state[0], state[1]) {
                    state = newstate;
                }
                // exit if integration could not continue
                else {break 'outer;}
            }
        }

        // output data in Desmos3D readable format
        println!("l_{n} = (r_{n} \\cos(p_{n}) \\sin(a_{n}), r_{n} \\sin(p_{n}) \\sin(a_{n}), r_{n} \\cos(a_{n}))");
        println!("l_{n}[1]");

        print!("r_{n} = [");
        for i in 0..saved_states.len() - 1 {
            print!("{:.3}, ", saved_states[i][0].y);
        }
        println!("{:.3}]", saved_states.last().unwrap()[0].y);

        print!("p_{n} = [");
        for i in 0..saved_states.len() - 1 {
            print!("{:.3}, ", saved_states[i][0].z);
        }
        println!("{:.3}]", saved_states.last().unwrap()[0].z);

        print!("a_{n} = [");
        for i in 0..saved_states.len() - 1 {
            print!("{:.3}, ", saved_states[i][0].w);
        }
        println!("{:.3}]", saved_states.last().unwrap()[0].w);
    }
}

// Spacetime metric
fn metric(x: Vector4<f32>) -> Matrix4<f32> {
    Matrix4::new(
        -(1. - RS / x.y), 0., 0., 0.,
        0., 1. / (1. - RS / x.y), 0., 0.,
        0., 0., x.y.powi(2), 0.,
        0., 0., 0., x.y.powi(2) * x.z.sin().powi(2)
    )
}

// Christoffel symbols, used to find geodesics in curved spacetime
fn christoffel(x: Vector4<f32>, index: [usize;3]) -> Option<f32> {
    let g = metric(x);
    let g_inv = g.try_inverse()?;

    let dg = [
        (metric(x + DT * Vector4::new(1., 0., 0., 0.)) - g) / DT,
        (metric(x + DT * Vector4::new(0., 1., 0., 0.)) - g) / DT,
        (metric(x + DT * Vector4::new(0., 0., 1., 0.)) - g) / DT,
        (metric(x + DT * Vector4::new(0., 0., 0., 1.)) - g) / DT,
    ];

    Some(0.5 * (0..=3).map(|a| {
        let derivatives = dg[index[1]].get((a, index[2])).unwrap()
            + dg[index[2]].get((a, index[1])).unwrap()
            - dg[a].get((index[1], index[2])).unwrap();
        
        g_inv.get((index[0], a)).unwrap() * derivatives
    }).sum::<f32>())
}

// One integration step, using euler's method, but normalizing the output so that steps taken are not too big or too small
fn geodesic_step(x: Vector4<f32>, v: Vector4<f32>) -> Option<[Vector4<f32>; 2]> {
    // imagine writing loops :3
    let dvdt = -Vector4::new(
        christoffel(x, [0, 0, 0])? * v.x * v.x 
            + 2. * christoffel(x, [0, 0, 1])? * v.x * v.y
            + 2. * christoffel(x, [0, 0, 2])? * v.x * v.z
            + 2. * christoffel(x, [0, 0, 3])? * v.x * v.w
            + christoffel(x, [0, 1, 1])? * v.y * v.y
            + 2. * christoffel(x, [0, 1, 2])? * v.y * v.z
            + 2. * christoffel(x, [0, 1, 3])? * v.y * v.w
            + christoffel(x, [0, 2, 2])? * v.z * v.z
            + 2. * christoffel(x, [0, 2, 3])? * v.z * v.w
            + christoffel(x, [0, 3, 3])? * v.w * v.w,
        christoffel(x, [1, 0, 0])? * v.x * v.x 
            + 2. * christoffel(x, [1, 0, 1])? * v.x * v.y
            + 2. * christoffel(x, [1, 0, 2])? * v.x * v.z
            + 2. * christoffel(x, [1, 0, 3])? * v.x * v.w
            + christoffel(x, [1, 1, 1])? * v.y * v.y
            + 2. * christoffel(x, [1, 1, 2])? * v.y * v.z
            + 2. * christoffel(x, [1, 1, 3])? * v.y * v.w
            + christoffel(x, [1, 2, 2])? * v.z * v.z
            + 2. * christoffel(x, [1, 2, 3])? * v.z * v.w
            + christoffel(x, [1, 3, 3])? * v.w * v.w,
        christoffel(x, [2, 0, 0])? * v.x * v.x 
            + 2. * christoffel(x, [2, 0, 1])? * v.x * v.y
            + 2. * christoffel(x, [2, 0, 2])? * v.x * v.z
            + 2. * christoffel(x, [2, 0, 3])? * v.x * v.w
            + christoffel(x, [2, 1, 1])? * v.y * v.y
            + 2. * christoffel(x, [2, 1, 2])? * v.y * v.z
            + 2. * christoffel(x, [2, 1, 3])? * v.y * v.w
            + christoffel(x, [2, 2, 2])? * v.z * v.z
            + 2. * christoffel(x, [2, 2, 3])? * v.z * v.w
            + christoffel(x, [2, 3, 3])? * v.w * v.w,
        christoffel(x, [3, 0, 0])? * v.x * v.x 
            + 2. * christoffel(x, [3, 0, 1])? * v.x * v.y
            + 2. * christoffel(x, [3, 0, 2])? * v.x * v.z
            + 2. * christoffel(x, [3, 0, 3])? * v.x * v.w
            + christoffel(x, [3, 1, 1])? * v.y * v.y
            + 2. * christoffel(x, [3, 1, 2])? * v.y * v.z
            + 2. * christoffel(x, [3, 1, 3])? * v.y * v.w
            + christoffel(x, [3, 2, 2])? * v.z * v.z
            + 2. * christoffel(x, [3, 2, 3])? * v.z * v.w
            + christoffel(x, [3, 3, 3])? * v.w * v.w,
    );

    // normalize so space steps are a specific size
    let v_space_magnitude = v.y*v.y / (1. - RS / x.y) + x.y*x.y*(v.z*v.z + x.z.sin().powi(2) * v.w*v.w);
    let dt = STEPSIZE / v_space_magnitude;

    let new_v = v + dt * dvdt;
    let new_x = x + dt * v;

    if new_x.magnitude().is_nan() || new_v.magnitude().is_nan() {
        None
    } else {
        Some([new_x, new_v])
    }
}