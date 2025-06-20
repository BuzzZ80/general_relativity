use std::f32::consts::PI;

use nalgebra::{Matrix4, Vector4};

const DT: f32 = 0.0001;
const STEPSIZE: f32 = 0.001;
const MAXSTEPS: usize = 150;

const RS: f32 = 1.;
const N: usize = 5;


fn main() {
    let mut state = [
        Vector4::new(1., 1.35, -PI/4., PI/4.),
        Vector4::new(f32::sqrt(2.) / 2., -f32::sqrt(2.) / 2., -PI/2., -PI/32.)
    ];
    let mut saved_states: Vec<[Vector4<f32>;2]> = Vec::with_capacity(150);
    'outer: for _ in 0..150 {
        saved_states.push(state);
        for _ in 0..MAXSTEPS {
            if let Some(newstate) = geodesic_step(state[0], state[1]) {
                state = newstate;
            }
            else {break 'outer;}
        }
    }

    println!("l_{N} = (r_{N} cos(p_{N}) sin(a_{N}), r_{N} sin(p_{N}) sin(a_{N}), r_{N} cos(a_{N}))");
    println!("l_{N}[1]");

    print!("r_{N} = [");
    for i in 0..saved_states.len() - 1 {
        print!("{:.3}, ", saved_states[i][0].y);
    }
    println!("{:.3}]", saved_states.last().unwrap()[0].y);

    print!("p_{N} = [");
    for i in 0..saved_states.len() - 1 {
        print!("{:.3}, ", saved_states[i][0].z);
    }
    println!("{:.3}]", saved_states.last().unwrap()[0].z);

    print!("a_{N} = [");
    for i in 0..saved_states.len() - 1 {
        print!("{:.3}, ", saved_states[i][0].w);
    }
    println!("{:.3}]", saved_states.last().unwrap()[0].w);
}

fn metric(x: Vector4<f32>) -> Matrix4<f32> {
    Matrix4::new(
        -(1. - RS / x.y), 0., 0., 0.,
        0., 1. / (1. - RS / x.y), 0., 0.,
        0., 0., x.y.powi(2), 0.,
        0., 0., 0., x.y.powi(2) * x.z.powi(2)
    )
}

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

fn geodesic_step(x: Vector4<f32>, v: Vector4<f32>) -> Option<[Vector4<f32>; 2]> {
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

    let v_magnitude = v.y*v.y + x.y*x.y*(v.z*v.z + x.z.sin().powi(2) * v.w*v.w);
    let dt = STEPSIZE / v_magnitude;

    let new_v = v + dt * dvdt;
    let new_x = x + dt * v;
    Some([new_x, new_v])
}