///UnimplementedVariant { variant:
/// "broadcasting for two stacks of matrixes
/// left side has shape 1x512x384:f32,
/// right side has shape 384x384:f32)
///  op: "MatMul"
use env_logger::Logger;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::{collections::HashMap, vec};
use wonnx::utils::InputTensor;

use wonnx::{
    utils::{graph, model, node, tensor, OutputTensor},
    SessionError, WonnxError,
};

async fn run() -> Result<(), WonnxError> {
    let (m, n, k) = (384, 384, 384);
    let x_data = ndarray::Array2::<f32>::random((m, k), Uniform::new(0f32, 1f32));
    let y_data = ndarray::Array2::<f32>::random((k, n), Uniform::new(0f32, 1f32));
    // Compute using the dot product
    let expected = x_data.dot(&y_data).as_slice().unwrap().to_owned();

    let result = execute_gpu_matmul(
        x_data.as_slice().unwrap().into(),
        y_data.as_slice().unwrap().into(),
        m as i64,
        n as i64,
        k as i64,
    )
    .await?;
    // let result = result.into_iter().next().unwrap().1;
    let z = result.get("Z").unwrap();
    match z {
        OutputTensor::F32(v) => {
            let diff: Vec<f32> = v.iter().zip(expected.iter()).map(|(l, r)| l - r).collect();
            let diff = diff
                .iter()
                .filter(|e| f32::abs(**e) > 1e-3f32)
                .collect::<Vec<_>>();
            if diff.len() > 0 {
                eprintln!(
                    "Percentage different elements: {:.3}%. ",
                    diff.len() as f32 / x_data.len() as f32,
                )
            }
        }
        _ => panic!("can't have another type"),
    }

    Ok(())
}
async fn execute_gpu_matmul<'a>(
    x_data: InputTensor<'a>,
    y_data: InputTensor<'a>,
    m: i64,
    n: i64,
    k: i64,
) -> Result<HashMap<String, OutputTensor>, SessionError> {
    let shape_x = [1, m, k];
    let shape_y = [k, n];
    let shape_z = [1, m as i64, n as i64];

    // Input Map
    let mut input_data = HashMap::new();
    input_data.insert("X".to_string(), x_data);
    input_data.insert("Y".to_string(), y_data);

    // Model graph
    log::info!("Creating model graph");
    let model = model(graph(
        vec![tensor("X", &shape_x), tensor("Y", &shape_y)],
        vec![tensor("Z", &shape_z)],
        vec![],
        vec![],
        vec![node(vec!["X", "Y"], vec!["Z"], "matmul", "MatMul", vec![])],
    ));

    let session = wonnx::Session::from_model(model)
        .await
        .expect("Session did not create");

    session.run(&input_data).await
}

fn main() {
    env_logger::init();
    let _ = Logger::from_default_env();
    log::info!("Running matmul !");

    pollster::block_on(run()).unwrap();
}
