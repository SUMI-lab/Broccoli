use crate::broccoli::{
    broccoli::BroccoliOutput,
    broccoli_helper_functions::broccoli_plot,
    environments::{
        environment::produce_successful_traces, environment_mountain_car::EnvironmentMountainCar,
    }
    ,
};

use crate::broccoli::environments::environment::Environment;

pub fn plot_mountain_car(b_output: BroccoliOutput, initial_states: &[Vec<f64>]) {
    let mut mountain_car_env = EnvironmentMountainCar::new();
    if let Some(score) = b_output.score {
        let decision_tree = b_output.decision_tree.unwrap();
        let traces = produce_successful_traces(
            &mut mountain_car_env,
            initial_states,
            decision_tree,
            score as u32,
        );
        for trace in traces.unwrap().iter().enumerate() {
            for f in 0..trace.1[0].len() {
                let x_values: Vec<f64> = (1..=trace.1.len()).map(|p| p as f64).collect();
                let x_name = "Time";
                let y_values: Vec<f64> = trace.1.iter().map(|v| v[f]).collect();
                let y_name = mountain_car_env.environment_info().feature_name(f);

                broccoli_plot(
                    &x_values,
                    x_name,
                    &y_values,
                    &y_name,
                    &format!("mc_{}_{}_{}", score, trace.0, y_name),
                );
            }
        }
    } else {
        println!("No tree found.");
    }
}
