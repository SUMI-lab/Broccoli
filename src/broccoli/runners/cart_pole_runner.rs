use std::io::Write;
use std::{fs::File, path::Path};

use crate::broccoli::{
    broccoli::BroccoliOutput,
    broccoli_helper_functions::broccoli_plot,
    environments::{
        environment::{produce_traces, Environment},
        environment_cartpole::EnvironmentCartPole,
    },
};

pub fn plot_cart_pole(
    b_output: BroccoliOutput,
    initial_states: &[Vec<f64>],
    num_simulation_iterations: u32,
) {
    let mut cartpole_env = EnvironmentCartPole::new();
    assert!(b_output.score.is_some());

    let decision_tree = b_output.decision_tree.unwrap();
    let mut decision_tree_2 = decision_tree.clone();
    let score = b_output.score.unwrap();
    let traces = produce_traces(
        &mut cartpole_env,
        initial_states,
        decision_tree,
        num_simulation_iterations,
    );

    for trace in traces.iter().enumerate() {
        let path = Path::new("actions").join(format!("cp_{}.txt", trace.1.len()));
        let mut output = File::create(path).unwrap();
        //print initial state
        for f in &trace.1[0] {
            let _ = write!(output, "{}", &format!("{} ", f));
        }

        let mut env = EnvironmentCartPole::new();
        env.reset(&trace.1[0]);

        //print actions
        for state in trace.1 {
            let ob_state = env.observe_state();
            for i in 0..ob_state.len() {
                if ob_state[i] != state[i] {
                    println!("{}\n{}\n{}", i, ob_state[i], state[i]);
                }
                assert_eq!(ob_state[i], state[i]);
            }

            let action = decision_tree_2.get_action(state);

            let _ = write!(output, "\n{}", &format!("{}", action));

            env.apply_action(action);
        }
    }

    for trace in traces.iter().enumerate() {
        for f in 0..trace.1[0].len() {
            let x_values: Vec<f64> = (1..=trace.1.len()).map(|p| p as f64).collect();
            let x_name = "Time";
            let y_values: Vec<f64> = trace.1.iter().map(|v| v[f]).collect();
            let y_name = cartpole_env.environment_info().feature_name(f);

            broccoli_plot(
                &x_values,
                x_name,
                &y_values,
                &y_name,
                &format!("cp_{}_{}_{}", score, trace.0, y_name),
            );
        }
    }
}
