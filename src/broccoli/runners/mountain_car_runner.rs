use crate::broccoli::{
    broccoli::{Broccoli, BroccoliOutput},
    broccoli_helper_functions::{broccoli_plot, check_predicate_increments},
    environments::{
        environment::produce_successful_traces, environment_mountain_car::EnvironmentMountainCar,
    },
    evaluators::environment_evaluator_minimise_iterations::EnvironmentEvaluatorMinimiseIterations,
};

use crate::broccoli::environments::environment::Environment;

pub fn run_mountain_car(
    depth: u32,
    num_nodes: u32,
    num_simulation_iterations: u32,
    initial_states: &[Vec<f64>],
    predicate_increments: &[f64],
    use_predicate_reasoning: bool,
) {
    println!("Starting: Mountain Car");

    //let initial_states: Vec<Vec<f64>> = extract_initial_states(initial_states_flattened, 2);
    check_predicate_increments(predicate_increments, 2);

    //construct supporting structs
    let evaluator = EnvironmentEvaluatorMinimiseIterations::new(
        EnvironmentMountainCar::new(),
        initial_states,
        num_simulation_iterations,
    );

    //run the main tree algorithm
    let b_output = Broccoli::compute_decision_tree(
        depth,
        num_nodes,
        Box::new(evaluator),
        predicate_increments,
        use_predicate_reasoning,
    );

    //process output
    b_output.print_basic_stats();
    plot_mountain_car(b_output, initial_states);
}

fn plot_mountain_car(b_output: BroccoliOutput, initial_states: &[Vec<f64>]) {
    if let Some(score) = b_output.score {
        let decision_tree = b_output.decision_tree.unwrap();
        let traces = produce_successful_traces(
            &mut EnvironmentMountainCar::new(),
            initial_states,
            decision_tree,
            score as u32,
        );
        for trace in traces.unwrap().iter().enumerate() {
            for f in 0..trace.1[0].len() {
                let x_values: Vec<f64> = (1..=trace.1.len()).map(|p| p as f64).collect();
                let x_name = "Time";
                let y_values: Vec<f64> = trace.1.iter().map(|v| v[f]).collect();
                let y_name = EnvironmentMountainCar::new()
                    .environment_info()
                    .feature_name(f);

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
