use std::io::Write;
use std::{fs::File, path::Path};

use crate::broccoli::{
    broccoli::{Broccoli, BroccoliOutput},
    broccoli_helper_functions::{broccoli_plot, check_predicate_increments},
    environments::{
        environment::{produce_successful_traces, Environment},
        environment_pendulum::EnvironmentPendulum,
    },
    evaluators::environment_evaluator_minimise_iterations::EnvironmentEvaluatorMinimiseIterations,
};

pub fn run_pendulum(
    depth: u32,
    num_nodes: u32,
    num_simulation_iterations: u32,
    initial_states: &[Vec<f64>],
    predicate_increments: &[f64],
    use_predicate_reasoning: bool,
) {
    println!("Starting: Pendulum");

    //let initial_states: Vec<Vec<f64>> = extract_initial_states(initial_states_flattened, 2);
    check_predicate_increments(predicate_increments, 2);

    //construct supporting structs
    let evaluator = EnvironmentEvaluatorMinimiseIterations::new(
        EnvironmentPendulum::new(),
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
    plot_pendulum(b_output, initial_states);
}

fn plot_pendulum(b_output: BroccoliOutput, initial_states: &[Vec<f64>]) {
    if let Some(score) = b_output.score {
        let decision_tree = b_output.decision_tree.unwrap();
        let mut decision_tree_2 = decision_tree.clone();
        let traces = produce_successful_traces(
            &mut EnvironmentPendulum::new(),
            initial_states,
            decision_tree,
            score as u32,
        )
        .unwrap();

        for trace in traces.iter().enumerate() {
            let path = Path::new("actions").join(format!("pen_{}.txt", trace.1.len()));
            let mut output = File::create(path).unwrap();
            //print initial state
            for f in &trace.1[0] {
                let _ = write!(output, "{}", &format!("{} ", f));
            }

            let mut env = EnvironmentPendulum::new();
            env.reset(&trace.1[0]);
            //println!("{}", env.observe_state()[0]);
            //println!("{}", env.observe_state()[1]);

            //print actions
            //let mut i = 0;
            for state in trace.1 {
                let ob_state = env.observe_state();
                for i in 0..ob_state.len() {
                    if ob_state[i] != state[i] {
                        println!("{}\n{}\n{}", i, ob_state[i], state[i]);
                    }
                    assert!(ob_state[i] == state[i]);
                }

                let action = decision_tree_2.get_action(state);

                let _ = write!(output, "\n{}", &format!("{}", action));

                env.apply_action(action);

                //println!("{}, {}", i, action);
                //println!("{} {}", env.observe_state()[0], env.observe_state()[1]);
                //println!("{}", env.observe_state()[1]);

                /*if action == 1 {
                    println!("I AM HERE {}", i);
                }*/

                //i += 1;
            }
        }

        for trace in traces.iter().enumerate() {
            for f in 0..trace.1[0].len() {
                let x_values: Vec<f64> = (1..=trace.1.len()).map(|p| p as f64).collect();
                let x_name = "Time";
                let y_values: Vec<f64> = trace.1.iter().map(|v| v[f]).collect();
                let y_name = EnvironmentPendulum::new()
                    .environment_info()
                    .feature_name(f);

                broccoli_plot(
                    &x_values,
                    x_name,
                    &y_values,
                    &y_name,
                    &format!("pen_{}_{}_{}", score, trace.0, y_name),
                );
            }
        }
    } else {
        println!("No tree found.");
    }
}
