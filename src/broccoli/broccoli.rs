use crate::broccoli::broccoli_helper_functions::{
    check_predicate_increments, extract_initial_states,
};
use crate::broccoli::environments::environment::Environment;
use crate::broccoli::trees::decision_tree_enumerator::DecisionTreeEnumerator;
use rand::prelude::SmallRng;
use rand::{Rng, SeedableRng};
use std::time::Duration;

use super::{evaluators::evaluator::Evaluator, trees::decision_tree::DecisionTree};

pub struct Broccoli {}

pub struct BroccoliOutput {
    pub decision_tree: Option<DecisionTree>,
    pub score: Option<f64>,
    pub runtime: Duration,
    pub num_trees_considered: usize,
    pub num_environment_calls: usize,
}

impl BroccoliOutput {
    pub fn print_basic_stats(&self) {
        println!("Runtime: {:.2?}", self.runtime);
        println!(
            "Num explicitly considered trees: {}",
            self.num_trees_considered
        );
        println!("Num environment calls: {}", self.num_environment_calls);

        if let Some(score) = self.score {
            println!("Score: {}", score);
            println!("Tree:\n{}", self.decision_tree.clone().unwrap());
        } else {
            println!("No tree found.");
        }
    }
}

//predicate generator

impl Broccoli {
    pub fn compute_decision_tree<'a>(
        decision_tree_depth: u32,
        num_predicate_nodes: u32,
        mut evaluator: Box<dyn Evaluator + 'a>,
        predicate_increments: &[f64],
        use_predicate_reasoning: bool,
        time_limit: Duration,
    ) -> BroccoliOutput {
        assert_eq!(
            evaluator.environment_info().num_features(),
            predicate_increments.len()
        );

        println!("Starting broccoli...");

        let mut trees_with_unused_nodes = 0;

        let time_tracker = std::time::Instant::now();
        let mut enumerator = DecisionTreeEnumerator::new(
            decision_tree_depth,
            num_predicate_nodes,
            //2_u32.pow(decision_tree_depth) - 1,
            predicate_increments.to_vec(),
            evaluator.environment_info().ranges(),
            evaluator.environment_info().num_actions() as u32,
            use_predicate_reasoning,
        );

        let mut best_score: Option<f64> = None;
        let mut best_tree: Option<DecisionTree> = None;

        while let Ok(decision_tree) = enumerator.next_tree() {
            let (decision_tree_with_info, new_score) = evaluator.evaluate(decision_tree);

            if let Ok(new_value) = new_score {
                best_score = Some(new_value);
                best_tree = Some(decision_tree_with_info.clone());
                evaluator.register_new_best_score(new_value);
                println!(
                    "...{} [{} nodes] ({:.2?})",
                    new_value,
                    decision_tree_with_info.num_predicate_nodes(),
                    time_tracker.elapsed()
                );
            }

            let mut num_unused_nodes = 0;
            for i in 0..decision_tree_with_info.get_nodes().len() {
                if decision_tree_with_info.get_nodes()[i].is_predicate()
                    && decision_tree_with_info.get_frequencies()[i] == 0
                {
                    num_unused_nodes += 1;
                }
            }

            if num_unused_nodes > 0 {
                trees_with_unused_nodes += 1;
                //println!("un: {}", decision_tree_with_info);
            }

            if use_predicate_reasoning {
                enumerator.apply_threshold_reasoning(&decision_tree_with_info);
            }

            if time_tracker.elapsed() > time_limit {
                println!("Time limit reached!");
                break
            }
        }

        println!("Trees with unused nodes: {}", trees_with_unused_nodes);

        BroccoliOutput {
            decision_tree: best_tree,
            score: best_score,
            runtime: time_tracker.elapsed(),
            num_trees_considered: enumerator.num_trees_generated(),
            num_environment_calls: evaluator.num_environment_calls(),
        }
    }
}

pub fn run_solver(
    depth: u32,
    num_nodes: u32,
    num_simulation_iterations: u32,
    predicate_increments: &Vec<f64>,
    use_predicate_reasoning: bool,
    neurips_parameters: Vec<u64>,
    initial_states_flattened: &Vec<f64>,
    environment: &mut dyn Environment,
    time_limit: Duration,
) -> (Vec<Vec<f64>>, BroccoliOutput) {
    let num_state_variables = environment.environment_info().feature_ranges.len();

    let initial_states = if neurips_parameters.is_empty() {
        extract_initial_states(initial_states_flattened, num_state_variables)
    } else {
        assert_eq!(
            neurips_parameters.len(),
            2,
            "Expected two values for the NeurIPS experiments."
        );
        //depending on the environment, create a vector of state ranges
        //  the initial states will be randomly sampled within these ranges
        let state_variable_ranges = &environment.environment_info().start_ranges;
        let seed = neurips_parameters[0];
        println!("Seed: {seed}");

        let num_initial_states = neurips_parameters[1];
        println!("Num initial states: {num_initial_states}");

        let mut random_generator = SmallRng::seed_from_u64(seed);

        let mut initial_states: Vec<Vec<f64>> = vec![];
        for _i in 0..num_initial_states {
            let mut state: Vec<f64> = vec![];
            for variable_range in state_variable_ranges {
                let mut val = random_generator.gen_range(variable_range.min..=variable_range.max);

                //rounding to two decimal places for simplicity
                val *= 1000.0;
                val = ((val as i64) as f64) / 1000.0;

                state.push(val);
            }
            initial_states.push(state);
        }
        initial_states
    };

    if initial_states.len() <= 100 {
        for state in initial_states.iter().enumerate() {
            println!("State {}: {:?}", state.0, state.1);
        }
    }

    println!("Starting: {}", environment.name());

    check_predicate_increments(predicate_increments, num_state_variables);

    //construct supporting structs
    let evaluator = environment.evaluator(&initial_states, num_simulation_iterations);

    //run the main tree algorithm
    let b_output = Broccoli::compute_decision_tree(
        depth,
        num_nodes,
        evaluator,
        predicate_increments,
        use_predicate_reasoning,
        time_limit,
    );

    //process output
    b_output.print_basic_stats();
    (initial_states, b_output)
}
