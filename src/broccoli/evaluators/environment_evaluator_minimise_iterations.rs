use crate::broccoli::{
    broccoli_helper_functions::{broccoli_greater_or_equal, broccoli_is_integer},
    environments::environment::{
        run_simulation_until_terminate_state, Environment, EnvironmentInfo,
    },
    trees::decision_tree::DecisionTree,
};

use super::evaluator::Evaluator;

pub struct EnvironmentEvaluatorMinimiseIterations<E: Environment> {
    environment: E,
    initial_states: Vec<Vec<f64>>,
    max_num_states: u32,
    num_environment_calls: usize,
}

impl<E: Environment> EnvironmentEvaluatorMinimiseIterations<E> {
    pub fn new(
        environment: E,
        initial_states: &[Vec<f64>],
        max_num_states: u32,
    ) -> EnvironmentEvaluatorMinimiseIterations<E> {
        EnvironmentEvaluatorMinimiseIterations {
            environment,
            initial_states: initial_states.to_vec(),
            max_num_states,
            num_environment_calls: 0,
        }
    }
}

impl<E: Environment> Evaluator for EnvironmentEvaluatorMinimiseIterations<E> {
    fn environment_info(&self) -> EnvironmentInfo {
        self.environment.environment_info()
    }

    fn evaluate(&mut self, decision_tree: DecisionTree) -> (DecisionTree, Result<f64, ()>) {
        let mut controller_global: Option<DecisionTree> = None;
        let mut global_score: Option<u32> = None;
        for initial_state in &self.initial_states {
            let mut controller = decision_tree.clone();
            let result = run_simulation_until_terminate_state(
                &mut self.environment,
                initial_state,
                &mut controller,
                self.max_num_states,
            );
            self.num_environment_calls += 1;
            match result {
                Ok(score) => {
                    //update the global score with respect to this simulation run
                    //  compute the maximum value amongst all runs
                    match global_score {
                        Some(global_score_value) => {
                            if score > global_score_value {
                                global_score = Some(score)
                            }
                        }
                        None => global_score = Some(score),
                    };

                    match controller_global.as_mut() {
                        Some(_) => {
                            controller_global
                                .as_mut()
                                .unwrap()
                                .merge_threshold_distances(&controller);
                        }
                        None => controller_global = Some(controller),
                    }
                }
                Err(_) => return (controller, Err(())),
            }
        }
        (controller_global.unwrap(), Ok(global_score.unwrap() as f64))
    }

    fn register_new_best_score(&mut self, new_score_upper_bound: f64) {
        assert!(broccoli_is_integer(new_score_upper_bound));
        assert!(broccoli_greater_or_equal(new_score_upper_bound, 1.0));
        assert!((new_score_upper_bound as u32 - 1) < self.max_num_states);
        self.max_num_states = new_score_upper_bound as u32 - 1;
    }

    fn num_environment_calls(&self) -> usize {
        self.num_environment_calls
    }
}
